"""Loop-level compaction hardening (ADR-0022): overflow-proof summarization, the
post-compact ``SessionStart(source="compact")`` event, and honest PreCompact triggers.

Field incident 2026-08-26 (131k local pod): an uncapped tool result pushed the session
past the window mid-turn; the reactive recovery compacted and retried — but the
summarize call itself carried the oversized history in one request (an overflow risk on
the very path that exists to fix overflows), the PreCompact payload said ``manual`` for
an automatic recovery, and no post-compact event existed for a framework to restore
serialized state (Claude Code fires ``SessionStart(source="compact")``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.compact import ELISION_MARKER, CompactionConfig, Compactor
from zakcode.agent.loop import AgentLoop
from zakcode.hooks import HookEvent, LifecyclePayload
from zakcode.messages import Message, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, RateLimited
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry


class _SummarizerProvider(Provider):
    """Canned completions; records every acomplete's messages; scriptable token counts."""

    def __init__(self, texts: list[str], *, tokens: int, window: int = 8192) -> None:
        self._texts = texts
        self._tokens = tokens
        self._window = window
        self.seen: list[list[Message]] = []

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.seen.append(list(messages))
        return LLMResult(text=self._texts[min(len(self.seen), len(self._texts)) - 1])

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return self._tokens

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=self._window)


def _loop(provider: Provider, tmp_path: Path, *, compactor: Compactor | None = None) -> AgentLoop:
    return AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        compactor=compactor,
    )


def _history(n: int) -> list[Message]:
    out: list[Message] = []
    for i in range(n):
        out.append(Message.user(f"question {i} " + "x" * 400))
        out.append(Message.assistant_text(f"answer {i} " + "y" * 400))
    return out


def test_summarize_sends_the_rendered_transcript_as_one_user_message(tmp_path: Path) -> None:
    # ADR-0082: never the raw role-tagged messages — a small model handed those continues
    # the dialogue instead of summarizing it (measured 2026-08-29, a 27B reducer).
    provider = _SummarizerProvider(["the summary"], tokens=100)
    loop = _loop(provider, tmp_path)
    history = _history(3)
    text = asyncio.run(loop._summarize_for_compaction(history))
    assert text == "the summary"
    assert len(provider.seen) == 1
    (sent,) = provider.seen[0]
    assert sent.role == "user"
    assert sent.text.startswith("Conversation transcript to summarize")
    assert "[user]\nquestion 0" in sent.text and "[assistant]\nanswer 2" in sent.text


def test_summary_drops_the_model_s_tool_call_and_thinking_markup(tmp_path: Path) -> None:
    # The field summary: the model's own last reply, then a text-format tool call.
    leaked = (
        "<think>should I summarize?</think>Phase 3 complete. Loaded 2 tree nodes.\n"
        '<tool_call>\n<function=update_plan>\n<parameter=tasks>\n[{"title": "Step 0"}]\n'
        "</parameter>\n</function>\n</tool_call>\nUnfinished: the aspirations loop."
    )
    provider = _SummarizerProvider([leaked], tokens=100)
    loop = _loop(provider, tmp_path)
    text = asyncio.run(loop._summarize_for_compaction(_history(2)))
    assert text == "Phase 3 complete. Loaded 2 tree nodes.\n\nUnfinished: the aspirations loop."


def test_summary_carries_the_harness_position_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zakcode import tasks
    from zakcode.tasks import skill_pages, skill_skeleton

    # Sections this small would pack into one page (ADR-0088); a 100-char budget keeps
    # the three-page shape the position note is about.
    monkeypatch.setattr(tasks, "PAGE_BUDGET_CHARS", 100)
    a, b, c = "alpha " * 10, "beta " * 12, "gamma " * 10
    body = (
        f"# Boot\n\nintro\n\n## Step 1: Alpha\n\n{a}\n\n## Step 2: Beta\n\n{b}\n\n"
        f"## Step 3: Gamma\n\n{c}\n"
    )
    provider = _SummarizerProvider(["the summary"], tokens=100)
    loop = _loop(provider, tmp_path)
    steps = skill_skeleton(body, skill="boot")
    loop.session.task_network.insert_before(None, steps)
    steps[0].status = "done"
    loop._skill_pages["boot"] = skill_pages(body, skill="boot")
    loop._skill_pages_delivered["boot"] = {1, 2}

    text = asyncio.run(loop._summarize_for_compaction(_history(2)))

    assert text.startswith("the summary\n\nHarness position")
    assert 'current step "Step 2: Beta" (1 of 3 steps closed)' in text
    assert "/boot: on section 2 of 3 (Step 2: Beta)" in text
    assert "do not re-load the skill" in text


def test_position_note_is_empty_without_a_plan_or_pages(tmp_path: Path) -> None:
    loop = _loop(_SummarizerProvider(["s"], tokens=100), tmp_path)
    assert loop._compaction_position_note() == ""


class _BusyThenSummarizing(_SummarizerProvider):
    """Rate-limited for the first ``busy`` calls, then a canned summary — a pod with
    every slot taken (five agents on four engines, coach 2026-08-29)."""

    def __init__(self, busy: int) -> None:
        super().__init__(["the summary"], tokens=100)
        self._busy = busy

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        if len(self.seen) < self._busy:
            self.seen.append(list(messages))
            raise RateLimited("qwen35a-gpu2 busy", retry_after=0.0)
        return await super().acomplete(messages, system=system, tools=tools, **kw)


def test_the_summarizer_waits_out_a_rate_limit_like_the_main_call(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # ADR-0083: the summarizer used to call the provider directly, so the first 429 of a
    # busy pod failed the compaction outright ("summarizer failed (RateLimited: …)").
    slept: list[float] = []

    async def no_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr("zakcode.agent.loop.asyncio.sleep", no_sleep)
    provider = _BusyThenSummarizing(busy=2)
    loop = _loop(provider, tmp_path)
    text = asyncio.run(loop._summarize_for_compaction(_history(2)))
    assert text == "the summary"
    assert len(provider.seen) == 3 and len(slept) == 2


def test_summarize_chunks_an_oversized_history(tmp_path: Path) -> None:
    # count_tokens says the history dwarfs the 8192 window, so the raw single call is
    # off the table; the rendered text (~24k chars) splits into 8192-char slices.
    provider = _SummarizerProvider(["part summary"], tokens=100_000)
    loop = _loop(provider, tmp_path)
    text = asyncio.run(loop._summarize_for_compaction(_history(28)))
    assert len(provider.seen) >= 2
    for call in provider.seen:
        assert len(call) == 1  # each slice travels as one plain user message
        assert "Part " in call[0].text
    assert "part summary" in text


def test_summarize_folds_long_part_summaries(tmp_path: Path) -> None:
    # Each part summary is near the slice budget, so the joined parts exceed it and a
    # final fold call produces the single summary.
    provider = _SummarizerProvider(["z" * 9000, "z" * 9000, "the folded summary"], tokens=100_000)
    loop = _loop(provider, tmp_path)
    text = asyncio.run(loop._summarize_for_compaction(_history(28)))
    assert text == "the folded summary"
    assert "Fold these part-summaries" in provider.seen[-1][0].text


def _lifecycle_recorder(loop: AgentLoop) -> list[LifecyclePayload]:
    captured: list[LifecyclePayload] = []

    async def record(payload: LifecyclePayload) -> None:
        captured.append(payload)

    for event in (HookEvent.PRE_COMPACT, HookEvent.SESSION_START):
        loop.hook_manager.register_lifecycle(event, record)
    return captured


def test_auto_compact_fires_pre_compact_then_session_start_compact(tmp_path: Path) -> None:
    provider = _SummarizerProvider(["summary"], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(_history(5))  # 10 messages; preserve_recent=6 keeps 6
    captured = _lifecycle_recorder(loop)

    notice = asyncio.run(loop._maybe_compact())

    events = [(p.event, p.trigger, p.source) for p in captured]
    assert events == [
        (HookEvent.PRE_COMPACT, "auto", ""),
        (HookEvent.SESSION_START, "", "compact"),
    ]
    assert notice == "context near the window — compacted 10 → 7 messages"


def test_compact_now_labels_the_recovery_trigger_auto(tmp_path: Path) -> None:
    provider = _SummarizerProvider(["summary"], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(_history(5))
    captured = _lifecycle_recorder(loop)

    assert asyncio.run(loop.compact_now(trigger="auto")) is True

    pre = [p for p in captured if p.event is HookEvent.PRE_COMPACT]
    post = [p for p in captured if p.event is HookEvent.SESSION_START]
    assert pre and pre[0].trigger == "auto"
    assert post and post[0].source == "compact"


def test_compact_now_default_stays_manual(tmp_path: Path) -> None:
    provider = _SummarizerProvider(["summary"], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(_history(5))
    captured = _lifecycle_recorder(loop)

    assert asyncio.run(loop.compact_now()) is True
    pre = [p for p in captured if p.event is HookEvent.PRE_COMPACT]
    assert pre and pre[0].trigger == "manual"


class _ExplodingProvider(_SummarizerProvider):
    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        raise RuntimeError("summarizer down")


def test_compact_now_reports_false_when_summarization_fails(tmp_path: Path) -> None:
    provider = _ExplodingProvider([], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(_history(5))
    before = list(loop.session.messages)

    assert asyncio.run(loop.compact_now(trigger="auto")) is False
    assert loop.session.messages == before  # history untouched on failure
    # ADR-0083: the failure is named, never swallowed into a bare False.
    assert loop.last_compaction == (
        "summarizer failed (RuntimeError: summarizer down) and no long tool output to elide"
    )


def _tool_pair(tool_id: str, output: str) -> list[Message]:
    return [
        Message(role="assistant", blocks=[ToolUseBlock(id=tool_id, name="read", input={})]),
        Message.tool_results([ToolResultBlock(tool_use_id=tool_id, output=output)]),
    ]


def _output(message: Message) -> str:
    block = message.blocks[0]
    assert isinstance(block, ToolResultBlock)
    return block.output


def test_compact_now_elides_old_tool_outputs_when_the_summarizer_fails(tmp_path: Path) -> None:
    # ADR-0083: the summarizer is a model call and can fail (a busy pod, a provider error,
    # an overflow of its own); the transcript still shrinks, and the failure is named.
    provider = _ExplodingProvider([], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend([*_history(3), *_tool_pair("t1", "x" * 5000), *_history(3)])

    assert asyncio.run(loop.compact_now(trigger="auto")) is True
    assert "summarizer failed (RuntimeError: summarizer down)" in loop.last_compaction
    assert (
        "elided 1 long tool output(s) instead — compacted 14 → 14 messages" in loop.last_compaction
    )
    assert _output(loop.session.messages[7]).startswith(ELISION_MARKER)
    assert len(loop.session.messages) == 14  # nothing summarized away, nothing dropped


def test_elide_now_reaches_the_tail(tmp_path: Path) -> None:
    # The shape that killed a worker Body (coach, 2026-08-29): the LAST tool result was an
    # 87 KB skill load, nothing was old enough to summarize, and every retry re-overflowed.
    provider = _SummarizerProvider(["summary"], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(
        [Message.user("load the skill"), *_tool_pair("t1", "body " * 20_000)]
    )
    captured = _lifecycle_recorder(loop)

    assert asyncio.run(loop.compact_now(trigger="auto")) is False
    assert loop.last_compaction == "nothing old enough to compact"
    assert asyncio.run(loop.elide_now(trigger="auto")) is True
    assert loop.last_compaction == "elided 1 long tool output(s) across all 3 messages"
    assert _output(loop.session.messages[2]).startswith(ELISION_MARKER)
    events = [(p.event, p.trigger, p.source) for p in captured]
    assert events[-2:] == [
        (HookEvent.PRE_COMPACT, "auto", ""),
        (HookEvent.SESSION_START, "", "compact"),
    ]


def test_maybe_compact_says_when_compaction_failed(tmp_path: Path) -> None:
    # The turn-start check used to log a warning to a logger with no handler and return
    # None — indistinguishable from "not needed" to anyone watching the session.
    provider = _ExplodingProvider([], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(_history(5))
    notice = asyncio.run(loop._maybe_compact())
    assert notice == (
        "compaction failed — summarizer failed (RuntimeError: summarizer down) and no long "
        "tool output to elide; continuing with the full history"
    )

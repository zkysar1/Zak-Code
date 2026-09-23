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
from zakcode.agent.loop import (
    _MAX_FOLD_PASSES,
    _MAX_SUMMARY_SLICES,
    _SUMMARY_OUTPUT_CHARS,
    AgentLoop,
    _clamp_middle,
    _pack_parts,
)
from zakcode.hooks import HookEvent, LifecyclePayload
from zakcode.messages import Message, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import (
    Capabilities,
    ContextWindowExceeded,
    LLMResult,
    Provider,
    RateLimited,
)
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


def test_a_streaming_turn_says_the_summarizer_is_waiting_on_a_rate_limit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """ADR-0100. The summarizer's retry loop returns a value, so a streaming turn used
    to show NOTHING for the whole wait — measured 2026-08-29 (coach reducer): 17 minutes
    with no socket, no event and no line, before "summarizer failed" printed. The
    streaming path now runs the compaction as a task and relays each retry notice as an
    AgentStatus WHILE the task is pending — before the compaction outcome, not after."""
    seen_pending: list[bool] = []
    # The backoff sleep becomes a handshake: each retry blocks until the consumer has
    # SEEN its notice. A no-op sleep would let the whole retry loop finish inside one
    # scheduling slice — a timeline production never has (its sleeps are seconds long)
    # — and the test could not tell "relayed live" from "dumped at the end".
    consumer_saw_it = asyncio.Event()

    async def handshake_sleep(delay: float) -> None:
        consumer_saw_it.clear()
        await consumer_saw_it.wait()

    monkeypatch.setattr("zakcode.agent.loop.asyncio.sleep", handshake_sleep)
    provider = _BusyThenSummarizing(busy=2)
    provider._tokens = 100_000  # over the 8192 window -> _maybe_compact runs the summarizer
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(_history(5))

    async def drive() -> tuple[list[str], str | None]:
        task = asyncio.ensure_future(loop._maybe_compact())
        notices: list[str] = []
        async for status in loop._statuses_until(task):
            seen_pending.append(not task.done())
            notices.append(status.message)
            consumer_saw_it.set()
        return notices, task.result()

    notices, note = asyncio.run(drive())
    assert len(notices) == 2 and all(
        n.startswith("provider rate-limited; retrying in") for n in notices
    )
    assert "into the 900s backoff budget" in notices[0]
    assert seen_pending == [True, True], "notices must arrive while the compaction is still running"
    assert note == "context near the window — compacted 10 → 7 messages"
    assert loop._status_sink is None, "the sink is scoped to the wait"


def test_statuses_until_cancels_a_compaction_the_turn_abandoned(tmp_path: Path) -> None:
    """A consumer that stops iterating (the turn was cancelled) must not leave the
    summarizer running against the transcript of a turn that no longer exists."""
    started = asyncio.Event()

    async def hangs() -> str:
        started.set()
        await asyncio.sleep(3600)
        return "never"

    loop = _loop(_SummarizerProvider(["s"], tokens=100), tmp_path)

    async def drive() -> bool:
        task = asyncio.ensure_future(hangs())
        gen = loop._statuses_until(task)
        pull = asyncio.ensure_future(gen.__anext__())
        await started.wait()
        pull.cancel()
        await asyncio.gather(pull, return_exceptions=True)
        await gen.aclose()
        await asyncio.sleep(0)
        return task.cancelled()

    assert asyncio.run(drive()) is True
    assert loop._status_sink is None


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


#: One slice at an 8192-token window: max(4096, int(8192 * 0.5) * 2) characters.
_SLICE = 8192
#: Prompt overhead a slice-sized call may add ("Part i of N …", the fold instruction).
_OVERHEAD = 128


class _StrictWindowProvider(_SummarizerProvider):
    """A summarizer that refuses any request over ``limit`` characters — what a window does."""

    def __init__(self, texts: list[str], *, tokens: int, window: int, limit: int) -> None:
        super().__init__(texts, tokens=tokens, window=window)
        self.limit = limit

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        if len(messages[0].text) > self.limit:
            raise ContextWindowExceeded(
                f"request ({len(messages[0].text)} chars) exceeds {self.limit}"
            )
        return await super().acomplete(messages, system=system, tools=tools, **kw)


class _ShrinkingProvider(_StrictWindowProvider):
    """Every slice summarizes to 5000 chars; each fold shrinks its input: z → y (3000), y → w
    (1000) — so the fold passes converge the way a real summarizer's do."""

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        text = messages[0].text
        if len(text) > self.limit:
            raise ContextWindowExceeded(f"request ({len(text)} chars) exceeds {self.limit}")
        self.seen.append(list(messages))
        if text.startswith("Part "):
            return LLMResult(text="z" * 5000)
        if "zzzz" in text:
            return LLMResult(text="y" * 3000)
        return LLMResult(text="w" * 1000)


def test_summarize_folds_long_part_summaries(tmp_path: Path) -> None:
    # Each part summary is over the slice budget, so the joined parts exceed it and the fold
    # produces the summary — in ADR-0183's packed form: a part over the budget is clamped and
    # every fold call fits one slice (the strict provider refuses one that does not).
    provider = _StrictWindowProvider(
        ["z" * 9000, "z" * 9000, "the folded summary"],
        tokens=100_000,
        window=8192,
        limit=_SLICE + _OVERHEAD,
    )
    loop = _loop(provider, tmp_path)
    text = asyncio.run(loop._summarize_for_compaction(_history(28)))
    assert text.startswith("the folded summary")
    assert any("Fold these part-summaries" in call[0].text for call in provider.seen)


def test_summarize_holds_an_enormous_history_to_the_slice_cap(tmp_path: Path) -> None:
    # ADR-0183: a session built on a large-window model resumed on an 8k local one used to cost
    # ceil(len / slice) summarize calls. The transcript is held to _MAX_SUMMARY_SLICES slices:
    # the first 2/3 and the last 1/3 kept around a note naming the elided middle, so both ends
    # of the conversation reach the summarizer and the note lands in the ninth slice.
    provider = _StrictWindowProvider(
        ["part summary"], tokens=10**6, window=8192, limit=_SLICE + _OVERHEAD
    )
    loop = _loop(provider, tmp_path)
    history = _history(160)
    assert len(loop._render_for_summary(history)) > _MAX_SUMMARY_SLICES * _SLICE  # the premise
    text = asyncio.run(loop._summarize_for_compaction(history))
    assert len(provider.seen) == _MAX_SUMMARY_SLICES
    assert "question 0 " in provider.seen[0][0].text
    assert "answer 159 " in provider.seen[-1][0].text
    assert [i for i, call in enumerate(provider.seen) if "transcript elided" in call[0].text] == [8]
    assert "part summary" in text


def test_summarize_fold_calls_never_exceed_one_slice(tmp_path: Path) -> None:
    # ADR-0183: the fold used to send every part-summary joined in ONE message with no size
    # check — the one request on the recovery path that could itself overflow. A join over one
    # slice budget is now folded in packed groups, each under the budget, pass by pass.
    provider = _ShrinkingProvider([], tokens=10**6, window=8192, limit=_SLICE + _OVERHEAD)
    loop = _loop(provider, tmp_path)
    text = asyncio.run(loop._summarize_for_compaction(_history(28)))
    n = sum(call[0].text.startswith("Part ") for call in provider.seen)
    assert n >= 2
    folds = [call[0].text for call in provider.seen[n:]]
    assert all(t.startswith("Fold these part-summaries") for t in folds)
    # pass 1: 5000-char parts cannot pair under 8192, one group each; pass 2: 3000-char parts
    # pack two to a group.
    assert [t.count("z" * 5000) for t in folds[:n]] == [1] * n
    assert [t.count("y" * 3000) for t in folds[n:]] == [2] * (n // 2) + [1] * (n % 2)
    assert text == "\n\n".join(["w" * 1000] * ((n + 1) // 2))


def test_summarize_clamps_when_the_fold_passes_are_spent(tmp_path: Path) -> None:
    # ADR-0183: a summarizer whose "summaries" never shrink (7000 chars back for every call)
    # cannot loop the fold: after _MAX_FOLD_PASSES the join is clamped — head, tail, a note
    # naming the loss — for one last fold that fits the slice budget.
    provider = _StrictWindowProvider(
        ["z" * 7000], tokens=10**6, window=8192, limit=_SLICE + _OVERHEAD
    )
    loop = _loop(provider, tmp_path)
    text = asyncio.run(loop._summarize_for_compaction(_history(28)))
    n = sum(call[0].text.startswith("Part ") for call in provider.seen)
    assert n >= 2
    assert len(provider.seen) == n + _MAX_FOLD_PASSES * n + 1
    last = provider.seen[-1][0].text
    assert last.startswith("Fold these part-summaries")
    assert "part-summaries elided" in last
    assert len(last) <= _SLICE + _OVERHEAD
    assert text == "z" * 7000


def test_pack_parts_keeps_every_group_under_the_budget() -> None:
    parts = ["a" * 300, "b" * 300, "c" * 500, "d" * 1200, "e" * 10]
    groups = _pack_parts(parts, 1000)
    assert [[p[0] for p in g] for g in groups] == [["a", "b"], ["c"], ["d"], ["e"]]
    assert all(len("\n\n".join(g)) <= 1000 for g in groups)
    assert "part-summary elided" in groups[2][0]  # the 1200-char part was clamped to fit


def test_clamp_middle_keeps_head_and_tail_within_budget() -> None:
    text = "H" * 600 + "M" * 600 + "T" * 600
    out = _clamp_middle(text, 900, "transcript")
    assert len(out) == 900
    assert out.startswith("H" * 600)
    assert out.endswith("T")
    assert "M" not in out
    assert "transcript elided: 1,800 characters exceed the summarizer's budget of 900" in out
    assert _clamp_middle("short", 900, "transcript") == "short"


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


def test_the_summarizer_reads_a_long_tool_output_by_its_head_and_tail(tmp_path: Path) -> None:
    """ADR-0232. Field 2026-09-23 (three 131k P40 bodies, 11 compactions): tool outputs were
    85-95% of what the summarizer re-read, and each summarizer run took 3-8 minutes."""
    output = "HEAD" + "m" * 40_000 + "TAIL"
    history = [Message.user("start"), *_tool_pair("t1", output), *_tool_pair("t2", "short one")]
    rendered = AgentLoop._render_for_summary(history)
    assert "HEAD" in rendered and "TAIL" in rendered
    assert "m" * (_SUMMARY_OUTPUT_CHARS * 2 // 3) not in rendered
    assert "[... 38,008 characters of this output left out ...]" in rendered
    assert "short one" in rendered
    assert len(rendered) < _SUMMARY_OUTPUT_CHARS + 300


def test_clipping_the_summarizer_s_copy_leaves_the_conversation_whole(tmp_path: Path) -> None:
    """Three 20,000-character outputs overflowed one 8,192-character slice, so the summarizer
    ran once per slice; clipped, the transcript fits one call. The messages it was handed
    keep every character."""
    provider = _StrictWindowProvider(
        ["the summary"], tokens=100_000, window=8192, limit=_SLICE + _OVERHEAD
    )
    loop = _loop(provider, tmp_path)
    history = [
        Message.user("read three files"),
        *_tool_pair("t1", "a" * 20_000),
        *_tool_pair("t2", "b" * 20_000),
        *_tool_pair("t3", "c" * 20_000),
    ]
    text = asyncio.run(loop._summarize_for_compaction(history))
    assert text == "the summary"
    assert len(provider.seen) == 1
    assert [_output(history[i]) for i in (2, 4, 6)] == ["a" * 20_000, "b" * 20_000, "c" * 20_000]


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


def test_auto_compact_holds_the_kept_tail_to_its_budget(tmp_path: Path) -> None:
    # ADR-0132: the loop hands the compactor the window and a token counter, so the kept
    # tail is a budget as well as a count, and the outcome names what was elided from it.
    # The counter floors the provider's estimate at 3 chars/token (the seam clamp's
    # density): three 5,000-char results are 5,000 tokens against an 8192-window budget
    # of 2,048, so the two older ones go and the newest stays whole.
    provider = _SummarizerProvider(["summary"], tokens=100)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(
        [
            *_history(3),
            *_tool_pair("t1", "x" * 5000),
            *_tool_pair("t2", "y" * 5000),
            *_tool_pair("t3", "z" * 5000),
        ]
    )

    assert asyncio.run(loop.compact_now()) is True
    assert loop.last_compaction == (
        "compacted 12 → 7 messages "
        "(2 long tool output(s) in the kept tail elided to fit its budget)"
    )
    assert _output(loop.session.messages[2]).startswith(ELISION_MARKER)
    assert _output(loop.session.messages[4]).startswith(ELISION_MARKER)
    assert _output(loop.session.messages[-1]) == "z" * 5000

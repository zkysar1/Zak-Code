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

from zakcode.agent.budget import IterationBudget
from zakcode.agent.compact import ELISION_MARKER, CompactionConfig, Compactor
from zakcode.agent.loop import (
    _FOLD_CLOSE,
    _FOLD_PROMPT,
    _MAX_FOLD_PASSES,
    _MAX_SUMMARY_SLICES,
    _SUMMARY_CLOSE,
    _SUMMARY_OUTPUT_CHARS,
    AgentLoop,
    _clamp_middle,
    _pack_parts,
    _summary_of,
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
from zakcode.usage import Usage


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
#: Prompt overhead a slice-sized call may add ("Part i of N …", the fold instruction, and
#: since ADR-0242 the closing instruction after the transcript).
_OVERHEAD = 384


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


def _skill_turn(name: str, body: str) -> Message:
    """A turn composed from ``/<name>``, typed or delivered by the harness: frame, then body."""
    return Message.user(
        f"<command-message>{name} is running</command-message>\n"
        f"<command-name>/{name}</command-name>\n\n{body}"
    )


def test_the_summarizer_reads_a_skill_turn_by_its_head_and_tail() -> None:
    """ADR-0238. Field 2026-09-23 (three 131k P40 bodies, 22 compactions): with tool outputs
    held (ADR-0232), a whole skill body was 67 to 81 percent of what the summarizer read
    wherever one sat in the summarized part, the 84,476-character loop skill three times on
    one worker. The same body loaded through use_skill is a tool output and was already held."""
    body = "RULES " + "s" * 40_000 + " RETURN"
    asked = "fix the parser " + "q" * 5_000
    history = [
        _skill_turn("worker-loop", body),
        Message.assistant_text("on it"),
        Message.user(asked),
    ]
    rendered = AgentLoop._render_for_summary(history)
    assert "<command-name>/worker-loop</command-name>" in rendered and " RETURN" in rendered
    assert "s" * (_SUMMARY_OUTPUT_CHARS * 2 // 3) not in rendered
    assert "characters of this skill's instructions left out ...]" in rendered
    assert asked in rendered  # what the operator wrote is never held
    assert history[0].text.endswith(body)  # only the summarizer's copy is clipped


def test_the_note_says_when_the_turn_s_skill_was_summarized_away(tmp_path: Path) -> None:
    """ADR-0238. On a worker Body the loop skill's only copy was summarized mid-lap at three
    compactions of three, and each time the Body ended its turn 7 to 19 rows later."""
    provider = _SummarizerProvider(["the summary"], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop._skill_pages["worker-loop"] = None  # a whole skill is listed without pages (ADR-0192)
    loop.session.messages.extend([_skill_turn("worker-loop", "step " * 6_000), *_history(5)])

    assert asyncio.run(loop.compact_now(trigger="auto")) is True

    (prompt,) = provider.seen[0]
    assert "step " * 1_000 not in prompt.text  # the body's middle never reached the summarizer
    assert (
        "- /worker-loop: its instructions (30,000 characters) were in the part summarized "
        "above, so they are no longer in your context; if the skill is needed again, load it "
        "with Skill"
    ) in loop.session.messages[0].text


def test_the_note_is_silent_while_the_newest_copy_is_kept(tmp_path: Path) -> None:
    # The loop's next lap delivered the skill again, and that copy is in the kept tail.
    provider = _SummarizerProvider(["the summary"], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    lap_two = _skill_turn("worker-loop", "lap two " * 3_000)
    loop.session.messages.extend(
        [_skill_turn("worker-loop", "lap one " * 3_000), *_history(5), lap_two]
    )

    assert asyncio.run(loop.compact_now(trigger="auto")) is True

    assert "its instructions" not in loop.session.messages[0].text
    assert loop.session.messages[-1] is lap_two


def test_an_operator_only_skill_is_named_with_the_operator_s_route_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _SummarizerProvider(["the summary"], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    monkeypatch.setattr(loop, "_user_only_skills", lambda: {"start"})
    loop.session.messages.extend([_skill_turn("start", "boot " * 4_000), *_history(5)])

    assert asyncio.run(loop.compact_now(trigger="auto")) is True

    assert (
        "- /start: its instructions (20,000 characters) were in the part summarized above, so "
        "they are no longer in your context; only the operator can run /start again, by typing "
        "it; Skill refuses it, so do not call Skill for it"
    ) in loop.session.messages[0].text


def test_a_paged_skill_keeps_its_section_line_and_gets_no_second_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zakcode import tasks
    from zakcode.tasks import skill_pages, skill_skeleton

    monkeypatch.setattr(tasks, "PAGE_BUDGET_CHARS", 100)
    alpha, beta = "alpha " * 10, "beta " * 12
    body = f"# Boot\n\nintro\n\n## Step 1: Alpha\n\n{alpha}\n\n## Step 2: Beta\n\n{beta}\n"
    provider = _SummarizerProvider(["the summary"], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.task_network.insert_before(None, skill_skeleton(body, skill="boot"))
    loop._skill_pages["boot"] = skill_pages(body, skill="boot")
    loop._skill_pages_delivered["boot"] = {1}
    loop.session.messages.extend([_skill_turn("boot", body), *_history(5)])

    assert asyncio.run(loop.compact_now(trigger="auto")) is True

    note = loop.session.messages[0].text
    assert "- /boot: on section 1 of 2" in note
    assert "/boot: its instructions" not in note


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


# ── ADR-0241: what a compaction cost ─────────────────────────────────────────────────

#: What every call of the priced summarizer below reports.
_SUMMARY_USAGE = Usage(
    prompt_tokens=900,
    completion_tokens=100,
    total_tokens=1000,
    cost_usd=0.01,
    cache_read_tokens=300,
)


class _PricedSummarizerProvider(_SummarizerProvider):
    """A summarizer whose every call reports its usage, as a real provider's does, and
    takes ``delay`` seconds."""

    delay = 0.0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        await asyncio.sleep(self.delay)
        result = await super().acomplete(messages, system=system, tools=tools, **kw)
        return result.model_copy(update={"usage": _SUMMARY_USAGE})


def _compaction_rows(loop: AgentLoop) -> list[dict[str, Any]]:
    return [
        e.data for e in loop._trace.of_kind("intervention") if e.data.get("kind") == "compaction"
    ]


def test_summarizer_calls_reach_the_session_and_the_budget(tmp_path: Path) -> None:
    # The plan judge and the quality gate always added their usage; the summarizer returned
    # only its text, so a compaction's calls reached neither /cost nor a cost cap.
    provider = _PricedSummarizerProvider(["summary"], tokens=100_000)
    budget = IterationBudget(100, max_cost_usd=5.0)
    loop = AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        compactor=Compactor(CompactionConfig()),
        budget=budget,
    )
    loop.session.messages.extend(_history(5))

    assert asyncio.run(loop.compact_now(trigger="auto")) is True
    assert len(provider.seen) == 1
    assert loop.session.cumulative_usage().total_tokens == 1000
    assert [usage.side_call for usage in loop.session.usages] == ["summarizer"]
    assert budget.tokens_spent == 1000
    assert budget.cost_spent == pytest.approx(0.01)


def test_the_compaction_row_says_what_it_cost(tmp_path: Path) -> None:
    # Measured 2026-09-23 (a 27B worker Body): 355 s between the last trace row and a
    # compaction's boundary, with nothing saying whether the PreCompact hooks or the
    # summarizer took it. The row now carries both.
    provider = _PricedSummarizerProvider(["summary"], tokens=100_000)
    provider.delay = 0.05
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(_history(5))

    async def serialize_state(payload: LifecyclePayload) -> None:
        await asyncio.sleep(0.05)  # a host's PreCompact hook, writing its checkpoint

    loop.hook_manager.register_lifecycle(HookEvent.PRE_COMPACT, serialize_state)

    assert asyncio.run(loop.compact_now(trigger="auto")) is True
    (row,) = _compaction_rows(loop)
    assert row["compacted"] is True
    assert row["summarizer_calls"] == 1
    assert row["summarizer_prompt_tokens"] == 900
    assert row["summarizer_cache_read_tokens"] == 300
    assert row["summarizer_completion_tokens"] == 100
    assert row["summarizer_s"] >= 0.04
    assert row["pre_compact_s"] >= 0.04


def test_every_slice_and_fold_is_counted(tmp_path: Path) -> None:
    # An oversized history is summarized in slices, then folded: every one of those calls
    # is the compaction's, and every one reaches the session.
    provider = _PricedSummarizerProvider(["part summary"], tokens=100_000)
    loop = _loop(provider, tmp_path)

    asyncio.run(loop._summarize_for_compaction(_history(28)))

    calls = len(provider.seen)
    assert calls >= 2
    assert loop._compaction_cost["summarizer_calls"] == calls
    assert loop._compaction_cost["summarizer_prompt_tokens"] == 900 * calls
    assert len(loop.session.usages) == calls


def test_a_failed_summarizer_call_still_counts_its_seconds(tmp_path: Path) -> None:
    # A summarizer that raised reported no usage, but it spent the time; a timed-out one
    # is the costliest compaction there is.
    provider = _ExplodingProvider([], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(_history(5))

    assert asyncio.run(loop.compact_now(trigger="auto")) is False
    (row,) = _compaction_rows(loop)
    assert row["compacted"] is False
    assert row["summarizer_calls"] == 1
    assert row["summarizer_prompt_tokens"] == 0
    assert "summarizer_s" in row
    assert loop.session.usages == []


def test_a_model_free_elision_row_shows_no_summarizer(tmp_path: Path) -> None:
    # The keys are always present, so a missing key means an older build, never "free".
    provider = _PricedSummarizerProvider(["summary"], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(
        [Message.user("load the skill"), *_tool_pair("t1", "body " * 20_000)]
    )

    assert asyncio.run(loop.elide_now(trigger="auto")) is True
    (row,) = _compaction_rows(loop)
    assert row["summarizer_calls"] == 0
    assert row["summarizer_s"] == 0.0
    assert provider.seen == []


def test_each_compaction_row_counts_only_its_own_calls(tmp_path: Path) -> None:
    # The record starts over at every PreCompact, so a second compaction in the same turn
    # does not inherit the first one's calls.
    provider = _PricedSummarizerProvider(["summary"], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(_history(5))
    assert asyncio.run(loop.compact_now(trigger="auto")) is True
    loop.session.messages.extend(_history(4))
    assert asyncio.run(loop.compact_now(trigger="auto")) is True

    assert [row["summarizer_calls"] for row in _compaction_rows(loop)] == [1, 1]


# ── ADR-0242: a summary, or asked for again ──────────────────────────────────────────


class _ScriptedSummarizer(_SummarizerProvider):
    """Answers with ``texts`` in order (the last repeats), each with a usage, and records
    each call's keyword arguments."""

    def __init__(self, texts: list[str], *, tokens: int, window: int = 8192) -> None:
        super().__init__(texts, tokens=tokens, window=window)
        self.kwargs: list[dict[str, Any]] = []

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.kwargs.append(kw)
        result = await super().acomplete(messages, system=system, tools=tools, **kw)
        return result.model_copy(update={"usage": _SUMMARY_USAGE})


def test_the_transcript_ends_with_the_instruction(tmp_path: Path) -> None:
    # Measured 2026-09-23 (a 27B model on the pod): 9 of 34 responses were the transcript's
    # next turn, a plan or a status line, while the instruction sat only above the transcript.
    provider = _SummarizerProvider(["<summary>summary</summary>"], tokens=100_000)
    loop = _loop(provider, tmp_path)

    asyncio.run(loop._summarize_for_compaction(_history(3)))

    (call,) = provider.seen
    text = call[0].text
    assert text.startswith("Conversation transcript to summarize")
    assert text.endswith(_SUMMARY_CLOSE)
    assert text.index("answer 2") < text.index("[end of transcript]")


def test_slices_and_folds_end_with_their_instruction(tmp_path: Path) -> None:
    provider = _SummarizerProvider(["x" * 5000], tokens=100_000)
    loop = _loop(provider, tmp_path)

    asyncio.run(loop._summarize_for_compaction(_history(40)))

    slices = [c[0].text for c in provider.seen if c[0].text.startswith("Part ")]
    folds = [c[0].text for c in provider.seen if c[0].text.startswith(_FOLD_PROMPT)]
    assert len(slices) >= 2 and folds
    assert all(text.endswith(_SUMMARY_CLOSE) for text in slices)
    assert all(text.endswith(_FOLD_CLOSE) for text in folds)


def test_the_summary_is_read_from_its_tags(tmp_path: Path) -> None:
    provider = _SummarizerProvider(
        ["Let me think.\n<summary>The user asked for X; Y is done.</summary>\nSomething else."],
        tokens=100_000,
    )
    loop = _loop(provider, tmp_path)

    text = asyncio.run(loop._summarize_for_compaction(_history(3)))

    assert text.startswith("The user asked for X; Y is done.")
    assert "Let me think." not in text and "Something else." not in text


def test_a_transcript_turn_is_asked_for_again(tmp_path: Path) -> None:
    # The field shape: the response opened with the renderer's own role label and went on
    # as the agent. It is thrown away, and the model is asked again at the rejection-retry
    # temperature; both responses' tokens were spent, so both are counted.
    provider = _ScriptedSummarizer(
        ["[assistant]\nI will now run the tests.", "<summary>The tests were fixed.</summary>"],
        tokens=100_000,
    )
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(_history(5))
    said: list[str] = []
    loop._status_sink = said.append

    assert asyncio.run(loop.compact_now(trigger="auto")) is True

    assert len(provider.seen) == 2
    assert "temperature" not in provider.kwargs[0]
    assert provider.kwargs[1]["temperature"] == pytest.approx(0.5)
    summary = loop.session.messages[0].text
    assert "The tests were fixed." in summary and "I will now run" not in summary
    (row,) = _compaction_rows(loop)
    assert row["summarizer_calls"] == 1
    assert row["summarizer_rejected"] == 1
    assert row["summarizer_prompt_tokens"] == 1800
    assert any("opened as the transcript's next turn" in line for line in said)


def test_a_short_response_to_a_long_transcript_is_asked_for_again(tmp_path: Path) -> None:
    provider = _ScriptedSummarizer(
        [
            "Phase 0.5: pre-selection.",
            "<summary>" + "The session did real work. " * 30 + "</summary>",
        ],
        tokens=100_000,
        window=131_072,
    )
    loop = _loop(provider, tmp_path)

    text = asyncio.run(loop._summarize_for_compaction(_history(40)))

    assert len(provider.seen) == 2
    assert len(provider.seen[0][0].text) >= 20_000  # the premise: one call, over the floor
    assert text.startswith("The session did real work.")
    assert loop._compaction_cost["summarizer_rejected"] == 1


def test_a_short_summary_of_a_short_transcript_stands(tmp_path: Path) -> None:
    provider = _ScriptedSummarizer(["Fine."], tokens=100_000, window=131_072)
    loop = _loop(provider, tmp_path)

    text = asyncio.run(loop._summarize_for_compaction(_history(3)))

    assert text.startswith("Fine.")
    assert len(provider.seen) == 1


def test_an_empty_response_is_not_a_summary(tmp_path: Path) -> None:
    # A thinking model that spent its answer thinking leaves nothing once the markup goes.
    provider = _ScriptedSummarizer(
        ["<think>just thinking</think>", "<summary>A real summary.</summary>"], tokens=100_000
    )
    loop = _loop(provider, tmp_path)
    said: list[str] = []
    loop._status_sink = said.append

    text = asyncio.run(loop._summarize_for_compaction(_history(3)))

    assert text.startswith("A real summary.")
    assert loop._compaction_cost["summarizer_rejected"] == 1
    assert any("(it was empty)" in line for line in said)


def test_two_non_summaries_fall_back_to_eliding_tool_outputs(tmp_path: Path) -> None:
    # One resample, then the ADR-0083 fallback: the old tool outputs are elided and the
    # conversation's own text stays, which keeps more than a non-summary would.
    provider = _ScriptedSummarizer(["[assistant] carrying on with the work"], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend([*_history(2), *_tool_pair("t1", "x" * 6000), *_history(4)])

    assert asyncio.run(loop.compact_now(trigger="auto")) is True

    assert len(provider.seen) == 2
    assert "SummaryNotWritten" in loop.last_compaction
    assert "opened as the transcript's next turn" in loop.last_compaction
    (row,) = _compaction_rows(loop)
    assert row["summarizer_calls"] == 1
    assert row["summarizer_rejected"] == 2


def test_a_summary_may_open_with_the_previous_summary_s_label() -> None:
    # On a re-compaction the transcript opens with the previous summary under [system]; a
    # model that echoes the label before its summary has still written one.
    summary, why = _summary_of("[system]\n[Conversation summary]\nThe merged summary.", 900)
    assert summary.endswith("The merged summary.") and why == ""
    assert _summary_of("[assistant]\nI will carry on.", 900) == (
        "",
        "it opened as the transcript's next turn",
    )

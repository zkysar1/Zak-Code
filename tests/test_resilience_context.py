"""ContextWindowExceeded compact-then-retry recovery (parity #1b/#9).

When a provider call overflows the model's window, the loop force-compacts and retries
the SAME call in place (no new iteration) rather than dying as provider_error — bounded by
``_MAX_CONTEXT_RECOVERY`` so an un-compactable session fails gracefully. Caught above the
failover branch so a context overflow never mis-routes into model-switching. Hermetic:
scripted providers + a stub compactor, no network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from zakcode.agent.loop import _MAX_CONTEXT_RECOVERY, AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    ContextWindowExceeded,
    LLMResult,
    Provider,
    ProviderError,
    ProviderStreamEvent,
    RequestFailed,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsage,
    ToolCall,
)
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.usage import Usage


class OverflowProvider(Provider):
    """Raises ContextWindowExceeded ``n`` times, then returns ``result`` forever."""

    def __init__(self, overflows: int, result: LLMResult | None = None) -> None:
        self._left = overflows
        self._result = result or LLMResult(text="recovered")
        self.calls = 0

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> LLMResult:
        self.calls += 1
        if self._left > 0:
            self._left -= 1
            raise ContextWindowExceeded("prompt is too long")
        return self._result

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> AsyncIterator[ProviderStreamEvent]:
        self.calls += 1
        if self._left > 0:
            self._left -= 1
            raise ContextWindowExceeded("prompt is too long")
        yield StreamTextDelta(text=self._result.text or "recovered")
        yield StreamDone(finish_reason="stop")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


class StubCompactor:
    """A compactor whose force-compaction always (or never) shrinks the transcript."""

    def __init__(self, *, succeeds: bool = True) -> None:
        self._succeeds = succeeds
        self.compact_calls = 0

    def should_compact(self, messages: list[Message], **kw: Any) -> bool:
        # Turn-start auto-compaction is a no-op in these tests; only the forced
        # compact_now() recovery path (which does not consult should_compact) runs.
        return False

    async def compact(self, messages: list[Message], *, summarize: Any) -> Any:
        from types import SimpleNamespace

        self.compact_calls += 1
        # Shrink to just the last message so a retry sees a smaller transcript.
        kept = list(messages[-1:]) if self._succeeds else list(messages)
        return SimpleNamespace(compacted=self._succeeds, messages=kept)


def _make_loop(provider: Provider, compactor: Any | None = None, **kw: Any) -> AgentLoop:
    settings = load_settings(workspace_root=Path.cwd())
    session = Session(cwd="/tmp/work", model="test/model")
    return AgentLoop(
        provider, ToolRegistry(), session, settings=settings, compactor=compactor, **kw
    )


async def _collect(loop: AgentLoop, text: str) -> list[Any]:
    return [ev async for ev in loop.astream_turn(text)]


# ── buffered ─────────────────────────────────────────────────────────────────


def test_context_overflow_compacts_and_retries_same_iteration() -> None:
    provider = OverflowProvider(overflows=1)
    compactor = StubCompactor(succeeds=True)
    loop = _make_loop(provider, compactor)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "completed"
    assert result.error == ""
    assert provider.calls == 2  # overflowed once, then succeeded on retry
    assert compactor.compact_calls == 1
    assert result.iterations == 1  # recovery is in-place, NOT a new iteration


def test_context_overflow_unrecoverable_ends_gracefully() -> None:
    """If compaction can't shrink the transcript, the turn ends as provider_error."""
    provider = OverflowProvider(overflows=5)
    compactor = StubCompactor(succeeds=False)  # nothing old enough to summarize
    loop = _make_loop(provider, compactor)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "provider_error"
    assert result.degraded
    assert "too long" in result.error
    assert provider.calls == 1  # no retry once compaction reports it can't help


def test_context_overflow_no_compactor_is_terminal() -> None:
    provider = OverflowProvider(overflows=5)
    loop = _make_loop(provider, compactor=None)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "provider_error"
    assert provider.calls == 1


def test_context_overflow_recovery_is_bounded() -> None:
    """After _MAX_CONTEXT_RECOVERY compactions in one turn, a further overflow is terminal."""
    provider = OverflowProvider(overflows=_MAX_CONTEXT_RECOVERY + 5)
    compactor = StubCompactor(succeeds=True)
    loop = _make_loop(provider, compactor)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "provider_error"
    # initial call + _MAX_CONTEXT_RECOVERY retries, then it gives up
    assert provider.calls == _MAX_CONTEXT_RECOVERY + 1
    assert compactor.compact_calls == _MAX_CONTEXT_RECOVERY


def test_context_overflow_never_triggers_model_failover() -> None:
    """The except-order guard: a ContextWindowExceeded must NOT reach the failover branch."""
    failover_calls: list[ProviderError] = []

    def _failover(exc: ProviderError) -> tuple[Provider, str] | None:
        failover_calls.append(exc)
        return (OverflowProvider(overflows=0), "should-not-happen")

    provider = OverflowProvider(overflows=1)
    compactor = StubCompactor(succeeds=True)
    loop = _make_loop(provider, compactor, model_failover=_failover)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "completed"
    assert failover_calls == []  # failover was never consulted for a context overflow


def test_non_context_provider_error_still_fails_over() -> None:
    """Control: a NON-context ProviderError still reaches failover (guard is surgical)."""

    class OneBoomProvider(Provider):
        def __init__(self) -> None:
            self.calls = 0

        async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
            self.calls += 1
            raise RequestFailed("boom")

        def count_tokens(self, messages, *, system=None) -> int:
            return 0

        def capabilities(self) -> Capabilities:
            return Capabilities(supports_tools=True, context_window=8192)

    seen: list[ProviderError] = []

    def _failover(exc: ProviderError) -> tuple[Provider, str] | None:
        seen.append(exc)
        return (OverflowProvider(overflows=0), "switched")  # recovers on the new provider

    loop = _make_loop(OneBoomProvider(), compactor=None, model_failover=_failover)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "completed"
    assert len(seen) == 1 and isinstance(seen[0], RequestFailed)


# ── streaming ────────────────────────────────────────────────────────────────


def test_streaming_context_overflow_compacts_and_retries() -> None:
    provider = OverflowProvider(overflows=1)
    compactor = StubCompactor(succeeds=True)
    loop = _make_loop(provider, compactor)
    events = asyncio.run(_collect(loop, "hi"))
    done = events[-1]
    assert done.stop_reason == "completed"
    assert provider.calls == 2
    assert compactor.compact_calls == 1
    # the compacted-and-retrying status surfaced on the stream, with before → after counts
    statuses = [ev.message for ev in events if type(ev).__name__ == "AgentStatus"]
    assert any("context window exceeded; compacted" in s and "and retrying" in s for s in statuses)


def test_streaming_context_overflow_unrecoverable_is_terminal() -> None:
    provider = OverflowProvider(overflows=5)
    compactor = StubCompactor(succeeds=False)
    loop = _make_loop(provider, compactor)
    done = asyncio.run(_collect(loop, "hi"))[-1]
    assert done.stop_reason == "provider_error"
    assert done.degraded


def test_streaming_context_overflow_no_compactor_is_terminal() -> None:
    provider = OverflowProvider(overflows=5)
    loop = _make_loop(provider, compactor=None)
    events = asyncio.run(_collect(loop, "hi"))
    done = events[-1]
    assert done.stop_reason == "provider_error"
    assert provider.calls == 1


def test_streaming_context_overflow_recovery_is_bounded() -> None:
    """After _MAX_CONTEXT_RECOVERY compactions in one turn, a further overflow is terminal."""
    provider = OverflowProvider(overflows=_MAX_CONTEXT_RECOVERY + 5)
    compactor = StubCompactor(succeeds=True)
    loop = _make_loop(provider, compactor)
    events = asyncio.run(_collect(loop, "hi"))
    done = events[-1]
    assert done.stop_reason == "provider_error"
    # initial call + _MAX_CONTEXT_RECOVERY retries, then it gives up
    assert provider.calls == _MAX_CONTEXT_RECOVERY + 1
    assert compactor.compact_calls == _MAX_CONTEXT_RECOVERY


def test_streaming_context_overflow_never_triggers_model_failover() -> None:
    """The except-order guard: a ContextWindowExceeded must NOT reach the failover branch."""
    failover_calls: list[ProviderError] = []

    def _failover(exc: ProviderError) -> tuple[Provider, str] | None:
        failover_calls.append(exc)
        return (OverflowProvider(overflows=0), "should-not-happen")

    provider = OverflowProvider(overflows=1)
    compactor = StubCompactor(succeeds=True)
    loop = _make_loop(provider, compactor, model_failover=_failover)
    events = asyncio.run(_collect(loop, "hi"))
    done = events[-1]
    assert done.stop_reason == "completed"
    assert failover_calls == []  # failover was never consulted for a context overflow


def test_streaming_non_context_provider_error_still_fails_over() -> None:
    """Control: a NON-context ProviderError still reaches failover (guard is surgical)."""

    class StreamOneBoomProvider(Provider):
        def __init__(self) -> None:
            self.calls = 0

        async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
            self.calls += 1
            raise RequestFailed("boom")

        async def astream(
            self, messages, *, system=None, tools=None, **kw
        ) -> AsyncIterator[ProviderStreamEvent]:
            self.calls += 1
            raise RequestFailed("boom")
            yield  # unreachable — makes Python treat this as an async generator

        def count_tokens(self, messages, *, system=None) -> int:
            return 0

        def capabilities(self) -> Capabilities:
            return Capabilities(supports_tools=True, context_window=8192)

    seen: list[ProviderError] = []

    def _failover(exc: ProviderError) -> tuple[Provider, str] | None:
        seen.append(exc)
        return (OverflowProvider(overflows=0), "switched")  # recovers on the new provider

    loop = _make_loop(StreamOneBoomProvider(), compactor=None, model_failover=_failover)
    events = asyncio.run(_collect(loop, "hi"))
    done = events[-1]
    assert done.stop_reason == "completed"
    assert len(seen) == 1 and isinstance(seen[0], RequestFailed)
    # verify the failover status surfaced on the stream
    statuses = [ev.message for ev in events if type(ev).__name__ == "AgentStatus"]
    assert any("switching model" in s for s in statuses)


# ── the bound is per CALL, and compaction is checked per call (ADR-0074) ─────────────


class PlanningProvider(Provider):
    """Answers a plan-tool call for the first ``steps`` calls, then text — a multi-iteration
    turn. With ``overflow`` set, every call's FIRST attempt overflows and the retry answers:
    each recovered overflow buys one more iteration, the shape a runner's single long turn
    takes, where a per-TURN bound of two ended the session on the third overflow (coach,
    2026-08-28)."""

    def __init__(self, steps: int, *, overflow: bool = False) -> None:
        self._steps = steps
        self._overflow = overflow
        self._answered = 0
        self._pending_overflow = overflow
        self.calls = 0
        self.overflows = 0

    def _answer(self, tools: list[dict[str, Any]] | None) -> LLMResult:
        if not tools:
            # A side call (the plan-quality judge shares the provider, tools=None): answer
            # blandly and leave the scripted main-loop sequence alone.
            return LLMResult(text="")
        self.calls += 1
        if self._pending_overflow:
            self._pending_overflow = False
            self.overflows += 1
            raise ContextWindowExceeded("prompt is too long")
        self._pending_overflow = self._overflow
        self._answered += 1
        if self._answered <= self._steps:
            tasks = [{"title": f"step {self._answered}", "status": "done"}]
            return LLMResult(
                tool_calls=[
                    ToolCall(
                        id=f"p{self._answered}", name="update_plan", arguments={"tasks": tasks}
                    )
                ]
            )
        return LLMResult(text="finished")

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> LLMResult:
        return self._answer(tools)

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> AsyncIterator[ProviderStreamEvent]:
        result = self._answer(tools)
        for call in result.tool_calls:
            yield StreamToolCallDelta(
                index=0, id=call.id, name=call.name, arguments_delta=json.dumps(call.arguments)
            )
        if result.text:
            yield StreamTextDelta(text=result.text)
        yield StreamDone(finish_reason="tool_calls" if result.tool_calls else "stop")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


class ThresholdCompactor:
    """Compacts once the transcript holds ``limit`` or more messages — the real trigger's
    shape (a count against a window), consulted wherever the loop asks."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.checks = 0
        self.compact_calls = 0

    def should_compact(self, messages: list[Message], **kw: Any) -> bool:
        self.checks += 1
        return len(messages) >= self._limit

    async def compact(self, messages: list[Message], *, summarize: Any) -> Any:
        from types import SimpleNamespace

        self.compact_calls += 1
        return SimpleNamespace(compacted=True, messages=list(messages[-2:]))


def _planning_loop(provider: Provider, compactor: Any) -> AgentLoop:
    settings = load_settings(workspace_root=Path.cwd())
    registry = ToolRegistry()
    registry.register(UpdatePlanTool())
    session = Session(cwd="/tmp/work", model="test/model")
    return AgentLoop(provider, registry, session, settings=settings, compactor=compactor)


def test_context_overflow_bound_is_per_call_not_per_turn() -> None:
    """More recoveries than _MAX_CONTEXT_RECOVERY in ONE turn — one per iteration, each
    on its own call — and the turn completes."""
    provider = PlanningProvider(steps=_MAX_CONTEXT_RECOVERY, overflow=True)
    compactor = StubCompactor(succeeds=True)
    result = asyncio.run(_planning_loop(provider, compactor).arun_turn("plan it"))
    assert result.stop_reason == "completed", result.error
    assert provider.overflows == _MAX_CONTEXT_RECOVERY + 1
    assert compactor.compact_calls == provider.overflows
    assert provider.calls == 2 * provider.overflows  # every overflow retried once, in place
    assert result.iterations == _MAX_CONTEXT_RECOVERY + 1


def test_streaming_context_overflow_bound_is_per_call_not_per_turn() -> None:
    provider = PlanningProvider(steps=_MAX_CONTEXT_RECOVERY, overflow=True)
    compactor = StubCompactor(succeeds=True)
    done = asyncio.run(_collect(_planning_loop(provider, compactor), "plan it"))[-1]
    assert done.stop_reason == "completed", done.error
    assert provider.overflows == _MAX_CONTEXT_RECOVERY + 1
    assert compactor.compact_calls == provider.overflows
    assert done.iterations == _MAX_CONTEXT_RECOVERY + 1


def test_auto_compaction_is_checked_before_every_call_not_only_at_turn_start() -> None:
    """A turn that crosses the threshold mid-way is compacted mid-way. The turn-start
    check alone saw an empty transcript and never looked again."""
    provider = PlanningProvider(steps=4)
    compactor = ThresholdCompactor(limit=5)  # user + two (assistant, tool) pairs
    loop = _planning_loop(provider, compactor)
    result = asyncio.run(loop.arun_turn("plan it"))
    assert result.stop_reason == "completed", result.error
    assert compactor.checks >= 5  # turn start + one per call
    assert compactor.compact_calls >= 1
    assert len(loop.session.messages) < 11  # the full transcript would be 1 + 2 * 5


def test_streaming_auto_compaction_is_checked_before_every_call() -> None:
    provider = PlanningProvider(steps=4)
    compactor = ThresholdCompactor(limit=5)
    events = asyncio.run(_collect(_planning_loop(provider, compactor), "plan it"))
    assert events[-1].stop_reason == "completed", events[-1].error
    assert compactor.compact_calls >= 1
    statuses = [ev.message for ev in events if type(ev).__name__ == "AgentStatus"]
    assert any("context near the window — compacted" in s for s in statuses)


class MeasuringProvider(PlanningProvider):
    """Reports a prompt size the local estimate cannot see: ``count_tokens`` says the
    transcript is a handful of tokens, the usage says it fills 90% of the window — the shape
    id-dense tool output takes against a chars/4 estimate (coach, 2026-08-28: the check read
    "fine" at 129k real on a 131k window)."""

    def __init__(self, steps: int, *, reported: int) -> None:
        super().__init__(steps)
        self._reported = reported

    def _answer(self, tools: list[dict[str, Any]] | None) -> LLMResult:
        result = super()._answer(tools)
        if not tools:
            return result
        usage = Usage(
            prompt_tokens=self._reported, completion_tokens=1, total_tokens=self._reported + 1
        )
        return result.model_copy(update={"usage": usage})

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> AsyncIterator[ProviderStreamEvent]:
        result = self._answer(tools)
        for call in result.tool_calls:
            yield StreamToolCallDelta(
                index=0, id=call.id, name=call.name, arguments_delta=json.dumps(call.arguments)
            )
        if result.text:
            yield StreamTextDelta(text=result.text)
        if tools:
            yield StreamUsage(usage=result.usage)
        yield StreamDone(finish_reason="tool_calls" if result.tool_calls else "stop")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return len(messages)  # a deliberately tiny estimate

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=1000)


class CountingCompactor:
    """The real trigger's shape — a count against a window — recording every count it is
    handed, so a test can see what the check BELIEVED the transcript weighed."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.counts: list[int] = []
        self.compact_calls = 0

    def should_compact(self, messages: list[Message], *, count_tokens: Any, **kw: Any) -> bool:
        n = count_tokens(messages)
        self.counts.append(n)
        return n > self._limit

    async def compact(self, messages: list[Message], *, summarize: Any) -> Any:
        from types import SimpleNamespace

        self.compact_calls += 1
        return SimpleNamespace(compacted=True, messages=list(messages[-2:]))


def test_the_compaction_check_is_floored_by_what_the_provider_last_measured() -> None:
    """ADR-0077: the check counts chars/4 locally, which ran ~25k under a 131k window on
    id-dense tool output. After a call, the provider's reported prompt size floors the next
    check, so the compaction fires where the threshold says — not where the window does."""
    provider = MeasuringProvider(steps=3, reported=900)
    compactor = CountingCompactor(limit=800)  # 0.8 of a 1000-token window
    loop = _planning_loop(provider, compactor)
    result = asyncio.run(loop.arun_turn("plan it"))
    assert result.stop_reason == "completed", result.error
    # Before the first call (turn start, then the pre-call check) nothing is measured yet —
    # the tiny estimate stands and no compaction fires.
    assert all(n < 800 for n in compactor.counts[:2]), compactor.counts
    # The first check after a call sees the measured 900 plus the appended messages.
    assert compactor.counts[2] >= 900, compactor.counts
    assert compactor.compact_calls >= 1
    # The anchor is the last main call's measurement, kept on the session for a resume.
    assert loop.session.prompt_anchor_tokens == 900
    assert 0 < loop.session.prompt_anchor_index <= len(loop.session.messages)


def test_streaming_compaction_check_is_floored_by_the_measured_prompt() -> None:
    provider = MeasuringProvider(steps=3, reported=900)
    compactor = CountingCompactor(limit=800)
    loop = _planning_loop(provider, compactor)
    events = asyncio.run(_collect(loop, "plan it"))
    assert events[-1].stop_reason == "completed", events[-1].error
    assert all(n < 800 for n in compactor.counts[:2]), compactor.counts
    assert compactor.counts[2] >= 900, compactor.counts
    assert compactor.compact_calls >= 1
    assert loop.session.prompt_anchor_tokens == 900


def test_the_anchor_only_ever_pulls_the_check_earlier_and_is_forgotten_on_compaction() -> None:
    session = Session(cwd="/tmp/work", model="test/model")
    session.add_message(Message.user("hi"))
    provider = MeasuringProvider(steps=0, reported=900)
    loop = AgentLoop(
        provider,
        ToolRegistry(),
        session,
        settings=load_settings(workspace_root=Path.cwd()),
        compactor=CountingCompactor(limit=800),
    )
    assert loop._count_tokens_anchored(session.messages) == 1  # bare estimate, no anchor
    loop._anchor_prompt(900)
    session.add_message(Message.user("more"))
    assert loop._count_tokens_anchored(session.messages) == 900 + 1  # anchor + the delta
    # An estimate larger than the anchor wins — the floor never lowers the count.
    session.prompt_anchor_tokens = 1
    assert loop._count_tokens_anchored(session.messages) == 2
    # A compaction rewrote the measured prefix: back to the bare estimate.
    loop._anchor_prompt(900)
    loop._forget_prompt_anchor()
    assert loop._count_tokens_anchored(session.messages) == 2
    # An anchor index past the transcript (a shrunk document) is ignored, not trusted.
    session.prompt_anchor_tokens, session.prompt_anchor_index = 900, 99
    assert loop._count_tokens_anchored(session.messages) == 2


def test_the_anchor_persists_with_the_session_and_an_old_document_loads_without_it() -> None:
    session = Session(cwd="/tmp/work", model="test/model")
    session.prompt_anchor_tokens, session.prompt_anchor_index = 900, 3
    reloaded = Session.model_validate(session.model_dump())
    assert (reloaded.prompt_anchor_tokens, reloaded.prompt_anchor_index) == (900, 3)
    older = Session.model_validate({"cwd": "/tmp/work", "model": "test/model"})
    assert (older.prompt_anchor_tokens, older.prompt_anchor_index) == (0, 0)


def test_module_constant_is_sane() -> None:
    assert _MAX_CONTEXT_RECOVERY >= 1

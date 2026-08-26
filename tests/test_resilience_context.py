"""ContextWindowExceeded compact-then-retry recovery (parity #1b/#9).

When a provider call overflows the model's window, the loop force-compacts and retries
the SAME call in place (no new iteration) rather than dying as provider_error — bounded by
``_MAX_CONTEXT_RECOVERY`` so an un-compactable session fails gracefully. Caught above the
failover branch so a context overflow never mis-routes into model-switching. Hermetic:
scripted providers + a stub compactor, no network.
"""

from __future__ import annotations

import asyncio
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
)
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry


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
    assert any(
        "context window exceeded; compacted" in s and "and retrying" in s for s in statuses
    )


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


def test_module_constant_is_sane() -> None:
    assert _MAX_CONTEXT_RECOVERY >= 1

"""Provider-failure resilience (audit P0-4; acceptance: ``test_rate_limit_retry``).

A rate-limited provider call is retried with ``retry_after``-aware backoff; any
provider failure that survives the retry budget ends the TURN gracefully
(``stop_reason="provider_error"``, ``degraded=True``, session consistent) instead of
unwinding the process — the difference between an unattended agent surviving a 429
burst and dying on it. Hermetic: scripted providers, no network; sleeps are patched.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

import zakcode.agent.loop as loop_module
from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import (
    AuthError,
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    RateLimited,
    StreamDone,
    StreamTextDelta,
)
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry


class FlakyProvider(Provider):
    """Raises the scripted exceptions in order, then returns ``result`` forever."""

    def __init__(self, failures: list[Exception], result: LLMResult | None = None) -> None:
        self._failures = list(failures)
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
        if self._failures:
            raise self._failures.pop(0)
        return self._result

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


class FlakyStreamProvider(FlakyProvider):
    """Streaming twin: scripted pre-stream failures, optional mid-stream failure."""

    def __init__(
        self,
        failures: list[Exception],
        *,
        fail_midstream: Exception | None = None,
    ) -> None:
        super().__init__(failures)
        self._fail_midstream = fail_midstream
        self.stream_calls = 0

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> AsyncIterator[ProviderStreamEvent]:
        self.stream_calls += 1
        if self._failures:
            raise self._failures.pop(0)
        yield StreamTextDelta(text="hello")
        if self._fail_midstream is not None:
            raise self._fail_midstream
        yield StreamDone(finish_reason="stop")


def _make_loop(provider: Provider, **overrides: object) -> AgentLoop:
    settings = load_settings(workspace_root=Path.cwd(), **overrides)
    session = Session(cwd="/tmp/work", model="test/model")
    return AgentLoop(provider, ToolRegistry(), session, settings=settings)


@pytest.fixture()
def fast_sleep(monkeypatch) -> list[float]:
    """Patch the loop's backoff sleep to record delays instead of waiting."""
    recorded: list[float] = []

    async def _instant(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(loop_module.asyncio, "sleep", _instant)
    return recorded


# ── the acceptance test (audit list #8) ──────────────────────────────────────


def test_rate_limit_retry(fast_sleep: list[float]) -> None:
    """A RateLimited error triggers backoff and the turn succeeds on retry."""
    provider = FlakyProvider([RateLimited("429", retry_after=0.5)])
    loop = _make_loop(provider)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "completed"
    assert result.error == ""
    assert not result.degraded
    assert provider.calls == 2  # initial + one retry
    assert fast_sleep == [0.5]  # honored the server-suggested delay


def test_rate_limit_exhausted_ends_turn_gracefully(fast_sleep: list[float]) -> None:
    provider = FlakyProvider([RateLimited("429", retry_after=0.0)] * 10)
    loop = _make_loop(provider, provider_max_retries=2)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "provider_error"
    assert result.degraded
    assert "429" in result.error
    assert provider.calls == 3  # initial + 2 retries
    # The session is consistent: the user message persisted, no half assistant turn.
    assert [m.role for m in loop.session.messages] == ["user"]


def test_non_retryable_provider_error_is_terminal_immediately(fast_sleep: list[float]) -> None:
    provider = FlakyProvider([AuthError("bad key")] * 2)
    loop = _make_loop(provider)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "provider_error"
    assert "bad key" in result.error
    assert provider.calls == 1  # never retried
    assert fast_sleep == []


def test_retries_disabled_by_config(fast_sleep: list[float]) -> None:
    provider = FlakyProvider([RateLimited("429")])
    loop = _make_loop(provider, provider_max_retries=0)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "provider_error"
    assert provider.calls == 1
    assert fast_sleep == []


def test_loop_still_usable_after_provider_error(fast_sleep: list[float]) -> None:
    """A failed turn must not poison the loop — the next turn runs normally."""
    provider = FlakyProvider([AuthError("transient outage")])
    loop = _make_loop(provider)
    first = asyncio.run(loop.arun_turn("hi"))
    assert first.stop_reason == "provider_error"
    second = asyncio.run(loop.arun_turn("again"))
    assert second.stop_reason == "completed"
    assert second.assistant_messages[0].text == "recovered"


# ── backoff math ─────────────────────────────────────────────────────────────


def test_retry_delay_honors_retry_after_and_caps() -> None:
    assert AgentLoop._retry_delay(RateLimited("x", retry_after=5.0), 1) == 5.0
    assert AgentLoop._retry_delay(RateLimited("x", retry_after=999.0), 1) == 30.0  # capped
    assert AgentLoop._retry_delay(RateLimited("x", retry_after=-1.0), 1) == 0.0  # floored
    # No retry_after → exponential from the base, capped.
    assert AgentLoop._retry_delay(RateLimited("x"), 1) == 1.0
    assert AgentLoop._retry_delay(RateLimited("x"), 2) == 2.0
    assert AgentLoop._retry_delay(RateLimited("x"), 3) == 4.0
    assert AgentLoop._retry_delay(RateLimited("x"), 10) == 30.0  # capped


# ── streaming path ───────────────────────────────────────────────────────────


async def _collect(loop: AgentLoop, text: str) -> list[Any]:
    return [ev async for ev in loop.astream_turn(text)]


def test_streaming_rate_limit_retries_before_first_event(fast_sleep: list[float]) -> None:
    provider = FlakyStreamProvider([RateLimited("429", retry_after=0.25)])
    loop = _make_loop(provider)
    events = asyncio.run(_collect(loop, "hi"))
    done = events[-1]
    assert done.stop_reason == "completed"
    assert provider.stream_calls == 2
    assert fast_sleep == [0.25]
    assert any(getattr(ev, "text", None) == "hello" for ev in events)


def test_streaming_midstream_failure_is_terminal_not_retried(fast_sleep: list[float]) -> None:
    """Once deltas reached the client, a retry would duplicate them — never retry."""
    provider = FlakyStreamProvider([], fail_midstream=RateLimited("429 mid-stream"))
    loop = _make_loop(provider)
    events = asyncio.run(_collect(loop, "hi"))
    done = events[-1]
    assert done.stop_reason == "provider_error"
    assert done.degraded
    assert provider.stream_calls == 1  # not retried despite being RateLimited
    assert fast_sleep == []
    statuses = [ev.message for ev in events if type(ev).__name__ == "AgentStatus"]
    assert any("provider error" in s for s in statuses)
    # Partial streamed text is not persisted: the failed turn leaves no assistant msg.
    assert [m.role for m in loop.session.messages] == ["user"]


def test_streaming_exhausted_retries_end_gracefully(fast_sleep: list[float]) -> None:
    provider = FlakyStreamProvider([RateLimited("429")] * 10)
    loop = _make_loop(provider, provider_max_retries=1)
    events = asyncio.run(_collect(loop, "hi"))
    done = events[-1]
    assert done.stop_reason == "provider_error"
    assert provider.stream_calls == 2  # initial + 1 retry

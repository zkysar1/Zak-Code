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
    StreamThinkingDelta,
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
            exc = self._fail_midstream
            self._fail_midstream = None  # consume once: a retried stream succeeds (recovery tests)
            raise exc
        yield StreamDone(finish_reason="stop")


class TempRecordingProvider(FlakyProvider):
    """Records the per-call ``temperature`` kwarg (``None`` when not overridden) for both
    the buffered (``acomplete``) and streaming (``astream``) paths, so a test can assert
    the rejection-retry resample. ``failures`` scripts ``acomplete``; ``fail_stream``
    scripts ``astream`` — each pops in order, then succeeds."""

    def __init__(
        self,
        failures: list[Exception],
        *,
        fail_stream: list[Exception] | None = None,
    ) -> None:
        super().__init__(failures)
        self.temps: list[float | None] = []
        self.stream_temps: list[float | None] = []
        self.stream_calls = 0
        self._fail_stream = list(fail_stream or [])

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        **kw: Any,
    ) -> LLMResult:
        self.calls += 1
        self.temps.append(temperature)
        if self._failures:
            raise self._failures.pop(0)
        return self._result

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        **kw: Any,
    ) -> AsyncIterator[ProviderStreamEvent]:
        self.stream_calls += 1
        self.stream_temps.append(temperature)
        if self._fail_stream:
            raise self._fail_stream.pop(0)
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


class ThinkingThenFailProvider(FlakyStreamProvider):
    """Streams a *reasoning* delta (never assistant text), then fails mid-stream."""

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> AsyncIterator[ProviderStreamEvent]:
        self.stream_calls += 1
        yield StreamThinkingDelta(text="weighing the options")
        if self._fail_midstream is not None:
            exc = self._fail_midstream
            self._fail_midstream = None  # consume once: the retried stream succeeds
            raise exc
        yield StreamTextDelta(text="recovered")
        yield StreamDone(finish_reason="stop")


def test_thinking_only_stream_stays_retryable(fast_sleep: list[float]) -> None:
    """Reasoning deltas must not consume the mid-stream retry budget.

    Twin of ``test_streaming_midstream_failure_is_terminal_not_retried`` above: there,
    a delta the client *rendered as an answer* had escaped, so retrying would duplicate
    it and the failure is terminal. A thinking delta is not an answer — nothing the
    retry would duplicate — so ``received_any`` stays False and the turn recovers.

    This is the property that makes a long reasoning phase safe: without it, a model
    that thinks for two minutes and then hits a 429 would burn the whole turn, and the
    longer it reasoned the more likely that became.
    """
    provider = ThinkingThenFailProvider([], fail_midstream=RateLimited("429 mid-stream"))
    loop = _make_loop(provider)
    events = asyncio.run(_collect(loop, "hi"))

    done = events[-1]
    assert done.stop_reason != "provider_error"
    assert provider.stream_calls == 2  # retried, unlike the assistant-text case
    assert any(getattr(ev, "text", None) == "recovered" for ev in events)


def test_streaming_exhausted_retries_end_gracefully(fast_sleep: list[float]) -> None:
    provider = FlakyStreamProvider([RateLimited("429")] * 10)
    loop = _make_loop(provider, provider_max_retries=1)
    events = asyncio.run(_collect(loop, "hi"))
    done = events[-1]
    assert done.stop_reason == "provider_error"
    assert provider.stream_calls == 2  # initial + 1 retry


def test_streaming_done_event_carries_error_detail(fast_sleep: list[float]) -> None:
    """A client consuming only the terminal event still learns WHY the turn failed."""
    provider = FlakyStreamProvider([AuthError("key revoked")])
    loop = _make_loop(provider)
    done = asyncio.run(_collect(loop, "hi"))[-1]
    assert done.stop_reason == "provider_error"
    assert "key revoked" in done.error
    # And a clean turn leaves it empty.
    ok = _make_loop(FlakyStreamProvider([]))
    done = asyncio.run(_collect(ok, "hi"))[-1]
    assert done.stop_reason == "completed" and done.error == ""


# ── shared-budget symmetry (review #5 finding: refund on both paths) ──────────


def test_provider_error_refunds_shared_budget_buffered(fast_sleep: list[float]) -> None:
    from zakcode.agent.budget import IterationBudget

    budget = IterationBudget(10)
    provider = FlakyProvider([AuthError("boom")])
    settings = load_settings(workspace_root=Path.cwd())
    loop = AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd="/tmp/work", model="test/model"),
        settings=settings,
        budget=budget,
    )
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "provider_error"
    assert budget.remaining == 10  # the no-work iteration was refunded


def test_provider_error_refunds_shared_budget_streaming(fast_sleep: list[float]) -> None:
    from zakcode.agent.budget import IterationBudget

    budget = IterationBudget(10)
    provider = FlakyStreamProvider([AuthError("boom")])
    settings = load_settings(workspace_root=Path.cwd())
    loop = AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd="/tmp/work", model="test/model"),
        settings=settings,
        budget=budget,
    )
    done = asyncio.run(_collect(loop, "hi"))[-1]
    assert done.stop_reason == "provider_error"
    assert budget.remaining == 10  # symmetric with the buffered path


def test_streaming_midstream_failure_still_refunds_budget(fast_sleep: list[float]) -> None:
    """Stack review minor #8: a MID-stream failure (events already received) also
    refunds the shared-budget unit — the partial output is discarded, so nothing the
    iteration consumed survives the turn."""
    from zakcode.agent.budget import IterationBudget

    budget = IterationBudget(10)
    provider = FlakyStreamProvider([], fail_midstream=RateLimited("429 mid-stream"))
    settings = load_settings(workspace_root=Path.cwd())
    loop = AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd="/tmp/work", model="test/model"),
        settings=settings,
        budget=budget,
    )
    done = asyncio.run(_collect(loop, "hi"))[-1]
    assert done.stop_reason == "provider_error"
    assert budget.remaining == 10  # refunded despite the partial stream


# -- groq "tool_use_failed": the model's own malformed tool call is retried ----


def test_model_output_rejected_is_retried_immediately(fast_sleep: list[float]) -> None:
    """A provider-rejected malformed tool call retries (delay 0) and recovers."""
    from zakcode.providers.base import ModelOutputRejected

    provider = FlakyProvider([ModelOutputRejected("malformed tool call (tool_use_failed)")])
    loop = _make_loop(provider)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "completed"
    assert not result.degraded
    assert provider.calls == 2  # initial + one retry
    assert fast_sleep == [0.0]  # nothing to wait for


def test_buffered_retry_log_names_the_real_cause(fast_sleep: list[float], caplog) -> None:
    """The buffered path's retry log mirrors the streaming notice: a rejected tool
    call is logged as what it is, never as 'rate-limited' (omni review of #13)."""
    import logging

    from zakcode.providers.base import ModelOutputRejected

    provider = FlakyProvider([ModelOutputRejected("malformed tool call (tool_use_failed)")])
    loop = _make_loop(provider)
    with caplog.at_level(logging.WARNING, logger="zakcode.agent.loop"):
        asyncio.run(loop.arun_turn("hi"))
    retry_logs = [r.getMessage() for r in caplog.records if "retry" in r.getMessage()]
    assert any("rejected a malformed tool call" in m for m in retry_logs)
    assert not any("rate-limited" in m for m in retry_logs)


def test_streaming_model_output_rejected_retries_with_clear_status(
    fast_sleep: list[float],
) -> None:
    """The operator-facing retry notice names the real cause, not 'rate limited'."""
    from zakcode.providers.base import ModelOutputRejected

    provider = FlakyStreamProvider([ModelOutputRejected("malformed tool call (tool_use_failed)")])
    loop = _make_loop(provider)
    events = asyncio.run(_collect(loop, "hi"))
    assert events[-1].stop_reason == "completed"
    assert provider.stream_calls == 2
    statuses = [getattr(ev, "message", "") for ev in events]
    assert any("malformed tool call" in s for s in statuses)
    assert not any("rate limited" in s for s in statuses)


def test_streaming_midstream_model_output_rejected_retries(fast_sleep: list[float]) -> None:
    """The real groq ``tool_use_failed`` bug: gpt-oss-120b emits a malformed tool call
    AFTER streaming deltas, so ``received_any`` is True and the generic gate would strand it
    as ``provider_error``. ModelOutputRejected is exempt from that gate — it retries and
    recovers — UNLIKE a generic mid-stream RateLimited (test above) which stays terminal.
    """
    from zakcode.providers.base import ModelOutputRejected

    provider = FlakyStreamProvider(
        [], fail_midstream=ModelOutputRejected("malformed tool call (tool_use_failed)")
    )
    loop = _make_loop(provider)
    events = asyncio.run(_collect(loop, "hi"))
    assert events[-1].stop_reason == "completed"  # recovered, NOT provider_error
    assert not events[-1].degraded
    assert provider.stream_calls == 2  # mid-stream rejection + one retry that succeeds
    statuses = [getattr(ev, "message", "") for ev in events]
    assert any("malformed tool call" in s for s in statuses)
    assert not any("rate limited" in s for s in statuses)


def test_streaming_midstream_rejection_retry_does_not_double_count_usage(
    fast_sleep: list[float],
) -> None:
    """Regression (fresh-eyes review of the mid-stream retry fix): a retried
    ModelOutputRejected must NOT bill the FAILED attempt's usage. Usage is folded only
    after an attempt streams to completion, so a provider that reports usage *before*
    raising mid-stream — then reports it again on the clean retry — leaves the turn
    charged for exactly ONE attempt, not two.
    """
    from zakcode.agent.budget import IterationBudget
    from zakcode.providers.base import ModelOutputRejected, StreamUsage
    from zakcode.usage import Usage

    class UsageThenRejectProvider(Provider):
        """Emits usage, then raises ModelOutputRejected mid-stream on the FIRST call;
        on the retry it re-emits the same usage and completes cleanly."""

        def __init__(self) -> None:
            self.stream_calls = 0

        async def astream(self, messages, *, system=None, tools=None, **kw):
            self.stream_calls += 1
            yield StreamTextDelta(text="partial")
            yield StreamUsage(usage=Usage(total_tokens=100, cost_usd=0.02))
            if self.stream_calls == 1:
                raise ModelOutputRejected("malformed tool call (tool_use_failed)")
            yield StreamDone(finish_reason="stop")

        async def acomplete(self, messages, *, system=None, tools=None, **kw):
            return LLMResult(text="unused")

        def count_tokens(self, messages, *, system=None) -> int:
            return 0

        def capabilities(self) -> Capabilities:
            return Capabilities(supports_tools=True, context_window=8192)

    budget = IterationBudget(10)
    provider = UsageThenRejectProvider()
    settings = load_settings(workspace_root=Path.cwd())
    loop = AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd="/tmp/work", model="test/model"),
        settings=settings,
        budget=budget,
    )
    done = asyncio.run(_collect(loop, "hi"))[-1]
    assert done.stop_reason == "completed"
    assert provider.stream_calls == 2  # mid-stream rejection + retry
    # The failed attempt's usage was dropped: exactly ONE attempt is billed.
    assert budget.cost_spent == pytest.approx(0.02)
    assert budget.tokens_spent == 100
    totals = loop.session.cumulative_usage()
    assert totals.total_tokens == 100
    assert totals.cost_usd == pytest.approx(0.02)


# -- Fix A: resample temperature on a rejection retry to break temp-0 determinism --


def test_rejection_retry_temperature_schedule() -> None:
    """The resample schedule escalates from the floor and clamps at 1.0."""
    from zakcode.agent.loop import AgentLoop

    assert AgentLoop._rejection_retry_temperature(1) == 0.5
    assert AgentLoop._rejection_retry_temperature(2) == pytest.approx(0.8)
    assert AgentLoop._rejection_retry_temperature(3) == 1.0  # 1.1 clamped
    assert AgentLoop._rejection_retry_temperature(9) == 1.0


def test_buffered_rejection_retry_resamples_at_raised_temperature(
    fast_sleep: list[float],
) -> None:
    """Fix A (buffered): a ModelOutputRejected retry re-issues at a raised temperature so a
    deterministic (temp-0) re-emit of the malformed call is broken. The first attempt uses
    the configured temperature (no override)."""
    from zakcode.providers.base import ModelOutputRejected

    provider = TempRecordingProvider([ModelOutputRejected("malformed (tool_use_failed)")])
    loop = _make_loop(provider)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "completed"
    assert provider.calls == 2
    assert provider.temps == [None, 0.5]  # default temp, then resampled at the floor


def test_buffered_rate_limit_retry_keeps_configured_temperature(
    fast_sleep: list[float],
) -> None:
    """A plain 429 retry must NOT perturb temperature — waiting, not resampling, is the
    remedy, and we want the identical request re-issued."""
    provider = TempRecordingProvider([RateLimited("429")])
    loop = _make_loop(provider)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "completed"
    assert provider.calls == 2
    assert provider.temps == [None, None]  # no override on either attempt


def test_streaming_rejection_retry_resamples_at_raised_temperature(
    fast_sleep: list[float],
) -> None:
    """Fix A (streaming twin): a pre-stream ModelOutputRejected retry re-issues astream at
    the raised temperature."""
    from zakcode.providers.base import ModelOutputRejected

    provider = TempRecordingProvider(
        [], fail_stream=[ModelOutputRejected("malformed (tool_use_failed)")]
    )
    loop = _make_loop(provider)
    done = asyncio.run(_collect(loop, "hi"))[-1]
    assert done.stop_reason == "completed"
    assert provider.stream_calls == 2
    assert provider.stream_temps == [None, 0.5]


def test_buffered_consecutive_rejections_escalate_temperature(
    fast_sleep: list[float],
) -> None:
    """Two consecutive rejections escalate the resample temperature (floor → floor+step)."""
    from zakcode.providers.base import ModelOutputRejected

    provider = TempRecordingProvider(
        [
            ModelOutputRejected("malformed 1 (tool_use_failed)"),
            ModelOutputRejected("malformed 2 (tool_use_failed)"),
        ]
    )
    loop = _make_loop(provider, provider_max_retries=3)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "completed"
    assert provider.calls == 3
    assert provider.temps == [None, 0.5, pytest.approx(0.8)]


def test_buffered_rejection_then_rate_limit_resets_temperature(
    fast_sleep: list[float],
) -> None:
    """Cross-bleed guard: a rejection raises temperature, but a SUBSEQUENT plain 429 retry
    resets it — waiting, not resampling, is the 429's remedy, so it re-issues at the
    configured temperature."""
    from zakcode.providers.base import ModelOutputRejected

    provider = TempRecordingProvider(
        [
            ModelOutputRejected("malformed (tool_use_failed)"),
            RateLimited("429"),
        ]
    )
    loop = _make_loop(provider, provider_max_retries=3)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "completed"
    assert provider.calls == 3
    assert provider.temps == [None, 0.5, None]

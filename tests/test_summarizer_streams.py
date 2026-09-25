"""The compaction summarizer streams its answer and collects it (ADR-0251).

Measured on the pod's request ledger (2026-09-15 to 09-25): every answer a worker lost —
the proxy wrote 200, the client surfaced nothing and waited the whole request timeout
before asking the same thing again — was a buffered call of summarizer size (11.9k-37k
prompt tokens, 4-17 minutes), and no streamed call in the same windows was lost. Streamed,
the summarizer rides the per-gap stall bound instead of the whole-call ceiling.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop, _collect_stream
from zakcode.config import Settings
from zakcode.messages import Message
from zakcode.providers import litellm_provider as provider_mod
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCallDelta,
    StreamUsage,
    TimedOut,
)
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.usage import Usage


class _StreamingSummarizer(Provider):
    """Streams a canned summary; its buffered path never returns — the lost-answer shape,
    where the server finished and this client sees nothing until its request timeout."""

    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas
        self.buffered_calls = 0
        self.streamed_calls = 0
        self.kwargs_seen: list[dict[str, Any]] = []

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.buffered_calls += 1
        await asyncio.sleep(3600)
        raise AssertionError("unreachable: the buffered answer never arrives")

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: Any = None,
        response_format: dict[str, Any] | None = None,
        **kw: Any,
    ) -> AsyncIterator[ProviderStreamEvent]:
        self.streamed_calls += 1
        self.kwargs_seen.append(dict(kw))
        yield StreamThinkingDelta(text="deciding what matters")
        for delta in self.deltas:
            yield StreamTextDelta(text=delta)
        yield StreamUsage(usage=Usage(prompt_tokens=37158, completion_tokens=42))
        yield StreamDone(finish_reason="stop")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 100

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _loop(provider: Provider, tmp_path: Path, *, summarizer: Provider | None = None) -> AgentLoop:
    return AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        summarizer_provider=summarizer,
    )


def _history(n: int) -> list[Message]:
    out: list[Message] = []
    for i in range(n):
        out.append(Message.user(f"question {i} " + "x" * 400))
        out.append(Message.assistant_text(f"answer {i} " + "y" * 400))
    return out


def test_the_summarizer_streams_its_answer_instead_of_waiting_on_a_buffered_one(
    tmp_path: Path,
) -> None:
    # On the pre-fix code this call awaits acomplete, whose answer never comes: the
    # wait_for expires. Streamed, the summary is back in milliseconds.
    provider = _StreamingSummarizer(["<summary>the ", "summary</summary>"])
    loop = _loop(provider, tmp_path)
    text = asyncio.run(asyncio.wait_for(loop._summarize_for_compaction(_history(2)), timeout=5))
    assert text == "the summary"
    assert provider.streamed_calls == 1 and provider.buffered_calls == 0
    # The session-affinity key rides the stream exactly as it rode the buffered call.
    assert "prompt_cache_key" in provider.kwargs_seen[0]
    # The stream's usage event is what the compaction accounts (ADR-0241).
    assert loop._compaction_cost["summarizer_prompt_tokens"] == 37158
    assert loop._compaction_cost["summarizer_completion_tokens"] == 42


def test_a_dedicated_summarizer_provider_streams_too(tmp_path: Path) -> None:
    main = _StreamingSummarizer(["never asked"])
    summarizer = _StreamingSummarizer(["<summary>from the side model</summary>"])
    loop = _loop(main, tmp_path, summarizer=summarizer)
    text = asyncio.run(asyncio.wait_for(loop._summarize_for_compaction(_history(2)), timeout=5))
    assert text == "from the side model"
    assert summarizer.streamed_calls == 1 and main.streamed_calls == main.buffered_calls == 0


async def _events(*events: ProviderStreamEvent) -> AsyncIterator[ProviderStreamEvent]:
    for event in events:
        yield event


def test_collect_stream_folds_the_events_into_the_buffered_result_shape() -> None:
    result = asyncio.run(
        _collect_stream(
            _events(
                StreamThinkingDelta(text="hm"),
                StreamTextDelta(text="a"),
                StreamTextDelta(text="b"),
                StreamToolCallDelta(index=0, id="c1", name="x", arguments_delta="{}"),
                StreamUsage(usage=Usage(prompt_tokens=3, completion_tokens=2)),
                StreamDone(finish_reason="stop"),
            )
        )
    )
    assert result.text == "ab" and result.thinking == "hm"
    assert result.usage.prompt_tokens == 3 and result.usage.completion_tokens == 2
    assert result.finish_reason == "stop"
    assert result.tool_calls == []  # a delta is not an answer; the callers offer no tools


def test_collect_stream_closes_a_stream_that_failed_midway() -> None:
    closed: list[bool] = []

    class _Failing:
        def __aiter__(self) -> _Failing:
            return self

        async def __anext__(self) -> ProviderStreamEvent:
            raise TimedOut("the stream stalled", bound="ZAKCODE_STREAM_STALL_TIMEOUT")

        async def aclose(self) -> None:
            closed.append(True)

    with pytest.raises(TimedOut):
        asyncio.run(_collect_stream(_Failing()))
    assert closed == [True]


class _StallingStream:
    """A backend that never sends a byte — the shape of a lost answer, or a dead prefill."""

    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self) -> _StallingStream:
        return self

    async def __anext__(self) -> Any:
        await asyncio.sleep(3600)
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


def test_the_summarizer_rides_the_per_gap_stall_bound_not_the_request_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A summarizer that never answers used to cost the whole ZAKCODE_REQUEST_TIMEOUT per
    # attempt (3600 s on the pod). Streamed, the provider's per-gap bound ends the wait,
    # the loop's fixed interrupt retries run, and every attempt's socket is released.
    streams: list[_StallingStream] = []

    async def serving(**_kwargs: Any) -> Any:
        stream = _StallingStream()
        streams.append(stream)
        return stream

    monkeypatch.setattr(provider_mod.litellm, "acompletion", serving)
    monkeypatch.setattr(provider_mod, "_MIN_REQUEST_INTERVAL_S", 0.0)
    summarizer = LiteLLMProvider(
        model="gpt-4o-mini",
        api_key="offline-test-key",
        settings=Settings(stream_stall_timeout=0.05, request_timeout=3600.0),
    )
    loop = _loop(_StreamingSummarizer(["unused"]), tmp_path, summarizer=summarizer)
    # The retry backoff is the loop's, not the provider's: zero it at the loop rather than
    # patching asyncio.sleep, which the stalling backend above also relies on.
    monkeypatch.setattr(loop, "_retry_delay", lambda exc, attempt: 0.0)
    started = time.monotonic()
    with pytest.raises(TimedOut) as excinfo:
        asyncio.run(loop._summarize_for_compaction(_history(2)))
    # The first attempt ends on the stall floor; each expiry teaches the first-chunk bound
    # to double (ADR-0248), so the attempt that finally raises names a learned bound. Never
    # the whole-call ceiling, which is what the buffered path charged.
    bound = excinfo.value.bound
    assert bound == "ZAKCODE_STREAM_STALL_TIMEOUT" or bound.startswith("first-chunk bound")
    assert bound != "ZAKCODE_REQUEST_TIMEOUT"
    assert time.monotonic() - started < 30.0  # bounded by the stall floor, not the hour
    assert len(streams) == 4  # the first attempt and the loop's three interrupt retries
    assert all(stream.closed for stream in streams)

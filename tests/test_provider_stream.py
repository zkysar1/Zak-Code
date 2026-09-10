"""Hermetic tests for true token streaming in :meth:`LiteLLMProvider.astream`.

No network, no live model: ``litellm.acompletion`` is monkeypatched to return an
async iterator of fake OpenAI-shaped chunk objects (plain ``SimpleNamespace``s, so
the provider's dict-or-attr ``_get`` helper exercises the attribute path). The
tests assert that:

* text deltas across multiple chunks pass through in order;
* a tool call split across chunks (id+name first, arguments later) passes through
  verbatim, keyed by a stable ``index``, for the loop to reassemble;
* usage on the final chunk becomes a :class:`StreamUsage`;
* a :class:`StreamDone` is always emitted last (even for an empty stream);
* an exception from the setup call — and one raised mid-iteration — is mapped to
  the provider error taxonomy rather than leaking the raw litellm exception.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

import zakcode.providers.litellm_provider as provider_mod
from zakcode.messages import Message
from zakcode.providers.base import (
    AuthError,
    ProviderStreamEvent,
    RateLimited,
    RequestFailed,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsage,
    TimedOut,
)
from zakcode.providers.litellm_provider import LiteLLMProvider

# ── Fake chunk builders ───────────────────────────────────────────────────────


def _delta(*, content: str | None = None, tool_calls: list[Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _choice(delta: SimpleNamespace, finish_reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(delta=delta, finish_reason=finish_reason)


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
) -> SimpleNamespace:
    """One streamed chunk with a single choice (and optional trailing usage)."""
    return SimpleNamespace(
        choices=[_choice(_delta(content=content, tool_calls=tool_calls), finish_reason)],
        usage=usage,
    )


def _usage_only_chunk(usage: Any) -> SimpleNamespace:
    """The final include_usage chunk: empty choices, usage populated."""
    return SimpleNamespace(choices=[], usage=usage)


def _tc(
    *,
    index: int,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _fake_stream(chunks: list[Any]) -> Any:
    """Return an awaitable that resolves to an async iterator over ``chunks``.

    Mirrors ``await litellm.acompletion(stream=True)`` -> async-iterable response.
    """

    async def _aiter() -> AsyncIterator[Any]:
        for c in chunks:
            yield c

    async def _acompletion(**_kwargs: Any) -> Any:
        return _aiter()

    return _acompletion


async def _collect(stream: AsyncIterator[ProviderStreamEvent]) -> list[ProviderStreamEvent]:
    return [event async for event in stream]


def _make_provider() -> LiteLLMProvider:
    return LiteLLMProvider(model="gpt-4o-mini", api_key="sk-test")


_MSGS = [Message.user("hello")]


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_text_deltas_across_multiple_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [
        _chunk(content="Hel"),
        _chunk(content="lo, "),
        _chunk(content="world"),
        _chunk(content="", finish_reason="stop"),  # empty content is skipped
    ]
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _fake_stream(chunks))

    events = await _collect(_make_provider().astream(_MSGS))

    texts = [e.text for e in events if isinstance(e, StreamTextDelta)]
    assert texts == ["Hel", "lo, ", "world"]
    assert isinstance(events[-1], StreamDone)
    assert events[-1].finish_reason == "stop"


async def test_astream_forwards_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _aiter() -> AsyncIterator[Any]:
        for c in [_chunk(content="hi", finish_reason="stop")]:
            yield c

    async def _acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _aiter()

    monkeypatch.setattr(provider_mod.litellm, "acompletion", _acompletion)
    rf = {"type": "json_object"}
    await _collect(_make_provider().astream(_MSGS, response_format=rf))
    assert captured["response_format"] == rf


async def test_kwargs_request_streaming_with_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _aiter() -> AsyncIterator[Any]:
        for c in [_chunk(content="hi", finish_reason="stop")]:
            yield c

    async def _acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _aiter()

    monkeypatch.setattr(provider_mod.litellm, "acompletion", _acompletion)

    await _collect(_make_provider().astream(_MSGS))

    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}


async def test_tool_call_split_across_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    # id + name arrive first; the JSON arguments dribble in over later chunks.
    chunks = [
        _chunk(tool_calls=[_tc(index=0, id="call_1", name="read_file", arguments="")]),
        _chunk(tool_calls=[_tc(index=0, arguments='{"path": ')]),
        _chunk(tool_calls=[_tc(index=0, arguments='"a.txt"}')]),
        _chunk(finish_reason="tool_calls"),
    ]
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _fake_stream(chunks))

    events = await _collect(_make_provider().astream(_MSGS))

    deltas = [e for e in events if isinstance(e, StreamToolCallDelta)]
    # One delta per chunk that carried tool_calls (the final finish-reason-only
    # chunk has no tool_calls, so it contributes no delta).
    assert len(deltas) == 3
    # All fragments share the same index.
    assert {d.index for d in deltas} == {0}
    # Only the first fragment carries id/name.
    assert deltas[0].id == "call_1"
    assert deltas[0].name == "read_file"
    assert all(d.id is None and d.name is None for d in deltas[1:])
    # Fragments pass through verbatim; concatenation yields the full JSON.
    assert "".join(d.arguments_delta for d in deltas) == '{"path": "a.txt"}'
    assert isinstance(events[-1], StreamDone)
    assert events[-1].finish_reason == "tool_calls"


async def test_parallel_tool_calls_keep_distinct_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _chunk(
            tool_calls=[
                _tc(index=0, id="a", name="grep", arguments='{"q":'),
                _tc(index=1, id="b", name="glob", arguments='{"p":'),
            ]
        ),
        _chunk(tool_calls=[_tc(index=1, arguments='"*.py"}')]),
        _chunk(tool_calls=[_tc(index=0, arguments='"x"}')]),
        _chunk(finish_reason="tool_calls"),
    ]
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _fake_stream(chunks))

    events = await _collect(_make_provider().astream(_MSGS))

    deltas = [e for e in events if isinstance(e, StreamToolCallDelta)]
    by_index: dict[int, str] = {0: "", 1: ""}
    for d in deltas:
        by_index[d.index] += d.arguments_delta
    assert by_index[0] == '{"q":"x"}'
    assert by_index[1] == '{"p":"*.py"}'


async def test_usage_on_final_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    chunks = [
        _chunk(content="hi", finish_reason="stop"),
        _usage_only_chunk(usage),
    ]
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _fake_stream(chunks))

    events = await _collect(_make_provider().astream(_MSGS))

    usages = [e for e in events if isinstance(e, StreamUsage)]
    assert len(usages) == 1
    assert usages[0].usage.prompt_tokens == 11
    assert usages[0].usage.completion_tokens == 7
    assert usages[0].usage.total_tokens == 18


async def test_done_always_emitted_for_empty_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _fake_stream([]))

    events = await _collect(_make_provider().astream(_MSGS))

    assert len(events) == 1
    assert isinstance(events[0], StreamDone)
    assert events[0].finish_reason is None


async def test_full_ordering_ends_with_done(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    chunks = [
        _chunk(content="Answer: "),
        _chunk(content="42"),
        _chunk(tool_calls=[_tc(index=0, id="c1", name="bash", arguments='{"cmd":"ls"}')]),
        _chunk(finish_reason="stop"),
        _usage_only_chunk(usage),
    ]
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _fake_stream(chunks))

    events = await _collect(_make_provider().astream(_MSGS))

    kinds = [e.event for e in events]
    # Text and tool-call deltas precede usage; done is strictly last.
    assert kinds[0] == "text_delta"
    assert kinds[-1] == "done"
    assert kinds.count("done") == 1
    # Exactly one usage event, and it precedes done.
    assert kinds.count("usage") == 1
    assert kinds.index("usage") < kinds.index("done")


async def test_setup_exception_mapped_to_taxonomy(monkeypatch: pytest.MonkeyPatch) -> None:
    class RateLimitError(Exception):
        pass

    async def _boom(**_kwargs: Any) -> Any:
        raise RateLimitError("slow down")

    monkeypatch.setattr(provider_mod.litellm, "acompletion", _boom)

    with pytest.raises(RateLimited):
        await _collect(_make_provider().astream(_MSGS))


async def test_setup_unknown_exception_degrades_to_request_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(**_kwargs: Any) -> Any:
        raise ValueError("weird")

    monkeypatch.setattr(provider_mod.litellm, "acompletion", _boom)

    with pytest.raises(RequestFailed):
        await _collect(_make_provider().astream(_MSGS))


async def test_midstream_exception_mapped_to_taxonomy(monkeypatch: pytest.MonkeyPatch) -> None:
    class AuthenticationError(Exception):
        pass

    async def _aiter() -> AsyncIterator[Any]:
        yield _chunk(content="partial")
        raise AuthenticationError("bad key")

    async def _acompletion(**_kwargs: Any) -> Any:
        return _aiter()

    monkeypatch.setattr(provider_mod.litellm, "acompletion", _acompletion)

    stream = _make_provider().astream(_MSGS)
    seen: list[ProviderStreamEvent] = []
    with pytest.raises(AuthError):
        async for event in stream:
            seen.append(event)

    # The text delta before the failure still surfaced; no raw vendor exception leaked.
    assert [e.text for e in seen if isinstance(e, StreamTextDelta)] == ["partial"]


async def test_stream_keeps_a_bounded_sample_of_raw_deltas(monkeypatch: Any) -> None:
    # The loop reads this when a completion delivers nothing (2026-08-28, coach: 622 tokens
    # generated, no text/thinking/tool call) — every field the backend sent, head and tail.
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=5, total_tokens=6)
    chunks = [_chunk(content=c) for c in "abcd"] + [
        _chunk(content="", finish_reason="stop"),
        _usage_only_chunk(usage),
    ]
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _fake_stream(chunks))
    provider = _make_provider()
    assert provider.last_stream_sample is None
    await _collect(provider.astream(_MSGS))
    sample = provider.last_stream_sample
    assert sample is not None
    assert sample["chunks"] == 6 and sample["finish_reason"] == "stop"
    # Three from the head, the last two of the five deltas from the tail; the usage-only
    # chunk carries no delta and is not sampled.
    assert len(sample["deltas"]) == 5
    assert '"content": "a"' in sample["deltas"][0]
    assert '"content": ""' in sample["deltas"][-1]


# ── the streaming stall deadline (ADR-0120, Zak-Code #176) ────────────────────
# Measured on the zc-03 pod: a streaming response's HEADERS arrive in 0.12-2.0s at every
# prompt size while the first DATA chunk waits out the whole prefill (0.29s at 5k prompt
# tokens, 25.5s at 22k, 126.1s at 60k). A socket-level read timeout is satisfied by the
# headers and never fires again, so a backend that sent headers and then nothing held one
# call for 45 minutes. The bound sits on the ITERATOR, and is per-GAP.


class _StallingStream:
    """Yields ``chunks`` with ``gap`` seconds between them, then hangs forever."""

    def __init__(self, chunks: list[Any], gap: float = 0.0, *, hang: bool = True) -> None:
        self._chunks, self._gap, self._hang = list(chunks), gap, hang
        self.closed = False
        self.i = 0

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        if self.i < len(self._chunks):
            await asyncio.sleep(self._gap)
            self.i += 1
            return self._chunks[self.i - 1]
        if self._hang:
            await asyncio.sleep(3600)  # the wedged backend: never another byte
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


def _serving(stream: Any) -> Any:
    async def _acompletion(**_kwargs: Any) -> Any:
        return stream

    return _acompletion


def _fast_provider(stall: float = 0.05) -> LiteLLMProvider:
    return LiteLLMProvider(model="gpt-4o-mini", api_key="sk-test", stream_stall_timeout=stall)


async def test_a_stream_that_never_sends_a_chunk_is_timed_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _StallingStream([])
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _serving(stream))
    with pytest.raises(TimedOut) as excinfo:
        await _collect(_fast_provider().astream(_MSGS))
    message = str(excinfo.value)
    assert "no stream data at all" in message
    assert "ZAKCODE_STREAM_STALL_TIMEOUT" in message
    assert stream.closed  # the socket is released, not leaked


async def test_a_stall_midway_reports_how_far_the_stream_got(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _StallingStream([_chunk(content="par"), _chunk(content="tial")])
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _serving(stream))
    with pytest.raises(TimedOut) as excinfo:
        await _collect(_fast_provider().astream(_MSGS))
    assert "stalled after 2 chunk(s)" in str(excinfo.value)


async def test_the_deadline_is_per_gap_so_a_long_stream_is_never_punished_for_its_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Six 20ms gaps = 120ms of streaming under a 60ms bound: a whole-call ceiling would
    # kill this healthy generation, a per-gap one must not. This is the property that
    # makes request_timeout the wrong knob for the job.
    chunks = [_chunk(content=str(i)) for i in range(5)]
    chunks.append(_chunk(content="", finish_reason="stop"))
    stream = _StallingStream(chunks, gap=0.02, hang=False)
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _serving(stream))
    events = await _collect(
        LiteLLMProvider(model="gpt-4o-mini", api_key="sk-test", stream_stall_timeout=0.06).astream(
            _MSGS
        )
    )
    assert [e.text for e in events if isinstance(e, StreamTextDelta)] == ["0", "1", "2", "3", "4"]
    assert isinstance(events[-1], StreamDone) and events[-1].finish_reason == "stop"


async def test_a_healthy_stream_is_unaffected_by_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [_chunk(content="fine"), _chunk(content="", finish_reason="stop")]
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _fake_stream(chunks))
    events = await _collect(_make_provider().astream(_MSGS))
    assert [e.text for e in events if isinstance(e, StreamTextDelta)] == ["fine"]


async def test_a_timed_out_stream_records_what_it_saw_for_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # last_stream_sample is how an empty completion is diagnosed; a stall must still fill
    # it, and chunks=0 is exactly the tell that the prefill never finished.
    stream = _StallingStream([])
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _serving(stream))
    provider = _fast_provider()
    with pytest.raises(TimedOut):
        await _collect(provider.astream(_MSGS))
    assert provider.last_stream_sample["chunks"] == 0


def test_stream_stall_timeout_resolution_prefers_the_explicit_value() -> None:
    from zakcode.config import Settings

    assert LiteLLMProvider(model="gpt-4o-mini", api_key="sk-test").stream_stall_timeout == 600.0
    assert (
        LiteLLMProvider(
            model="gpt-4o-mini", api_key="sk-test", stream_stall_timeout=12.5
        ).stream_stall_timeout
        == 12.5
    )
    settings = Settings(stream_stall_timeout=42.0)
    assert (
        LiteLLMProvider(
            model="gpt-4o-mini", api_key="sk-test", settings=settings
        ).stream_stall_timeout
        == 42.0
    )
    # …and it is INDEPENDENT of request_timeout, which operators raise for slow backends
    # (the pod runs 3600): a per-gap bound that large would not catch a 45-minute stall.
    both = LiteLLMProvider(
        model="gpt-4o-mini", api_key="sk-test", settings=Settings(request_timeout=3600.0)
    )
    assert both.request_timeout == 3600.0
    assert both.stream_stall_timeout == 600.0

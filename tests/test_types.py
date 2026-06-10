"""Tests for the provider ABC, value objects, streaming events, and error taxonomy."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from tests.conftest import StubProvider
from zakcode.messages import Message
from zakcode.providers.base import (
    AuthError,
    Capabilities,
    ContextWindowExceeded,
    LLMResult,
    Provider,
    ProviderError,
    ProviderStreamEvent,
    RateLimited,
    RequestFailed,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsage,
    ToolCall,
)
from zakcode.usage import Usage


def test_toolcall_defaults() -> None:
    tc = ToolCall(id="c1", name="read_file")
    assert tc.arguments == {}
    assert tc.id == "c1"
    assert tc.name == "read_file"


def test_llmresult_has_tool_calls() -> None:
    empty = LLMResult(text="hello")
    assert not empty.has_tool_calls

    with_calls = LLMResult(
        text="",
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
    )
    assert with_calls.has_tool_calls


def test_capabilities_defaults() -> None:
    caps = Capabilities()
    assert caps.supports_tools is True
    assert caps.supports_vision is False
    assert caps.supports_caching is False
    assert caps.context_window == 8192
    assert caps.max_output is None


def test_streaming_event_discriminator() -> None:
    adapter = TypeAdapter(ProviderStreamEvent)

    text_delta = adapter.validate_python({"event": "text_delta", "text": "hi"})
    assert isinstance(text_delta, StreamTextDelta)
    assert text_delta.text == "hi"

    tool_delta = adapter.validate_python(
        {"event": "tool_call_delta", "index": 0, "id": "c1", "name": "read"}
    )
    assert isinstance(tool_delta, StreamToolCallDelta)
    assert tool_delta.index == 0

    usage_ev = adapter.validate_python({"event": "usage"})
    assert isinstance(usage_ev, StreamUsage)
    assert usage_ev.usage == Usage()

    done = adapter.validate_python({"event": "done", "finish_reason": "stop"})
    assert isinstance(done, StreamDone)
    assert done.finish_reason == "stop"


def test_provider_abc_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        Provider()  # type: ignore[abstract]


async def test_provider_astream_default() -> None:
    """The default astream wraps acomplete into a stream of events."""
    result = LLMResult(
        text="hello",
        tool_calls=[ToolCall(id="c1", name="read", arguments={"path": "a"})],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        finish_reason="stop",
    )
    stub = StubProvider(result=result)

    events = []
    async for event in stub.astream([Message.user("hi")]):
        events.append(event)

    assert len(events) == 4
    assert isinstance(events[0], StreamTextDelta)
    assert events[0].text == "hello"
    assert isinstance(events[1], StreamToolCallDelta)
    assert events[1].name == "read"
    assert isinstance(events[2], StreamUsage)
    assert events[2].usage.prompt_tokens == 10
    assert isinstance(events[3], StreamDone)
    assert events[3].finish_reason == "stop"


async def test_provider_astream_default_forwards_response_format() -> None:
    """The default astream folds response_format into its acomplete call, so structured output
    works on the streaming path for a minimal Provider that only implements acomplete."""

    class _Recording(StubProvider):
        def __init__(self) -> None:
            super().__init__(result=LLMResult(text="ok"))
            self.seen: list[dict | None] = []

        async def acomplete(  # noqa: ANN001
            self, messages, *, system=None, tools=None, response_format=None, **kw
        ):
            self.seen.append(response_format)
            return self._result

    rf = {"type": "json_object"}
    stub = _Recording()
    async for _ in stub.astream([Message.user("hi")], response_format=rf):
        pass
    assert stub.seen == [rf]


def test_error_hierarchy() -> None:
    assert issubclass(AuthError, ProviderError)
    assert issubclass(ContextWindowExceeded, ProviderError)
    assert issubclass(RateLimited, ProviderError)
    assert issubclass(RequestFailed, ProviderError)
    assert issubclass(ProviderError, Exception)


def test_ratelimited_retry_after() -> None:
    err = RateLimited("slow down", retry_after=30.0)
    assert str(err) == "slow down"
    assert err.retry_after == 30.0

    err_no_retry = RateLimited("slow")
    assert err_no_retry.retry_after is None

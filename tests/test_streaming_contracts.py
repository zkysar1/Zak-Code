"""Tests for the M1 streaming contracts (provider stream events, agent events,
and the default non-streaming Provider.astream implementation)."""

from __future__ import annotations

import pydantic
import pytest

from zakcode.events import (
    AgentDone,
    AgentTextDelta,
    AgentToolCall,
)
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsage,
    ToolCall,
)
from zakcode.usage import Usage

# ── union round-trips (the property the future SSE/WS transport relies on) ────

_PROVIDER_ADAPTER = pydantic.TypeAdapter(
    StreamTextDelta | StreamToolCallDelta | StreamUsage | StreamDone
)


def test_provider_stream_event_union_roundtrip() -> None:
    for original in [
        StreamTextDelta(text="hi"),
        StreamToolCallDelta(index=0, id="c1", name="read_file", arguments_delta='{"path"'),
        StreamUsage(usage=Usage(total_tokens=5)),
        StreamDone(finish_reason="stop"),
    ]:
        restored = _PROVIDER_ADAPTER.validate_json(original.model_dump_json())
        assert restored == original


_AGENT_ADAPTER = pydantic.TypeAdapter(AgentTextDelta | AgentToolCall | AgentDone)


def test_agent_event_union_roundtrip() -> None:
    for original in [
        AgentTextDelta(text="thinking..."),
        AgentToolCall(id="c1", name="bash", arguments={"command": "ls"}),
        AgentDone(stop_reason="completed", iterations=2, usage=Usage(total_tokens=9)),
    ]:
        restored = _AGENT_ADAPTER.validate_json(original.model_dump_json())
        assert restored == original
        assert restored.event == original.event


# ── the default Provider.astream wraps acomplete into a one-shot stream ────────


class _ScriptedProvider(Provider):
    """Returns a fixed LLMResult; uses the inherited default astream()."""

    def __init__(self, result: LLMResult) -> None:
        self._result = result

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        return self._result

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


async def test_default_astream_emits_text_then_usage_then_done() -> None:
    provider = _ScriptedProvider(
        LLMResult(text="hello", usage=Usage(total_tokens=3), finish_reason="stop")
    )
    events = [e async for e in provider.astream([Message.user("hi")])]
    kinds = [e.event for e in events]
    assert kinds == ["text_delta", "usage", "done"]
    assert isinstance(events[0], StreamTextDelta) and events[0].text == "hello"
    assert isinstance(events[-1], StreamDone) and events[-1].finish_reason == "stop"


async def test_default_astream_emits_tool_call_deltas() -> None:
    provider = _ScriptedProvider(
        LLMResult(
            tool_calls=[
                ToolCall(id="c1", name="read_file", arguments={"path": "a.py"}),
                ToolCall(id="c2", name="bash", arguments={"command": "ls"}),
            ],
            usage=Usage(total_tokens=7),
        )
    )
    events = [e async for e in provider.astream([Message.user("go")])]
    deltas = [e for e in events if isinstance(e, StreamToolCallDelta)]
    assert [d.index for d in deltas] == [0, 1]
    assert deltas[0].name == "read_file"
    assert '"path"' in deltas[0].arguments_delta
    assert events[-1].event == "done"


async def test_default_astream_empty_completion_still_terminates() -> None:
    provider = _ScriptedProvider(LLMResult())
    events = [e async for e in provider.astream([Message.user("hi")])]
    # No text, no tools: just usage + done — never hangs, always terminates.
    assert [e.event for e in events] == ["usage", "done"]


def test_agent_done_mirrors_turn_summary_fields() -> None:
    done = AgentDone(stop_reason="max_iterations", iterations=50, usage=Usage(total_tokens=1))
    assert done.stop_reason == "max_iterations"
    assert done.iterations == 50


def test_unknown_event_is_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        _AGENT_ADAPTER.validate_python({"event": "nope"})

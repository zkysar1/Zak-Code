"""Tests for ClaudeCodeProvider — the Claude Code session bridge provider.

Every test mocks the bridge as an async callable returning a canned string. Zero API
cost, zero network. The provider imports no Claude Code internals, so it is fully
exercised here in isolation.
"""

from __future__ import annotations

import pytest

from zds_llm_provider.claude_code import (
    ClaudeCodeProvider,
    CompletionBridge,
    _format_messages,
)
from zds_llm_provider.messages import Message, ToolResultBlock, ToolUseBlock
from zds_llm_provider.types import (
    LLMResult,
    RequestFailed,
    StreamDone,
    StreamTextDelta,
    StreamUsage,
)


def _canned_bridge(response: str, *, record: list[str] | None = None) -> CompletionBridge:
    """A stub bridge that returns ``response`` and optionally records prompts it saw."""

    async def bridge(prompt: str) -> str:
        if record is not None:
            record.append(prompt)
        return response

    return bridge


async def test_acomplete_simple_text() -> None:
    provider = ClaudeCodeProvider(_canned_bridge("hello from the bridge"))
    result = await provider.acomplete([Message.user("hi")])
    assert isinstance(result, LLMResult)
    assert result.text == "hello from the bridge"
    assert result.tool_calls == []
    assert result.finish_reason == "stop"
    assert result.usage.total_tokens == 0  # Claude Code exposes no token counts


async def test_acomplete_with_tools() -> None:
    response = '<tool_call>\n{"name": "read_file", "arguments": {"path": "x.py"}}\n</tool_call>'
    seen: list[str] = []
    provider = ClaudeCodeProvider(_canned_bridge(response, record=seen))
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    result = await provider.acomplete([Message.user("read x.py")], tools=tools)
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "read_file"
    assert call.arguments == {"path": "x.py"}
    assert result.finish_reason == "tool_calls"
    # The text tool protocol was appended to the prompt the bridge received.
    assert "read_file" in seen[0]


async def test_acomplete_unoffered_tool_not_recovered() -> None:
    """A tool-call naming a tool we never offered is left as text, not executed."""
    response = '<tool_call>\n{"name": "rm_rf", "arguments": {"path": "/"}}\n</tool_call>'
    provider = ClaudeCodeProvider(_canned_bridge(response))
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    result = await provider.acomplete([Message.user("go")], tools=tools)
    assert result.tool_calls == []


async def test_acomplete_bridge_error() -> None:
    async def failing(prompt: str) -> str:
        raise RuntimeError("bridge exploded")

    provider = ClaudeCodeProvider(failing)
    with pytest.raises(RequestFailed):
        await provider.acomplete([Message.user("hi")])


def test_format_messages_includes_system() -> None:
    out = _format_messages([Message.user("hello")], system="You are a test.")
    assert "[System]" in out
    assert "You are a test." in out
    assert "[User]" in out
    assert "hello" in out


def test_format_messages_tool_results() -> None:
    msg = Message.tool_results([ToolResultBlock(tool_use_id="call_0", output="42 matches")])
    out = _format_messages([msg])
    assert "[TOOL_RESULT call_0]" in out
    assert "42 matches" in out


def test_format_messages_renders_tool_use() -> None:
    msg = Message(
        role="assistant",
        blocks=[ToolUseBlock(id="call_0", name="grep", input={"q": "x"})],
    )
    out = _format_messages([msg])
    assert "[ASSISTANT TOOL_CALL] grep" in out
    assert '"q": "x"' in out


def test_count_tokens_rough_estimate() -> None:
    provider = ClaudeCodeProvider(_canned_bridge(""))
    n = provider.count_tokens([Message.user("a" * 40)], system="b" * 40)
    assert isinstance(n, int)
    assert n == 20  # (40 + 40) // 4


def test_capabilities_defaults() -> None:
    caps = ClaudeCodeProvider(_canned_bridge("")).capabilities()
    assert caps.context_window == 200_000
    assert caps.supports_tools is True
    assert caps.supports_vision is True


def test_capabilities_custom() -> None:
    provider = ClaudeCodeProvider(_canned_bridge(""), context_window=64_000, max_output=4096)
    caps = provider.capabilities()
    assert caps.context_window == 64_000
    assert caps.max_output == 4096


async def test_astream_default_works() -> None:
    provider = ClaudeCodeProvider(_canned_bridge("streamed text"))
    events = [event async for event in provider.astream([Message.user("hi")])]
    assert any(isinstance(e, StreamTextDelta) and e.text == "streamed text" for e in events)
    assert any(isinstance(e, StreamUsage) for e in events)
    assert isinstance(events[-1], StreamDone)

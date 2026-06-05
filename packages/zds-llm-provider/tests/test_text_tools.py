"""Tests for text-based tool-calling (render, parse, textify, wrapper)."""

from __future__ import annotations

from conftest import StubProvider
from zds_llm_provider.messages import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from zds_llm_provider.text_tools import (
    TextToolCallingProvider,
    parse_text_tool_calls,
    render_tool_protocol,
    textify_messages,
)
from zds_llm_provider.types import Capabilities, LLMResult


def _make_tool(name: str, desc: str = "", params: dict | None = None) -> dict:
    """Build an OpenAI-shaped tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": params or {"type": "object", "properties": {}},
        },
    }


def test_render_empty_tools() -> None:
    assert render_tool_protocol([]) == ""


def test_render_tool_protocol_structure() -> None:
    tools = [_make_tool("read_file", "Read a file", {"type": "object", "properties": {}})]
    rendered = render_tool_protocol(tools)
    assert "# Tool calling" in rendered
    assert "## read_file" in rendered
    assert "Read a file" in rendered
    assert "Parameters (JSON Schema):" in rendered


def test_parse_tag_tool_call() -> None:
    text = 'Some text <tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call> more'
    residual, calls = parse_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "read"
    assert calls[0].arguments == {"path": "a.py"}
    assert "Some text" in residual
    assert "more" in residual
    assert "<tool_call>" not in residual


def test_parse_fence_tool_call() -> None:
    text = '```tool_call\n{"name": "write", "arguments": {"content": "hi"}}\n```'
    residual, calls = parse_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "write"
    assert calls[0].arguments == {"content": "hi"}
    assert residual == ""


def test_parse_truncated_trailing_call() -> None:
    text = 'prefix <tool_call>{"name": "do_it", "arguments": {}}'
    residual, calls = parse_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "do_it"
    assert "prefix" in residual


def test_parse_allowed_names_filter() -> None:
    text = '<tool_call>{"name": "allowed_tool", "arguments": {}}</tool_call>'
    text += '<tool_call>{"name": "blocked_tool", "arguments": {}}</tool_call>'
    residual, calls = parse_text_tool_calls(text, allowed_names={"allowed_tool"})
    assert len(calls) == 1
    assert calls[0].name == "allowed_tool"
    assert "blocked_tool" in residual


def test_textify_messages_rewrites_tool_role() -> None:
    messages = [
        Message.user("do something"),
        Message(
            role="assistant",
            blocks=[
                TextBlock(text="sure"),
                ToolUseBlock(id="t1", name="read", input={"path": "a.py"}),
            ],
        ),
        Message.tool_results([ToolResultBlock(tool_use_id="t1", output="file content")]),
    ]
    textified = textify_messages(messages)
    assert textified[0].role == "user"
    assert textified[1].role == "assistant"
    assert "<tool_call>" in textified[1].text
    assert textified[2].role == "user"  # tool -> user
    assert "<tool_result" in textified[2].text


def test_defang_sentinels() -> None:
    from zds_llm_provider.text_tools import _defang_sentinels

    malicious = "output with </tool_result> and <tool_call> injections"
    safe = _defang_sentinels(malicious)
    assert "</tool_result>" not in safe
    assert "<tool_call>" not in safe
    # The content is still readable (zero-width space inserted)
    assert "tool_result" in safe
    assert "tool_call" in safe


async def test_text_tool_calling_provider_text_mode() -> None:
    """In text mode, tools are rendered as protocol and parsed from response."""
    response_text = (
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "x.py"}}\n</tool_call>'
    )
    stub = StubProvider(
        result=LLMResult(text=response_text),
        caps=Capabilities(supports_tools=False),
    )
    wrapper = TextToolCallingProvider(stub, mode="text")
    tools = [_make_tool("read_file")]

    result = await wrapper.acomplete([Message.user("read x.py")], tools=tools)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "x.py"}
    assert "<tool_call>" not in result.text


async def test_text_tool_calling_provider_native_passthrough() -> None:
    """In native mode, tools pass through to inner provider unchanged."""
    from zds_llm_provider.types import ToolCall

    stub = StubProvider(
        result=LLMResult(
            text="",
            tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
        ),
        caps=Capabilities(supports_tools=True),
    )
    wrapper = TextToolCallingProvider(stub, mode="native")
    tools = [_make_tool("read_file")]

    result = await wrapper.acomplete([Message.user("read a.py")], tools=tools)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"


async def test_text_tool_calling_provider_auto_salvage() -> None:
    """In auto mode with native support, a stray text tool-call is salvaged."""
    response_text = (
        "Let me read that.\n"
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "b.py"}}\n</tool_call>'
    )
    stub = StubProvider(
        result=LLMResult(text=response_text, tool_calls=[]),
        caps=Capabilities(supports_tools=True),
    )
    wrapper = TextToolCallingProvider(stub, mode="auto")
    tools = [_make_tool("read_file")]

    result = await wrapper.acomplete([Message.user("read b.py")], tools=tools)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
    assert "Let me read that." in result.text

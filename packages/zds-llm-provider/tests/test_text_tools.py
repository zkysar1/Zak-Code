"""Tests for text-based tool-calling (render, parse, textify, wrapper)."""

from __future__ import annotations

import json

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
from zds_llm_provider.types import Capabilities, LLMResult, StreamDone


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


def test_parse_argument_value_containing_close_marker() -> None:
    # A write whose content legitimately contains the literal </tool_call> must NOT
    # truncate the body: the balanced-brace scan ignores the marker inside the string.
    content = "Docs about the protocol: emit </tool_call> to end a call."
    payload = json.dumps({"name": "write_file", "arguments": {"content": content}})
    text = f"<tool_call>\n{payload}\n</tool_call>"
    residual, calls = parse_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert calls[0].arguments["content"] == content
    assert "<tool_call>" not in residual
    assert "</tool_call>" not in residual


def test_parse_nested_braces_in_arguments() -> None:
    # Nested JSON objects in arguments must brace-balance, not stop at the first }.
    payload = json.dumps({"name": "configure", "arguments": {"opts": {"a": 1, "b": {"c": 2}}}})
    residual, calls = parse_text_tool_calls(f"<tool_call>{payload}</tool_call>")
    assert len(calls) == 1
    assert calls[0].arguments == {"opts": {"a": 1, "b": {"c": 2}}}


def test_parse_trailing_prose_inside_block_is_salvaged() -> None:
    # A weak model often appends prose after the JSON object inside the block; the
    # leading JSON object is salvaged via raw_decode and the prose is dropped.
    text = '<tool_call>{"name": "run", "arguments": {"x": 1}} done now, let me run it</tool_call>'
    residual, calls = parse_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "run"
    assert calls[0].arguments == {"x": 1}
    assert "done now" not in residual


def test_parse_trailing_prose_without_close_tag() -> None:
    # Same salvage on the truncated path (no close tag): leading object recovered.
    text = '<tool_call>{"name": "run", "arguments": {}} ok'
    _residual, calls = parse_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "run"


def test_parse_prose_before_json_is_not_misparsed() -> None:
    # raw_decode only consumes a LEADING object: prose before the JSON leaves the
    # block unrecovered (no false positive), and the name-gate still applies.
    text = "<tool_call>let me think {not json here}</tool_call>"
    residual, calls = parse_text_tool_calls(text)
    assert calls == []
    assert "let me think" in residual


def test_parse_two_calls_one_with_embedded_marker() -> None:
    # The trailing-close-tag absorption must not swallow a following separate call.
    first = json.dumps({"name": "write_file", "arguments": {"content": "x </tool_call> y"}})
    second = json.dumps({"name": "read_file", "arguments": {"path": "a.py"}})
    text = f"<tool_call>{first}</tool_call>\n<tool_call>{second}</tool_call>"
    _residual, calls = parse_text_tool_calls(text)
    assert [c.name for c in calls] == ["write_file", "read_file"]
    assert calls[0].arguments["content"] == "x </tool_call> y"


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


async def test_forwards_response_format_to_inner() -> None:
    # The wrapper must forward response_format to the inner provider on BOTH the tool-less
    # passthrough and the text-mode path (structured output composes with the text protocol).
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
    inner = _Recording()
    wrapper = TextToolCallingProvider(inner, mode="text")
    await wrapper.acomplete([Message.user("hi")], response_format=rf)  # tool-less passthrough
    await wrapper.acomplete(  # text-mode path (protocol injected)
        [Message.user("hi")], tools=[_make_tool("read_file")], response_format=rf
    )
    assert inner.seen == [rf, rf]


async def test_astream_forwards_response_format_to_inner() -> None:
    # astream is the production streaming path the agent loop uses, so the forwarding it relies
    # on must be pinned: response_format must reach the inner provider on BOTH the text-mode
    # branch (buffered via the wrapper's acomplete) and the native auto branch (inner.astream).
    class _Recording(StubProvider):
        def __init__(self) -> None:
            super().__init__(result=LLMResult(text="ok"), caps=Capabilities(supports_tools=True))
            self.seen: list[dict | None] = []

        async def acomplete(  # noqa: ANN001
            self, messages, *, system=None, tools=None, response_format=None, **kw
        ):
            self.seen.append(response_format)
            return self._result

        async def astream(  # noqa: ANN001
            self, messages, *, system=None, tools=None, response_format=None, **kw
        ):
            self.seen.append(response_format)
            yield StreamDone(finish_reason="stop")

    rf = {"type": "json_object"}
    tools = [_make_tool("read_file")]

    inner_text = _Recording()  # text mode buffers via the wrapper's acomplete -> inner.acomplete
    async for _ in TextToolCallingProvider(inner_text, mode="text").astream(
        [Message.user("hi")], tools=tools, response_format=rf
    ):
        pass
    assert inner_text.seen == [rf]

    inner_native = _Recording()  # native auto path streams inner.astream directly
    async for _ in TextToolCallingProvider(inner_native, mode="auto").astream(
        [Message.user("hi")], tools=tools, response_format=rf
    ):
        pass
    assert inner_native.seen == [rf]


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

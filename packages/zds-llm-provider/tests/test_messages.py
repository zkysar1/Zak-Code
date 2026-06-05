"""Tests for the canonical conversation model (messages module)."""

from __future__ import annotations

from zds_llm_provider.messages import (
    ContentBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def test_user_message_constructor() -> None:
    msg = Message.user("hello")
    assert msg.role == "user"
    assert len(msg.blocks) == 1
    assert isinstance(msg.blocks[0], TextBlock)
    assert msg.blocks[0].text == "hello"


def test_system_message_constructor() -> None:
    msg = Message.system("you are helpful")
    assert msg.role == "system"
    assert msg.text == "you are helpful"


def test_assistant_text_constructor() -> None:
    msg = Message.assistant_text("sure thing")
    assert msg.role == "assistant"
    assert msg.text == "sure thing"


def test_tool_results_constructor() -> None:
    results = [
        ToolResultBlock(tool_use_id="t1", output="ok"),
        ToolResultBlock(tool_use_id="t2", output="done", is_error=False),
    ]
    msg = Message.tool_results(results)
    assert msg.role == "tool"
    assert len(msg.blocks) == 2


def test_text_accessor_concatenates() -> None:
    msg = Message(
        role="assistant",
        blocks=[
            TextBlock(text="hello "),
            ToolUseBlock(id="t1", name="read", input={}),
            TextBlock(text="world"),
        ],
    )
    assert msg.text == "hello world"


def test_tool_uses_accessor() -> None:
    tool1 = ToolUseBlock(id="t1", name="read", input={"path": "a.py"})
    tool2 = ToolUseBlock(id="t2", name="write", input={"path": "b.py"})
    msg = Message(
        role="assistant",
        blocks=[TextBlock(text="let me"), tool1, TextBlock(text="then"), tool2],
    )
    assert msg.tool_uses == [tool1, tool2]


def test_content_block_discriminated_union() -> None:
    """ContentBlock deserializes correctly based on the 'type' discriminator."""
    from pydantic import TypeAdapter

    adapter = TypeAdapter(ContentBlock)

    text = adapter.validate_python({"type": "text", "text": "hi"})
    assert isinstance(text, TextBlock)

    tool_use = adapter.validate_python(
        {"type": "tool_use", "id": "t1", "name": "read", "input": {}}
    )
    assert isinstance(tool_use, ToolUseBlock)

    tool_result = adapter.validate_python(
        {"type": "tool_result", "tool_use_id": "t1", "output": "ok"}
    )
    assert isinstance(tool_result, ToolResultBlock)

    thinking = adapter.validate_python({"type": "thinking", "text": "hmm"})
    assert isinstance(thinking, ThinkingBlock)

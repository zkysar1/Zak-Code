"""Tests for the Claude Code transcript projection (``hooks/transcript.py``).

The projection's contract is *the consumers' read pattern*. So the central test does not
assert on an invented schema — it reads the rendered ``.jsonl`` back the SAME way Claude
Code's Stop-hook reader (``trailing-text-detector.py``) does (find the last ``type ==
"assistant"`` line, concatenate its ``message.content`` ``text`` blocks) and an audit
reader does (scan ``content`` for ``tool_use`` / ``tool_result``), and asserts the data is
findable that way. Pure, offline, no API.
"""

from __future__ import annotations

import json

from zakcode.hooks.transcript import render_claude_code_transcript
from zakcode.messages import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def _lines(transcript: str) -> list[dict]:
    """Parse every line as JSON the way a line-oriented reader would (``for line in f``)."""
    return [json.loads(line) for line in transcript.splitlines() if line.strip()]


def _last_assistant_text(transcript: str) -> str:
    """Re-express ``trailing-text-detector.py``'s read: last assistant turn's text blocks.

    This mirrors the consumer's logic rather than importing it — the test passes iff the
    projection is readable exactly how that hook reads a real Claude Code transcript.
    """
    for evt in reversed(_lines(transcript)):
        if evt.get("type") != "assistant":
            continue
        msg = evt.get("message") or {}
        content = msg.get("content") or []
        if not isinstance(content, list):
            continue
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _session() -> list[Message]:
    """A representative turn: user prompt, assistant text + a tool call, then a result."""
    return [
        Message.user("please grep for TODO"),
        Message(
            role="assistant",
            blocks=[
                TextBlock(text="On it. Let me search the tree."),
                ToolUseBlock(id="call_0", name="grep", input={"pattern": "TODO"}),
            ],
        ),
        Message.tool_results(
            [ToolResultBlock(tool_use_id="call_0", output="3 matches", is_error=False)]
        ),
    ]


def test_every_line_is_valid_json() -> None:
    transcript = render_claude_code_transcript(_session(), session_id="s1", cwd="/repo")
    raw = transcript.splitlines()
    assert raw, "expected at least one line"
    for line in raw:
        json.loads(line)  # raises if any line is not valid JSON
    assert transcript.endswith("\n"), "transcript should be newline-terminated"


def test_assistant_text_found_the_way_the_stop_hook_reads_it() -> None:
    transcript = render_claude_code_transcript(_session(), session_id="s1", cwd="/repo")
    # Read it back exactly as trailing-text-detector.py does.
    assert _last_assistant_text(transcript) == "On it. Let me search the tree."


def test_assistant_line_shape_matches_reader_expectations() -> None:
    transcript = render_claude_code_transcript(_session())
    assistant = [e for e in _lines(transcript) if e.get("type") == "assistant"]
    assert len(assistant) == 1
    msg = assistant[0]["message"]
    assert msg["role"] == "assistant"
    assert isinstance(msg["content"], list)
    types = {b["type"] for b in msg["content"]}
    assert types == {"text", "tool_use"}


def test_tool_use_block_is_findable_with_name_and_input() -> None:
    """Mirrors aspirations-rejection-audit.py: scan content for a named tool_use + input."""
    transcript = render_claude_code_transcript(_session())
    found = None
    for evt in _lines(transcript):
        msg = evt.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    found = item
    assert found is not None
    assert found["name"] == "grep"
    assert (found.get("input") or {}).get("pattern") == "TODO"


def test_tool_result_rides_on_a_user_line() -> None:
    transcript = render_claude_code_transcript(_session())
    results = []
    for evt in _lines(transcript):
        if evt.get("type") != "user":
            continue
        content = (evt.get("message") or {}).get("content")
        if isinstance(content, list):
            results.extend(b for b in content if b.get("type") == "tool_result")
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "call_0"
    assert results[0]["content"] == "3 matches"
    assert results[0]["is_error"] is False


def test_plain_user_message_content_is_a_string() -> None:
    """An audit reader accepts string ``content`` for user turns — emit prose as a string."""
    transcript = render_claude_code_transcript([Message.user("hello there")])
    line = _lines(transcript)[0]
    assert line["type"] == "user"
    assert line["message"]["content"] == "hello there"


def test_empty_input_renders_empty_string() -> None:
    assert render_claude_code_transcript([]) == ""


def test_system_messages_are_omitted() -> None:
    transcript = render_claude_code_transcript(
        [Message.system("you are helpful"), Message.user("hi")]
    )
    lines = _lines(transcript)
    assert len(lines) == 1
    assert lines[0]["type"] == "user"


def test_thinking_blocks_are_omitted_but_text_survives() -> None:
    transcript = render_claude_code_transcript(
        [
            Message(
                role="assistant",
                blocks=[
                    ThinkingBlock(text="secret reasoning", signature="sig"),
                    TextBlock(text="visible answer"),
                ],
            )
        ]
    )
    assert _last_assistant_text(transcript) == "visible answer"
    # The thinking text must not leak into the transcript at all.
    assert "secret reasoning" not in transcript


def test_never_raises_on_unusual_or_empty_blocks() -> None:
    """A message with no blocks, an empty-text block, and an odd role must not crash."""
    messages = [
        Message(role="assistant", blocks=[]),  # empty content
        Message(role="assistant", blocks=[TextBlock(text="")]),  # empty text
        Message(role="tool", blocks=[ToolResultBlock(tool_use_id="x", output="")]),
        Message(role="system", blocks=[TextBlock(text="dropped")]),
    ]
    transcript = render_claude_code_transcript(messages)
    # Whatever it produced, every line must still be valid JSON.
    for line in transcript.splitlines():
        json.loads(line)


def test_empty_blocks_only_still_renders_valid_lines() -> None:
    transcript = render_claude_code_transcript([Message(role="assistant", blocks=[])])
    lines = _lines(transcript)
    assert len(lines) == 1
    assert lines[0]["type"] == "assistant"
    assert lines[0]["message"]["content"] == []

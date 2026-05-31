"""Hermetic tests for the streaming TUI renderer.

No network, no model, no real provider. Events are fed through a fake async
generator into a :class:`StreamRenderer` whose console writes to an in-memory
``io.StringIO`` (``force_terminal=False`` so no ANSI escapes are emitted).
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator

import pytest
from rich.console import Console

from zakcode.cli.render import StreamRenderer, split_on_safe_boundary
from zakcode.events import (
    AgentDone,
    AgentEvent,
    AgentStatus,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.usage import Usage


def _usage(prompt: int = 0, completion: int = 0, cost: float = 0.0) -> Usage:
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cost_usd=cost,
    )


async def _astream(events: list[AgentEvent]) -> AsyncIterator[AgentEvent]:
    for event in events:
        yield event


def _make_renderer() -> tuple[StreamRenderer, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=200, no_color=True)
    return StreamRenderer(console=console), buffer


# --------------------------------------------------------------------------
# split_on_safe_boundary (pure function)
# --------------------------------------------------------------------------


def test_split_plain_text_flushes_to_last_newline() -> None:
    emit, keep = split_on_safe_boundary("alpha\nbeta\npartial")
    assert emit == "alpha\nbeta\n"
    assert keep == "partial"


def test_split_empty_string() -> None:
    assert split_on_safe_boundary("") == ("", "")


def test_split_no_newline_keeps_everything() -> None:
    assert split_on_safe_boundary("no newline here") == ("", "no newline here")


def test_split_all_complete_lines_flush_fully() -> None:
    emit, keep = split_on_safe_boundary("one\ntwo\n")
    assert emit == "one\ntwo\n"
    assert keep == ""


def test_split_closed_fence_flushes_fully() -> None:
    text = "before\n```python\nx = 1\n```\nafter\n"
    emit, keep = split_on_safe_boundary(text)
    # A fully closed fence is safe; everything up to the last newline emits.
    assert emit == text
    assert keep == ""


def test_split_open_fence_keeps_fenced_part_buffered() -> None:
    text = "intro\n```python\nx = 1\n"
    emit, keep = split_on_safe_boundary(text)
    # The line before the fence is safe; the open fence + body stays buffered.
    assert emit == "intro\n"
    assert keep == "```python\nx = 1\n"


def test_split_open_fence_with_no_preceding_text() -> None:
    text = "```\ncode line\n"
    emit, keep = split_on_safe_boundary(text)
    assert emit == ""
    assert keep == text


def test_split_indented_fence_is_recognized() -> None:
    # A fence marker may be preceded by whitespace; stripped text starts ```.
    text = "text\n   ```\nbody\n"
    emit, keep = split_on_safe_boundary(text)
    assert emit == "text\n"
    assert keep == "   ```\nbody\n"


def test_split_invariant_emit_plus_keep_equals_input() -> None:
    samples = [
        "",
        "abc",
        "abc\n",
        "a\nb\nc",
        "```\nx\n```\n",
        "pre\n```\nx\n",
        "   ```bash\nls\n```\ndone\n",
    ]
    for sample in samples:
        emit, keep = split_on_safe_boundary(sample)
        assert emit + keep == sample


# --------------------------------------------------------------------------
# StreamRenderer.render (end-to-end over a fake stream)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_returns_agent_done() -> None:
    renderer, _buffer = _make_renderer()
    done_event = AgentDone(stop_reason="stop", iterations=2, usage=_usage(10, 5))
    result = await renderer.render(_astream([done_event]))
    assert result is done_event


@pytest.mark.asyncio
async def test_render_empty_stream_returns_none() -> None:
    renderer, buffer = _make_renderer()
    result = await renderer.render(_astream([]))
    assert result is None
    assert buffer.getvalue() == ""


@pytest.mark.asyncio
async def test_render_text_appears() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentTextDelta(text="Hello, "),
        AgentTextDelta(text="world!\n"),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage(1, 1)),
    ]
    await renderer.render(_astream(events))
    assert "Hello, world!" in buffer.getvalue()


@pytest.mark.asyncio
async def test_render_flushes_trailing_partial_line_at_end() -> None:
    renderer, buffer = _make_renderer()
    # No trailing newline -> partial line buffered until end of turn, then flushed.
    events: list[AgentEvent] = [
        AgentTextDelta(text="dangling tail"),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    assert "dangling tail" in buffer.getvalue()


@pytest.mark.asyncio
async def test_render_open_fence_only_completes_at_end() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentTextDelta(text="intro\n```\n"),
        AgentTextDelta(text="code = 1\n"),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    # Even though the fence never closed, buffered content is flushed at end of
    # turn so nothing is lost.
    assert "intro" in out
    assert "code = 1" in out
    assert "```" in out


@pytest.mark.asyncio
async def test_render_tool_call_line() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="t1", name="read_file", arguments={"path": "a.txt"}),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "tool read_file(" in out
    assert "path=a.txt" in out


@pytest.mark.asyncio
async def test_render_tool_call_args_abbreviated_single_line() -> None:
    renderer, buffer = _make_renderer()
    long_val = "x" * 200
    events: list[AgentEvent] = [
        AgentToolCall(
            id="t1",
            name="write_file",
            arguments={"content": f"line1\nline2\n{long_val}"},
        ),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    # Collapsed to the first line and truncated; full long value absent.
    assert "tool write_file(" in out
    assert long_val not in out
    assert "line2" not in out  # second line collapsed away


@pytest.mark.asyncio
async def test_render_tool_result_ok_and_err() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolResult(tool_use_id="t1", output="all good\nmore", is_error=False),
        AgentToolResult(tool_use_id="t2", output="boom", is_error=True),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "-> ok" in out
    assert "all good" in out
    assert "-> err" in out
    assert "boom" in out


@pytest.mark.asyncio
async def test_render_status_line() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentStatus(message="thinking hard"),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    assert "thinking hard" in buffer.getvalue()


@pytest.mark.asyncio
async def test_render_footer_shows_iterations_and_tokens() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentDone(stop_reason="stop", iterations=3, usage=_usage(40, 60)),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "3 iter" in out
    assert "100 tok" in out


@pytest.mark.asyncio
async def test_render_footer_shows_cost_when_present() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentDone(stop_reason="stop", iterations=1, usage=_usage(10, 10, cost=0.1234)),
    ]
    await renderer.render(_astream(events))
    assert "$0.1234" in buffer.getvalue()


@pytest.mark.asyncio
async def test_render_footer_omits_cost_when_zero() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentDone(stop_reason="stop", iterations=1, usage=_usage(10, 10, cost=0.0)),
    ]
    await renderer.render(_astream(events))
    assert "$" not in buffer.getvalue()


@pytest.mark.asyncio
async def test_render_footer_falls_back_to_accumulated_usage() -> None:
    renderer, buffer = _make_renderer()
    # AgentDone carries no usage tokens; the renderer accumulated some via
    # AgentUsage events, which the footer falls back to.
    events: list[AgentEvent] = [
        AgentUsage(usage=_usage(5, 5)),
        AgentUsage(usage=_usage(5, 5)),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    assert "20 tok" in buffer.getvalue()


@pytest.mark.asyncio
async def test_render_footer_prefers_done_usage() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentUsage(usage=_usage(5, 5)),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage(20, 20)),
    ]
    await renderer.render(_astream(events))
    # Footer uses the authoritative AgentDone usage (40 tok), not the running 10.
    assert "40 tok" in buffer.getvalue()

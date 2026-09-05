"""Hermetic tests for the streaming TUI renderer.

No network, no model, no real provider, no wall-clock. Events are fed through a
fake async generator into a :class:`StreamRenderer` whose console writes to an
in-memory ``io.StringIO`` (``force_terminal=False`` so no ANSI escapes are
emitted) and whose clock is a :class:`FakeClock` advanced by the generator —
durations are exact regardless of how often the renderer samples the clock.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator

import pytest
from rich.console import Console, Group
from rich.text import Text

from zakcode.cli._glyphs import resolve_glyphs
from zakcode.cli._layout import panel
from zakcode.cli._theme import ZAK_THEME
from zakcode.cli.render import StreamRenderer, display_call, split_on_safe_boundary
from zakcode.events import (
    AgentDone,
    AgentEvent,
    AgentStatus,
    AgentTaskUpdate,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.usage import Usage


class FakeClock:
    """Injectable monotonic clock: advances only when the test says so."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


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


def _make_renderer(
    width: int = 200, clock: FakeClock | None = None
) -> tuple[StreamRenderer, io.StringIO]:
    buffer = io.StringIO()
    console = Console(
        file=buffer, force_terminal=False, width=width, no_color=True, theme=ZAK_THEME
    )
    return StreamRenderer(console=console, clock=clock or FakeClock()), buffer


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


# ── tool blocks: headline, receipt, rail ─────────────────────────────────────


@pytest.mark.asyncio
async def test_render_tool_call_headline() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="t1", name="read_file", arguments={"path": "a.txt"}),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    assert "● Read(a.txt)" in buffer.getvalue()


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
    assert "Write(" in out
    assert long_val not in out
    assert "line2" not in out  # second line collapsed away


@pytest.mark.asyncio
async def test_render_receipt_read_mockup_pin() -> None:
    # The mockup's Read block, byte for byte: headline, synthesized receipt with a
    # FakeClock duration, no preview rail for Read.
    clock = FakeClock()
    renderer, buffer = _make_renderer(width=90, clock=clock)

    async def stream() -> AsyncIterator[AgentEvent]:
        yield AgentToolCall(id="t1", name="read_file", arguments={"path": "src/config.py"})
        clock.advance(0.1)
        yield AgentToolResult(
            tool_use_id="t1",
            output="\n".join(f"line {i}" for i in range(134)),
            is_error=False,
        )
        yield AgentDone(stop_reason="completed", iterations=1, usage=_usage(5, 5))

    await renderer.render(stream())
    out = buffer.getvalue()
    assert "● Read(src/config.py)" in out
    assert "└ 134 lines · 0.1s" in out
    assert "line 7" not in out  # Read results carry no preview


@pytest.mark.asyncio
async def test_render_receipt_attaches_without_blank_or_repeated_name() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="t1", name="read_file", arguments={"path": "a.txt"}),
        AgentToolResult(tool_use_id="t1", output="42 lines", is_error=False),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    lines = buffer.getvalue().splitlines()
    i_call = next(i for i, ln in enumerate(lines) if "Read(a.txt)" in ln)
    assert "└" in lines[i_call + 1]  # contiguous: zero blanks inside the group
    assert buffer.getvalue().count("Read") == 1  # name on the call line only


@pytest.mark.asyncio
async def test_render_durations_keyed_by_id_for_parallel_calls() -> None:
    clock = FakeClock()
    renderer, buffer = _make_renderer(clock=clock)

    async def stream() -> AsyncIterator[AgentEvent]:
        yield AgentToolCall(id="t1", name="read_file", arguments={"path": "a.txt"})
        clock.advance(0.1)
        yield AgentToolCall(id="t2", name="read_file", arguments={"path": "b.txt"})
        clock.advance(0.1)
        yield AgentToolResult(tool_use_id="t2", output="x", is_error=False)  # 0.1s
        clock.advance(0.1)
        yield AgentToolResult(tool_use_id="t1", output="x", is_error=False)  # 0.3s
        yield AgentDone(stop_reason="completed", iterations=1, usage=_usage())

    await renderer.render(stream())
    out = buffer.getvalue()
    assert "1 line · 0.1s" in out
    assert "1 line · 0.3s" in out


@pytest.mark.asyncio
async def test_render_receipt_omits_duration_for_unknown_id() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolResult(tool_use_id="ghost", output="a\nb", is_error=False),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    receipt = next(ln for ln in buffer.getvalue().splitlines() if "└" in ln)
    # No duration when the id is unknown; a result with no preceding call line is
    # DETACHED, so the receipt names its tool (UX.md rule 3 exception).
    assert receipt.strip() == "└ Tool · 2 lines"


@pytest.mark.asyncio
async def test_render_detached_result_names_tool_and_gaps() -> None:
    # An interleaved result (not directly under its own call) must not glue to the
    # wrong call line: it detaches with a gap and carries its tool name.
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="t1", name="read_file", arguments={"path": "a.py"}),
        AgentToolCall(id="t2", name="grep", arguments={"pattern": "foo"}),
        AgentToolResult(tool_use_id="t1", output="x\ny\nz", is_error=False),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    lines = buffer.getvalue().splitlines()
    receipt_idx = next(i for i, ln in enumerate(lines) if "└" in ln)
    assert "Read · 3 lines" in lines[receipt_idx]  # detached: named
    assert lines[receipt_idx - 1].strip() == ""  # detached: gapped off Search's call line


@pytest.mark.asyncio
async def test_render_run_output_rails_with_real_lines() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="r1", name="bash", arguments={"command": "py fizzbuzz.py"}),
        AgentToolResult(tool_use_id="r1", output="1\n2\nFizz\n4\nBuzz", is_error=False),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage(5, 5)),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "Run($ py fizzbuzz.py)" in out
    assert "5 lines" in out
    assert "Fizz" in out and "Buzz" in out
    rail_lines = [ln for ln in out.splitlines() if "│" in ln]
    assert len(rail_lines) == 5  # every output line is a rail row


@pytest.mark.asyncio
async def test_render_run_truncates_head_tail() -> None:
    # >12 lines: first 6, a dim hidden-count row, last 4 — terminal summaries
    # like "42 passed" always survive; blank lines render as bare rail rows.
    renderer, buffer = _make_renderer()
    body_lines = [f"line {i:02d}" for i in range(19)] + ["42 passed in 3.41s"]
    body_lines[2] = ""
    events: list[AgentEvent] = [
        AgentToolCall(id="r1", name="bash", arguments={"command": "uv run pytest -q"}),
        AgentToolResult(tool_use_id="r1", output="\n".join(body_lines), is_error=False),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "20 lines" in out
    assert "… +10 lines …" in out
    assert "line 05" in out  # head end
    assert "line 06" not in out  # hidden middle
    assert "42 passed in 3.41s" in out  # tail survives
    assert any(ln.strip() == "│" for ln in out.splitlines())  # bare rail row


@pytest.mark.asyncio
async def test_render_run_no_output() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="r1", name="bash", arguments={"command": "true"}),
        AgentToolResult(tool_use_id="r1", output="", is_error=False),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    assert "no output" in buffer.getvalue()


@pytest.mark.asyncio
async def test_render_search_receipt_and_preview() -> None:
    clock = FakeClock()
    renderer, buffer = _make_renderer(width=90, clock=clock)

    async def stream() -> AsyncIterator[AgentEvent]:
        yield AgentToolCall(
            id="s1", name="grep", arguments={"pattern": "os\\.environ\\[", "glob": "**/*.py"}
        )
        clock.advance(0.2)
        yield AgentToolResult(
            tool_use_id="s1",
            output="src/config.py:42:    value = os.environ[name]",
            is_error=False,
        )
        yield AgentDone(stop_reason="completed", iterations=1, usage=_usage())

    await renderer.render(stream())
    out = buffer.getvalue()
    assert "● Search(os\\.environ\\[ in **/*.py)" in out
    assert "└ 1 match · 0.2s" in out
    assert "│ src/config.py:42:" in out


@pytest.mark.asyncio
async def test_render_search_preview_caps_at_five() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="s1", name="grep", arguments={"pattern": "x"}),
        AgentToolResult(
            tool_use_id="s1",
            output="\n".join(f"hit {i:02d}" for i in range(8)),
            is_error=False,
        ),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "8 matches" in out
    assert "hit 04" in out
    assert "hit 05" not in out
    assert "… +3 more" in out


@pytest.mark.asyncio
async def test_render_edit_diff_receipt_and_band_rows() -> None:
    renderer, buffer = _make_renderer()
    diff = "\n".join(
        [
            "@@ -41,9 +41,12 @@ def load_settings():",
            "-    value = os.environ[name]",
            "-    return value",
            "+    value = os.environ.get(name)",
            "+    if value is None:",
            "+        value = _dotenv_lookup(name)",
            "+    if value is None:",
            "+        raise KeyError(name)",
            "+    return value",
        ]
    )
    events: list[AgentEvent] = [
        AgentToolCall(id="e1", name="edit_file", arguments={"path": "src/config.py"}),
        AgentToolResult(tool_use_id="e1", output=diff, is_error=False),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "● Edit(src/config.py)" in out
    assert "+6 -2" in out
    assert "│ @@ -41,9 +41,12 @@" in out
    assert "value = os.environ.get(name)" in out


@pytest.mark.asyncio
async def test_render_edit_non_diff_falls_back_to_line_count() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="e1", name="edit_file", arguments={"path": "a.py"}),
        AgentToolResult(tool_use_id="e1", output="ok\ndone", is_error=False),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    assert "2 lines" in buffer.getvalue()


@pytest.mark.asyncio
async def test_render_write_receipt() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="w1", name="write_file", arguments={"path": "a.py"}),
        AgentToolResult(tool_use_id="w1", output="wrote 99 bytes", is_error=False),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    assert "written" in buffer.getvalue()


@pytest.mark.asyncio
async def test_render_todo_glyph_mapping() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="td", name="todo_write", arguments={}),
        AgentToolResult(
            tool_use_id="td", output="[x] design\n[ ] implement\nfree note", is_error=False
        ),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "3 items" in out
    assert "✓ design" in out
    assert "○ implement" in out
    assert "free note" in out  # format-tolerant passthrough


@pytest.mark.asyncio
async def test_render_todo_collapses_a_complete_plan() -> None:
    # A finished checklist (every step done or cancelled) collapses to one summary line so
    # the completed plan does not linger under the answer (ADR-0108); the real update_plan
    # render is indented and carries a header, which the collapse tolerates.
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="td", name="update_plan", arguments={}),
        AgentToolResult(
            tool_use_id="td",
            output="Current plan (2/2 steps done):\n  [x] 1 design\n  [-] 2 implement — dropped",
            is_error=False,
        ),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "complete" in out
    assert "2 steps" in out
    assert "design" not in out  # the finished rows are not re-listed


@pytest.mark.asyncio
async def test_render_todo_collapse_ignores_bracketed_tags_that_are_not_rows() -> None:
    # ADR-0110: only glyph ROWS decide the collapse. A hook's "[plan-completion-verdict] …"
    # provenance tag in the tool output used to count as an open step and kept the finished
    # plan on screen (measured 2026-09-05 in a Mind workspace).
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="td", name="update_plan", arguments={}),
        AgentToolResult(
            tool_use_id="td",
            output=(
                "Current plan (2/2 steps done):\n  [x] 1 design — the shape is settled\n"
                "  [x] 2 implement — landed\n\n[plan-completion-verdict] answer the request"
            ),
            is_error=False,
        ),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "complete" in out
    assert "2 steps" in out
    assert "design" not in out


@pytest.mark.asyncio
async def test_render_draws_a_harness_plan_change_once() -> None:
    # ADR-0112: a plan the harness changed (a request anchor, a skill skeleton) arrives as a
    # task_update with no tool call. It draws as a detached "Plan · N items" receipt with the
    # same glyph-mapped rows — once; a repeat of the same rows stays silent.
    renderer, buffer = _make_renderer()
    plan = "Current plan (0/1 steps done):\n  [~] 1 Audit the pipeline — the request itself"
    events: list[AgentEvent] = [
        AgentTaskUpdate(plan=plan, finished=0, total=1),
        AgentTaskUpdate(plan=plan, finished=0, total=1),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "Plan" in out and "2 items" in out  # header row + the step row
    assert out.count("Audit the pipeline") == 1


@pytest.mark.asyncio
async def test_render_task_update_repeating_the_todo_result_is_silent() -> None:
    # The loop emits a task_update at the next iteration for a plan the model just authored
    # through update_plan; the Todo receipt already showed it, so the update draws nothing.
    renderer, buffer = _make_renderer()
    plan = "Current plan (0/2 steps done):\n  [~] 1 design\n  [ ] 2 implement"
    events: list[AgentEvent] = [
        AgentToolCall(id="td", name="update_plan", arguments={}),
        AgentToolResult(tool_use_id="td", output=plan, is_error=False),
        AgentTaskUpdate(plan=plan, finished=0, total=2),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert out.count("design") == 1 and "Plan" not in out


@pytest.mark.asyncio
async def test_render_error_restructures_with_detail_cap() -> None:
    # └ ✗ first-line receipt + at most 8 full-brightness detail rows behind the rail.
    renderer, buffer = _make_renderer()
    detail = [f"detail {i:02d}" for i in range(1, 12)]
    events: list[AgentEvent] = [
        AgentToolCall(id="r1", name="bash", arguments={"command": "./boom.sh"}),
        AgentToolResult(
            tool_use_id="r1",
            output="\n".join(["exited with code 1", *detail]),
            is_error=True,
        ),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "✗ exited with code 1" in out
    assert "│ detail 01" in out
    assert "detail 08" in out
    assert "detail 09" not in out  # capped at 8 detail rows


@pytest.mark.asyncio
async def test_render_result_value_with_brackets_is_literal() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="r1", name="bash", arguments={"command": "grep tag"}),
        AgentToolResult(tool_use_id="r1", output="found [bold] tag", is_error=False),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage(5, 5)),
    ]
    await renderer.render(_astream(events))
    assert "[bold]" in buffer.getvalue()  # rendered literally, never parsed as markup


# ── prose: markers, inline markdown, code blocks ─────────────────────────────


@pytest.mark.asyncio
async def test_render_inline_markdown() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentTextDelta(text="- first item\n"),
        AgentTextDelta(text="see **bold** and `code` here\n"),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage(5, 5)),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "first item" in out
    assert "•" in out  # the bullet glyph replaced "- "
    assert "bold" in out and "**" not in out  # bold markers consumed
    assert "code" in out and "`" not in out  # inline-code markers consumed


@pytest.mark.asyncio
async def test_render_fenced_code_block_with_language_tag() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentTextDelta(text="intro\n\n```python\nx = 1\n```\n\nafter\n"),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage(5, 5)),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "intro" in out
    assert "· python" in out  # the dim language tag at col 4
    assert "x = 1" in out  # code body rendered (via Syntax)
    assert "```" not in out  # fence markers consumed, not printed raw
    # prose after a code block re-anchors with its own marker
    after = next(ln for ln in out.splitlines() if "after" in ln)
    assert after.startswith("  ● ")


@pytest.mark.asyncio
async def test_render_prose_continuation_keeps_empty_gutter() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentTextDelta(text="first line\nsecond line\n"),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    lines = buffer.getvalue().splitlines()
    first = next(ln for ln in lines if "first line" in ln)
    second = next(ln for ln in lines if "second line" in ln)
    assert first.startswith("  ● first line")
    assert second.startswith("    second line")  # empty gutter, same column math


@pytest.mark.asyncio
async def test_render_prose_after_tool_block_gets_one_gap() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentToolCall(id="t1", name="read_file", arguments={"path": "a.txt"}),
        AgentToolResult(tool_use_id="t1", output="ok", is_error=False),
        AgentTextDelta(text="after the tool\n"),
        AgentDone(stop_reason="completed", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    lines = buffer.getvalue().splitlines()
    idx = next(i for i, ln in enumerate(lines) if "after the tool" in ln)
    assert lines[idx].startswith("  ● ")  # re-anchored with its own marker
    assert lines[idx - 1].strip() == ""  # exactly one separating blank
    assert lines[idx - 2].strip() != ""


@pytest.mark.asyncio
async def test_render_status_line() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentStatus(message="provider rate-limited — retrying in 2s"),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage()),
    ]
    await renderer.render(_astream(events))
    assert "· provider rate-limited — retrying in 2s" in buffer.getvalue()


# ── the six pinned tests (docs/UX.md regressions) ────────────────────────────


@pytest.mark.asyncio
async def test_pinned_wrap_hangs_under_body_column() -> None:
    # Narrow console: wrapped prose lands at col 4, never under the marker, and
    # the group carries exactly one ● (no footer event, so no footer marker).
    renderer, buffer = _make_renderer(width=40)
    long_line = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
        "mike november oscar papa\n"
    )
    await renderer.render(_astream([AgentTextDelta(text=long_line)]))
    out = buffer.getvalue()
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) > 1  # it did wrap
    assert lines[0].startswith("  ● alpha")
    for continuation in lines[1:]:
        assert continuation.startswith("    ")
        assert continuation[4] != " "  # body exactly at col 4
    assert out.count("●") == 1


@pytest.mark.asyncio
async def test_pinned_blank_rhythm() -> None:
    # Interleaved prose/tool/status/code: never two consecutive blank lines,
    # exactly one blank between blocks, zero blanks inside a tool group.
    clock = FakeClock()
    renderer, buffer = _make_renderer(width=90, clock=clock)

    async def stream() -> AsyncIterator[AgentEvent]:
        yield AgentTextDelta(text="Checking the loader.\n")
        yield AgentToolCall(id="t1", name="read_file", arguments={"path": "a.py"})
        clock.advance(0.1)
        yield AgentToolResult(tool_use_id="t1", output="x\ny", is_error=False)
        yield AgentStatus(message="retrying")
        yield AgentTextDelta(text="```python\nx = 1\n```\n")
        yield AgentTextDelta(text="All good.\n")
        yield AgentDone(stop_reason="completed", iterations=2, usage=_usage(5, 5))

    await renderer.render(stream())
    out = buffer.getvalue()
    assert "\n\n\n" not in out  # the no-triple-newline invariant
    lines = out.splitlines()
    i_call = next(i for i, ln in enumerate(lines) if "Read(a.py)" in ln)
    assert "└" in lines[i_call + 1]  # receipt contiguous with its call
    assert lines[i_call - 1].strip() == ""  # one blank above the tool block
    i_status = next(i for i, ln in enumerate(lines) if "retrying" in ln)
    assert lines[i_status - 1].strip() == ""
    assert lines[i_status - 2].strip() != ""  # exactly one blank, not two
    assert not out.endswith("\n\n")  # no trailing blank after the footer


@pytest.mark.asyncio
async def test_pinned_ascii_mode_same_columns(monkeypatch) -> None:
    async def run(ascii_mode: bool) -> str:
        if ascii_mode:
            monkeypatch.setenv("ZAKCODE_ASCII", "1")
        else:
            monkeypatch.delenv("ZAKCODE_ASCII", raising=False)
        clock = FakeClock()
        renderer, buffer = _make_renderer(width=90, clock=clock)

        async def stream() -> AsyncIterator[AgentEvent]:
            yield AgentToolCall(id="t1", name="read_file", arguments={"path": "src/config.py"})
            clock.advance(0.1)
            yield AgentToolResult(
                tool_use_id="t1",
                output="\n".join(f"line {i:02d}" for i in range(134)),
                is_error=False,
            )
            yield AgentToolCall(id="r1", name="bash", arguments={"command": "uv run pytest -q"})
            clock.advance(0.1)
            yield AgentToolResult(
                tool_use_id="r1",
                output="\n".join(f"out {i:02d}" for i in range(20)),
                is_error=False,
            )
            yield AgentToolCall(id="r2", name="bash", arguments={"command": "./boom.sh"})
            clock.advance(0.1)
            yield AgentToolResult(tool_use_id="r2", output="boom\nwhy it broke", is_error=True)
            yield AgentDone(stop_reason="completed", iterations=1, usage=_usage(5, 5))

        await renderer.render(stream())
        return buffer.getvalue()

    def offsets(out: str) -> tuple[int, int, int]:
        lines = out.splitlines()
        call = next(ln for ln in lines if "Read(" in ln)
        receipt = next(ln for ln in lines if "134 lines" in ln)
        preview = next(ln for ln in lines if "out 00" in ln)
        return (call.index("Read("), receipt.index("134 lines"), preview.index("out 00"))

    unicode_out = await run(ascii_mode=False)
    ascii_out = await run(ascii_mode=True)

    assert "o Read(" in ascii_out  # marker_tool falls back to a distinct 'o'
    assert "\\ 134 lines" in ascii_out  # single-char elbow keeps the receipt at col 6
    assert "| " in ascii_out  # the rail
    assert "x boom" in ascii_out  # single-char fail glyph
    assert "... +" in ascii_out  # inline-only multi-char ellipsis
    for glyph in ("●", "└", "│", "✗", "…", "·"):
        assert glyph not in ascii_out
    assert offsets(ascii_out) == offsets(unicode_out)  # identical column grid


@pytest.mark.asyncio
async def test_pinned_permission_interleaving() -> None:
    # The prompter owns its own spacing (one blank before the panel, one after the
    # decision echo) and never touches renderer state; the receipt still attaches
    # to the call line per the group-binding rule. Sequence per the mockup:
    # call line / blank / panel / permit echo / blank / └ ✗ receipt.
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=90, no_color=True, theme=ZAK_THEME)
    clock = FakeClock()
    renderer = StreamRenderer(console=console, clock=clock)
    g = resolve_glyphs(console)

    def prompter_prints() -> None:
        console.print()
        body = Group(
            display_call("bash", {"command": "./scripts/release.sh prod"}),
            Text("full access — touches paths outside the workspace", style="perm.reason"),
        )
        console.print(panel(console, "permission", body, border_style="perm.border"))
        console.print(Text(f"  permit (1-3 or y/a/n) {g['prompt']} a"))
        console.print()

    async def stream() -> AsyncIterator[AgentEvent]:
        yield AgentToolCall(
            id="r1", name="bash", arguments={"command": "./scripts/release.sh prod"}
        )
        prompter_prints()
        clock.advance(2.4)
        yield AgentToolResult(
            tool_use_id="r1",
            output="exited with code 1\nrelease.sh: line 14: AWS_PROFILE: unbound variable",
            is_error=True,
        )
        yield AgentDone(
            stop_reason="max_iterations", iterations=8, usage=_usage(1050, 1050, cost=0.004)
        )

    await renderer.render(stream())
    out = buffer.getvalue()
    lines = out.splitlines()
    i_call = next(i for i, ln in enumerate(lines) if "●" in ln and "release.sh" in ln)
    assert lines[i_call + 1].strip() == ""  # prompter's leading blank
    assert "permission" in lines[i_call + 2]  # the panel title line
    i_permit = next(i for i, ln in enumerate(lines) if "permit (1-3 or y/a/n)" in ln)
    assert lines[i_permit + 1].strip() == ""  # prompter's trailing blank
    assert "✗ exited with code 1" in lines[i_permit + 2]  # receipt directly after
    assert "2.4s" in lines[i_permit + 2]
    assert "full access — touches paths outside the workspace" in out
    assert "DANGER_FULL_ACCESS" not in out
    assert "\n\n\n" not in out
    assert "stopped early — max iterations" in out  # footer carries the outcome


@pytest.mark.asyncio
async def test_footer_surfaces_prompt_cache_share() -> None:
    # ADR-0021: when the backend reports cache-read tokens, the footer shows what share
    # of the prompt side was cached — the at-a-glance signal that a big token number was
    # mostly discounted re-sends (and, at 0%, that caching is NOT hitting). Silent at 0.
    async def footer_for(usage: Usage) -> str:
        renderer, buffer = _make_renderer()
        await renderer.render(
            _astream([AgentDone(stop_reason="completed", iterations=3, usage=usage)])
        )
        return buffer.getvalue()

    cached = Usage(
        prompt_tokens=1000, completion_tokens=50, total_tokens=1050, cache_read_tokens=750
    )
    out = await footer_for(cached)
    assert "(75% cached)" in out
    out = await footer_for(_usage(1000, 50))
    assert "cached" not in out  # zero cache reads -> no annotation, no clutter


@pytest.mark.asyncio
async def test_pinned_footer_states() -> None:
    async def footer_for(done: AgentDone) -> str:
        renderer, buffer = _make_renderer()
        await renderer.render(_astream([done]))
        return buffer.getvalue()

    out = await footer_for(
        AgentDone(stop_reason="max_iterations", iterations=8, usage=_usage(5, 5))
    )
    assert "stopped early — max iterations" in out
    out = await footer_for(AgentDone(stop_reason="doom_loop", iterations=4, usage=_usage(5, 5)))
    assert "stopped — repeating itself" in out
    out = await footer_for(
        AgentDone(
            stop_reason="provider_error",
            iterations=1,
            usage=_usage(5, 5),
            error="rate limit hit\nmore detail",
        )
    )
    assert "provider error — rate limit hit" in out
    out = await footer_for(AgentDone(stop_reason="some_new_reason", iterations=1, usage=_usage()))
    assert "some new reason" in out
    out = await footer_for(AgentDone(stop_reason="completed", iterations=1, usage=_usage(5, 5)))
    assert "done ·" in out
    assert "stopped" not in out
    # gave_up gets its own honest label (the model went silent mid-task, 2026-08-26).
    out = await footer_for(AgentDone(stop_reason="gave_up", iterations=3, usage=_usage(5, 5)))
    assert "stopped — gave up (no output)" in out
    # A DEGRADED completion must not print an indistinguishable clean "done": the
    # degraded flag was computed and silently dropped by the footer (field incident
    # 2026-08-25 — a stuck-nudged give-up rendered exactly like a good turn).
    out = await footer_for(
        AgentDone(stop_reason="completed", iterations=5, usage=_usage(5, 5), degraded=True)
    )
    assert "done — struggled" in out


def test_pinned_permission_parser_legacy_synonyms() -> None:
    from zakcode.cli import _parse_permission_answer
    from zakcode.permissions import PermissionOutcome

    for answer in ("y", "yes", "o", "once", "allow", "allow once", " Y "):
        assert _parse_permission_answer(answer) is PermissionOutcome.ALLOW_ONCE
    for answer in (
        "a",
        "always",
        "session",
        "allow session",
        "allow for session",
        "allow for the session",
        "allow always",
    ):
        assert _parse_permission_answer(answer) is PermissionOutcome.ALLOW_SESSION
    for answer in ("n", "no", "d", "deny"):
        assert _parse_permission_answer(answer) is PermissionOutcome.DENY_ONCE
    assert _parse_permission_answer("frobnicate") is None


def test_pinned_permission_parser_numbered_keys() -> None:
    from zakcode.cli import _parse_permission_answer
    from zakcode.permissions import PermissionOutcome

    assert _parse_permission_answer("1") is PermissionOutcome.ALLOW_ONCE
    assert _parse_permission_answer("2") is PermissionOutcome.ALLOW_SESSION
    assert _parse_permission_answer("3") is PermissionOutcome.DENY_ONCE


# ── footer: tokens, cost, elapsed ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_render_footer_mockup_pin() -> None:
    clock = FakeClock()
    renderer, buffer = _make_renderer(width=90, clock=clock)

    async def stream() -> AsyncIterator[AgentEvent]:
        clock.advance(41.2)
        yield AgentDone(
            stop_reason="completed", iterations=3, usage=_usage(10000, 5300, cost=0.023)
        )

    await renderer.render(stream())
    assert "● done · 3 iterations · 15.3k tokens · $0.023 · 41.2s" in buffer.getvalue()


@pytest.mark.asyncio
async def test_render_footer_shows_iterations_and_tokens() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentDone(stop_reason="stop", iterations=3, usage=_usage(40, 60)),
    ]
    await renderer.render(_astream(events))
    out = buffer.getvalue()
    assert "3 iterations" in out
    assert "100 tokens" in out


@pytest.mark.asyncio
async def test_render_footer_shows_cost_when_present() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentDone(stop_reason="stop", iterations=1, usage=_usage(10, 10, cost=0.1234)),
    ]
    await renderer.render(_astream(events))
    assert "$0.1234" in buffer.getvalue()


@pytest.mark.asyncio
async def test_render_footer_always_shows_cost_slot() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentDone(stop_reason="stop", iterations=1, usage=_usage(10, 10, cost=0.0)),
    ]
    await renderer.render(_astream(events))
    # The cost slot is always present ($0.00 at zero) so the footer never jitters.
    assert "$0.00" in buffer.getvalue()


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
    assert "20 tokens" in buffer.getvalue()


@pytest.mark.asyncio
async def test_render_footer_prefers_done_usage() -> None:
    renderer, buffer = _make_renderer()
    events: list[AgentEvent] = [
        AgentUsage(usage=_usage(5, 5)),
        AgentDone(stop_reason="stop", iterations=1, usage=_usage(20, 20)),
    ]
    await renderer.render(_astream(events))
    # Footer uses the authoritative AgentDone usage (40 tokens), not the running 10.
    assert "40 tokens" in buffer.getvalue()


# ── helpers: display_call, formatting, diff preview ──────────────────────────


def test_display_call_shapes() -> None:
    assert display_call("read_file", {"path": "a.py"}).plain == "Read(a.py)"
    assert display_call("bash", {"command": "ls -la"}).plain == "Run($ ls -la)"
    assert display_call("powershell", {"command": "dir"}).plain == "Run($ dir)"
    assert (
        display_call("grep", {"pattern": "os\\.environ\\[", "glob": "**/*.py"}).plain
        == "Search(os\\.environ\\[ in **/*.py)"
    )
    assert display_call("unknown_tool", {"x": "y"}).plain == "UnknownTool(y)"
    assert display_call("todo_write", {}).plain == "Todo()"
    assert display_call("update_plan", {}).plain == "Todo()"  # the real plan tool renders as Todo


def test_display_call_truncation_directions() -> None:
    # Commands middle-truncate (head AND tail survive); paths left-truncate so
    # the filename survives.
    long_cmd = "uv run " + "x" * 100 + " --flag tail-end"
    cmd = display_call("bash", {"command": long_cmd}).plain
    assert "uv run" in cmd and "tail-end" in cmd and "…" in cmd
    assert long_cmd not in cmd

    long_path = "C:\\very\\" + "deep\\" * 30 + "config.py"
    path = display_call("read_file", {"path": long_path}).plain
    assert path.startswith("Read(…")
    assert path.endswith("config.py)")


def test_fmt_cost_pins() -> None:
    from zakcode.cli.render import _fmt_cost

    assert _fmt_cost(0.023) == "$0.023"
    assert _fmt_cost(0.0004) == "$0.0004"
    assert _fmt_cost(0.0) == "$0.00"
    assert _fmt_cost(1.5) == "$1.50"


def test_fmt_duration_pins() -> None:
    from zakcode.cli.render import _fmt_duration

    assert _fmt_duration(0.1) == "0.1s"
    assert _fmt_duration(3.61) == "3.6s"
    assert _fmt_duration(41.2) == "41.2s"
    assert _fmt_duration(75.0) == "1m 15s"


def test_diff_preview_ignores_non_diff_output() -> None:
    # Output whose lines merely begin with -/+ (markdown bullets, an ls -l listing, a file
    # read) is NOT a diff and must not be colorized/truncated (audit #6).
    from zakcode.cli.render import _diff_preview

    assert _diff_preview("- installed pkg\n- removed old file\n- done") is None
    assert _diff_preview("-rw-r--r-- 1 me 0 a.txt\n-rw-r--r-- 1 me 0 b.txt") is None


def test_diff_preview_colorizes_real_unified_diff() -> None:
    from zakcode.cli.render import _diff_preview

    # @@ hunk signature.
    rows = _diff_preview("@@ -1,2 +1,2 @@\n-old line\n+new line\n context")
    assert rows is not None
    plain = "\n".join(row.plain for row in rows)
    assert "old line" in plain and "new line" in plain
    # ---/+++ file-header signature.
    assert _diff_preview("--- a/x.py\n+++ b/x.py\n-removed\n+added") is not None


def test_diff_preview_marks_truncation() -> None:
    from zakcode.cli.render import _diff_preview

    body = "\n".join(["@@ -1 +1 @@"] + [f"+line {i}" for i in range(13)])
    rows = _diff_preview(body)
    assert rows is not None
    assert len(rows) == 13  # 12 shown + the hidden-count row
    assert rows[-1].plain == "… +2 lines"

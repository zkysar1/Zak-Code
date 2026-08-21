"""SDK ⇄ CLI RENDERING correctness — the one LOSSY interface, held to a semantic contract.

The fourth axis, alongside transport (``test_sdk_iface_parity.py``), config
(``test_sdk_iface_config_parity.py``), and permission
(``test_sdk_iface_permission_parity.py``). Those three pin interfaces that must
relay the SDK's :class:`~zakcode.events.AgentEvent` stream FAITHFULLY. The CLI's
terminal :class:`~zakcode.cli.render.StreamRenderer` is deliberately NOT one of
them: guard-4547 names it a separate axis — a lossy, human-facing SINK, not a
relay — so byte-exact event parity is the wrong contract to hold it to.

THE CLAIM
---------
The renderer is lossy, but it must never lose what MATTERS. Fed the REAL SDK event
stream (a live :class:`~zakcode.Agent` + ``ScriptedProvider`` — NOT hand-built
events; that is what ``test_render.py`` already does, and the distinction is the
point: this file proves the renderer is correct against the stream the SDK
actually produces for a canonical scenario), the rendered transcript must surface
every semantically-important thing the SDK did — each tool call by its human
display name + argument, each result (summarized), the assistant's reply verbatim,
a faithful error — and :meth:`StreamRenderer.render` must return the true terminal
:class:`~zakcode.events.AgentDone`.

WHAT "LOSSY BUT CORRECT" MEANS (the boundary this axis pins)
-----------------------------------------------------------
The renderer SUMMARIZES on purpose, and that is correct, not a defect:

* tool names render as DISPLAY names — ``write_file`` -> ``Write``,
  ``read_file`` -> ``Read``.
* a ``Write`` result renders ``written``; the SDK's ``Wrote N bytes to P`` byte
  count is intentionally dropped.
* a ``Read`` result renders ``N line``; the file CONTENT is intentionally not
  echoed back to the terminal.
* an error result restructures to ``✗ <message>`` plus a detail rail.

So this file asserts the important content IS surfaced *in its display form*; it
does NOT assert byte-exact parity — that is transport's job, and holding a human
sink to it would be a category error. That asymmetry is exactly why rendering is
its own axis instead of another layer in the transport harness.

WHY THE LONG-HORIZON CASE
-------------------------
:func:`test_long_horizon_render_drops_and_reorders_nothing` is the rendering
analogue of the transport harness's ``long_horizon`` scenario: over a long stream
(six tool calls) the renderer must surface EVERY call, in order — a sink that
coalesced, truncated, or dropped a middle event would still "look rendered".

HERMETIC
--------
``ScriptedProvider`` never touches a network (no inference — no GPU, no Groq). The
renderer's ``Console`` writes to an in-memory ``io.StringIO`` with
``force_terminal=False`` + ``no_color=True`` (no ANSI escapes), and its clock is a
frozen fake, so the rendered text is fully deterministic. Same console/clock idiom
as ``test_render.py``'s ``_make_renderer``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pytest
from rich.console import Console

from zakcode import Agent
from zakcode.cli._theme import ZAK_THEME
from zakcode.cli.render import StreamRenderer
from zakcode.evals.harness import ScriptedProvider, call_tool, reply
from zakcode.events import AgentDone
from zakcode.permissions import PermissionPolicy
from zakcode.providers.base import LLMResult

# Labeled so a stray real turn (a mis-wired agent) would look obviously wrong; the
# input itself never appears in the rendered output — the renderer renders EVENTS.
_CANONICAL_INPUT = "__canonical render input__"


class _FakeClock:
    """Injectable monotonic clock frozen at 0.0 so every synthesized duration renders
    as ``0.0s`` — durations are exact regardless of how often the renderer samples it."""

    def __call__(self) -> float:
        return 0.0


# A deliberately long turn (mirrors the transport harness's ``long_horizon``): six
# sequential write_file calls before the reply, so the render must surface six
# distinct call lines in order.
_LONG_HORIZON_SCRIPT: tuple[LLMResult, ...] = tuple(
    call_tool("write_file", {"path": f"note-{i}.txt", "content": "x" * i}, id=f"w{i}")
    for i in range(1, 7)
) + (reply("Wrote all six notes."),)


# ── the semantic-content cases (expected substrings captured from the live renderer) ──


@dataclass(frozen=True)
class RenderCase:
    id: str
    script: tuple[LLMResult, ...]
    # Human-facing content that MUST appear in the rendered transcript, in display
    # form (Write/Read, not write_file/read_file). Substrings, not layout: this axis
    # pins that meaning survives, not the exact grid (that is test_render.py's job).
    expect_substrings: tuple[str, ...]
    expect_stop_reason: str
    expect_iterations: int


RENDER_CASES: list[RenderCase] = [
    RenderCase(
        id="text_only",
        script=(reply("Hello - parity holds."),),
        expect_substrings=("Hello - parity holds.",),
        expect_stop_reason="completed",
        expect_iterations=1,
    ),
    RenderCase(
        # The tool call surfaces by DISPLAY name + arg (Write(parity.txt)); the
        # result surfaces as the summarized receipt (written); the reply verbatim.
        id="tool_then_reply",
        script=(
            call_tool("write_file", {"path": "parity.txt", "content": "hi"}, id="w1"),
            reply("Wrote parity.txt."),
        ),
        expect_substrings=("Write(parity.txt)", "written", "Wrote parity.txt."),
        expect_stop_reason="completed",
        expect_iterations=2,
    ),
    RenderCase(
        # Long horizon: first + last call must both surface (the middle four and
        # the ordering are pinned by test_long_horizon_render_drops_and_reorders_nothing).
        id="long_horizon",
        script=_LONG_HORIZON_SCRIPT,
        expect_substrings=(
            "Write(note-1.txt)",
            "Write(note-6.txt)",
            "Wrote all six notes.",
        ),
        expect_stop_reason="completed",
        expect_iterations=7,
    ),
    RenderCase(
        # Two DIFFERENT tools in one turn — both must surface by their own display
        # name. (The Read receipt summarizes to a line count, not the file content;
        # this axis checks the call is surfaced, not that content is echoed back.)
        id="read_after_write",
        script=(
            call_tool("write_file", {"path": "note.txt", "content": "hello horizon"}, id="w1"),
            call_tool("read_file", {"path": "note.txt"}, id="r1"),
            reply("Read the note back."),
        ),
        expect_substrings=("Write(note.txt)", "Read(note.txt)", "Read the note back."),
        expect_stop_reason="completed",
        expect_iterations=3,
    ),
    RenderCase(
        # The error render path: a failed tool result restructures to ✗ + message,
        # and the human MUST see the actual error text, not a swallowed failure. The
        # turn still completes (the agent recovers and replies), so stop_reason stays
        # "completed" — the tool failed, the TURN did not.
        id="tool_error",
        script=(
            call_tool("read_file", {"path": "does-not-exist-xyz.txt"}, id="r1"),
            reply("Tried to read."),
        ),
        expect_substrings=(
            "Read(does-not-exist-xyz.txt)",
            "File not found: does-not-exist-xyz.txt",
            "Tried to read.",
        ),
        expect_stop_reason="completed",
        expect_iterations=2,
    ),
]


# ── the one render helper: real SDK stream -> real renderer -> (transcript, done) ─────


async def _render(
    script: tuple[LLMResult, ...], *, workspace_root: Path
) -> tuple[str, AgentDone | None]:
    """Feed the REAL SDK event stream through the REAL ``StreamRenderer``.

    A live ``Agent`` (hermetic ``ScriptedProvider``) produces the stream; the
    renderer consumes it over a ``StringIO``-backed console with a frozen clock.
    Returns the rendered transcript and the terminal ``AgentDone`` render() returns.
    The agent's ``astream_turn`` generator is handed straight to ``render()`` — the
    same wiring the CLI's local chat path uses (SDK stream in, rendered text out).
    """
    agent = Agent(
        provider=ScriptedProvider(list(script)),
        permission_policy=PermissionPolicy("allow"),
        default_model="scripted/render",
        workspace_root=str(workspace_root),
        max_iterations=8,
    )
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=200, no_color=True, theme=ZAK_THEME)
    renderer = StreamRenderer(console=console, clock=_FakeClock())
    done = await renderer.render(agent.astream_turn(_CANONICAL_INPUT))
    return buffer.getvalue(), done


# ── tests ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", RENDER_CASES, ids=lambda c: c.id)
async def test_render_surfaces_the_sdk_stream_semantics(case: RenderCase, tmp_path: Path) -> None:
    """The rendered transcript surfaces every semantically-important thing the SDK did,
    and ``render()`` returns the true terminal ``AgentDone``.

    The break point mirrors the transport harness's: if the renderer drops a tool
    call, swallows an error, or loses the reply, the matching substring is absent and
    THIS case fails while the others stay green — localizing the loss to one shape.
    """
    text, done = await _render(case.script, workspace_root=tmp_path)

    # render() returns the true terminal event (the REPL reads it for post-turn signals).
    assert done is not None
    assert done.stop_reason == case.expect_stop_reason
    assert done.iterations == case.expect_iterations

    # ...and the transcript surfaces every semantically-important thing, in display form.
    for needle in case.expect_substrings:
        assert needle in text, f"{case.id}: expected {needle!r} in rendered transcript:\n{text}"

    # the terminal receipt renders with the right outcome word + iteration count.
    assert "done" in text
    assert f"{case.expect_iterations} iterations" in text


async def test_long_horizon_render_drops_and_reorders_nothing(tmp_path: Path) -> None:
    """Over a long stream the renderer surfaces EVERY tool call, IN ORDER.

    The rendering analogue of the transport ``long_horizon`` scenario: a sink that
    coalesced, truncated, or dropped a middle event would still "look rendered", so
    the check is positional — all six call lines present, their offsets strictly
    increasing. Distinct paths (note-1..note-6) mean no line can stand in for another.
    """
    text, done = await _render(_LONG_HORIZON_SCRIPT, workspace_root=tmp_path)

    assert done is not None
    assert done.iterations == 7

    positions = [text.find(f"note-{i}.txt") for i in range(1, 7)]
    assert all(pos != -1 for pos in positions), f"a call was dropped from the render: {positions}"
    assert positions == sorted(positions), f"calls rendered out of order: {positions}"

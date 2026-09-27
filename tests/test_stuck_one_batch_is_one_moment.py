"""The stuck ladder counts identical results inside ONE batch as one observation
(ADR-0038, amended 2026-09-27).

Measured in the field on an unattended session with a small local model: its last model call
asked for eight edits to one test file in a single batch. Seven applied (the file's diff was
nine changed lines, so they were distinct edits), and each answered "Made 1 replacement in
<path>". The tracker reads the edit epoch once, after the batch, so all seven shared it, the
repeated-outcome signal counted seven sightings, and the seventh sighting's rung is a STOP: a
155-iteration turn ended ``stuck`` on its most productive step. Across ten such sessions, 8 of
13 ladder steps recorded in their traces had their whole repeat count inside one batch.

A re-observation needs a known result, and inside one batch the model has seen none of the
answers yet. Every case here has its control beside it: the same result coming back in later
batches still climbs, so what these tests pin is the per-batch count and not a ladder that
stopped working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.agent.stuck import SIG_REPEATED_OUTCOME, StuckAction, StuckTracker
from zakcode.config import load_settings
from zakcode.evals.harness import ScriptedProvider, reply
from zakcode.messages import ToolResultBlock
from zakcode.permissions import PermissionMode, PermissionPolicy
from zakcode.providers.base import LLMResult, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.tools.builtins.edit import EditFileTool

ROWS = 8
PROBE_OUTPUT = "status: 3 workers idle, queue depth 0, last error none"


def _edit_calls(n: int) -> list[ToolCall]:
    """``n`` distinct single-line edits to one file, as one batch asks for them."""
    return [
        ToolCall(
            id=f"e{i}",
            name="Edit",
            arguments={
                "path": "table.txt",
                "old_string": f"row {i} = old",
                "new_string": f"row {i} = new",
            },
        )
        for i in range(n)
    ]


def _answered(calls: list[ToolCall], output: str) -> list[ToolResultBlock]:
    return [ToolResultBlock(tool_use_id=c.id, output=output) for c in calls]


def _probe(i: int) -> ToolCall:
    return ToolCall(id=f"p{i}", name="probe", arguments={"q": f"variant {i}"})


# ── the tracker ───────────────────────────────────────────────────────────────


def test_identical_results_inside_one_batch_are_one_observation() -> None:
    tracker = StuckTracker()
    calls = _edit_calls(ROWS)
    tracker.observe(calls, _answered(calls, "Made 1 replacement in table.txt"), epoch=ROWS)
    assert SIG_REPEATED_OUTCOME not in tracker.last_signals
    assert tracker.next_action() is StuckAction.CONTINUE
    assert tracker.evidence() == {"signals": ""}


def test_control_the_same_result_in_three_batches_still_climbs() -> None:
    tracker = StuckTracker()
    for i in range(1, 4):
        tracker.observe([_probe(i)], _answered([_probe(i)], PROBE_OUTPUT))
    assert SIG_REPEATED_OUTCOME in tracker.last_signals
    assert tracker.next_action() is StuckAction.NUDGE
    assert tracker.evidence()["repeats"] == 3


def test_a_batch_counts_once_toward_the_repeats_of_later_batches() -> None:
    # Three identical answers in the first batch are ONE sighting, so the same answer must come
    # back in two more batches before the third sighting draws the nudge.
    tracker = StuckTracker()
    first = [_probe(i) for i in range(3)]
    tracker.observe(first, _answered(first, PROBE_OUTPUT))
    assert tracker.next_action() is StuckAction.CONTINUE
    tracker.observe([_probe(3)], _answered([_probe(3)], PROBE_OUTPUT))
    assert tracker.next_action() is StuckAction.CONTINUE
    tracker.observe([_probe(4)], _answered([_probe(4)], PROBE_OUTPUT))
    assert tracker.next_action() is StuckAction.NUDGE
    assert tracker.evidence()["repeats"] == 3


# ── the loop, with the real edit tool ─────────────────────────────────────────


def _loop(tmp_path: Path, script: list[Any]) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(EditFileTool())
    return AgentLoop(
        ScriptedProvider(script),
        registry,
        Session(cwd=str(tmp_path), model="test"),
        settings=load_settings(workspace_root=tmp_path),
        workspace_root=tmp_path,
        permission_policy=PermissionPolicy(PermissionMode.ALLOW),
        max_iterations=20,
    )


async def _run(loop: AgentLoop, prompt: str, path: str) -> None:
    if path == "buffered":
        await loop.arun_turn(prompt)
        return
    async for _ in loop.astream_turn(prompt):
        pass


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_one_batch_of_edits_to_one_file_finishes_the_turn(tmp_path: Path, path: str) -> None:
    table = tmp_path / "table.txt"
    table.write_text("".join(f"row {i} = old\n" for i in range(ROWS)))
    batch = LLMResult(tool_calls=_edit_calls(ROWS), finish_reason="tool_calls")
    loop = _loop(tmp_path, [batch, reply("done")])
    await _run(loop, "mark every row new", path)
    assert table.read_text() == "".join(f"row {i} = new\n" for i in range(ROWS))
    assert [e.data for e in loop._trace.events if e.data.get("kind") == "stuck"] == []
    assert loop.session.last_stop_reason == "completed"

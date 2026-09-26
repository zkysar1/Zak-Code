"""An outcome on a step sent without a status closes it (ADR-0254).

Measured on the bench (2026-09-26, a 27B on the plugin-conventions task, one run of three): the
model never sent a ``status`` on any step — 80 of 80 steps across 17 plan calls — and wrote an
``outcome`` on 71 of them. Each call read "Plan updated: 0/4 steps done" back; only an exact resend
advanced a step (lever N, ADR-0168), one per round trip, and every edit to an outcome's text reset
that. The run took 24 turns and 2133 s where its two siblings took 6 turns each.

The schema already said both halves: omit the status for pending, set the outcome when you mark
the step done. A step carrying an outcome and no status contradicts itself, and the outcome is the
positive statement — the model wrote what the step PRODUCED. The builder now reads it as done, in
the call that carries it. Only the ABSENT status closes: an explicit pending or in_progress beside
an outcome stays the model's call, an unknown status string stays pending as before, and a compound
step's status is derived from its children. In the rest of the measured population (332 steps over
57 runs of the previous campaign; 1,369 steps over three served loops in 48 h) a status-less step
carried an outcome once, and lever N advanced exactly that step — so outside the doom-loop shape
this reading changes nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.tasks import TaskNetwork
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.update_plan import (
    _COMPLETE_HINT,
    UpdatePlanTool,
    _build_task,
    _task_schema,
)

STEPS = ["read the conventions", "write the plugin doc", "add the check", "run the tests"]


def _ctx() -> tuple[ToolContext, TaskNetwork]:
    net = TaskNetwork()
    return ToolContext(workspace_root=Path("/tmp"), task_network=net), net


def _plan(outcomes: dict[int, str], statuses: dict[int, str] | None = None) -> list[dict[str, str]]:
    """The four-step plan with an outcome on the given steps and a status only where said."""
    plan: list[dict[str, str]] = []
    for i, title in enumerate(STEPS):
        step = {"title": title}
        if i in outcomes:
            step["outcome"] = outcomes[i]
        if statuses and i in statuses:
            step["status"] = statuses[i]
        plan.append(step)
    return plan


def test_the_builder_reads_an_outcome_with_no_status_as_done() -> None:
    assert _build_task({"title": "x", "outcome": "did it"}, 1).status == "done"
    assert _build_task({"title": "x"}, 1).status == "pending"
    # An explicit status beside the outcome is the model's call.
    pending = _build_task({"title": "x", "status": "pending", "outcome": "did it"}, 1)
    assert pending.status == "pending"
    working = _build_task({"title": "x", "status": "in_progress", "outcome": "half"}, 1)
    assert working.status == "in_progress"
    # An unknown status string stays pending as before — the rule is about ABSENCE.
    assert _build_task({"title": "x", "status": "complete", "outcome": "did it"}, 1).status == (
        "pending"
    )
    # A compound step's status is derived from its children, never forced by its own outcome.
    parent = _build_task({"title": "p", "outcome": "did it", "subtasks": [{"title": "c"}]}, 2)
    assert parent.status == "pending" and parent.children[0].status == "pending"
    # The collector names exactly the primitive steps read this way, at any depth.
    closed: list[str] = []
    _build_task(
        {
            "title": "p",
            "outcome": "o",
            "subtasks": [{"title": "c1", "outcome": "done c1"}, {"title": "c2"}],
        },
        2,
        closed,
    )
    assert closed == ["c1"]


def test_the_schema_says_so() -> None:
    desc = _task_schema(1)["properties"]["status"]["description"]
    assert "Omit for pending" in desc
    assert "a step sent with an 'outcome' and no status is read as done" in desc


async def test_steps_sent_with_an_outcome_and_no_status_are_done_in_that_call() -> None:
    ctx, net = _ctx()
    first = await UpdatePlanTool().execute({"tasks": _plan({})}, ctx)
    assert first.output.startswith("Plan updated: 0/4 steps done")
    assert first.data is not None and first.data["closed_on_outcome"] == 0
    assert "no status as done" not in first.output

    # The measured shape: outcomes appear on steps 1-3, a status on none of them.
    result = await UpdatePlanTool().execute(
        {"tasks": _plan({0: "read", 1: "wrote", 2: "added"})}, ctx
    )
    assert result.output.startswith("Plan updated: 3/4 steps done · current: 4 run the tests"), (
        result.output
    )
    assert result.data is not None and result.data["closed_on_outcome"] == 3
    assert "read 3 step(s) sent with an outcome and no status as done" in result.output
    assert "'read the conventions'" in result.output
    assert "send status 'done' with it" in result.output
    assert [t.status for t in net.tasks] == ["done", "done", "done", "pending"]
    assert [t.outcome for t in net.tasks[:3]] == ["read", "wrote", "added"]
    # Every close is logged like any other transition (ADR-0110).
    closes = [e for e in net.log if e.kind == "step" and "-> done" in e.detail]
    assert len(closes) == 3


async def test_the_measured_doom_loop_completes_in_the_call_carrying_the_last_outcome() -> None:
    ctx, net = _ctx()
    tool = UpdatePlanTool()
    await tool.execute({"tasks": _plan({})}, ctx)
    net.attach_evidence(net.current(), "Write plugin.md ✓")

    r1 = await tool.execute({"tasks": _plan({0: "read"})}, ctx)
    assert r1.output.startswith("Plan updated: 1/4 steps done · current: 2 write the plugin doc")
    r2 = await tool.execute({"tasks": _plan({0: "read", 1: "wrote", 2: "added"})}, ctx)
    assert r2.output.startswith("Plan updated: 3/4 steps done · current: 4 run the tests")
    full = _plan({0: "read", 1: "wrote", 2: "added", 3: "green"})
    r3 = await tool.execute({"tasks": full}, ctx)
    assert r3.output.startswith("Plan updated: 4/4 steps done — complete."), r3.output
    assert r3.hint == _COMPLETE_HINT
    assert net.is_complete()
    # A blind resend of the same status-less plan is unchanged, and there is nothing to advance.
    r4 = await tool.execute({"tasks": full}, ctx)
    assert r4.output.startswith("Plan unchanged: 4/4 steps done — complete."), r4.output
    assert r4.data is not None and "autoadvanced" not in r4.data


async def test_an_explicit_status_beside_an_outcome_is_the_models_call() -> None:
    ctx, net = _ctx()
    result = await UpdatePlanTool().execute(
        {"tasks": _plan({0: "half of it"}, {0: "in_progress"})}, ctx
    )
    assert result.output.startswith("Plan updated: 0/4 steps done · current: 1")
    assert result.data is not None and result.data["closed_on_outcome"] == 0
    assert "no status as done" not in result.output
    assert net.tasks[0].status == "in_progress" and net.tasks[0].outcome == "half of it"


async def test_the_todowrite_alias_always_carries_a_status_so_nothing_changes() -> None:
    ctx, _net = _ctx()
    todos = [{"content": "a", "status": "in_progress"}, {"content": "b", "status": "pending"}]
    result = await UpdatePlanTool().execute({"todos": todos}, ctx)
    assert result.output.startswith("Plan updated: 0/2 steps done")
    assert result.data is not None and result.data["closed_on_outcome"] == 0


@pytest.mark.asyncio
async def test_the_loop_notes_the_close_for_the_census() -> None:
    from tests.test_loop_planning import _done, _judge_ok, _loop, _plan_call, _Scripted

    plan = [{"title": "A"}, {"title": "B"}]
    closing = [{"title": "A", "outcome": "did A"}, {"title": "B"}]
    provider = _Scripted([_plan_call(plan), _judge_ok(), _plan_call(closing), _done()])
    loop, session = _loop(provider)
    await loop.arun_turn("do a two-step thing")

    notes = [
        e
        for e in loop._trace.events
        if e.kind == "intervention" and e.data.get("kind") == "plan_outcome_close"
    ]
    assert len(notes) == 1
    outputs = [
        b.output
        for m in session.messages
        for b in m.blocks
        if type(b).__name__ == "ToolResultBlock" and "Plan " in b.output
    ]
    assert [o.split(":")[0] for o in outputs] == ["Plan updated", "Plan updated"]
    assert "read 1 step(s) sent with an outcome and no status as done" in outputs[1]

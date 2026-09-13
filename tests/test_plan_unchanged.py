"""A resend of the plan in force is answered "unchanged", never "updated" (ADR-0168).

Measured on the bench (arm K, 2026-09-13, a 35B on task 10, basin-sampled): after finishing a
step the model sent the plan back byte-for-byte four times, read "Plan updated: 0/4 steps done"
each time — the tool had replaced the plan with itself and said so as if something had moved —
and the turn ended as a doom loop. The receipt now says what happened, on the first resend, and
names the way forward for the step in hand; the loop notes the rail so the census can count it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.tasks import Task, TaskNetwork
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.update_plan import _COMPLETE_HINT, UpdatePlanTool

PLAN = [
    {"title": "Create utils/duration.py", "status": "in_progress", "note": "module exists"},
    {"title": "Export parse_duration", "blocked_by": ["1"]},
    {"title": "Add tests"},
]


def _ctx() -> tuple[ToolContext, TaskNetwork]:
    net = TaskNetwork()
    return ToolContext(workspace_root=Path("/tmp"), task_network=net), net


async def test_a_resend_of_the_plan_in_force_is_unchanged_not_updated() -> None:
    ctx, net = _ctx()
    first = await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    assert first.output.startswith("Plan updated: 0/3 steps done")
    events = len(net.log)

    again = await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    assert not again.is_error
    assert again.output.startswith("Plan unchanged: 0/3 steps done")
    assert "current: 1 Create utils/duration.py" in again.output
    assert "Nothing was updated" in again.output
    assert again.data is not None and again.data["unchanged"] is True
    assert again.hint is not None
    assert "If step 1 is finished" in again.hint and "status 'done'" in again.hint
    assert "do it now with a tool call" in again.hint
    # A resend is not an event: the record and the progress are exactly as they were.
    assert len(net.log) == events
    assert net.progress() == (0, 3)
    assert [t.title for t in net.tasks] == [t["title"] for t in PLAN]


async def test_a_status_tick_a_note_or_an_outcome_edit_is_an_update() -> None:
    ctx, _ = _ctx()
    await UpdatePlanTool().execute({"tasks": PLAN}, ctx)

    ticked = [dict(PLAN[0], status="done", outcome="wrote the module"), *PLAN[1:]]
    result = await UpdatePlanTool().execute({"tasks": ticked}, ctx)
    assert result.output.startswith("Plan updated: 1/3 steps done")
    assert result.data is not None and "unchanged" not in result.data

    noted = [ticked[0], dict(ticked[1], note="the symbol imports"), ticked[2]]
    result = await UpdatePlanTool().execute({"tasks": noted}, ctx)
    assert result.output.startswith("Plan updated: 1/3 steps done")

    reworded = [dict(noted[0], outcome="wrote utils/duration.py"), *noted[1:]]
    result = await UpdatePlanTool().execute({"tasks": reworded}, ctx)
    assert result.output.startswith("Plan updated: 1/3 steps done")

    # And the same plan once more IS a resend.
    result = await UpdatePlanTool().execute({"tasks": reworded}, ctx)
    assert result.output.startswith("Plan unchanged: 1/3 steps done · current: 2 Export")


async def test_the_rail_names_a_recorded_outcome_whose_status_never_moved() -> None:
    # The measured shape: every step still pending, the finished one carrying an outcome.
    plan = [
        {"title": "Create utils/duration.py", "outcome": "wrote the module"},
        {"title": "Export parse_duration"},
    ]
    ctx, _ = _ctx()
    await UpdatePlanTool().execute({"tasks": plan}, ctx)
    again = await UpdatePlanTool().execute({"tasks": plan}, ctx)
    assert again.output.startswith("Plan unchanged: 0/2 steps done · current: 1 ")
    assert again.hint is not None
    assert again.hint.startswith(
        "Step 1 already carries an outcome but its status is still 'pending'. "
    )


async def test_a_finished_plan_resent_gets_the_verdict_rail() -> None:
    done = [{"title": "a", "status": "done"}, {"title": "b", "status": "done"}]
    ctx, _ = _ctx()
    first = await UpdatePlanTool().execute({"tasks": done}, ctx)
    assert first.output.startswith("Plan updated: 2/2 steps done — complete.")
    again = await UpdatePlanTool().execute({"tasks": done}, ctx)
    assert again.output.startswith("Plan unchanged: 2/2 steps done — complete.")
    assert again.hint == _COMPLETE_HINT
    assert again.data is not None and again.data["complete"] is True


def test_state_signature_sees_what_progress_signature_ignores() -> None:
    net = TaskNetwork()
    net.replace_from_author([Task(title="a", note="n"), Task(title="b")])
    progress, state = net.progress_signature(), net.state_signature()
    net.tasks[0].note = "tests pass"
    assert net.progress_signature() == progress
    assert net.state_signature() != state
    state = net.state_signature()
    net.tasks[1].outcome = "found it"
    assert net.state_signature() != state
    state = net.state_signature()
    net.tasks[1].blocked_by = ["1"]
    assert net.state_signature() != state


@pytest.mark.asyncio
async def test_the_loop_notes_the_unchanged_receipt_as_an_intervention() -> None:
    from tests.test_loop_planning import _done, _judge_ok, _loop, _plan_call, _Scripted

    plan = [{"title": "A", "status": "in_progress"}, {"title": "B"}]
    provider = _Scripted([_plan_call(plan), _judge_ok(), _plan_call(plan), _done()])
    loop, session = _loop(provider)
    await loop.arun_turn("do a two-step thing")

    notes = [
        e
        for e in loop._trace.events
        if e.kind == "intervention" and e.data.get("kind") == "plan_unchanged"
    ]
    assert len(notes) == 1
    outputs = [
        b.output
        for m in session.messages
        for b in m.blocks
        if type(b).__name__ == "ToolResultBlock" and "Plan " in b.output
    ]
    assert [o.split(":")[0] for o in outputs] == ["Plan updated", "Plan unchanged"]
    assert "Hint: Do not resend the same plan." in outputs[1]

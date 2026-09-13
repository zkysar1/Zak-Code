"""Lever N (ADR-0168, opt-in ``plan_autoadvance``): when the model resends the plan unchanged and
the current step has been worked on but left non-terminal, the harness marks it done and advances.

Measured on the bench (arm M, 2026-09-13, a 35B on task 10): the model wrote the module, exported
it, wrote passing tests, fixed a test bug — then resent an all-``pending`` plan six times, saying
"The work is already done … I just need to mark the plan steps complete" before resending it again,
and the turn doom-looped. A text rail does not move a model that will not emit ``status: done``;
this lever does it for the model, deterministically, and makes the advance stick against the next
full-replace so the model cannot undo it by resending the same plan.
"""

from __future__ import annotations

from pathlib import Path

from zakcode.tasks import TaskNetwork
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.update_plan import UpdatePlanTool

PLAN = [
    {"title": "write the module", "note": "file exists"},
    {"title": "export it", "note": "symbol imports"},
    {"title": "add tests", "note": "tests pass"},
]


def _ctx(*, autoadvance: bool) -> tuple[ToolContext, TaskNetwork]:
    net = TaskNetwork()
    return ToolContext(
        workspace_root=Path("/tmp"), task_network=net, plan_autoadvance=autoadvance
    ), net


async def _lay_out_and_work(ctx: ToolContext, net: TaskNetwork) -> None:
    """Send the plan once, then record that the current step was worked on (evidence attached)."""
    await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    net.attach_evidence(net.current(), "wrote /tmp/utils/duration.py")


async def test_off_by_default_a_worked_step_resent_unchanged_stays_unchanged() -> None:
    ctx, net = _ctx(autoadvance=False)
    await _lay_out_and_work(ctx, net)
    result = await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    assert result.output.startswith("Plan unchanged")
    assert result.data is not None and not result.data.get("autoadvanced")
    assert net.current().title == "write the module"  # nothing advanced


async def test_on_a_worked_step_resent_unchanged_is_advanced_for_the_model() -> None:
    ctx, net = _ctx(autoadvance=True)
    await _lay_out_and_work(ctx, net)
    result = await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    assert result.output.startswith("Advanced step 1 ('write the module') to done")
    assert "current: 2 export it" in result.output
    assert result.data is not None and result.data["autoadvanced"] is True
    assert result.data["advanced_step"] == "1"
    assert result.hint is not None and "Do step 2 now" in result.hint
    step = net.tasks[0]
    assert step.status == "done" and step.harness_done is True
    assert step.outcome  # filled from the evidence line
    assert net.current().title == "export it"


async def test_the_advance_is_sticky_across_a_blind_resend_of_the_same_plan() -> None:
    ctx, net = _ctx(autoadvance=True)
    await _lay_out_and_work(ctx, net)
    await UpdatePlanTool().execute({"tasks": PLAN}, ctx)  # advances step 1
    assert net.tasks[0].status == "done"

    # The model resends its ORIGINAL all-pending plan (it will not emit status:done). The full
    # replace would set step 1 back to pending, but harness_done keeps it done — and the frontier
    # advances again (the plan has been worked on).
    result = await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    assert net.tasks[0].status == "done"  # NOT undone by the resend
    assert result.output.startswith("Advanced step 2 ('export it') to done")
    assert net.current().title == "add tests"


async def test_it_walks_the_frontier_to_completion_then_hands_over_the_verdict() -> None:
    ctx, net = _ctx(autoadvance=True)
    await _lay_out_and_work(ctx, net)
    # The model keeps resending the same all-pending plan; the harness walks each worked step done.
    r1 = await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    assert r1.data["advanced_step"] == "1"
    r2 = await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    assert r2.data["advanced_step"] == "2"
    r3 = await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    assert r3.data["advanced_step"] == "3"
    assert net.is_complete()
    assert r3.data["complete"] is True
    assert "complete." in r3.output
    # The last advance carries the verdict rail (ADR-0108): answer the request, don't report "done".
    assert r3.hint is not None and "Re-read the user's original request" in r3.hint


async def test_a_never_worked_plan_resent_unchanged_is_not_advanced() -> None:
    # No evidence anywhere and no outcome: the harness invents no progress it has no sign of.
    ctx, net = _ctx(autoadvance=True)
    await UpdatePlanTool().execute({"tasks": PLAN}, ctx)  # laid out, never worked
    result = await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    assert result.output.startswith("Plan unchanged")
    assert result.data is not None and not result.data.get("autoadvanced")
    assert net.current().title == "write the module"


async def test_an_outcome_recorded_but_status_left_pending_is_advanced() -> None:
    # The "recorded the result, forgot the status" shape — advanced on the model's own outcome.
    ctx, net = _ctx(autoadvance=True)
    plan = [dict(PLAN[0], outcome="wrote the module"), PLAN[1], PLAN[2]]
    await UpdatePlanTool().execute({"tasks": plan}, ctx)
    result = await UpdatePlanTool().execute({"tasks": plan}, ctx)
    assert result.output.startswith("Advanced step 1")
    assert net.tasks[0].status == "done"


async def test_an_edit_is_still_an_edit_not_an_advance() -> None:
    # Autoadvance only fires on an UNCHANGED resend; a real status tick is a normal update.
    ctx, net = _ctx(autoadvance=True)
    await _lay_out_and_work(ctx, net)
    ticked = [dict(PLAN[0], status="done"), PLAN[1], PLAN[2]]
    result = await UpdatePlanTool().execute({"tasks": ticked}, ctx)
    assert result.output.startswith("Plan updated: 1/3 steps done")
    assert result.data is not None and not result.data.get("autoadvanced")

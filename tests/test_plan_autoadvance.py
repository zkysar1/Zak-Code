"""Lever N (ADR-0168): when the model resends the plan unchanged and the current step has been
worked on but left non-terminal, the harness marks it done and advances. UNCONDITIONAL since
ADR-0202 — there is no setting, no env var, and no context flag.

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


def _ctx() -> tuple[ToolContext, TaskNetwork]:
    net = TaskNetwork()
    return ToolContext(workspace_root=Path("/tmp"), task_network=net), net


async def _lay_out_and_work(ctx: ToolContext, net: TaskNetwork) -> None:
    """Send the plan once, then record that the current step was worked on (evidence attached)."""
    await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    net.attach_evidence(net.current(), "wrote /tmp/utils/duration.py")


async def test_there_is_no_switch_the_advance_is_unconditional() -> None:
    # ADR-0202. This lever fixes a defect (a doom loop that burns a whole turn), and a defect fix
    # does not get a switch: two behaviours means every future reader has to establish which one
    # they are looking at, and every bug report has to say which side it came from. The pin is on
    # the ABSENCE of the knob in both places it used to live, so re-introducing one fails here.
    from zakcode.config import Settings

    assert not hasattr(Settings(), "plan_autoadvance")
    assert "plan_autoadvance" not in ToolContext.model_fields
    # And the behaviour reaches a context built with no Settings at all — the bare/embedder path
    # that used to default to the conservative side and silently get the doom loop instead.
    ctx, net = _ctx()
    await _lay_out_and_work(ctx, net)
    result = await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    assert result.data is not None and result.data["autoadvanced"] is True


async def test_a_worked_step_resent_unchanged_is_advanced_for_the_model() -> None:
    ctx, net = _ctx()
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
    ctx, net = _ctx()
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
    ctx, net = _ctx()
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
    # No evidence anywhere and no outcome: the harness invents no progress it has no sign of. This
    # is what bounds an unconditional lever — the trigger is the evidence, not a setting.
    ctx, net = _ctx()
    await UpdatePlanTool().execute({"tasks": PLAN}, ctx)  # laid out, never worked
    result = await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    assert result.output.startswith("Plan unchanged")
    assert result.data is not None and not result.data.get("autoadvanced")
    assert net.current().title == "write the module"


async def test_an_outcome_recorded_but_status_left_pending_is_advanced() -> None:
    # The "recorded the result, said pending anyway" shape — advanced on the model's own outcome.
    # The status is EXPLICIT here: a step sent with an outcome and no status at all closes in the
    # call that carries it (ADR-0254, test_plan_outcome_closes.py), so it never reaches this lever.
    ctx, net = _ctx()
    plan = [dict(PLAN[0], status="pending", outcome="wrote the module"), PLAN[1], PLAN[2]]
    first = await UpdatePlanTool().execute({"tasks": plan}, ctx)
    assert first.output.startswith("Plan updated: 0/3 steps done")
    result = await UpdatePlanTool().execute({"tasks": plan}, ctx)
    assert result.output.startswith("Advanced step 1")
    assert net.tasks[0].status == "done"


async def test_an_edit_is_still_an_edit_not_an_advance() -> None:
    # Autoadvance only fires on an UNCHANGED resend; a real status tick is a normal update.
    ctx, net = _ctx()
    await _lay_out_and_work(ctx, net)
    ticked = [dict(PLAN[0], status="done"), PLAN[1], PLAN[2]]
    result = await UpdatePlanTool().execute({"tasks": ticked}, ctx)
    assert result.output.startswith("Plan updated: 1/3 steps done")
    assert result.data is not None and not result.data.get("autoadvanced")


async def test_a_same_title_child_does_not_break_the_advance_or_its_stickiness() -> None:
    # The measured arm-N shape (ON b3 r3): the 35B nested a parent "Create utils/duration.py" over
    # a same-named child leaf, then resent the all-pending plan 9x and doom-looped. The carryover
    # was keyed by title alone, so the parent and child collided; the child leaf inherited the
    # parent's empty evidence and harness_done=False on every full-replace, and the advance never
    # stuck (progress frozen at 0/3). The carryover now keys on (title, is-parent), so the child
    # keeps its own memory. Regression for that collision.
    ctx, net = _ctx()
    plan = [
        {
            "title": "Create utils/duration.py",
            "subtasks": [{"title": "Create utils/duration.py", "note": "file exists"}],
        },
        {"title": "Export parse_duration"},
        {"title": "Add tests"},
    ]
    await UpdatePlanTool().execute({"tasks": plan}, ctx)
    net.attach_evidence(net.current(), "wrote /tmp/utils/duration.py")
    r1 = await UpdatePlanTool().execute({"tasks": plan}, ctx)
    assert r1.data is not None and r1.data.get("autoadvanced")  # fires despite the title collision
    child = net.tasks[0].children[0]
    assert child.status == "done" and child.harness_done is True
    # STICKS and walks the frontier: the next identical resend advances step 2, not 1.1 again.
    r2 = await UpdatePlanTool().execute({"tasks": plan}, ctx)
    assert net.tasks[0].children[0].status == "done"  # the collision no longer undoes it
    assert r2.output.startswith("Advanced step 2")


async def test_the_doom_guard_lets_the_harness_walk_a_resent_plan_to_completion() -> None:
    # Fix (b), ADR-0168 lever N: the doom-loop guard keys on the model's identical tool-call batch,
    # so a model that resends the same plan trips it. But the harness ADVANCES the plan on each
    # resend — progress, not a stall — so the guard resets its counter on a harness advance and the
    # frontier walks to completion instead of dying mid-walk. Measured (arm N ON b3 r3): the
    # advance fired but the run still doom-looped. Loop-level regression: if the reset regresses,
    # the walk dies partway and is_complete() is False, which is exactly what this asserts.
    from tests.test_loop_planning import _judge_ok, _plan_call, _Scripted
    from zakcode.agent.loop import AgentLoop
    from zakcode.config import Settings
    from zakcode.session import Session
    from zakcode.tools import default_registry

    # Every step carries an outcome but says pending: each identical resend advances the current
    # step (the doom-loop shape where the model does the work but won't emit ``status: done``).
    # The pending is explicit: sent with no status at all, an outcome-bearing step closes in the
    # call that carries it (ADR-0254) and there would be no walk to test.
    plan = [
        {"title": "A", "status": "pending", "outcome": "did A"},
        {"title": "B", "status": "pending", "outcome": "did B"},
        {"title": "C", "status": "pending", "outcome": "did C"},
    ]
    # author, decomposition judge, then resend the identical plan (Scripted repeats the last).
    provider = _Scripted([_plan_call(plan), _judge_ok(), _plan_call(plan)])
    session = Session(cwd="/tmp", model="test/model")
    loop = AgentLoop(provider, default_registry(), session, max_iterations=20, settings=Settings())
    await loop.arun_turn("do a three-step thing")

    net = session.task_network
    assert net.is_complete()  # the harness walked all three steps done across identical resends
    assert [t.status for t in net.tasks] == ["done", "done", "done"]

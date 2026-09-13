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


def test_author_signature_tracks_model_fields_not_harness_derived_outcome() -> None:
    from zakcode.tasks import author_signature

    def a(**over: object) -> list[Task]:
        first = {"title": "a", "status": "in_progress", "note": "n"}
        first.update(over)
        return [Task(**first), Task(title="b")]  # type: ignore[arg-type]

    base = author_signature(a())
    assert author_signature(a()) == base  # same submission
    assert author_signature(a(status="done")) != base  # a status tick
    assert author_signature(a(note="m")) != base  # a note edit
    assert author_signature(a(outcome="x")) != base  # an outcome edit
    assert (
        author_signature(
            [Task(title="a", status="in_progress", note="n"), Task(title="b", blocked_by=["1"])]
        )
        != base
    )  # noqa: E501


async def test_the_rail_fires_on_the_FIRST_resend_despite_non_idempotent_carryover() -> None:
    # The measured arm-M shape: a compound parent whose child carries an outcome the harness
    # propagates on the SECOND apply. A state compare only converges then; the submission compare
    # fires now. (Regression for the fires-one-call-late defect.)
    plan = [
        {
            "title": "Create utils/duration.py",
            "outcome": "utils/duration.py created with parse_duration",
            "subtasks": [{"title": "write parse_duration", "note": "file has parse_duration"}],
        },
        {"title": "Export it"},
    ]
    ctx, net = _ctx()
    first = await UpdatePlanTool().execute({"tasks": plan}, ctx)
    assert first.output.startswith("Plan updated")
    # The harness has NOT propagated the outcome to the child yet (non-idempotent) — prove it so the
    # test fails loudly if that ever changes and the point is moot.
    child = net.tasks[0].children[0]
    assert child.outcome == ""
    second = await UpdatePlanTool().execute({"tasks": plan}, ctx)  # the FIRST resend
    assert second.output.startswith("Plan unchanged"), second.output
    assert second.data is not None and second.data["unchanged"] is True


async def test_a_delayed_challenge_close_reads_updated_not_unchanged() -> None:
    # A null-result close is reopened once (ADR-0116); resending the same done-close APPLIES it the
    # second time. The submission is identical but the plan advances, so the event-count half of the
    # predicate must keep it "updated".
    ctx, net = _ctx()
    start = {"tasks": [{"title": "find the config", "status": "in_progress"}]}
    await UpdatePlanTool().execute(start, ctx)
    net.attach_evidence(net.tasks[0], "search ∅ No files found matching the query.")
    close = {"tasks": [{"title": "find the config", "status": "done"}]}
    reopened = await UpdatePlanTool().execute(close, ctx)
    assert "REOPENED" in reopened.output  # the challenge fired
    assert net.tasks[0].status == "in_progress"
    applied = await UpdatePlanTool().execute(close, ctx)  # same submission, but the close now takes
    assert applied.output.startswith("Plan updated"), applied.output
    assert net.tasks[0].status == "done"


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

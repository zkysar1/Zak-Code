"""A step keeps the note and outcome the model already gave it (ADR-0235).

Measured on three served bodies (2026-09-23, their last 24 hours of transcripts): an
``update_plan`` call carried 2,272 to 2,812 characters of arguments, and calls that did nothing
but update the plan took 9.5 to 27.3% of a body's model time, almost all of it decoding those
characters. Notes and outcomes were 41 to 57% of them, and on the busiest body 83% of the notes
and 74% of the outcomes were the same as in the call before. The model now sends every step's
title and status; a note or outcome it leaves out is kept, and leaving it out is not an edit.
"""

from __future__ import annotations

from pathlib import Path

from zakcode.tasks import Task, TaskNetwork, author_fields, author_signature
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.update_plan import UpdatePlanTool

LONG = [
    {"title": "Create utils/duration.py", "status": "in_progress", "note": "the module imports"},
    {"title": "Export parse_duration", "note": "the symbol imports from utils"},
    {"title": "Add tests", "note": "pytest passes"},
]
#: The same plan as the model may now send it: every step's title and status, nothing else.
SHORT = [{"title": step["title"], "status": step.get("status", "pending")} for step in LONG]


def _ctx() -> tuple[ToolContext, TaskNetwork]:
    net = TaskNetwork()
    return ToolContext(workspace_root=Path("/tmp"), task_network=net), net


async def test_a_resend_that_leaves_out_notes_and_outcomes_keeps_them() -> None:
    ctx, net = _ctx()
    await UpdatePlanTool().execute({"tasks": LONG}, ctx)
    closed = [
        dict(LONG[0], status="done", outcome="wrote the module"),
        dict(LONG[1], status="in_progress"),
        LONG[2],
    ]
    await UpdatePlanTool().execute({"tasks": closed}, ctx)

    short = [
        {"title": "Create utils/duration.py", "status": "done"},
        {"title": "Export parse_duration", "status": "done"},
        {"title": "Add tests", "status": "in_progress"},
    ]
    result = await UpdatePlanTool().execute({"tasks": short}, ctx)

    assert result.output.startswith("Plan updated: 2/3 steps done"), result.output
    assert [t.note for t in net.tasks] == [step["note"] for step in LONG]
    assert net.tasks[0].outcome == "wrote the module"
    assert "3 Add tests — pytest passes" in net.render()  # the done-condition stays in view


async def test_leaving_out_what_did_not_change_is_not_an_edit_in_either_form() -> None:
    ctx, net = _ctx()
    await UpdatePlanTool().execute({"tasks": LONG}, ctx)
    events = len(net.log)

    # The short form, then switching back and forth: one plan, so every resend is unchanged.
    for form in (SHORT, LONG, SHORT):
        again = await UpdatePlanTool().execute({"tasks": form}, ctx)
        assert again.output.startswith("Plan unchanged: 0/3 steps done"), again.output
        assert again.data is not None and again.data["unchanged"] is True
    assert len(net.log) == events
    assert [t.note for t in net.tasks] == [step["note"] for step in LONG]


async def test_the_short_form_still_reaches_the_advance() -> None:
    # The doom-loop rail (ADR-0168 lever N) must fire on the FIRST unchanged resend, and a model
    # that switched to the short form has changed nothing. Read literally, the short form would
    # differ from the long one and read "Plan updated", putting the advance off by a call.
    ctx, net = _ctx()
    await UpdatePlanTool().execute({"tasks": LONG}, ctx)
    net.attach_evidence(net.tasks[0], "Write utils/duration.py ✓")

    advanced = await UpdatePlanTool().execute({"tasks": SHORT}, ctx)

    assert advanced.output.startswith("Advanced step 1"), advanced.output
    assert net.tasks[0].status == "done" and net.tasks[0].note == "the module imports"
    assert net.tasks[0].outcome.startswith("last action:")  # the harness wrote this one

    # Keep resending: every resend walks the frontier. What the MODEL sent is what a left-out
    # field stands for, never an outcome the harness wrote into the step; the map read after
    # the replace held one, and the third resend read as an edit instead of advancing.
    for step in ("2", "3"):
        again = await UpdatePlanTool().execute({"tasks": SHORT}, ctx)
        assert again.data is not None and again.data.get("advanced_step") == step, again.output
    assert net.is_complete()
    assert net.last_author_fields["leaf:create utils/duration.py"] == ["the module imports", ""]


async def test_a_new_note_in_the_short_form_is_an_edit_and_is_then_what_was_sent() -> None:
    ctx, net = _ctx()
    await UpdatePlanTool().execute({"tasks": LONG}, ctx)

    renoted = [SHORT[0], dict(SHORT[1], note="parse_duration is in __all__"), SHORT[2]]
    result = await UpdatePlanTool().execute({"tasks": renoted}, ctx)

    assert result.output.startswith("Plan updated: 0/3 steps done"), result.output
    assert [t.note for t in net.tasks] == [
        "the module imports",
        "parse_duration is in __all__",
        "pytest passes",
    ]
    again = await UpdatePlanTool().execute({"tasks": SHORT}, ctx)
    assert again.output.startswith("Plan unchanged"), again.output
    assert net.tasks[1].note == "parse_duration is in __all__"


def test_a_kept_done_condition_still_counts_when_the_close_leaves_it_out() -> None:
    # ADR-0116 challenges a close on a null result only when the step has NO done-condition. A
    # model that gave one earlier and leaves it out of the closing call still has one.
    note = "a hit lists a .tar.gz, or the root listing shows any file"
    net = TaskNetwork()
    net.replace_from_author(
        [Task(title="search the drive", status="in_progress", note=note), Task(title="report")]
    )
    net.attach_evidence(net.tasks[0], "grep tar.gz ∅ No matches found")
    net.replace_from_author(
        [Task(title="search the drive", status="done"), Task(title="report", status="in_progress")]
    )
    assert net.tasks[0].status == "done" and net.tasks[0].challenged is False
    assert net.tasks[0].note == note

    # Positive control: a step that never had a done-condition is still challenged.
    bare = TaskNetwork()
    bare.replace_from_author(
        [Task(title="search the drive", status="in_progress"), Task(title="report")]
    )
    bare.attach_evidence(bare.tasks[0], "grep tar.gz ∅ No matches found")
    bare.replace_from_author(
        [Task(title="search the drive", status="done"), Task(title="report", status="in_progress")]
    )
    assert bare.tasks[0].challenged is True


def test_what_was_sent_is_keyed_like_the_carry_over() -> None:
    # A parent and a same-titled child (the measured shape ADR-0168 keys the carry-over apart
    # for) keep separate fields, and a short resend of the tree reads as the same plan.
    long = [
        Task(
            title="Build",
            kind="compound",
            note="the package builds",
            children=[Task(title="build", note="make exits 0")],
        )
    ]
    sent = author_fields(long)
    assert sent == {"parent:build": ["the package builds", ""], "leaf:build": ["make exits 0", ""]}
    short = [Task(title="Build", kind="compound", children=[Task(title="build")])]
    assert author_signature(short, sent) == author_signature(long)
    assert author_signature(short) != author_signature(long)  # read literally, it differs


async def test_what_was_sent_survives_a_restore_and_a_cleared_plan_forgets_it() -> None:
    ctx, net = _ctx()
    await UpdatePlanTool().execute({"tasks": LONG}, ctx)

    restored = TaskNetwork.model_validate_json(net.model_dump_json())
    assert restored.last_author_fields == net.last_author_fields != {}
    resumed = ToolContext(workspace_root=Path("/tmp"), task_network=restored)
    again = await UpdatePlanTool().execute({"tasks": SHORT}, resumed)
    assert again.output.startswith("Plan unchanged"), again.output

    # A session saved before the field existed loads as "nothing sent yet".
    older = net.model_dump()
    del older["last_author_fields"]
    assert TaskNetwork.model_validate(older).last_author_fields == {}

    await UpdatePlanTool().execute({"tasks": []}, ctx)
    assert net.last_author_fields == {} and net.last_author_signature == ""

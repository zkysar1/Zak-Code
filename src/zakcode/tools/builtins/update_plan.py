"""The ``update_plan`` tool — the model's handle on the hierarchical task network.

This is Zak Code's TodoWrite analog: the model calls it to lay out, refine, and track the
near-term plan for the current goal. The proven **full-replace** contract (the model hands the
*entire* current plan every call, with each step's status) keeps weak local models robust —
there is no fragile per-id patching — and lets the harness re-number ids and re-derive the HTN
invariants (derived parent status, single focus) from scratch on every edit (see
:mod:`zakcode.tasks`).

The tool only mutates the in-memory :class:`~zakcode.tasks.TaskNetwork` on the
:class:`~zakcode.tools.base.ToolContext`; the loop persists it and re-injects the rendered plan
into context each iteration. Decomposition *discipline* (decompose a goal to primitive steps,
finish the actionable ones before ending) is enforced by the loop's plan gate; the *method* for
how to break down a given kind of task is domain knowledge that lives in skills, never here.
"""

from __future__ import annotations

from typing import Any

from zakcode.config import PermissionTier
from zakcode.tasks import Task, TaskNetwork, TaskStatus, author_fields, author_signature, clip
from zakcode.tools.base import (
    RECEIPT_OF_CHANGE,
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)

#: Maximum decomposition depth the schema exposes. The near-term layer rarely needs more than
#: goal → step → sub-step; bounding it keeps the schema concrete (no recursive ``$ref`` that
#: weak models and some providers mishandle) while still allowing real hierarchy.
_MAX_DEPTH = 3

_STATUS_VALUES = ["pending", "in_progress", "done", "blocked", "cancelled"]

#: How the plan advances (ADR-0237, ADR-0416): inside the next step's first call, never in a
#: response of its own. Every model call is a round trip, and on a slow backend a response that
#: only updates the plan is the most expensive kind: measured 2026-09-23 on the 131k P40 pod, it
#: took 17.8 and 22.3 percent of the two worker sessions' model time (median 104 and 146 seconds,
#: against 38 and 35 for a work call). ADR-0237 asked the model to pair ``update_plan`` with the
#: work in one response; measured 2026-11-27 on eleven worker sessions of one 27B model, not one
#: of 108 update_plan calls shared a response with another call. So the close rides INSIDE the
#: call (ADR-0416): any tool but the plan's takes an optional ``step_done``, which the harness
#: applies before the call runs (the call's evidence is credited to the step it starts, ADR-0110).
#: One sentence, said by the tool description, this tool's receipt, the system prompt and the plan
#: the loop re-shows each call, so the model reads the same rule everywhere it looks. A paged
#: skill's section is the exception, and its page says so: the next section arrives only in the
#: reply to an update_plan call.
PLAN_ADVANCE = (
    'by adding "step_done": "<what it produced>" to the next step\'s first call, on any tool: '
    "a response that only updates the plan costs a whole model call"
)

#: The tools that edit or read the plan itself (ADR-0110 / ADR-0111): their calls are never a
#: step's evidence, and (ADR-0416) they never take ``step_done``.
PLAN_TOOL_NAMES = frozenset({"update_plan", "plan", "todo", "plan_recall", "plan_history"})

#: The argument that closes the plan's current step inside another call (ADR-0416), and the
#: schema property that declares it. A backend builds its tool-call grammar from the schema, so
#: the property must be declared on every tool that may carry it or the model cannot emit it.
STEP_DONE_ARG = "step_done"
STEP_DONE_PROPERTY: dict[str, Any] = {
    "type": "string",
    "description": (
        "Optional. Closes the plan's current step before this call runs and starts the next: "
        "what the finished step produced."
    ),
}

#: The short form (ADR-0388): close the current step and start the next without resending the
#: plan. Measured 2026-11-17 on ten 27B worker sessions behind one pod: the
#: responses that only updated the plan were 19 to 28 percent of each session's model time, a
#: median 44 s and 668 streamed tokens each, about 450 of them the resent plan (median 1,551
#: chars). Of 201 on four sessions, 70 resent the plan unchanged (lever N advanced most) and
#: 22 only closed a step and started the next, 45 percent of that time between them; an
#: ``advance`` says the same in a few dozen tokens. Said wherever the advance itself is
#: (:data:`PLAN_ADVANCE`).
#:
#: As the call itself (ADR-0388 amendment). Measured 2026-11-17 on the two trial sessions: in
#: their first 35 and 45 minutes on the short form, 5 responses only updated the plan and none
#: used it, every one resending the whole plan (median 833 and 5,036 chars). Every place that
#: named the short form named the whole-plan update first and the short form after it, in
#: words. A cue moves a small model where it lands on what the model already does: the call it
#: writes.
PLAN_ADVANCE_CALL = '{"advance": {"outcome": "<what the step produced>"}}'
PLAN_SHORT_FORM = (
    "To close the current step with no other call to make, call update_plan with "
    f"{PLAN_ADVANCE_CALL} instead of the whole plan."
)

#: Next-step rail: after planning, the model should act on the current step, not re-plan. The
#: short form leads (ADR-0388 amendment); the whole plan is for changing the steps.
_PLANNED_HINT = (
    "Plan updated. Now do the step marked '<- current'. When it is done, close it "
    f"{PLAN_ADVANCE}; update_plan with {PLAN_ADVANCE_CALL} does the same on its own. "
    "Send the whole plan only to add, split, drop or reorder steps."
)

#: Verdict rail (ADR-0108): the call that closes the LAST step is the one moment the harness
#: knows the plan is done, and until this hint existed it said nothing — an open plan got a
#: next-step rail, a finished one got silence, and a small model filled the silence with "the
#: plan is complete, no further action is needed" instead of the answer the user asked for.
_COMPLETE_HINT = (
    "Plan complete — every step is terminal. The plan was the means, not the deliverable: do "
    "not report that it is finished. Re-read the user's original request and answer it — lead "
    "with the conclusion or verdict, then the evidence the steps produced."
)

#: Verdict rail for a session that runs a loop (ADR-0392). A turn-end hook has named the skill
#: that re-enters this session's loop (``Session.loop_skill``, ADR-0187), so a finished plan was
#: one pass of that loop and no request is waiting on it. Served loop sessions given the rail
#: above went looking for one: they added steps to deliver a final verdict on "the original
#: request" and re-read rules about answering it. This rail names the loop and says nothing about
#: a request or a verdict, because those words were what the sessions acted on.
_LOOP_COMPLETE_HINT = (
    "Plan complete — every step is terminal. This session runs the /{skill} loop, so the plan was "
    "one pass of it, not a deliverable: do not report that it is finished. Do what /{skill} says "
    "comes next — its next step, its re-entry or its close."
)


def _complete_hint(ctx: ToolContext) -> str:
    """The verdict rail for this session: the loop's once a hook has named one (ADR-0392)."""
    loop = (ctx.loop_skill() if ctx.loop_skill is not None else "").split()
    return _LOOP_COMPLETE_HINT.format(skill=loop[0]) if loop else _COMPLETE_HINT


#: Challenge rail (ADR-0116): the harness just REOPENED a step the model closed on a null
#: result with no done-condition (the advisory in the output says which and why). The next
#: action is that step, with a positive control — not the step the model had moved on to.
_CHALLENGED_HINT = (
    "A step was reopened (see the note above): do it first — run a positive control (show the "
    "same tool sees something known to exist in that scope) or a query of a different shape — "
    "then close it with a done-condition in 'note' and what you found in 'outcome'."
)

#: Unchanged rail (ADR-0168): the model sent back the plan already in force — the same steps,
#: statuses, notes, outcomes, dependencies — so nothing was updated, and the receipt says so
#: instead of "Plan updated". Measured on the bench (arm K, a 35B on task 10, basin-sampled):
#: after finishing a step the model resent the plan four times without marking the step done,
#: read "Plan updated: 0/4 steps done" each time, and the turn ended as a doom loop. The rail
#: names the two ways forward in the step's own terms; it fires on the FIRST resend, an
#: iteration before the loop's generic exact-repeat guard.
_UNCHANGED_HINT = (
    "Do not resend the same plan. If step {id} is finished, call update_plan with 'advance' (its "
    "'outcome' says what it produced); if it is not, do it now with a tool call."
)

#: Advance rail (ADR-0168 lever N): the harness just marked a worked-on step done because the model
#: resent the plan unchanged and would not close it. Point at the step now current — do it, do not
#: resend the plan unchanged again.
_ADVANCED_HINT = (
    "Do step {id} now with a tool call. When a step is genuinely done, call update_plan with "
    "'advance' (its 'outcome' says what it produced); do not resend the plan unchanged."
)


def _task_schema(depth: int, *, described: bool = True) -> dict[str, Any]:
    """JSON schema for one task node, nesting ``subtasks`` to ``depth`` levels.

    The field descriptions are written once, on the outermost node (``described``). The nested
    nodes carry the same field names, types, status values and required list without them: the
    schema is sent on every model call, so a repeated description costs context every time.
    """
    properties: dict[str, Any] = {
        "title": {
            "type": "string",
            "description": "Short imperative description of the step (e.g. 'Add the route').",
        },
        "status": {
            "type": "string",
            "enum": _STATUS_VALUES,
            "description": (
                "pending (not started), in_progress (working on it now — keep at most ONE), "
                "done, blocked (cannot proceed, say why in note), or cancelled. Omit for pending; "
                "a step sent with an 'outcome' and no status is read as done."
            ),
        },
        "note": {
            "type": "string",
            "description": (
                "The step's done-condition: a one-line, checkable acceptance criterion you will "
                "verify against (e.g. 'tests pass', 'GET /health returns 200'). For a step that "
                "searches, lists, or looks something up, say what a hit looks like AND what "
                "proves the scope was visible — a null result never closes such a step on its "
                "own. For a blocked step, say why instead. Recommended on every primitive step; "
                "omit only if truly none applies. Kept once given: leave it out of later calls "
                "unless it changes."
            ),
        },
        "blocked_by": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional: ids of earlier steps that must finish before this one can start, using "
                "the step numbers shown in the plan (top-level 1, 2, 3…; sub-steps 2.1, 2.2…), "
                'e.g. ["1", "2"]. Omit when the step has no prerequisites.'
            ),
        },
        "outcome": {
            "type": "string",
            "description": (
                "What the step PRODUCED or found, one line — set it when you mark the step done "
                "('the flake is a stale cache', 'route added in app/users.py'). This is the "
                "record a later step, a resumed session, or the user reads back. Kept once given: "
                "leave it out of later calls unless it changes."
            ),
        },
    }
    if depth > 1:
        properties["subtasks"] = {
            "type": "array",
            "description": (
                "Sub-steps this step decomposes into, with the same fields as a step; a plan nests "
                f"{_MAX_DEPTH} levels at most. A step WITH subtasks is a compound goal (its status "
                "is derived from them); a step WITHOUT subtasks is a primitive action."
            ),
            "items": _task_schema(depth - 1, described=False),
        }
    if not described:
        properties = {
            name: {key: value for key, value in prop.items() if key != "description"}
            for name, prop in properties.items()
        }
    return {"type": "object", "properties": properties, "required": ["title"]}


def _build_task(
    raw: dict[str, Any], depth: int, closed_on_outcome: list[str] | None = None
) -> Task:
    """Build a :class:`Task` from a model-supplied node. ``kind`` is inferred from subtasks.

    ``closed_on_outcome`` collects the titles of the primitive steps this call read as done
    because they carried an ``outcome`` and no ``status`` (ADR-0254), so the receipt can say so.
    """
    title = str(raw.get("title", "")).strip() or "(untitled)"
    raw_status = raw.get("status")
    note = str(raw.get("note", "")).strip()
    outcome = str(raw.get("outcome", "")).strip()
    raw_deps = raw.get("blocked_by")
    blocked_by = (
        [str(d) for d in raw_deps if isinstance(d, str | int)] if isinstance(raw_deps, list) else []
    )
    children_raw = raw.get("subtasks") if depth > 1 else None
    children: list[Task] = []
    if isinstance(children_raw, list):
        children = [
            _build_task(c, depth - 1, closed_on_outcome)
            for c in children_raw
            if isinstance(c, dict)
        ]
    status: TaskStatus
    if raw_status in _STATUS_VALUES:
        status = raw_status
    elif raw_status is None and outcome and not children:
        # ADR-0254: an outcome on a step sent WITHOUT a status closes it. The schema says to omit
        # the status for pending and to set the outcome when marking the step done, so a step
        # carrying an outcome and no status contradicts itself, and the outcome is the positive
        # statement: the model wrote what the step PRODUCED. Reading it as pending fed a measured
        # doom loop — a model that never sent a status wrote outcomes on every step, read "0/4
        # steps done" back, and resent the plan until lever N walked the frontier one round trip
        # per step (17 plan calls, 24 turns, for a task its other runs finished in 6). Only the
        # ABSENT status closes: an explicit pending or in_progress beside an outcome is the model's
        # call, an unknown status string stays pending as before, and a compound step's status
        # is derived from its children.
        status = "done"
        if closed_on_outcome is not None:
            closed_on_outcome.append(title)
    else:
        status = "pending"
    return Task(
        title=title,
        status=status,
        note=note,
        outcome=outcome,
        blocked_by=blocked_by,
        kind="compound" if children else "primitive",
        children=children,
    )


#: Claude Code's ``TodoWrite`` statuses → this plan's (``completed`` is ``done`` here).
_TODO_STATUS = {
    "pending": "pending",
    "in_progress": "in_progress",
    "completed": "done",
    "done": "done",
    "cancelled": "cancelled",
}


def _from_todos(todos: list[Any]) -> list[dict[str, Any]]:
    """Claude Code's ``TodoWrite`` shape (``todos: [{content, status, activeForm}]``) as this
    plan's flat step list — the ``TodoWrite`` alias resolves here (ADR-0190). A todo with no
    usable text is dropped rather than refused: the plan it describes is still the model's."""
    steps: list[dict[str, Any]] = []
    for todo in todos:
        if not isinstance(todo, dict):
            continue
        title = todo.get("content") or todo.get("title") or todo.get("activeForm")
        if not isinstance(title, str) or not title.strip():
            continue
        status = _TODO_STATUS.get(str(todo.get("status", "pending")), "pending")
        steps.append({"title": title.strip(), "status": status})
    return steps


def _all_finished(tasks: list[Task]) -> bool:
    """Whether every primitive step in ``tasks`` is done or cancelled (ADR-0390)."""
    return bool(tasks) and all(
        _all_finished(t.children) if t.children else t.status in ("done", "cancelled")
        for t in tasks
    )


def _leaf_count(tasks: list[Task]) -> int:
    """How many primitive steps ``tasks`` holds."""
    return sum(_leaf_count(t.children) if t.children else 1 for t in tasks)


def plan_steps(args: dict[str, Any]) -> list[Any] | None:
    """The step list a call's arguments carry: ``tasks``, else the ``TodoWrite`` alias's
    ``todos`` (ADR-0190). ``None`` when neither is a list."""
    tasks = args.get("tasks")
    if tasks is None and isinstance(args.get("todos"), list):
        tasks = _from_todos(args["todos"])
    return tasks if isinstance(tasks, list) else None


def authors_a_plan(args: dict[str, Any]) -> bool:
    """Whether a call with these arguments leaves a plan of the model's on the board: at least
    one step object. An empty list clears the board and a list with no objects is refused, so
    neither counts. The loop's plan-first gate reads this before the call runs (ADR-0231)."""
    steps = plan_steps(args)
    return steps is not None and any(isinstance(step, dict) for step in steps)


class UpdatePlanTool(Tool):
    """Lay out or update the hierarchical task plan for the current goal."""

    spec = ToolSpec(
        name="update_plan",
        description=(
            "Maintain a hierarchical plan for a multi-step task. Call it FIRST on any task beyond "
            "a single obvious action, or that asks for several separate things: decompose the "
            "goal into ordered, primitive steps — each with a clear done-condition and no hidden "
            "'figure out how' (a step whose approach you have not worked out is not primitive: "
            "break it into 'subtasks' or make working out the how its own step, and use "
            "'blocked_by' when a step depends on earlier ones). Then, as "
            f"you go, mark a step done and the next one in_progress {PLAN_ADVANCE}. "
            f"{PLAN_SHORT_FORM} Otherwise send the WHOLE plan each time: every step's title and "
            "status. A step keeps the note "
            "and outcome you already gave it, so send those only when they are new or changed. "
            "When you mark a step done, record what it produced in its 'outcome'. Skip it only "
            "for a request for one straightforward thing you can do in a single action."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "The full, ordered plan — every step, with current statuses.",
                    "items": _task_schema(_MAX_DEPTH),
                },
                "advance": {
                    "type": "object",
                    "description": (
                        "The short form, sent INSTEAD of 'tasks' when the only change is that the "
                        "current step is finished: it closes that step and starts the next, "
                        'e.g. {"advance": {"outcome": "route added in app/users.py"}}.'
                    ),
                    "properties": {
                        "outcome": {
                            "type": "string",
                            "description": "What the step produced or found, one line.",
                        }
                    },
                },
            },
            # No ``required``: a call carries 'tasks' or 'advance' (ADR-0388), and a backend that
            # builds its tool-call grammar from the schema would otherwise force 'tasks' out.
        },
        # Planning never touches the workspace or system, so it is always available (READ_ONLY) —
        # the model should never be gated out of thinking. NEVER_PARALLEL: it mutates shared plan
        # state, so it must not be reordered against itself or run on the concurrent batch path.
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        network = ctx.task_network
        if network is None:
            return ToolResult.error(
                "planning is not available here (no task network on the context)"
            )
        tasks = plan_steps(args)
        advance = args.get("advance")
        advancing = advance is not None and advance is not False
        ignored: list[str] = []
        closing: Task | None = None
        if advancing and tasks is not None:
            ignored.append(
                "'advance' was ignored: this call also sent 'tasks', and the whole plan wins. "
                "Send 'advance' on its own to close the current step and start the next."
            )
        elif advancing:
            short = self._advance_steps(network, advance)
            if isinstance(short, ToolResult):
                return short
            tasks, closing = short
        if tasks is None:
            return ToolResult.error(
                "'tasks' must be an array of step objects ({title, status?, note?, subtasks?}), "
                "or 'advance' must close the current step",
                fix='Pass the full plan as an array, e.g. [{"title": "...", "status": "pending"}], '
                'or close the current step with {"advance": {"outcome": "what it produced"}}.',
            )
        if not tasks:
            # An empty plan clears the board (the goal is a single action, or work is abandoned).
            # The record keeps what was dropped (ADR-0110).
            if network.tasks:
                dropped = network.actionable_remaining()
                network.record(
                    "cleared",
                    detail=f"{len(dropped)} open step(s) dropped by the model: "
                    + "; ".join(clip(t.title, 40) for t in dropped[:6]),
                )
            network.tasks = []
            network.last_author_signature = ""
            network.last_author_fields = {}
            network.normalize()
            return ToolResult.ok("Plan cleared.", data={"task_count": 0})

        closed_on_outcome: list[str] = []
        built = [
            _build_task(t, _MAX_DEPTH, closed_on_outcome) for t in tasks if isinstance(t, dict)
        ]
        if not built:
            # Every item was malformed (no objects). Don't wipe an existing plan over a bad
            # call — leave it untouched and tell the model how to shape the input.
            return ToolResult.error(
                "no valid steps found: each item in 'tasks' must be an object with a 'title'",
                fix='e.g. [{"title": "first step"}, {"title": "second step"}]',
            )
        # Full-replace, but the steps' MEMORY (evidence, outcome, origin) carries over by title
        # and every transition is logged (ADR-0110) — the model resends the plan, not its record.
        # A note or outcome it leaves out is one it did not change (ADR-0235): the signature reads
        # the submission through what it last sent, and the network keeps the field itself.
        sent = network.last_author_fields
        submitted = author_signature(built, sent)
        # Read before the replace: it installs these very objects and writes the harness's
        # carry-over into them, and a "last sent" map holding a harness-filled outcome makes
        # the next identical resend read as an edit.
        fields = author_fields(built, sent)
        prior_author = network.last_author_signature
        if network.is_empty() and submitted == prior_author and _all_finished(built):
            # ADR-0390: the plan the model last sent finished, and the turn-start reset took it off
            # the board (ADR-0110). Sent back unchanged it would go up again, and with it the
            # "answer now" line on every call of a turn that has nothing to do with it. (An
            # advance never gets here with the board empty: it is refused first, ADR-0388.)
            return self._finished_resend(built)
        events = network.log_folded + len(network.log)
        advisories = ignored + network.replace_from_author(built)
        network.last_author_signature = submitted
        network.last_author_fields = fields

        finished, total = network.progress()
        if submitted == prior_author and network.log_folded + len(network.log) == events:
            # Unchanged rail (ADR-0168): the model resent the plan already in force — the same tree
            # it last sent — so nothing was updated, and "Plan updated" was the receipt that fed a
            # measured doom loop. Comparing the SUBMISSION (author_signature), not the network
            # state, fires on the FIRST resend: replace_from_author's outcome carry-over is non-
            # idempotent and a state compare only converges on the second apply (arm M). The
            # event-count half is still load-bearing: a null close the harness hands back (ADR-0116)
            # resends the same tree but records the challenge / applies the delayed close: updated.
            current = network.current()
            if (
                current is not None
                and current.status in ("pending", "in_progress")
                and (
                    current.evidence
                    or current.outcome
                    or any(leaf.evidence for leaf in network.leaves())
                )
            ):
                # Lever N (ADR-0168, unconditional since ADR-0202): the plan has been worked on
                # and the model will not close the step — mark it done and move the frontier,
                # instead of the rail it ignores (arm M). The trigger is the EVIDENCE, not a
                # setting: with nothing worked this falls through to the plain unchanged receipt,
                # which is what keeps an always-on advance from inventing progress.
                return self._autoadvance(network, current, _complete_hint(ctx))
            return self._unchanged(network, finished, total, _complete_hint(ctx))
        if closed_on_outcome:
            # ADR-0254: say what was read, in the receipt the model gets back — a harness that
            # closes a step on the model's behalf tells the model so, and names the contract it
            # applied, so the next plan can carry the status itself.
            named = "; ".join(f"'{clip(t, 40)}'" for t in closed_on_outcome[:4])
            more = f" (+{len(closed_on_outcome) - 4} more)" if len(closed_on_outcome) > 4 else ""
            advisories.append(
                f"read {len(closed_on_outcome)} step(s) sent with an outcome and no status as "
                f"done: {named}{more} — an outcome means the step is finished; send status 'done' "
                "with it."
            )
        quality, deficiencies = network.quality()
        # The result is a RECEIPT, not the plan (ADR-0124). The model just sent the whole plan
        # (full-replace), and the loop re-injects the live checklist as an ephemeral tail
        # message every iteration anyway — so echoing it here put a third copy into the
        # PERSISTED history on every call, compounding for the rest of the session (measured:
        # 37 iterations on a 40-step plan, 11.68M tokens). What stays is what the model cannot
        # get elsewhere: the step to act on now, this call's advisories, this edit's score.
        current = network.current()
        if network.is_complete():
            output = f"Plan updated: {finished}/{total} steps done — complete."
        elif current is not None:
            output = (
                f"Plan updated: {finished}/{total} steps done · current: "
                f"{current.id} {clip(current.title, 80)}"
            )
        else:
            output = f"Plan updated: {finished}/{total} steps done."
        closed = network.get(closing.id) if closing is not None else None
        if closed is not None and closed.status == "done":
            # ADR-0388: the short form names the step it closed. The model never sent the plan,
            # so this line is how it learns WHICH step the advance closed — the one the plan
            # marked current, which need not be the one the model had in mind.
            output = f"Step {closed.id} done ({clip(closed.title, 60)!r}). {output}"
        if advisories:
            output += "\n\nNotes:\n" + "\n".join(f"- {a}" for a in advisories)
        if deficiencies:
            # Structural quality (ADR-0050): the evaluate_candidate port scores every edit
            # for free; the named deficiencies make the number actionable.
            output += f"\n\nPlan quality {round(quality * 100)}%: " + "; ".join(deficiencies[:3])
        return ToolResult.ok(
            output,
            data={
                "task_count": total,
                "finished": finished,
                "advisories": advisories,
                "quality": quality,
                "deficiencies": deficiencies,
                "complete": network.is_complete(),
                # ADR-0254: how many steps this call read as done on their outcome alone. The
                # loop notes it so the census can count the shape (as it counts lever N).
                "closed_on_outcome": len(closed_on_outcome),
                # ADR-0388: the step this call closed through the short form ("" when the call
                # sent the plan), so a census can count how often a model advances that way.
                "short_form": closing.id if closing is not None else "",
                # ADR-0209: this receipt acknowledges an edit that CHANGED the plan, and it
                # reads the same whenever the done count and the current step do. Measured
                # in a served loop: 14 of 47 stuck rungs, two of them STOPs, fell on it while
                # the model was journalling into its plan between real steps. The two
                # receipts for a plan sent back UNCHANGED (below) do not carry the flag:
                # they are the churn, and the stuck ladder must keep counting them.
                RECEIPT_OF_CHANGE: True,
            },
            hint=self._hint(network, _complete_hint(ctx)),
        )

    @staticmethod
    def _advance_steps(
        network: TaskNetwork, advance: Any
    ) -> tuple[list[dict[str, Any]], Task] | ToolResult:
        """The short form (ADR-0388) as the whole plan it stands for, and the step it closes.

        Every step as it stands, the current one ``done`` with the outcome given, and the step
        that is current once it is done ``in_progress``: the full-replace the model would have
        sent, so the path that installs it runs unchanged — the carry-over, the null-close
        challenge (ADR-0116), the log, the receipt and the record of what the model last sent.
        Notes and outcomes are left out, so every step keeps its own (ADR-0235); only the closed
        step's outcome is new. ``advance`` is ``{"outcome": "..."}``, a bare outcome string, or
        anything else for a close with no outcome (the harness then fills it from the step's
        last evidence line, as it does for a full-replace).
        """
        if network.is_anchor_only():
            return ToolResult.error(
                "there is no plan of yours to advance yet",
                fix="Lay out the plan first: send update_plan with 'tasks', the ordered steps.",
            )
        step = network.current()
        if step is None:
            why = (
                "every step is already finished"
                if network.is_complete()
                else "every open step is blocked"
            )
            return ToolResult.error(
                f"there is no current step to advance: {why}",
                fix="Send the whole plan with 'tasks' to add, reopen or unblock a step.",
            )
        outcome = advance.get("outcome") if isinstance(advance, dict) else advance
        nodes: dict[str, dict[str, Any]] = {}

        def node(task: Task) -> dict[str, Any]:
            raw: dict[str, Any] = {
                "title": task.title,
                "status": task.status,
                "blocked_by": list(task.blocked_by),
            }
            if task.children:
                raw["subtasks"] = [node(child) for child in task.children]
            nodes[task.id] = raw
            return raw

        steps = [node(task) for task in network.tasks]
        nodes[step.id]["status"] = "done"
        if isinstance(outcome, str) and outcome.strip():
            nodes[step.id]["outcome"] = outcome.strip()
        # The next step is the one current once this one is done, dependencies and derived
        # parent statuses included: read it from the plan as it will stand, ids and all.
        after = TaskNetwork(tasks=[_build_task(raw, _MAX_DEPTH) for raw in steps])
        after.normalize()
        following = after.current()
        if following is not None and following.status == "pending" and following.id in nodes:
            nodes[following.id]["status"] = "in_progress"
        return steps, step

    @staticmethod
    def _autoadvance(network: TaskNetwork, step: Task, complete_hint: str) -> ToolResult:
        """Lever N (ADR-0168): mark a worked step done for the model and name the next (opt-in)."""
        advanced_id, advanced_title = step.id, step.title
        new_current = network.harness_advance(step)
        finished, total = network.progress()
        data = {
            "task_count": total,
            "finished": finished,
            "unchanged": True,
            "autoadvanced": True,
            "advanced_step": advanced_id,
            "complete": network.is_complete(),
        }
        lead = (
            f"Advanced step {advanced_id} ({clip(advanced_title, 60)!r}) to done for you — it was "
            "already worked on and you resent the plan unchanged."
        )
        if new_current is None:
            tail = " — complete." if network.is_complete() else "."
            output = f"{lead} Plan now {finished}/{total} steps done{tail}"
            hint = complete_hint if network.is_complete() else None
            return ToolResult.ok(output, data=data, hint=hint)
        output = (
            f"{lead} Plan now {finished}/{total} steps done · current: {new_current.id} "
            f"{clip(new_current.title, 80)}."
        )
        return ToolResult.ok(output, data=data, hint=_ADVANCED_HINT.format(id=new_current.id))

    @staticmethod
    def _finished_resend(built: list[Task]) -> ToolResult:
        """The receipt for a finished plan sent back after the board was cleared (ADR-0390)."""
        total = _leaf_count(built)
        return ToolResult.ok(
            f"Not reopened: this plan already finished ({total}/{total} steps done) and was "
            "closed, so nothing was updated. Send update_plan again only for new work; otherwise "
            "go on with your next action.",
            data={"task_count": 0, "unchanged": True, "reopened": False},
        )

    @staticmethod
    def _unchanged(
        network: TaskNetwork, finished: int, total: int, complete_hint: str
    ) -> ToolResult:
        """The receipt for a resend of the plan already in force (ADR-0168)."""
        data = {
            "task_count": total,
            "finished": finished,
            "unchanged": True,
            "complete": network.is_complete(),
        }
        current = network.current()
        if current is None:
            tail = " — complete." if network.is_complete() else "."
            output = f"Plan unchanged: {finished}/{total} steps done{tail} Nothing was updated."
            hint = complete_hint if network.is_complete() else None
            return ToolResult.ok(output, data=data, hint=hint)
        output = (
            f"Plan unchanged: {finished}/{total} steps done · current: {current.id} "
            f"{clip(current.title, 80)}. Nothing was updated — this is the plan already in force."
        )
        # No "you recorded an outcome but left the status" prefix here any more: since ADR-0202
        # made the advance unconditional, a current step carrying an outcome is ADVANCED rather
        # than answered with this rail. ``current()`` only ever returns a pending or in_progress
        # leaf, which is exactly the set the advance fires on, so that branch became unreachable.
        # What reaches this rail is a plan nobody has worked on yet — hence one hint, not two.
        return ToolResult.ok(output, data=data, hint=_UNCHANGED_HINT.format(id=current.id))

    @staticmethod
    def _hint(network: Any, complete_hint: str) -> str:
        if network.log and network.log[-1].kind == "challenged":
            return _CHALLENGED_HINT
        return complete_hint if network.is_complete() else _PLANNED_HINT


__all__ = ["UpdatePlanTool"]

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

#: Next-step rail: after planning, the model should act on the current step, not re-plan.
_PLANNED_HINT = (
    "Plan updated. Now do the step marked '<- current'; call update_plan again only to mark it "
    "done and move on, or to refine the plan as you learn more."
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
    "Do not resend the same plan. If step {id} is finished, resend the plan with its status "
    "'done' (and its result in 'outcome') and the next step 'in_progress'; if it is not, do it "
    "now with a tool call."
)

#: Advance rail (ADR-0168 lever N): the harness just marked a worked-on step done because the model
#: resent the plan unchanged and would not close it. Point at the step now current — do it, do not
#: resend the plan unchanged again.
_ADVANCED_HINT = (
    "Do step {id} now with a tool call. When a step is genuinely done, resend the plan with its "
    "status 'done' and its result in 'outcome'; do not resend the plan unchanged."
)


def _task_schema(depth: int) -> dict[str, Any]:
    """JSON schema for one task node, nesting ``subtasks`` to ``depth`` levels."""
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
                "done, blocked (cannot proceed, say why in note), or cancelled. Omit for pending."
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
                "Sub-steps this step decomposes into. A step WITH subtasks is a compound goal "
                "(its status is derived from them); a step WITHOUT subtasks is a primitive action."
            ),
            "items": {
                "type": "object",
                "properties": _task_schema(depth - 1)["properties"],
                "required": ["title"],
            },
        }
    return {"type": "object", "properties": properties, "required": ["title"]}


def _build_task(raw: dict[str, Any], depth: int) -> Task:
    """Build a :class:`Task` from a model-supplied node. ``kind`` is inferred from subtasks."""
    title = str(raw.get("title", "")).strip() or "(untitled)"
    raw_status = raw.get("status")
    status: TaskStatus = raw_status if raw_status in _STATUS_VALUES else "pending"
    note = str(raw.get("note", "")).strip()
    outcome = str(raw.get("outcome", "")).strip()
    raw_deps = raw.get("blocked_by")
    blocked_by = (
        [str(d) for d in raw_deps if isinstance(d, str | int)] if isinstance(raw_deps, list) else []
    )
    children_raw = raw.get("subtasks") if depth > 1 else None
    children: list[Task] = []
    if isinstance(children_raw, list):
        children = [_build_task(c, depth - 1) for c in children_raw if isinstance(c, dict)]
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
            "Maintain a hierarchical plan for a multi-step task. Call it FIRST on any task that "
            "needs three or more distinct actions, or that asks for several separate things: "
            "decompose the goal into ordered, primitive steps — each with a clear done-condition "
            "and no hidden 'figure out how' (break a step into 'subtasks' when it is itself "
            "several actions, and use 'blocked_by' when a step depends on earlier ones). Then "
            "call it again to mark a step done and the next one in_progress as you go. Always "
            "send the WHOLE plan each time: every step's title and status. A step keeps the note "
            "and outcome you already gave it, so send those only when they are new or changed. "
            "When you mark a step done, record what it produced in its 'outcome'. Skip it only "
            "for a request that asks one thing needing one or two actions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "The full, ordered plan — every step, with current statuses.",
                    "items": _task_schema(_MAX_DEPTH),
                }
            },
            "required": ["tasks"],
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
        if tasks is None:
            return ToolResult.error(
                "'tasks' must be an array of step objects ({title, status?, note?, subtasks?})",
                fix='Pass the full plan as an array, e.g. [{"title": "...", "status": "pending"}].',
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

        built = [_build_task(t, _MAX_DEPTH) for t in tasks if isinstance(t, dict)]
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
        events = network.log_folded + len(network.log)
        advisories = network.replace_from_author(built)
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
                return self._autoadvance(network, current)
            return self._unchanged(network, finished, total)
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
                # ADR-0209: this receipt acknowledges an edit that CHANGED the plan, and it
                # reads the same whenever the done count and the current step do. Measured
                # in a served loop: 14 of 47 stuck rungs, two of them STOPs, fell on it while
                # the model was journalling into its plan between real steps. The two
                # receipts for a plan sent back UNCHANGED (below) do not carry the flag:
                # they are the churn, and the stuck ladder must keep counting them.
                RECEIPT_OF_CHANGE: True,
            },
            hint=self._hint(network),
        )

    @staticmethod
    def _autoadvance(network: TaskNetwork, step: Task) -> ToolResult:
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
            hint = _COMPLETE_HINT if network.is_complete() else None
            return ToolResult.ok(output, data=data, hint=hint)
        output = (
            f"{lead} Plan now {finished}/{total} steps done · current: {new_current.id} "
            f"{clip(new_current.title, 80)}."
        )
        return ToolResult.ok(output, data=data, hint=_ADVANCED_HINT.format(id=new_current.id))

    @staticmethod
    def _unchanged(network: TaskNetwork, finished: int, total: int) -> ToolResult:
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
            hint = _COMPLETE_HINT if network.is_complete() else None
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
    def _hint(network: Any) -> str:
        if network.log and network.log[-1].kind == "challenged":
            return _CHALLENGED_HINT
        return _COMPLETE_HINT if network.is_complete() else _PLANNED_HINT


__all__ = ["UpdatePlanTool"]

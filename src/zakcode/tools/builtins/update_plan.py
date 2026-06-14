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
from zakcode.tasks import Task, TaskStatus
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec

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
            "description": "Optional one-line detail or acceptance criterion (e.g. 'tests pass').",
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
    children_raw = raw.get("subtasks") if depth > 1 else None
    children: list[Task] = []
    if isinstance(children_raw, list):
        children = [_build_task(c, depth - 1) for c in children_raw if isinstance(c, dict)]
    return Task(
        title=title,
        status=status,
        note=note,
        kind="compound" if children else "primitive",
        children=children,
    )


class UpdatePlanTool(Tool):
    """Lay out or update the hierarchical task plan for the current goal."""

    spec = ToolSpec(
        name="update_plan",
        description=(
            "Maintain a hierarchical plan for a multi-step task. Call it FIRST on any task that "
            "needs more than one action: decompose the goal into ordered, primitive steps (break "
            "a step into 'subtasks' when it is itself several actions). Then call it again to "
            "mark a step done and the next one in_progress as you go. Always send the WHOLE plan "
            "each time, with every step's status. Skip it for a single trivial action."
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
        tasks = args.get("tasks")
        if not isinstance(tasks, list):
            return ToolResult.error(
                "'tasks' must be an array of step objects ({title, status?, note?, subtasks?})",
                fix='Pass the full plan as an array, e.g. [{"title": "...", "status": "pending"}].',
            )
        if not tasks:
            # An empty plan clears the board (the goal is a single action, or work is abandoned).
            network.tasks = []
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
        network.tasks = built
        advisories = network.normalize()

        rendered = network.render()
        finished, total = network.progress()
        output = rendered
        if advisories:
            output += "\n\nNotes:\n" + "\n".join(f"- {a}" for a in advisories)
        return ToolResult.ok(
            output,
            data={
                "task_count": total,
                "finished": finished,
                "advisories": advisories,
                "complete": network.is_complete(),
            },
            hint=None if network.is_complete() else _PLANNED_HINT,
        )


__all__ = ["UpdatePlanTool"]

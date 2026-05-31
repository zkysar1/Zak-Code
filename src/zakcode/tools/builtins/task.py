"""The ``task`` tool — delegate self-contained subtasks to isolated sub-agents (M4).

When the model calls ``task``, each requested subtask is run by an isolated
sub-agent (fresh history, filtered tools, shared budget) and the **summaries** are
returned — not the children's raw transcripts — so the parent's context stays
compact. Independent subtasks in one call run **concurrently** (``asyncio.gather``).

The tool is a thin adapter over the :class:`~zakcode.tools.base.SubAgentSpawner`
placed on the :class:`~zakcode.tools.base.ToolContext` by the loop. With no spawner
(an ordinary turn, or inside a sub-agent — one-level nesting), it returns a
recoverable error rather than raising, so the model can adapt.

Delegation itself requires only :data:`READ_ONLY` permission: the *child's* tool
calls are gated by the same permission policy as the parent, so spawning cannot be
used to escape the gate.
"""

from __future__ import annotations

import asyncio
from typing import Any

from zakcode.agent.subagent import SubAgentResult
from zakcode.config import PermissionTier
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec


class TaskTool(Tool):
    """Run one or more subtasks as isolated, concurrent sub-agents."""

    spec = ToolSpec(
        name="task",
        description=(
            "Delegate one or more self-contained subtasks to isolated sub-agents that run "
            "concurrently and return short summaries (not full transcripts). Use it to "
            "parallelize independent research or work. Each subtask gets a fresh context and "
            "a filtered toolset; they share one iteration budget."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "The subtasks to run concurrently (one sub-agent each).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subagent_type": {
                                "type": "string",
                                "description": (
                                    "Which sub-agent type to use; omit for the default."
                                ),
                            },
                            "prompt": {
                                "type": "string",
                                "description": "The self-contained instruction for this subtask.",
                            },
                        },
                        "required": ["prompt"],
                    },
                }
            },
            "required": ["tasks"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        spawner = ctx.spawner
        if spawner is None:
            return ToolResult.error(
                "delegation is not available here (no sub-agent spawner on the context)"
            )

        tasks = args.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            return ToolResult.error(
                "'tasks' must be a non-empty array of {subagent_type?, prompt} objects"
            )

        available = spawner.available_types()
        default_type = available[0] if available else "general-purpose"

        prepared: list[tuple[str, str]] = []
        for i, task in enumerate(tasks):
            if not isinstance(task, dict) or "prompt" not in task:
                return ToolResult.error(f"task[{i}] must be an object containing a 'prompt'")
            type_name = task.get("subagent_type") or default_type
            if available and type_name not in available:
                return ToolResult.error(
                    f"task[{i}]: unknown subagent_type {type_name!r}; available: {available}"
                )
            prepared.append((type_name, str(task["prompt"])))

        # Run every subtask concurrently. return_exceptions=True so one child's
        # failure (e.g. the budget's child cap) is reported, not propagated as a
        # crash that would lose the other children's results.
        results = await asyncio.gather(
            *(spawner.spawn(type_name=tn, prompt=p) for tn, p in prepared),
            return_exceptions=True,
        )

        lines: list[str] = []
        errors = 0
        for i, (type_name, _prompt) in enumerate(prepared):
            res = results[i]
            if isinstance(res, BaseException):
                errors += 1
                lines.append(f"[task {i + 1}] {type_name}: ERROR: {type(res).__name__}: {res}")
            elif isinstance(res, SubAgentResult):
                note = "" if res.stop_reason == "completed" else f" (stopped: {res.stop_reason})"
                lines.append(f"[task {i + 1}] {res.name}{note}:\n{res.summary}")

        return ToolResult(
            output="\n\n".join(lines),
            is_error=errors == len(results),
            data={"count": len(results), "errors": errors},
        )


__all__ = ["TaskTool"]

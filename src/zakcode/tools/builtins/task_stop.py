"""The ``TaskStop`` tool — kill a background command started with
``Bash(run_in_background=true)`` (ADR-0191). Claude Code's shape: ``task_id``."""

from __future__ import annotations

from typing import Any

from zakcode.config import PermissionTier
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec


class TaskStopTool(Tool):
    """Stop a running background command (its whole process group)."""

    spec = ToolSpec(
        name="TaskStop",
        description=(
            "Stop a background command started with Bash(run_in_background=true): kills it "
            "and everything it spawned. Its output so far stays in its output file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The id Bash returned when it started the command.",
                },
            },
            "required": ["task_id"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        tasks = ctx.background_tasks
        if tasks is None:
            return ToolResult.error("background commands are not available here (no session).")
        task_id = str(args.get("task_id") or "").strip()
        if not task_id:
            return ToolResult.error("'task_id' is required.")
        stopped, reason = await tasks.stop(task_id)
        if not stopped:
            return ToolResult.error(reason, data={"task_id": task_id, "stopped": False})
        task = tasks.get(task_id)
        return ToolResult.ok(
            f"Background task {task_id} stopped."
            + (f" Output so far: {task.output_file}" if task is not None else ""),
            data={"task_id": task_id, "stopped": True},
        )

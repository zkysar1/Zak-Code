"""The ``TaskOutput`` tool — the output and status of a background command started with
``Bash(run_in_background=true)`` (ADR-0191). Claude Code's shape: ``task_id``, ``block``
(wait for it to exit first), ``timeout`` in milliseconds."""

from __future__ import annotations

from typing import Any

from zakcode.background import DEFAULT_BLOCK_MS, MAX_BLOCK_MS
from zakcode.config import PermissionTier
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec


class TaskOutputTool(Tool):
    """Read a background command's output, waiting for it to exit when asked."""

    spec = ToolSpec(
        name="TaskOutput",
        description=(
            "Retrieve the output of a background command started with "
            "Bash(run_in_background=true): its status, its exit code once it has exited, and "
            "the latest output (the last 64KB). block=true (the default) waits up to timeout "
            "milliseconds for it to exit first; block=false returns what is there now. You are "
            "notified when it exits anyway — do not poll it with ScheduleWakeup."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The id Bash returned when it started the command.",
                },
                "block": {
                    "type": "boolean",
                    "description": "Wait for the command to exit before returning (default true).",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"How long to wait, in milliseconds (default {DEFAULT_BLOCK_MS}, "
                        f"max {MAX_BLOCK_MS}); ignored when block=false."
                    ),
                },
            },
            "required": ["task_id"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        tasks = ctx.background_tasks
        if tasks is None:
            return ToolResult.error("background commands are not available here (no session).")
        task_id = str(args.get("task_id") or "").strip()
        task = tasks.get(task_id) if task_id else None
        if task is None:
            known = ", ".join(t.id for t in tasks.records()) or "none"
            return ToolResult.error(
                f"no background task with id {task_id!r} (known: {known}).",
                data={"task_id": task_id},
            )
        block = args.get("block")
        block = True if block is None else block is True
        timeout = args.get("timeout")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            timeout = DEFAULT_BLOCK_MS
        timeout = min(timeout, MAX_BLOCK_MS)
        if block:
            status, code = await tasks.wait(task, timeout / 1000.0)
        else:
            status, code = tasks.status(task)
        output = tasks.output(task)
        head = f"[task {task.id}] status: {status}"
        if code is not None:
            head += f", exit code: {code}"
        head += f"\ncommand: {task.command.strip().splitlines()[0] if task.command.strip() else ''}"
        head += f"\noutput file: {task.output_file}"
        body = output if output.strip() else "(no output yet)"
        text = f"{head}\n\n{body}"
        data = {
            "task_id": task.id,
            "status": status,
            "exit_code": code,
            "output_file": task.output_file,
        }
        if status == "completed" and code not in (0, None):
            return ToolResult.error(text, data=data)
        return ToolResult.ok(text, data=data)

"""Run a shell command within the workspace (DANGER_FULL_ACCESS)."""

from __future__ import annotations

import asyncio
import subprocess

from zakcode.config import PermissionTier
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)

# Default and hard-cap timeouts, in seconds.
_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 60
# Maximum number of characters of combined output to return.
_MAX_OUTPUT = 64 * 1024


class BashTool(Tool):
    """Execute an arbitrary shell command with the workspace as the cwd."""

    spec = ToolSpec(
        name="bash",
        description=(
            "Run a shell command with the workspace as the working directory. "
            "stdout and stderr are combined. Times out after 60 seconds. Returns a "
            "non-zero exit code as an error."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command line to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (capped at 60).",
                    "minimum": 1,
                    "maximum": _MAX_TIMEOUT,
                },
            },
            "required": ["command"],
        },
        required_permission=PermissionTier.DANGER_FULL_ACCESS,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Run ``command`` and return combined output plus the exit code."""
        command = args.get("command")
        if not isinstance(command, str) or not command:
            return ToolResult.error("'command' is required and must be a string.")

        timeout = args.get("timeout")
        if not isinstance(timeout, int) or timeout <= 0:
            timeout = _DEFAULT_TIMEOUT
        timeout = min(timeout, _MAX_TIMEOUT)

        try:
            completed = await asyncio.to_thread(
                self._run, command, str(ctx.workspace_root), timeout
            )
        except subprocess.TimeoutExpired:
            return ToolResult.error(
                f"Command timed out after {timeout}s: {command}",
                data={"command": command, "timed_out": True},
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to run command: {exc}", data={"command": command})

        output = completed.stdout or ""
        truncated = False
        if len(output) > _MAX_OUTPUT:
            output = output[:_MAX_OUTPUT] + "\n\n[... output truncated ...]"
            truncated = True

        exit_code = completed.returncode
        combined = output
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += f"[exit code: {exit_code}]"

        data = {
            "command": command,
            "exit_code": exit_code,
            "truncated": truncated,
        }
        if exit_code != 0:
            return ToolResult.error(combined, data=data)
        return ToolResult.ok(combined, data=data)

    @staticmethod
    def _run(command: str, cwd: str, timeout: int) -> subprocess.CompletedProcess[str]:
        """Synchronously run ``command`` via the platform shell, combining streams."""
        return subprocess.run(
            command,
            cwd=cwd,
            shell=True,  # noqa: S602 - this is the explicit shell tool
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            errors="replace",
        )

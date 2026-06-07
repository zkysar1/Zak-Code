"""Run a shell command within the workspace (DANGER_FULL_ACCESS)."""

from __future__ import annotations

import os

from zakcode.config import PermissionTier
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins._proc import CommandTimeout, run_capturing

# Default and hard-cap timeouts, in seconds.
_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 60
# Maximum number of characters of combined output to return.
_MAX_OUTPUT = 64 * 1024


def _windows_shell_fix(command: str, output: str) -> str | None:
    """A remedy hint for a likely Windows shell-quoting / command-not-found failure, else None.

    On Windows ``create_subprocess_shell`` runs commands through **cmd.exe**, where bash-isms
    (single-quote quoting, ``;`` chaining) do not parse — a model trained on bash hits this and,
    seeing only a cryptic error, tends to retry the identical command until the stuck guard
    halts it. Naming the real fix (the rails channel) breaks that loop. Conservative: only fires
    on Windows, on a strong signal, so it doesn't mislabel an ordinary command failure.
    """
    if os.name != "nt":
        return None
    low = output.lower()
    if "'" in command or "unterminated string literal" in low:
        return (
            "On Windows the bash tool runs under cmd.exe, where bash-style single-quote quoting "
            "(and ';' chaining) do not parse. Use the powershell tool, double-quote the code, "
            "or write a script file and run it."
        )
    if "is not recognized" in low:
        return (
            "cmd.exe did not find that command (the bash tool runs under cmd.exe on Windows). "
            "Check the name, or use the powershell tool."
        )
    return None


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
        if not isinstance(command, str) or not command.strip():
            return ToolResult.error("'command' is required and must be a non-empty string.")

        # ``bool`` is an ``int`` subclass; treat True/False as "no timeout given".
        timeout = args.get("timeout")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            timeout = _DEFAULT_TIMEOUT
        timeout = min(timeout, _MAX_TIMEOUT)

        # run_capturing spawns the child in its own process group so a timeout OR a turn
        # cancellation kills the whole tree (no orphaned grandchildren); CancelledError is
        # NOT caught here (it is BaseException) so a cancel propagates after teardown.
        try:
            output, exit_code = await run_capturing(
                shell_command=command,
                cwd=str(ctx.workspace_root),
                timeout=timeout,
            )
        except CommandTimeout:
            return ToolResult.error(
                f"Command timed out after {timeout}s: {command}",
                data={"command": command, "timed_out": True},
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to run command: {exc}", data={"command": command})

        truncated = False
        if len(output) > _MAX_OUTPUT:
            hidden = len(output) - _MAX_OUTPUT
            output = output[:_MAX_OUTPUT] + (
                f"\n\n[... output truncated at 64KB; {hidden} more chars hidden — "
                "narrow the command (grep/head/tail) to see the rest ...]"
            )
            truncated = True

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
            return ToolResult.error(combined, data=data, fix=_windows_shell_fix(command, output))
        return ToolResult.ok(combined, data=data)

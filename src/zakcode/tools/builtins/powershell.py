"""Run a PowerShell command within the workspace (DANGER_FULL_ACCESS).

A first-class sibling of :mod:`~zakcode.tools.builtins.bash` for Windows-centric
workflows. ``bash`` runs through the platform shell (``cmd.exe`` on Windows), so it
cannot run PowerShell cmdlets; this tool invokes PowerShell directly. It prefers
PowerShell 7+ (``pwsh``, cross-platform) and falls back to Windows PowerShell
(``powershell.exe``); on a host with neither it returns a recoverable error rather
than raising, so the agent can adapt (e.g. fall back to ``bash``).

Safety mirrors the bash tool: the same workspace cwd, the same output cap and
timeout, combined stdout+stderr, and — because its argument key is ``command`` — the
same deny-first permission gate and catastrophic-command blocklist
(:data:`~zakcode.permissions.DANGEROUS_PATTERNS`, which includes PowerShell idioms
like ``Remove-Item -Recurse -Force`` and ``Format-Volume``).
"""

from __future__ import annotations

import shutil

from zakcode.config import PermissionTier
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins._proc import CommandTimeout, run_capturing

# Default and hard-cap timeouts, in seconds (kept in step with the bash tool).
_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 60
# Maximum number of characters of combined output to return.
_MAX_OUTPUT = 64 * 1024


def _find_powershell() -> str | None:
    """Locate a PowerShell executable: prefer cross-platform ``pwsh``, then Windows."""
    for candidate in ("pwsh", "powershell"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


class PowerShellTool(Tool):
    """Execute a PowerShell command line with the workspace as the cwd."""

    spec = ToolSpec(
        name="powershell",
        description=(
            "Run a PowerShell command with the workspace as the working directory "
            "(uses pwsh / powershell.exe — prefer this over 'bash' on Windows for "
            "cmdlets and PowerShell syntax). stdout and stderr are combined. Times out "
            "after 60 seconds. Returns a non-zero exit code as an error."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The PowerShell command line to execute.",
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
        """Run ``command`` through PowerShell and return combined output + exit code."""
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult.error("'command' is required and must be a non-empty string.")

        exe = _find_powershell()
        if exe is None:
            return ToolResult.error(
                "PowerShell is not available on this host (neither 'pwsh' nor "
                "'powershell' is on PATH). Use the 'bash' tool instead.",
                data={"command": command, "powershell_missing": True},
            )

        # ``bool`` is an ``int`` subclass; treat True/False as "no timeout given".
        timeout = args.get("timeout")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            timeout = _DEFAULT_TIMEOUT
        timeout = min(timeout, _MAX_TIMEOUT)

        # -NoProfile for a clean/fast session, -NonInteractive so a cmdlet never blocks on a
        # prompt, and -Command - to read the script from stdin (avoids argv quoting pitfalls).
        # run_capturing spawns in its own process group so a timeout or turn cancellation
        # kills the whole tree; CancelledError propagates (not caught) after teardown.
        try:
            output, exit_code = await run_capturing(
                argv=[exe, "-NoProfile", "-NonInteractive", "-Command", "-"],
                cwd=str(ctx.workspace_root),
                timeout=timeout,
                stdin_text=command,
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
            output = output[:_MAX_OUTPUT] + "\n\n[... output truncated ...]"
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
            return ToolResult.error(combined, data=data)
        return ToolResult.ok(combined, data=data)

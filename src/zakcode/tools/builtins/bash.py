"""Run a shell command within the workspace (DANGER_FULL_ACCESS)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from zakcode._subprocess import find_bash
from zakcode.config import PermissionTier
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins._proc import CommandTimeout, run_capturing

# Default and hard-cap timeouts, in seconds. The default stays short (most commands are quick),
# but the cap is generous so a real build/test suite (often >60s) can finish with an explicit
# ``timeout`` instead of always failing -- the prior 60s hard cap surfaced as a false stall.
_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 600
# Maximum number of characters of combined output to return.
_MAX_OUTPUT = 64 * 1024


def _windows_shell_fix(command: str, output: str) -> str | None:
    """A remedy hint for a likely Windows shell-quoting / command-not-found failure, else None.

    Only relevant on the **cmd.exe fallback** — when no Git Bash is found, the bash tool runs
    commands through cmd.exe, where bash-isms (single-quote quoting, ``;`` chaining) do not parse;
    a bash-trained model hits this and tends to retry the identical command until the stuck guard
    halts it, so naming the real fix breaks that loop. When real Git Bash IS present the tool runs
    bash, so these hints don't apply. Conservative: Windows + strong signal only.
    """
    if os.name != "nt" or find_bash() is not None:
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


#: ``bash: line 1: name: command not found`` / dash ``sh: 1: name: not found``.
_NOT_FOUND_RE = re.compile(r"(?:line )?\d*:?\s*([^\s:]+): (?:command )?not found")
#: ``bash: line 1: ./x.sh: Permission denied``.
_PERM_DENIED_RE = re.compile(r"(?:line )?\d*:?\s*([^\s:]+): Permission denied")
#: Directories never worth descending into when locating a script by basename.
_SKIP_DIRS = frozenset({"node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache"})
#: Bounded search: depth below the workspace root, and total directories visited.
_FIND_MAX_DEPTH = 4
_FIND_MAX_DIRS = 800


def _locate_basename(root: Path, name: str) -> str | None:
    """Workspace-relative path of the first file named ``name``, else None (bounded walk).

    Hidden dirs and dependency trees are pruned, depth and visited-dir count are capped,
    so the search stays cheap even in a large repo — this only runs on a failed command.
    """
    if not name or "/" in name or "\\" in name:
        return None
    root = root.resolve()
    for visited, (dirpath, dirnames, filenames) in enumerate(os.walk(root), start=1):
        rel_depth = len(Path(dirpath).relative_to(root).parts)
        if visited > _FIND_MAX_DIRS or rel_depth >= _FIND_MAX_DEPTH:
            dirnames[:] = []
        else:
            dirnames[:] = sorted(
                d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS
            )
        if name in filenames:
            return (Path(dirpath) / name).relative_to(root).as_posix()
    return None


#: An inline-program Python invocation: ``python -c`` / ``python3 -c`` / ``py -3 -c``.
#: Intermediate tokens must look like options so ``python3 x.py && grep -c foo`` never matches.
_PY_INLINE_RE = re.compile(r"(?:^|[\s;&|(])(?:python[0-9.]*|py)(?:\s+-\S+)*\s+-c(?=\s)")


def _python_inline_fix(command: str, output: str) -> str | None:
    """A remedy hint when an inline ``python -c`` program failed to parse, else None.

    A multi-line program passed through ``-c`` gets mangled by shell quoting — most
    famously an apostrophe inside a single-quoted program (a comment like "we'll…")
    ends the quote and truncates the code, so Python reports a Syntax/IndentationError
    on a line that looks perfectly fine. Models then retry the identical command
    verbatim (measured 2026-08-26: three identical IndentationError retries, then a
    dead turn) — naming the real cause and the file-based escape breaks that loop.
    """
    if "SyntaxError" not in output and "IndentationError" not in output:
        return None
    if not _PY_INLINE_RE.search(command):
        return None
    return (
        "The inline -c program likely got mangled by shell quoting — an apostrophe "
        'inside a single-quoted program (e.g. a comment like "we\'ll") ends the quote '
        "and truncates the code, so the reported syntax error is not the real problem. "
        "Do not retry the same command: write the program to a file with the write_file "
        "tool and run `python3 <file>` instead."
    )


def _posix_exit_fix(command: str, output: str, exit_code: int, root: Path) -> str | None:
    """A remedy hint for the two classic script-invocation failures, else None.

    * exit 127 — a bare script name not on PATH: locate the basename in the workspace and
      name the working invocation (measured 2026-08-25: a mind agent burned an error +
      find + retry ritual per script, dozens of times, because ``x.sh`` lived at
      ``core/scripts/x.sh``).
    * exit 126 — the file exists but is not executable: name the chmod (or ``bash path``)
      escape, once, instead of letting the model rediscover it per file.
    """
    if exit_code == 127:
        m = _NOT_FOUND_RE.search(output)
        if m:
            found = _locate_basename(root, m.group(1))
            if found:
                return (
                    f"'{m.group(1)}' is not on PATH but exists in the workspace at {found} — "
                    f"run it as `bash {found}` (or add its directory to PATH for every future "
                    "command via a <workspace>/.zakcode/env line like "
                    f'`PATH="$PWD/{Path(found).parent.as_posix()}:$PATH"`).'
                )
        return None
    if exit_code == 126:
        m = _PERM_DENIED_RE.search(output)
        if m:
            return (
                f"{m.group(1)} exists but is not executable — run it as `bash {m.group(1)}`, "
                f"or fix the whole class once with `chmod +x` on the scripts directory "
                "instead of one file at a time."
            )
    return None


class BashTool(Tool):
    """Execute an arbitrary shell command with the workspace as the cwd."""

    spec = ToolSpec(
        name="bash",
        description=(
            "Run a shell command with the workspace as the working directory. "
            "stdout and stderr are combined. Default 60s timeout (max 600). Returns a "
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
                    "description": "Timeout in seconds (default 60, max 600).",
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
                extra_env=ctx.egress_env,
                drop_env=ctx.scrub_env,
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
            fix = (
                _posix_exit_fix(command, output, exit_code, Path(str(ctx.workspace_root)))
                or _python_inline_fix(command, output)
                or _windows_shell_fix(command, output)
            )
            return ToolResult.error(combined, data=data, fix=fix)
        return ToolResult.ok(combined, data=data)

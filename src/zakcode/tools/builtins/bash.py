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
from zakcode.tools.builtins._suggest import suggest

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
#: Directories never worth descending into when locating a file by basename: VCS,
#: dependency trees, virtualenvs and tool caches. Other dot-dirs ARE walked — a Mind
#: deployment keeps its domain data and scripts under a hidden `.mind-data/`, and
#: pruning every dot-dir left both the 127 hint and the ENOENT hint blind to the very
#: directory the model was guessing at (measured 2026-08-29, zc-03).
_SKIP_DIRS = frozenset(
    {
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".git",
        ".hg",
        ".svn",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        ".zakcode",
    }
)
#: Bounded search: depth below the workspace root, and total directories visited.
_FIND_MAX_DEPTH = 4
_FIND_MAX_DIRS = 800


def _locate_all(root: Path, name: str, limit: int = 3) -> list[str]:
    """Workspace-relative paths of up to ``limit`` files named ``name`` (bounded walk).

    VCS/dependency/cache trees are pruned, depth and visited-dir count are capped, so
    the search stays cheap even in a large repo — this only runs on a failed command.
    """
    if not name or "/" in name or "\\" in name:
        return []
    root = root.resolve()
    hits: list[str] = []
    for visited, (dirpath, dirnames, filenames) in enumerate(os.walk(root), start=1):
        rel_depth = len(Path(dirpath).relative_to(root).parts)
        if visited > _FIND_MAX_DIRS or rel_depth >= _FIND_MAX_DEPTH:
            dirnames[:] = []
        else:
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        if name in filenames:
            hits.append((Path(dirpath) / name).relative_to(root).as_posix())
            if len(hits) >= limit:
                break
    return hits


def _locate_basename(root: Path, name: str) -> str | None:
    """Workspace-relative path of the first file named ``name``, else None."""
    hits = _locate_all(root, name, limit=1)
    return hits[0] if hits else None


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


#: A script fed to the wrong interpreter (ADR-0093): ``python3 x.sh`` / ``py -3 x.sh`` (a shell
#: script parsed as Python) or ``bash x.py`` / ``sh x.py`` (Python run as shell). The script
#: must be the first non-option argument, as a bare path with the mismatched extension — an
#: inline ``-c "…"`` program (``-c`` is excluded from the options run) or a script passed
#: later (``python3 tool.py x.sh``) never matches.
_PY_ON_SHELL_RE = re.compile(
    r"(?:^|[\s;&|(])(?:python[0-9.]*|py)(?:\s+-[^c\s]\S*)*\s+([\w./\\-]+\.sh)(?=$|[\s;&|)])"
)
_SHELL_ON_PY_RE = re.compile(
    r"(?:^|[\s;&|(])(?:bash|sh|zsh|dash)(?:\s+-[^c\s]\S*)*\s+([\w./\\-]+\.py)(?=$|[\s;&|)])"
)
#: What the wrong interpreter says: Python's parse errors on a shell script; a shell's on a
#: Python file (``import`` is no command; ``def f():`` is a syntax error near ``(``).
_INTERPRETER_ERROR_RE = re.compile(
    r"SyntaxError|IndentationError|syntax error near unexpected token|import: command not found"
)

#: Ubuntu's apport installs a Python excepthook that itself crashes on an inline program (it
#: ``stat``s the "binary", which is ``-c``), so every traceback from ``python3 -c`` is followed
#: by the hook's own ~20-line traceback and, under "Original exception was:", a re-print of
#: the original. Measured 2026-08-29 (zc-03): 20 of the fleet's 61 tracebacks that day — and a
#: small model reads the hook's failure as a second, unrelated error (ADR-0096).
_APPORT_BLOCK_RE = re.compile(
    r"\nError in sys\.excepthook:\n(?P<hook>(?:.*\n)*?)Original exception was:\n"
    r"(?P<orig>Traceback \(most recent call last\):\n(?:[ \t].*\n)*.*\n?)?"
)


def _strip_apport_noise(output: str) -> str:
    """``output`` without apport's excepthook failure — and without the re-print of the
    original traceback that follows it, when the original already stands above."""

    def cut(found: re.Match[str]) -> str:
        if "apport_python_hook" not in found.group("hook"):
            return found.group(0)  # some other hook's failure: real output, kept
        orig = found.group("orig") or ""
        if orig.strip() and orig.strip() in output[: found.start()]:
            orig = ""
        return "\n" + orig

    return _APPORT_BLOCK_RE.sub(cut, output)


def _interpreter_mismatch_fix(command: str) -> str | None:
    """A remedy hint when a script was run through the wrong interpreter, else None.

    Python parsing a shell script reports a SyntaxError at the first ``case`` arm or
    ``fi`` — a traceback that reads like a broken script, not a wrong command. Measured
    2026-08-29 (coach reducer): ``python3 core/scripts/aspirations-update-goal.sh …`` four
    times verbatim, each a SyntaxError on the .sh's line 65, until the stuck guard limited
    the turn to read-only tools and the iteration's state update never ran. Naming the
    interpreter breaks that loop; nothing else in the output does.
    """
    py_on_shell = _PY_ON_SHELL_RE.search(command)
    if py_on_shell:
        path = py_on_shell.group(1)
        return (
            f"{path} is a shell script; Python parsed it as Python (that is the SyntaxError). "
            f"Run it as `bash {path} …` with the same arguments."
        )
    shell_on_py = _SHELL_ON_PY_RE.search(command)
    if shell_on_py:
        path = shell_on_py.group(1)
        return (
            f"{path} is a Python file; the shell ran it as shell. "
            f"Run it as `python3 {path} …` with the same arguments."
        )
    return None


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


#: The "No such file" shapes shell tools and Python print, each capturing the path the
#: command named. The earliest match in the output wins.
_ENOENT_RES = (
    # python3 x.py
    re.compile(r"can't open file '([^']+)': \[Errno 2\] No such file or directory"),
    # a Python program's own open()/read_text()
    re.compile(r"FileNotFoundError: \[Errno 2\] No such file or directory: '([^']+)'"),
    # ls
    re.compile(r"cannot access '([^']+)': No such file or directory"),
    # pytest
    re.compile(r"ERROR: file or directory not found: (\S+)"),
    # bash / cat / cd / head / source ... : `<tool>: <path>: No such file or directory`
    re.compile(
        r"(?m)^[\w.\-/]+: (?:line \d+: )?((?:[A-Za-z]:)?[^\s:'\"]+): No such file or directory"
    ),
)
#: Names that are never a file the model meant: stdin markers and apport's `-c` artefact.
_NOT_A_FILE = frozenset({"-", "-c", "<stdin>", "<string>"})


def _enoent_fix(output: str, root: Path, extra_roots: list[Path]) -> str | None:
    """Name where a missing file actually is, or its nearest names — else None.

    Measured 2026-08-29 (zc-03, eight Bodies): 15 of the day's 73 failed commands were
    ENOENT and every one was a guessed path — `world/scripts/reasoning-bank.py`,
    `core/scripts/wm-list.sh`, `core/scripts/aspirations-write.sh`, `world/forged-skills.yaml`
    for `.mind-data/world/forged-skills.yaml` — each followed by the model's own
    find -> retry ritual, or a second guess. The file tools already answer a not-found
    with the workspace's closest paths (ADR-0040); a shell command deserves the same
    answer. No lead, no hint: a genuinely absent file stays a plain error.
    """
    best: re.Match[str] | None = None
    for rx in _ENOENT_RES:
        m = rx.search(output)
        if m and (best is None or m.start() < best.start()):
            best = m
    if best is None:
        return None
    path = best.group(1).rstrip(".,;:")
    name = Path(path).name
    if not name or name in _NOT_A_FILE or name.startswith("<"):
        return None
    roots = [Path(root), *(Path(r) for r in extra_roots)]
    found: list[tuple[Path, str]] = []  # (root, workspace-relative hit)
    for r in roots:
        found.extend((r, rel) for rel in _locate_all(r, name))
    parent = _existing_parent(path, roots)
    # A hit that ENDS with the guessed path is a wrong-prefix guess (`world/x.yaml` for
    # `.mind-data/world/x.yaml`); a same-named file in an unrelated directory is only a
    # lead when the guessed directory does not exist at all. When the directory is real,
    # the file under another agent's dir is noise, not a lead.
    guess = Path(path).as_posix().lstrip("./")
    found.sort(key=lambda fr: not (fr[1] == guess or fr[1].endswith("/" + guess)))
    wrong_prefix = bool(found) and (
        parent is None or found[0][1] == guess or found[0][1].endswith("/" + guess)
    )
    if wrong_prefix:
        # Its neighbours with the same leading token are usually the family the model
        # wanted (`reasoning-bank.py` found beside `reasoning-bank-add.sh`).
        hits = [rel if r == roots[0] else (r / rel).as_posix() for r, rel in found]
        first_root, first_rel = found[0]
        kin = _prefix_siblings((first_root / first_rel).parent, name)
        kin_note = f" (similar names there: {', '.join(kin[:5])})" if kin else ""
        return (
            f"'{path}' does not exist from the workspace root, but a file named '{name}' "
            f"does: {', '.join(hits[:3])}{kin_note} — use that path (or `cd` there first) "
            "instead of guessing another."
        )
    if parent is not None:
        # The directory is real and the file is not: a typo'd name (its siblings share
        # the leading token: `wm-list.sh` beside `wm-read.sh`), or a deliberate check
        # of an optional file — which gets no hint, because there is no lead.
        kin = _prefix_siblings(parent, name)
        if kin:
            return (
                f"'{path}' does not exist; that directory holds "
                f"{', '.join(kin[:5])} — use one of those instead of inventing a name."
            )
        return None
    by_name, _ = suggest(path, roots[0], roots[1:], soft=True)
    if by_name:
        return (
            f"'{path}' does not exist and nothing in the workspace is named '{name}'; "
            f"closest names: {', '.join(by_name[:5])} — pick one of those or `ls` the "
            "directory before inventing a third."
        )
    return None


_TOKEN_SPLIT_RE = re.compile(r"[-_.]")


def _existing_parent(path: str, roots: list[Path]) -> Path | None:
    """The missing path's parent directory, if it exists (absolute, or under a root)."""
    p = Path(path)
    candidates = [p.parent] if p.is_absolute() else [r / p.parent for r in roots]
    for c in candidates:
        try:
            if c.is_dir():
                return c
        except OSError:
            continue
    return None


def _prefix_siblings(directory: Path, name: str) -> list[str]:
    """Files in ``directory`` sharing ``name``'s leading token (`wm` of `wm-list.sh`)."""
    token = _TOKEN_SPLIT_RE.split(name, 1)[0].lower()
    if len(token) < 2:
        return []
    try:
        entries = sorted(e.name for e in directory.iterdir() if e.is_file())
    except OSError:
        return []
    return [e for e in entries if e.lower().startswith(token) and e != name]


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

        output = _strip_apport_noise(output)  # before the budget: the noise must not spend it
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
                _interpreter_mismatch_fix(command)
                or _posix_exit_fix(command, output, exit_code, Path(str(ctx.workspace_root)))
                or _enoent_fix(output, Path(str(ctx.workspace_root)), ctx.extra_workspace_roots)
                or _python_inline_fix(command, output)
                or _windows_shell_fix(command, output)
            )
            return ToolResult.error(combined, data=data, fix=fix)
        # Exit 0 can be a trailing pipe's (`python3 x.sh … | tail -40`): the interpreter
        # choked and `tail` reported success (ADR-0093, measured on the reducer the same day
        # the hint shipped — the pipe hid the failure the hint was written for). The
        # mismatched command plus the interpreter's own error text is the signal; the
        # result is the failure it was.
        mismatch = _interpreter_mismatch_fix(command)
        if mismatch and _INTERPRETER_ERROR_RE.search(output):
            return ToolResult.error(
                combined,
                data=data,
                fix=f"{mismatch} (The exit code 0 is the pipe's, not the script's.)",
            )
        return ToolResult.ok(combined, data=data)

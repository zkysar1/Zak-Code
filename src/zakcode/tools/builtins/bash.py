"""Run a shell command within the workspace (DANGER_FULL_ACCESS)."""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import Any

from zakcode._subprocess import find_bash
from zakcode.background import BackgroundTask, BackgroundTasks
from zakcode.config import PermissionTier
from zakcode.tools.base import (
    STDOUT_CHARS,
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins._proc import CommandTimeout, run_capturing
from zakcode.tools.builtins._suggest import suggest

# Default and hard-cap timeouts, in seconds. The cap is generous so a real build/test suite
# can finish with an explicit ``timeout`` instead of always failing -- the prior 60s hard cap
# surfaced as a false stall. The default is Claude Code's: two minutes (its schema counts
# milliseconds, default 120000, max 600000). It was 60s here, and a Mind's own scripts can
# outlast that on a slow box: measured 2026-09-23, an alpha worker's goal-selector run on
# zc-02 was killed at 60s and the model spent a call retrying it with a longer timeout.
_DEFAULT_TIMEOUT = 120
_MAX_TIMEOUT = 600
# How much of a command's output reaches the model: Claude Code's limits, from its public tools
# reference (ADR-0234). A command that succeeded is shown whole up to _INLINE_CHARS; past that,
# the whole output is saved to a file in the session's output directory and the result is its
# first _PREVIEW_CHARS plus the file's path. A command that failed is shown whole up to
# _FAILURE_CHARS; past that, its start and end, and the path too. It was 64 KB here, cut at the
# head for success and failure alike: measured 2026-09-23 on two worker Bodies over 16 hours,
# five outputs hit that cap (a goal selector's JSON, an aspirations dump, a cat), each put about
# 22k tokens into a 131k window, and four of the ten compactions kept mostly such outputs.
_INLINE_CHARS = 30_000
_PREVIEW_CHARS = 2_000
_FAILURE_CHARS = 10_000


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
            "On Windows the Bash tool runs under cmd.exe, where bash-style single-quote quoting "
            "(and ';' chaining) do not parse. Use the powershell tool, double-quote the code, "
            "or write a script file and run it."
        )
    if "is not recognized" in low:
        return (
            "cmd.exe did not find that command (the Bash tool runs under cmd.exe on Windows). "
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


def _locate_all(root: Path, name: str, limit: int = 3, *, dirs: bool = False) -> list[str]:
    """Workspace-relative paths of up to ``limit`` files named ``name`` (bounded walk).

    VCS/dependency/cache trees are pruned, depth and visited-dir count are capped, so
    the search stays cheap even in a large repo — this only runs on a failed command.
    With ``dirs=True`` it matches directories instead (never a pruned one).
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
        if name in (dirnames if dirs else filenames):
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
        "Do not retry the same command: write the program to a file with the Write "
        "tool and run `python3 <file>` instead."
    )


#: Script output piped into a Python program: ``… | python3 -c '…'`` / ``… | python3 -``.
_PIPE_INTO_PY_RE = re.compile(r"\|\s*(?:python[0-9.]*|py)(?=\s|$)")
#: ``json.loads`` on a first line that is a lone ``[`` or ``{``: the upstream printed a
#: pretty-printed document (one record over many lines), and the program read one line.
#: ``char 1`` when the line was stripped, ``line 2 … char 2`` when its newline came along.
_JSON_FIRST_LINE_RE = re.compile(
    r"JSONDecodeError: Expecting value: line (?:1 column 2 \(char 1\)|2 column 1 \(char 2\))"
)


_MODULE_NOT_FOUND_RE = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")


def _importable_dir(path: Path) -> bool:
    """A directory Python can import as a package: has ``__init__.py`` or any module."""
    try:
        return (path / "__init__.py").is_file() or any(
            p.suffix == ".py" for p in path.iterdir() if p.is_file()
        )
    except OSError:
        return False


def _module_not_found_fix(
    command: str, output: str, root: Path, extra_roots: list[Path]
) -> str | None:
    """Name where a package of that name lives in the workspace, else None.

    Measured 2026-08-30 (zc-03, coach Bodies): five ``ModuleNotFoundError: No module
    named 'yahoo'`` in 24 h across four sessions — every one ``cd <workspace> && python3
    …`` after the package had been consolidated under ``.mind-data/world/scripts/yahoo``
    — and one identical retry, because the error names the module and nothing names the
    directory Python would have had to be run from. A dotted name whose top package IS
    found but whose submodule is not gets the package's real module names instead. A
    name found nowhere in the workspace is an invented one when the command itself
    declared a root — a literal ``sys.path.insert``, a ``PYTHONPATH=``, a ``cd`` — and
    the closest real names under that root are then the lead (g-353-80); with no
    declared root it stays a plain error: install guesses are not this hint's business.
    """
    m = _MODULE_NOT_FOUND_RE.search(output)
    if m is None:
        return None
    parts = m.group(1).split(".")
    top = parts[0]
    roots = [Path(root), *(Path(r) for r in extra_roots)]
    hits: list[tuple[Path, str]] = []  # (root, workspace-relative package dir or module file)
    for r in roots:
        hits.extend((r, rel) for rel in _locate_all(r, top, dirs=True) if _importable_dir(r / rel))
        hits.extend((r, rel) for rel in _locate_all(r, f"{top}.py"))
    if not hits:
        return _nearest_module_fix(command, top, roots[0])
    first_root, first_rel = hits[0]
    if len(parts) > 1 and (first_root / first_rel).is_dir():
        pkg = first_root / first_rel
        sub = parts[1]
        if not ((pkg / f"{sub}.py").is_file() or (pkg / sub).is_dir()):
            modules = sorted(
                p.stem for p in pkg.iterdir() if p.suffix == ".py" and p.stem != "__init__"
            )
            shown = ", ".join(modules[:8]) or "no modules"
            return (
                f"Package '{top}' is at {first_rel} but has no module '{sub}' — it holds: "
                f"{shown}. Import one of those; do not invent a module name."
            )
    shown_hits = [rel if r == roots[0] else (r / rel).as_posix() for r, rel in hits[:3]]
    parent = Path(shown_hits[0]).parent.as_posix()
    run_from = (
        f"`cd {parent} && python3 …`"
        if parent not in ("", ".")
        else (f"the workspace root (`cd {roots[0]}`)")
    )
    where = f"PYTHONPATH={parent}" if parent not in ("", ".") else f"PYTHONPATH={roots[0]}"
    return (
        f"No module named '{top}' on sys.path from this cwd, but the workspace has it: "
        f"{', '.join(shown_hits)}. Python imports it from its parent directory — run from "
        f"there ({run_from}) or prefix `{where}`; do not move or copy the package."
    )


#: A sys.path root the command itself declared, as a string literal — ``sys.path.insert(0,
#: "x")``, ``sys.path.append('x')``, quotes escaped or not — or a ``PYTHONPATH=x[:y]``
#: prefix. A computed root (``str(Path(__file__).parent)``, ``os.path.join(…)``) cannot be
#: read off the command.
_SYS_PATH_LITERAL_RE = re.compile(
    r"""sys\.path\.(?:insert\s*\(\s*\d+\s*,|append\s*\()\s*\\?(['"])([^'"]+?)\\?\1\s*\)"""
)
_PYTHONPATH_RE = re.compile(r"(?:^|[\s;&|(])PYTHONPATH=([^\s;&|]+)")
#: Names under one root are bounded so the closest-name pass stays cheap.
_NEAREST_MAX_NAMES = 1500


def _split_path_list(value: str) -> list[str]:
    """``PYTHONPATH=a:b`` entries. The command is a shell command on every platform, so
    ``:`` separates (``;`` too) — except, where drives exist, the colon of a drive letter
    (``C:\\x``, ``D:/x``): one letter into its entry and followed by a slash. Not
    ``os.pathsep``: that is ``;`` on Windows, where it read ``core/scripts:$PYTHONPATH`` as
    one entry (CI, 2026-09-17)."""
    parts: list[str] = []
    cur = ""
    for i, ch in enumerate(value):
        drive = (
            os.name == "nt"
            and len(cur) == 1
            and cur.isalpha()
            and value[i + 1 : i + 2] in ("/", "\\")
        )
        if ch == ";" or (ch == ":" and not drive):
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [part for part in parts if part]


def _command_roots(command: str, root: Path) -> list[tuple[Path, str]]:
    """Directories the command put on sys.path, each with why, in the order it named them.

    Literal ``sys.path.insert``/``append`` roots and ``PYTHONPATH`` entries come first —
    the model named them, so they are where it believed the module lived — then the
    directory a ``cd`` prefix moved into (``sys.path[0]`` for a ``python3 -c`` program).
    Relative roots resolve against the cwd the ``cd``s produce; an unexpandable one
    (``$VAR``, ``~``) is skipped rather than guessed.
    """
    cwd = _cwd_before(command, root)
    found: list[tuple[Path, str]] = []

    def add(raw: str, why: str) -> None:
        raw = raw.strip("\"'")
        if not raw or raw.startswith(("$", "~")):
            return
        p = Path(raw)
        if not p.is_absolute():
            if cwd is None:
                return
            p = cwd / p
        if all(p != q for q, _ in found):
            found.append((p, why))

    for m in _SYS_PATH_LITERAL_RE.finditer(command):
        add(m.group(2), "the sys.path root this command added")
    for m in _PYTHONPATH_RE.finditer(command):
        for entry in _split_path_list(m.group(1).strip("\"'")):
            add(entry, "the PYTHONPATH this command set")
    if cwd is not None and _CD_RE.search(command):
        add(str(cwd), "the directory this command cd'd into — sys.path[0] for `python3 -c`")
    return found


def _importable_names(root: Path) -> tuple[list[str], list[str]]:
    """What ``import <name>`` could resolve to directly under ``root`` — module stems and
    package directories — and, separately, every other file by its full name, so a shell
    script mistaken for a module surfaces as ``pipeline-read.sh`` rather than as nothing."""
    modules: list[str] = []
    others: list[str] = []
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError:
        return modules, others
    for e in entries[:_NEAREST_MAX_NAMES]:
        if e.name in _SKIP_DIRS or e.name == "__init__.py":
            continue
        if e.is_dir():
            if _importable_dir(Path(e.path)):
                modules.append(e.name)
        elif e.name.endswith(".py"):
            modules.append(e.name[:-3])
        else:
            others.append(e.name)
    return modules, others


def _shown_root(p: Path, root: Path) -> str:
    """``p`` workspace-relative when it is inside the workspace, else as given."""
    try:
        rel = p.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return p.as_posix()
    return "the workspace root" if rel == "." else rel


def _nearest_module_fix(command: str, top: str, root: Path) -> str | None:
    """The lead for a module that exists nowhere in the workspace, else None.

    Measured 2026-08-30 (zc-03, coach-w7): ``python3 -c`` with ``sys.path.insert(0,
    "core/scripts")`` then ``from pipeline_read import …`` — a module name invented the
    way script paths are invented (ADR-0106 refuses those before running), but an import
    inside ``-c`` is not a path a preflight can stat. The error carries the missing NAME
    and the command carries the root the model believed it lived under, so the same
    sibling-lead shape applies: the closest real names under that root. Nothing fires
    without a declared root — a plain ``python3 -c "import x"`` missing a third-party
    package is an install question, not this hint's.
    """
    roots = _command_roots(command, root)
    if not roots:
        return None
    lead = f"No module named '{top}' exists anywhere in the workspace — the import never resolved. "
    existing = [(p, why) for p, why in roots if p.is_dir()]
    if not existing:
        p, why = roots[0]
        return lead + (
            f"`{_shown_root(p, root)}` ({why}) does not exist, so nothing under it could be "
            "imported: `ls` its parent before guessing another path."
        )
    for p, why in existing:
        modules, others = _importable_names(p)
        close = difflib.get_close_matches(top, [*modules, *others], n=4, cutoff=0.6)
        if close:
            note = (
                " — a name with an extension is a file to run or read, not a module"
                if any("." in c for c in close)
                else ""
            )
            return lead + (
                f"Under {_shown_root(p, root)} ({why}) the closest names are: "
                f"{', '.join(close)}{note}. Import one that exists; do not invent a module name."
            )
    p, why = existing[0]
    modules, _ = _importable_names(p)
    modules.sort(key=lambda n: (n.startswith("_"), n))  # public names first
    shown = ", ".join(modules[:8]) or "no Python modules at all"
    more = f", … ({len(modules)} in all)" if len(modules) > 8 else ""
    return lead + (
        f"Nothing under {_shown_root(p, root)} ({why}) is close to '{top}'; it holds: "
        f"{shown}{more}. `ls` it and import a name that exists; do not invent one."
    )


def _json_first_line_fix(command: str, output: str) -> str | None:
    """A remedy hint when a piped-in Python parser choked on the first line of a
    pretty-printed JSON document, else None.

    Measured 2026-08-30 (zc-03, two sessions): ``aspirations-query.sh … | python3 -c`` and
    ``goal-selector.sh … | python3 -c`` both died with ``Expecting value: line 1 column 2
    (char 1)`` — the signature of ``json.loads("[")``: the wrapper prints an indented
    document, the program parsed it line by line as JSONL. The error names a column, not
    the cause, so the model reads it as broken output and re-runs the wrapper.
    """
    if not _JSON_FIRST_LINE_RE.search(output):
        return None
    if not _PIPE_INTO_PY_RE.search(command):
        return None
    return (
        "The parser choked on the FIRST LINE of the piped-in output, which was a lone `[` "
        "or `{`: the upstream prints a pretty-printed JSON document (one record spread over "
        "many lines), not JSONL. Parse the whole stream — `json.load(sys.stdin)` — instead "
        "of `json.loads` per line or on `readline()`; the result may be a list, so index or "
        "iterate it. The upstream output is fine; do not re-run it."
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


def _fit_output(
    output: str, *, failed: bool, tasks: BackgroundTasks | None
) -> tuple[str, Path | None]:
    """``output`` as the model sees it, and the file holding all of it when it was too long
    to show whole (ADR-0234).

    Within the limit (_INLINE_CHARS for a success, _FAILURE_CHARS for a failure) it comes back
    unchanged. Past it the whole output is saved to the session's output directory, and:

    * a success shows its first _PREVIEW_CHARS, cut back to a line end, and the path — Claude
      Code's shape. The start is where a listing, a report or a JSON document puts what
      matters; the rest is one Read or grep away.
    * a failure shows its start and end, 2/3 and 1/3 of _FAILURE_CHARS (the loop's seam-clamp
      shape): the first error at one end, the verdict at the other, and the path.

    With no session to save into (a bare tool context), or when the save fails, a long success
    is shown the way a failure is, start and end, at _INLINE_CHARS: there is no path to give.
    """
    limit = _FAILURE_CHARS if failed else _INLINE_CHARS
    if len(output) <= limit:
        return output, None
    saved: Path | None = None
    if tasks is not None:
        try:
            saved = tasks.save_output(output)
        except OSError:
            saved = None
    lines = output.count("\n") + (0 if output.endswith("\n") else 1)
    size = f"{len(output):,} characters in {lines:,} lines"
    where = (
        # Named by tool, not by verb: "search it with grep in the shell" sent a lesser model
        # to the Grep tool, which ADR-0234 keeps out of this directory (Ayoai-Mind g-375-12).
        # Read opens the file, grep reaches it through the Bash tool, the Grep tool does not.
        f"The whole output is in {saved}: use the Read tool on it (offset/limit), or run grep "
        "on it through the Bash tool. The Grep tool cannot open this directory."
        if saved is not None
        else "Run the command again, narrower (grep, head, tail), for the rest."
    )
    if failed or saved is None:
        head = limit * 2 // 3
        tail = limit - head
        note = (
            f"\n\n[... the output is {size}; its first {head:,} and last {tail:,} "
            f"characters are shown. {where} ...]\n\n"
        )
        return output[:head] + note + output[-tail:], saved
    preview = output[:_PREVIEW_CHARS]
    line_end = preview.rfind("\n")
    if line_end >= _PREVIEW_CHARS // 2:
        preview = preview[: line_end + 1]
    note = (
        f"[... the output is {size}; its first {len(preview):,} characters are shown. {where} ...]"
    )
    return preview + ("" if preview.endswith("\n") else "\n") + note, saved


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
    # coreutils: `ls: cannot access 'x'`, `touch: cannot touch 'x'`, `mkdir: cannot create
    # directory 'x'`, `stat: cannot statx 'x'`, `rm: cannot remove 'x'`, `cp: cannot stat 'x'`
    re.compile(r"cannot [a-z]+(?: [a-z]+)* '([^']+)': No such file or directory"),
    # pytest
    re.compile(r"ERROR: file or directory not found: (\S+)"),
    # bash / cat / cd / head / source ... : `<tool>: <path>: No such file or directory`
    re.compile(
        r"(?m)^[\w.\-/]+: (?:line \d+: )?((?:[A-Za-z]:)?[^\s:'\"]+): No such file or directory"
    ),
    # the same frame with the path QUOTED — newer coreutils (Git Bash on Windows CI, 2026-08-29:
    # `cat: 'C:/Users/.../forged-skills.yaml': No such file or directory`)
    re.compile(r"(?m)^[\w.\-/]+: '([^']+)': No such file or directory"),
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
    guess = _guess_relative(path, roots)
    found.sort(key=lambda fr: not (fr[1] == guess or fr[1].endswith("/" + guess)))
    suffix_hit = bool(found) and (found[0][1] == guess or found[0][1].endswith("/" + guess))
    if suffix_hit:
        return _wrong_prefix_hint(path, name, found, roots)
    if parent is not None:
        # The directory is real and the file is not: a typo'd name (its siblings share
        # the leading token: `wm-list.sh` beside `wm-read.sh`), a directory guessed at
        # the wrong place (`ls world/` for `.mind-data/world`), or a deliberate check of
        # an optional file — which gets no hint, because there is no lead.
        same_words = _reordered_siblings(parent, name)
        kin = [k for k in _prefix_siblings(parent, name) if k not in same_words]
        if same_words:
            # The words are right and the order is not: name the real file FIRST, as a
            # path the model can paste, and keep the leading-token family as an aside.
            rel_parent = _guess_relative(str(parent), roots)
            paths = [f"{rel_parent}/{s}" if rel_parent not in ("", ".") else s for s in same_words]
            kin_note = f" (other names there: {', '.join(kin[:4])})" if kin else ""
            return (
                f"'{path}' does not exist, but the same words in another order do: "
                f"{', '.join(paths[:3])} — use that exactly as written{kin_note}."
            )
        if kin:
            return (
                f"'{path}' does not exist; that directory holds "
                f"{', '.join(kin[:5])} — use one of those instead of inventing a name."
            )
        dir_hits = [h for r in roots for h in _locate_all(r, name, dirs=True)]
        if dir_hits:
            return (
                f"'{path}' does not exist, but a directory named '{name}' does: "
                f"{', '.join(dir_hits[:3])} — use that path instead of guessing another."
            )
        return None
    # The guessed DIRECTORY does not exist. Name the first component that is missing and
    # where a directory of that name really is: `.mind-data/agents/coach/sessions/<sid>/x`
    # fails at `.mind-data/agents`, and `agents/` lives at the workspace root (measured
    # 2026-08-29, zc-03: touch/grep/ls on that invented prefix, four times in one hour,
    # none of them a name the file search could lead on — the file was about to be created).
    # A same-named FILE elsewhere is the more specific lead and keeps the first word; the
    # prefix diagnosis rides beside it, and stands alone only when there is no file lead.
    prefix_note = ""
    missing = _first_missing_component(path, roots)
    if missing is not None:
        anchor, part = missing
        dir_hits = [h for r in roots for h in _locate_all(r, part, dirs=True)]
        if dir_hits:
            anchor_rel = _guess_relative(str(anchor), roots)
            missing_txt = f"{anchor_rel}/{part}" if anchor_rel not in ("", ".") else part
            prefix_note = (
                f"'{missing_txt}' is the first missing part of that path, but a directory "
                f"named '{part}' does exist: {', '.join(dir_hits[:3])} — rebuild the path "
                "from there instead of guessing another prefix"
            )
    hint: str | None = None
    if found:
        hint = _wrong_prefix_hint(path, name, found, roots)
    else:
        by_name, _ = suggest(path, roots[0], roots[1:], soft=True)
        if by_name:
            hint = (
                f"'{path}' does not exist and nothing in the workspace is named '{name}'; "
                f"closest names: {', '.join(by_name[:5])} — pick one of those or `ls` the "
                "directory before inventing a third."
            )
    if hint:
        return f"{hint} ({prefix_note}.)" if prefix_note else hint
    if prefix_note:
        return f"'{path}' does not exist: {prefix_note}."
    return None


_TOKEN_SPLIT_RE = re.compile(r"[-_.]")


def _wrong_prefix_hint(
    path: str, name: str, found: list[tuple[Path, str]], roots: list[Path]
) -> str:
    """The hint for a same-named file found elsewhere — with its neighbours sharing the
    leading token, which are usually the family the model wanted (`reasoning-bank.py`
    found beside `reasoning-bank-add.sh`)."""
    hits = [rel if r == roots[0] else (r / rel).as_posix() for r, rel in found]
    first_root, first_rel = found[0]
    kin = _prefix_siblings((first_root / first_rel).parent, name)
    kin_note = f" (similar names there: {', '.join(kin[:5])})" if kin else ""
    lead = (
        f"'{path}' does not exist from the workspace root, but a file named '{name}' "
        f"does: {', '.join(hits[:3])}{kin_note} — "
    )
    guess = _guess_relative(path, roots)
    if first_root == roots[0] and first_rel.endswith("/" + guess):
        # The guess is the real path minus its leading directory. Say exactly that: the
        # generic "(or `cd` there first)" read as a cwd problem — measured 2026-08-30
        # (zc-03): a Body answered this hint with `cd <workspace root> && <same command>`,
        # was refused again, and only then used the path the hint had already named.
        prefix = first_rel[: -len(guess) - 1]
        return lead + (
            f"that is the same path missing its leading '{prefix}/' — use "
            f"'{first_rel}' exactly as written; the cwd is already the workspace root, "
            "so a `cd` will not help."
        )
    return lead + "use that path (or `cd` there first) instead of guessing another."


def _guess_relative(path: str, roots: list[Path]) -> str:
    """The guessed path as a root-relative POSIX string when it lies under a root, else as
    written minus any leading `./` — the form the suffix match compares against hits.
    Bodies guess ABSOLUTE paths as often as relative ones (measured 2026-08-29), and an
    absolute `<root>/world/x.yaml` must match the hit `.mind-data/world/x.yaml` too."""
    p = Path(path)
    if p.is_absolute():
        for r in roots:
            try:
                return p.resolve().relative_to(Path(r).resolve()).as_posix()
            except (ValueError, OSError):
                continue
        return p.as_posix()
    return p.as_posix().lstrip("./")


def _first_missing_component(path: str, roots: list[Path]) -> tuple[Path, str] | None:
    """(deepest existing ancestor, first missing component) of a guessed path, or None
    when only its last component is missing — that is the typo / optional-file case,
    which the sibling branch answers. Relative paths are read against the first root."""
    p = Path(path)
    if p.is_absolute():
        base, parts = Path(p.anchor), p.parts[1:]
    else:
        base, parts = Path(roots[0]), tuple(x for x in p.parts if x not in (".", ""))
    cur = base
    for i, part in enumerate(parts):
        nxt = cur / part
        try:
            exists = nxt.exists()
        except OSError:
            return None
        if not exists:
            return None if i == len(parts) - 1 else (cur, part)
        cur = nxt
    return None


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


def _reordered_siblings(directory: Path, name: str) -> list[str]:
    """Files in ``directory`` made of exactly ``name``'s words in another order
    (`create-blocker.sh` for a guessed `blocker-create.sh`).

    Measured 2026-08-30 (zc-03): a Body guessed `core/scripts/blocker-create.sh`; the
    leading-token family offered `blocker-create-gate.sh`, `blocker-recheck.sh` — none of
    them the script — and the Body spent six more commands (`ls`, three `grep -rl`, two
    reads) finding `create-blocker.sh` on its own. Same multiset of tokens, extension
    included, is a stronger lead than a shared first word and is listed first.
    """
    want = sorted(t for t in _TOKEN_SPLIT_RE.split(name.lower()) if t)
    if len(want) < 2:
        return []
    try:
        entries = sorted(e.name for e in directory.iterdir() if e.is_file())
    except OSError:
        return []
    return [
        e
        for e in entries
        if e != name and sorted(t for t in _TOKEN_SPLIT_RE.split(e.lower()) if t) == want
    ]


#: ``Name(arg…`` at the very start of a command: a tool CALL written as shell. A shell
#: function definition (``name() {``) has nothing between the parens and does not match.
_CALL_SYNTAX_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*[^)\s]")


def _tool_typed_as_command(command: str, registry: Any) -> str | None:
    """The refusal for a registered tool written as a shell call, else None.

    Measured 2026-08-29 (zc-03, eight Bodies on a 27B local model): the loop's skills
    show the deadman net as ``ScheduleWakeup(prompt=…, delaySeconds=600)``, and the
    Bodies typed exactly that into the bash tool — five times in one session, each a
    shell syntax error and a lost turn, followed by "that needs to be a direct tool
    call" and then no call at all: zero ``schedule_wakeup`` invocations fleet-wide, so
    the net was never armed. The registry knows the name (and the Claude-Code-shaped
    alias); the bash tool can say so BEFORE running anything, with the tool's real
    name and its parameters, instead of letting bash report a syntax error.
    """
    m = _CALL_SYNTAX_RE.match(command)
    if m is None or registry is None:
        return None
    typed = m.group(1)
    try:
        tool = registry.get(typed)
        if tool is None or not registry.is_active(tool.name):
            return None
        params = list((getattr(tool.spec, "parameters", None) or {}).get("properties", {}))
    except Exception:  # noqa: BLE001 - a hint on the error path never raises
        return None
    shape = "{" + ", ".join(params) + "}" if params else "its arguments"
    return (
        f"`{typed}(…)` is the `{tool.spec.name}` TOOL written as a shell command; bash "
        f"cannot run it and nothing was run. Call the `{tool.spec.name}` tool directly "
        f"with {shape}."
    )


#: An interpreter (or ``source``/``.``) followed by a RELATIVE script path — at least one
#: slash, a script extension, no `$`/quote (an unexpandable path is not checked). This is
#: the shape a Body types when it names a Mind script from memory: `bash core/scripts/x.sh`.
#:
#: Leading ``VAR=value`` assignments are stepped over. They are not decoration on this
#: fleet: measured 2026-08-30 (zc-03, eight Bodies, 24 h) 165 of 454 script invocations
#: were ``cd … && MIND_AGENT=coach AYOAI_AGENT=coach STORAGE_BACKEND=local bash
#: core/scripts/x.sh`` — 36 %, invisible to the start-of-command anchor — and five of them
#: named a script that does not exist (``loop-orchestrator-entry-battery.sh``,
#: ``runner-heartbeat-tick.sh``, ``goal-scorer.sh``, ``wm-list.sh``, ``parse-flags.sh``),
#: each reaching bash as a 127 the model then spent a ~7-minute step on.
_SCRIPT_INVOCATION_RE = re.compile(
    r"(?:^|[;&|(]\s*|\bthen\s+|\bdo\s+)\s*(?:\w+=[^\s;&|]*\s+)*(bash|sh|python3?|source|\.)\s+"
    r"((?:[\w.\-]+/)+[\w.\-]+\.(?:sh|bash|py))(?=\s|$|[;&|)])"
)
#: ``cd <target>`` — the only cwd change this preflight follows (an absolute or
#: workspace-relative literal; a `$VAR`, `~` or `-` target means "cannot tell": fail open).
_CD_RE = re.compile(r"(?:^|[;&|(]\s*)cd\s+([^\s;&|)]+)")


def _cwd_before(prefix: str, root: Path) -> Path | None:
    """The directory a relative path resolves against after the ``cd``s in ``prefix``."""
    cds = list(_CD_RE.finditer(prefix))
    if not cds:
        return root
    target = cds[-1].group(1).strip("\"'")
    if target.startswith(("$", "~", "-")):
        return None
    p = Path(target)
    return p if p.is_absolute() else root / p


def _script_path_missing(command: str, root: Path, extra_roots: list[Path]) -> str | None:
    """The refusal for a script invocation naming a file that does not exist, else None.

    Measured 2026-08-29 (zc-03, eight Bodies, 24 h): 13 of 340 `bash|python3 <path>`
    invocations named a script that does not exist — `core/scripts/recurring-goal-detectors.sh`,
    `core/scripts/aspirations-read-goal.sh`, `core/scripts/worker-close-unit.sh` — and 5 of
    them piped the output (`… 2>&1 | python3 -c "json.loads(…)"`), so bash's own "No such
    file" went down the pipe, the parser raised JSONDecodeError, and the ENOENT hint
    (ADR-0097) that answers exactly this never saw the frame it keys on. Checking the path
    BEFORE running costs one `stat` and cannot be swallowed by a pipe; the refusal says
    nothing ran and carries the same lead the post-run hint would have.

    Fail-open by construction: a path this preflight cannot resolve (a `$VAR`, a `cd` to
    an unexpandable target, a heredoc body) is not checked. A file written earlier in the
    SAME command (`… > x.sh && bash x.sh`) is not checked either.
    """
    head = command.split("<<", 1)[0]
    for m in _SCRIPT_INVOCATION_RE.finditer(head):
        interp, path = m.group(1), m.group(2)
        before = head[: m.start()]
        base = _cwd_before(before, root)
        if base is None or re.search(r">\s*" + re.escape(path), before):
            continue
        candidates = [base / path, *(Path(r) / path for r in extra_roots)]
        if any(_exists(c) for c in candidates):
            continue
        where = f"'{path}' does not exist" + ("" if base == root else f" under {base}")
        lead = _enoent_fix(f"bash: {path}: No such file or directory", base, list(extra_roots))
        return f"{where}, so `{interp} {path}` was not run — nothing in this command ran. " + (
            lead or "`ls` the directory before guessing another name."
        )
    return None


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except OSError:
        return False


class BashTool(Tool):
    """Execute an arbitrary shell command with the workspace as the cwd."""

    spec = ToolSpec(
        name="Bash",
        description=(
            "Run a shell command with the workspace as the working directory. "
            "stdout and stderr are combined. Default 120s timeout (max 600): a command still "
            "running then is moved to the background, not killed, and the result says how to "
            "wait for it. Returns a non-zero exit code as an error. Output past 30,000 "
            "characters (10,000 for a failed command) is saved to a file: the result shows "
            "part of it and the path."
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
                    "description": "Timeout in seconds (default 120, max 600).",
                    "minimum": 1,
                    "maximum": _MAX_TIMEOUT,
                },
                "description": {
                    "type": "string",
                    "description": "Optional one-line note on what the command does (recorded).",
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "true starts the command detached and returns at once with a task id "
                        "and an output file; the session is notified when it exits (a "
                        "<task-notification> harness line at its next idle prompt). Read its "
                        "output with TaskOutput, kill it with TaskStop. No '&' needed."
                    ),
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

        typed = _tool_typed_as_command(command, ctx.tool_registry)
        if typed is not None:
            return ToolResult.error(
                typed, data={"command": command, "tool_typed_as_command": True}, fix=typed
            )
        missing = _script_path_missing(
            command, Path(str(ctx.workspace_root)), list(ctx.extra_workspace_roots)
        )
        if missing is not None:
            return ToolResult.error(
                missing, data={"command": command, "script_path_missing": True}, fix=missing
            )

        if args.get("run_in_background") is True:
            return await self._start_background(command, args, ctx)

        # ``bool`` is an ``int`` subclass; treat True/False as "no timeout given".
        timeout = args.get("timeout")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            timeout = _DEFAULT_TIMEOUT
        timeout = min(timeout, _MAX_TIMEOUT)

        # A command still running at its timeout is moved to the background, not killed
        # (ADR-0236): the session's task table takes it and the result says how to wait. With
        # no session to hold it, run_capturing kills it at the timeout. Either way the child
        # runs in its own process group, so a turn cancellation kills the whole tree (no
        # orphaned grandchildren); CancelledError is NOT caught here (it is BaseException) so a
        # cancel propagates after teardown.
        tasks = ctx.background_tasks
        try:
            if tasks is None:
                output, exit_code = await run_capturing(
                    shell_command=command,
                    cwd=str(ctx.workspace_root),
                    timeout=timeout,
                    extra_env=ctx.egress_env,
                    drop_env=ctx.scrub_env,
                )
            else:
                description = args.get("description")
                ran = await tasks.run_foreground(
                    command,
                    cwd=str(ctx.workspace_root),
                    timeout_seconds=timeout,
                    description=description if isinstance(description, str) else "",
                    extra_env=ctx.egress_env,
                    drop_env=list(ctx.scrub_env),
                )
                if isinstance(ran, BackgroundTask):
                    # TaskOutput is the way to wait: the exit notification arrives only at an
                    # idle prompt, and a worker's one long turn may never reach one.
                    return ToolResult.ok(
                        f"Command did not finish within its {timeout}s timeout and was moved "
                        f"to the background, still running, with ID: {ran.id}. Output is being "
                        f"written to: {ran.output_file}. To wait for it, call "
                        f'TaskOutput(task_id="{ran.id}", timeout=600000): it returns when the '
                        "command exits or after 10 minutes; call it again if it is still "
                        f'running. TaskOutput(task_id="{ran.id}", block=false) or a Read of '
                        "that file shows the output so far; TaskStop kills it.",
                        data={
                            "command": command,
                            "background": True,
                            "moved_to_background": True,
                            "timeout": timeout,
                            "task_id": ran.id,
                            "output_file": ran.output_file,
                        },
                    )
                output, exit_code = ran
        except CommandTimeout:
            return ToolResult.error(
                f"Command timed out after {timeout}s: {command}",
                data={"command": command, "timed_out": True},
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to run command: {exc}", data={"command": command})

        return self._finish(command, output, exit_code, ctx)

    async def _start_background(self, command: str, args: dict, ctx: ToolContext) -> ToolResult:
        """Claude Code's ``run_in_background`` (ADR-0191): spawn detached, return at once with
        the task id and the output file; the exit is reported at the session's next idle
        prompt as a ``<task-notification>`` harness line."""
        tasks = ctx.background_tasks
        if tasks is None:
            return ToolResult.error(
                "background commands are not available here (no session to hold one); run "
                "the command in the foreground.",
                data={"command": command, "background": False},
            )
        description = args.get("description")
        try:
            task = await tasks.start(
                command,
                cwd=str(ctx.workspace_root),
                description=description if isinstance(description, str) else "",
                extra_env=ctx.egress_env,
                drop_env=list(ctx.scrub_env),
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(
                f"Failed to start command in the background: {exc}", data={"command": command}
            )
        # The notification comes at an idle prompt, so the result also names the call that
        # waits: a worker's one long turn may never reach that prompt (ADR-0236).
        return ToolResult.ok(
            f"Command running in background with ID: {task.id}. Output is being written to: "
            f"{task.output_file}. You will be notified when it completes, at the session's next "
            f'idle prompt; to wait for it now, call TaskOutput(task_id="{task.id}", '
            f'timeout=600000). To check interim output, call TaskOutput(task_id="{task.id}", '
            "block=false) or Read that file.",
            data={
                "command": command,
                "background": True,
                "task_id": task.id,
                "output_file": task.output_file,
            },
        )

    def _finish(self, command: str, output: str, exit_code: int, ctx: ToolContext) -> ToolResult:
        """The foreground result: the combined output within the limits, the exit code, and
        the fix a failure suggests. The fix hints read the WHOLE output, before the limits."""
        output = _strip_apport_noise(output)  # before the limits: the noise must not spend them
        failed = exit_code != 0
        fix: str | None = None
        if failed:
            fix = (
                _interpreter_mismatch_fix(command)
                or _posix_exit_fix(command, output, exit_code, Path(str(ctx.workspace_root)))
                or _enoent_fix(output, Path(str(ctx.workspace_root)), ctx.extra_workspace_roots)
                or _module_not_found_fix(
                    command, output, Path(str(ctx.workspace_root)), ctx.extra_workspace_roots
                )
                or _python_inline_fix(command, output)
                or _json_first_line_fix(command, output)
                or _windows_shell_fix(command, output)
            )
        else:
            # Exit 0 can be a trailing pipe's (`python3 x.sh … | tail -40`): the interpreter
            # choked and `tail` reported success (ADR-0093, measured on the reducer the same
            # day the hint shipped — the pipe hid the failure the hint was written for). The
            # mismatched command plus the interpreter's own error text is the signal; the
            # result is the failure it was.
            mismatch = _interpreter_mismatch_fix(command)
            if mismatch and _INTERPRETER_ERROR_RE.search(output):
                failed = True
                fix = f"{mismatch} (The exit code 0 is the pipe's, not the script's.)"

        shown, saved = _fit_output(output, failed=failed, tasks=ctx.background_tasks)
        combined = shown
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += f"[exit code: {exit_code}]"

        data: dict[str, Any] = {
            "command": command,
            "exit_code": exit_code,
            "truncated": shown != output,
            # Where the command's own output ends and the exit-code line begins: the
            # PostToolUse wire cuts Claude Code's ``tool_response.stdout`` here (ADR-0210).
            STDOUT_CHARS: len(shown),
        }
        if saved is not None:
            data["output_file"] = str(saved)
            data["output_chars"] = len(output)
        if failed:
            return ToolResult.error(combined, data=data, fix=fix)
        return ToolResult.ok(combined, data=data)

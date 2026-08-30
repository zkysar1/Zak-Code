"""Run a shell command within the workspace (DANGER_FULL_ACCESS)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

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
        "Do not retry the same command: write the program to a file with the write_file "
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


def _module_not_found_fix(output: str, root: Path, extra_roots: list[Path]) -> str | None:
    """Name where a package of that name lives in the workspace, else None.

    Measured 2026-08-30 (zc-03, coach Bodies): five ``ModuleNotFoundError: No module
    named 'yahoo'`` in 24 h across four sessions — every one ``cd <workspace> && python3
    …`` after the package had been consolidated under ``.mind-data/world/scripts/yahoo``
    — and one identical retry, because the error names the module and nothing names the
    directory Python would have had to be run from. A dotted name whose top package IS
    found but whose submodule is not gets the package's real module names instead. A
    genuinely absent package (nothing in the workspace by that name) stays a plain
    error: install guesses are not this hint's business.
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
        return None
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
                or _module_not_found_fix(
                    output, Path(str(ctx.workspace_root)), ctx.extra_workspace_roots
                )
                or _python_inline_fix(command, output)
                or _json_first_line_fix(command, output)
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

"""The Recipe Cursor (Slice 2): a verify-before-finish gate for create-and-run tasks.

A weak local model writes a file, claims success, and ends the turn over code it never
actually ran (or that is broken) — failures #2 (incoherent plan) and #5 (broken final
state). The cursor makes the HARNESS, not the model, own "done": once the model has
written a **runnable script** this turn (a file whose extension maps to a known interpreter
— ``.py``/``.js``/``.ts``/``.sh``/``.rb``/``.ps1``/... see :data:`_INTERPRETER_BY_EXT`),
the turn may not end ``completed`` until the model has RUN that file successfully (a
``bash``/``powershell`` result with no error whose command actually *executes* the written
file — run by an interpreter or directly, not merely ``echo``/``cat``-ing it). If it cannot
get there within ``attempt_cap`` nudges, the turn ends gracefully as ``recipe_stalled``
rather than deadlocking or claiming a false success.

The gate is **always on and self-arms from the observed write** — it is not a feature flag.
A turn that writes no runnable script is wholly unaffected (``needs_verification`` stays
False), so "always on" costs nothing on non-create-and-run turns.

Pure per-turn state: the loop creates one cursor per turn, feeds it each iteration's
``(calls, results)`` via :meth:`observe`, and consults :meth:`needs_verification` at the
turn's natural completion point. No provider/transport/vendor knowledge — the *model*
performs the run (so the correct interpreter, e.g. ``py``, is used via the workspace
rules), and the harness only enforces that a real run happened.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import sys

from zakcode.messages import ToolResultBlock
from zakcode.providers.base import ToolCall

_WRITE_TOOLS = {"write_file", "edit_file"}
_RUN_TOOLS = {"bash", "powershell"}

# Programs that actually EXECUTE a script (vs. merely naming it). Used to require a real
# run before the gate is satisfied, so ``echo``/``cat``/``ls``/``rm <file>`` no longer
# count. Cross-shell, structural names — not a vendor branch.
# INVARIANT: every bare interpreter that :func:`resolve_run_command` can emit as a command
# HEAD must appear here, or a successful harness/model run would not be credited (the gate
# would falsely stall). I.e. ``_INTERPRETERS`` must be a superset of the head-position
# executables in :data:`_INTERPRETER_BY_EXT` (``deno`` is the lone exception — it runs via
# the ``deno run`` subcommand, so ``deno`` is the head and ``run`` is just a body token).
_INTERPRETERS = {
    "py",
    "python",
    "python2",
    "python3",
    "node",
    "deno",
    "bun",
    "tsx",
    "ts-node",
    "ruby",
    "bash",
    "sh",
    "zsh",
    "pwsh",
    "powershell",
}
#: Package runners that execute via a ``run`` subcommand (e.g. ``uv run app.py``).
_RUNNERS = {"uv", "poetry", "pdm", "hatch", "rye", "pipenv"}
#: Runnable script extensions → the interpreters (in preference order) that execute them.
#: A successful write/edit of a file with one of these extensions ARMS the verify gate, and
#: the harness-issued run uses the first interpreter present on PATH. Generalizes the gate
#: beyond Python; an extension with no available interpreter simply falls back to nudging.
_INTERPRETER_BY_EXT: dict[str, tuple[str, ...]] = {
    ".py": ("py", "python3", "python"),
    ".js": ("node",),
    ".mjs": ("node",),
    ".cjs": ("node",),
    ".ts": ("deno", "bun", "tsx", "ts-node"),
    ".sh": ("bash", "sh"),
    ".bash": ("bash", "sh"),
    ".rb": ("ruby",),
    ".ps1": ("pwsh", "powershell"),
}
#: Shell tokens that separate one command from the next; matching is per-segment so an
#: interpreter in one segment never blesses a filename merely named in another.
_SEGMENT_SEPARATORS = {"&&", "||", ";", "|", "&", "\n"}
#: Windows executable suffixes stripped from a command's head before the interpreter check,
#: so a Windows invocation (``python.exe x.py``, ``node.cmd x.js``, and the very common
#: ``sys.executable`` form) is recognized as a real run — not just the bare ``python``/``py``.
_EXE_SUFFIXES = (".exe", ".cmd", ".bat", ".com")

# Conservative acceptance extraction (Slice 2b-C): a stated expected-output literal in
# the request, captured only when unambiguous so a wrong guess can never over-gate.
_ACCEPT_RE = re.compile(
    r"(?:prints?|outputs?|displays?|says?|should\s+(?:print|output|say|return))"
    r"[^`\"'\n]{0,40}"
    r"(?:`([^`\n]{1,200})`"
    r"|\"([^\"\n]{1,200})\""
    r"|'([^'\n]{1,200})'"
    r"|“([^”\n]{1,200})”)",
    re.IGNORECASE,
)
# A value that looks like a single ``name.ext`` filename token (e.g. ``result.csv``,
# ``data.xml``) is almost never an expected-stdout literal, so it is rejected. The
# extension must be ALPHABETIC so genuine numeric outputs like ``3.14`` / ``v1.2`` (whose
# "extension" is digits) are NOT mistaken for filenames. Generic by design — it replaces a
# hand-maintained extension allowlist that silently missed ``.csv``/``.xml``/``.log``/...
_FILENAME_RE = re.compile(r"^\S+\.[A-Za-z]{1,8}$")


def extract_acceptance(user_text: str) -> str | None:
    """Conservatively extract a stated expected-output literal from the request.

    Returns the string the program should print, or ``None`` when the request does not
    clearly and unambiguously state one (the common case -> exit-0-only verification).
    High precision by design: any ambiguity (no match, more than one distinct candidate,
    a path/filename-looking literal, multi-line, or over-long) returns ``None`` so a
    wrong extraction can never trap the turn in an unsatisfiable acceptance check.
    """
    candidates: list[str] = []
    for m in _ACCEPT_RE.finditer(user_text):
        literal = next((g for g in m.groups() if g is not None), None)
        if literal is not None:
            candidates.append(literal)
    distinct = list(dict.fromkeys(candidates))
    if len(distinct) != 1:
        return None
    value = distinct[0].strip()
    if not value or len(value) > 200 or "\n" in value:
        return None
    if "/" in value or "\\" in value or _FILENAME_RE.match(value):
        return None
    return value


def _basename_any(token: str) -> str:
    """Last path component of ``token``, splitting on BOTH separators (``/`` and ``\\``).

    Quotes are stripped first, and both separators are honored so a Windows or POSIX
    path resolves to the same basename regardless of the OS the harness runs on.
    """
    token = token.strip().strip("'\"")
    return re.split(r"[\\/]", token)[-1]


def _interpreter_name(token: str) -> str:
    """The command head normalized for the interpreter check: basename, lowercased, and
    with a trailing Windows executable suffix (``.exe``/``.cmd``/``.bat``/``.com``) removed.

    This is why ``C:\\...\\python.exe x.py`` (the ``sys.executable`` form weak models emit
    on Windows) counts as a real run of ``x.py`` exactly like bare ``python x.py``.
    """
    name = _basename_any(token).lower()
    for suffix in _EXE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _tokenize(command: str) -> list[str]:
    """Split a shell command into tokens, tolerant of quotes and Windows backslashes.

    Uses ``shlex`` in non-POSIX mode so quoted paths with spaces stay one token and
    backslashes are not treated as escapes; falls back to a plain whitespace split if
    the command has unbalanced quotes (``shlex`` would raise).
    """
    try:
        return shlex.split(command, posix=False)
    except ValueError:
        return command.split()


def _executed_targets(command: str, targets: set[str]) -> set[str]:
    """The subset of ``targets`` (basenames) that ``command`` actually *executes*.

    Unlike a naive ``basename in command`` substring test, each target must appear as a
    whole path token in an execution position — run by an interpreter
    (``py``/``python``/``node``/...), by a package runner (``uv run x.py``), or directly
    (``./x.py`` / ``.\\x.py``) — within a single command segment. So a command that merely
    *names* the file (``echo``/``cat``/``ls``/``rm`` ``x.py``) does not count, and a longer
    unrelated filename (``aa.py`` for target ``a.py``) no longer false-positives.
    """
    if not command or not targets:
        return set()
    segments: list[list[str]] = [[]]
    for tok in _tokenize(command):
        if tok in _SEGMENT_SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(tok)

    executed: set[str] = set()
    for seg in segments:
        if not seg:
            continue
        head = _interpreter_name(seg[0])
        if head in _INTERPRETERS:
            body = seg[1:]
        elif head in _RUNNERS and len(seg) >= 2 and _interpreter_name(seg[1]) == "run":
            body = seg[2:]
        else:
            body = []
        executed |= {b for tok in body if (b := _basename_any(tok)) in targets}
        # Direct execution: a ./x.py or .\x.py token anywhere in the segment.
        for tok in seg:
            stripped = tok.strip().strip("'\"")
            if stripped.startswith(("./", ".\\")) and (b := _basename_any(stripped)) in targets:
                executed.add(b)
    return executed


def _runnable_path(call: ToolCall, result: ToolResultBlock) -> str | None:
    """The path a successful write/edit touched IF it is a runnable script, else ``None``.

    "Runnable" = an extension in :data:`_INTERPRETER_BY_EXT` (``.py``/``.js``/``.ts``/
    ``.sh``/``.rb``/``.ps1``/...). A write of such a file arms the verify-before-finish gate.
    """
    path: str | None = None
    if isinstance(result.data, dict):
        candidate = result.data.get("path")
        if isinstance(candidate, str):
            path = candidate
    if path is None:
        candidate = call.arguments.get("path")
        path = candidate if isinstance(candidate, str) else None
    if path is None:
        return None
    return path if os.path.splitext(path)[1].lower() in _INTERPRETER_BY_EXT else None


def resolve_run_command(path: str) -> str | None:
    """A shell command that runs ``path`` with an available interpreter, or None if none.

    Picks the interpreter from the file's extension (:data:`_INTERPRETER_BY_EXT`) and the
    first candidate present on PATH (``shutil.which``) — e.g. ``py "x.py"``, ``node "x.js"``,
    ``bash "x.sh"``. Used by the harness-issued verification run so the right interpreter is
    chosen deterministically; returns None when none resolves so the caller falls back to
    nudging the model rather than manufacturing a failing run.

    For ``.py`` there is a guaranteed last resort: the interpreter currently running Zak Code
    (``sys.executable``) can always run a written ``.py``, even when no bare ``py``/``python``
    is on PATH (a Windows venv whose ``Scripts`` dir isn't on PATH, or an embedded/isolated
    interpreter launched by absolute path). ``sys.executable`` is an absolute path; its
    ``.exe`` suffix is normalized off by :func:`_executed_targets`, so the run is still
    credited as executing the file.
    """
    ext = os.path.splitext(path)[1].lower()
    for exe in _INTERPRETER_BY_EXT.get(ext, ()):
        if shutil.which(exe):
            if exe == "deno":  # deno runs a script via the `run` subcommand, not directly
                return f'deno run "{path}"'
            return f'{exe} "{path}"'
    if ext == ".py" and sys.executable:
        return f'"{sys.executable}" "{path}"'
    return None


class RecipeCursor:
    """Per-turn 'verify what you wrote before finishing' gate (see the module docstring)."""

    def __init__(
        self, *, enabled: bool, attempt_cap: int = 3, acceptance: str | None = None
    ) -> None:
        self.enabled = enabled
        self.attempt_cap = max(0, attempt_cap)
        self.acceptance = acceptance  # required substring in the run output, or None
        self.wrote_runnable = False  # a runnable script was created/edited this turn
        self.nudges = 0  # verification attempts spent (nudges + harness runs) toward the cap
        self._targets: set[str] = set()  # basenames of runnable scripts written this turn
        self._abs_targets: list[str] = []  # their paths, in write order (for the harness run)
        self._verified: set[str] = set()  # basenames that have been run successfully
        self.harness_runs = 0  # how many harness-issued verification runs were issued
        # Recovery tracking (research R1): per-target, the command whose run of a created file
        # FAILED — so a later SUCCESS that runs the SAME target is recorded as a recovery, and a
        # never-failed file's first-try success is never misattributed as one. (review2 #3)
        self._failed_by_target: dict[str, str] = {}
        self._recovered_command: str | None = None

    @property
    def verified(self) -> bool:
        """True once EVERY runnable written this turn has been run successfully.

        Per-target (not one turn-wide flag), so writing two files and running only one no
        longer marks the whole turn verified — the gate keeps its 'run what you wrote'
        promise across multiple files.
        """
        return bool(self._targets) and self._targets <= self._verified

    def observe(self, calls: list[ToolCall], results: list[ToolResultBlock]) -> None:
        """Update state from one iteration's *successful* tool calls."""
        if not self.enabled:
            return
        by_id = {r.tool_use_id: r for r in results}
        for call in calls:
            result = by_id.get(call.id)
            if result is None:
                continue
            if result.is_error:
                # Record a FAILED run PER executed target so a later success of the SAME file is
                # a recovery (research R1). Does NOT affect verification — a failed run still
                # verifies nothing, so the gate behaves exactly as before.
                if call.name in _RUN_TOOLS and self.wrote_runnable:
                    command = call.arguments.get("command")
                    if isinstance(command, str):
                        for base in _executed_targets(command, self._targets):
                            self._failed_by_target[base] = command
                continue
            if call.name in _WRITE_TOOLS:
                path = _runnable_path(call, result)
                if path is not None:
                    self.wrote_runnable = True
                    base = os.path.basename(path)
                    self._targets.add(base)
                    self._abs_targets.append(path)
                    self._verified.discard(base)  # a fresh write must be re-verified
            elif call.name in _RUN_TOOLS and self.wrote_runnable:
                command = call.arguments.get("command")
                if not isinstance(command, str):
                    continue
                # Only a command that actually EXECUTES a written file verifies it (not one
                # that merely names it). With an acceptance string, the run output must ALSO
                # contain it (catches "exits 0 but prints the wrong thing"); without one,
                # exit-0 suffices.
                executed = _executed_targets(command, self._targets)
                output_ok = self.acceptance is None or self.acceptance in (result.output or "")
                if executed and output_ok:
                    self._verified |= executed
                    # A recovery: this success ran a target whose EARLIER run failed. Bind it
                    # per-target so a never-failed file's first-try success is not misattributed,
                    # and clear the recovered targets' failure record. (review2 #3)
                    recovered = executed & set(self._failed_by_target)
                    if recovered:
                        self._recovered_command = command
                        for base in recovered:
                            self._failed_by_target.pop(base, None)

    @property
    def recovered_command(self) -> str | None:
        """A run command that SUCCEEDED after an earlier run of a created file FAILED this turn
        (else ``None``) — the corrective step a recipe-recovery lesson records (research R1)."""
        return self._recovered_command

    def needs_verification(self) -> bool:
        """True when the turn should not end yet: a runnable file written, not yet run."""
        return self.enabled and self.wrote_runnable and not (self._targets <= self._verified)

    def can_nudge(self) -> bool:
        """Whether another verification attempt is allowed before giving up (recipe_stalled)."""
        return self.nudges < self.attempt_cap

    def pending_target(self) -> str | None:
        """The most-recently-written runnable script still needing verification, else None."""
        for path in reversed(self._abs_targets):
            if os.path.basename(path) not in self._verified:
                return path
        return None

    def consume_attempt(self) -> None:
        """Count one verification attempt (e.g. a harness-issued run) toward the cap."""
        self.nudges += 1

    @staticmethod
    def _run_hint(target: str | None) -> str:
        """A language-correct 'how to run it' clause for the nudge, derived from the target.

        Prefers the exact command the harness itself would use (:func:`resolve_run_command`,
        which resolves an interpreter actually on PATH); falls back to the preferred
        interpreter for the extension when none is installed. So a ``.js`` target yields
        ``node "app.js"`` and a ``.sh`` target ``bash "build.sh"`` — never a hardcoded ``py``
        for a non-Python file (which a literal-minded weak model would run and fail). (review2 #1)
        """
        if not target:
            return "run it with the right interpreter"
        cmd = resolve_run_command(target)
        if cmd is not None:
            return f"run it now, e.g. `{cmd}`"
        exes = _INTERPRETER_BY_EXT.get(os.path.splitext(target)[1].lower(), ())
        if exes:
            prog = "deno run" if exes[0] == "deno" else exes[0]
            return f'run it now, e.g. `{prog} "{target}"`'
        return "run it with the right interpreter"

    def nudge(self) -> str:
        """Consume one attempt and return the one-step corrective instruction to inject."""
        self.nudges += 1
        message = (
            "You created or edited a runnable script but have not run it successfully yet. "
            f"{self._run_hint(self.pending_target())} to verify it works; if it errors, fix "
            "the file and run it again. Do not finish until the program runs without error."
        )
        if self.acceptance is not None:
            message += (
                f" The program must print `{self.acceptance}`; if it ran but did not "
                "produce that, fix it and run again."
            )
        return message

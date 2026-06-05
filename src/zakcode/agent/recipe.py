"""The Recipe Cursor (Slice 2): a verify-before-finish gate for create-and-run tasks.

A weak local model writes a file, claims success, and ends the turn over code it never
actually ran (or that is broken) — failures #2 (incoherent plan) and #5 (broken final
state). The cursor makes the HARNESS, not the model, own "done": once the model has
written a runnable (``.py``) file this turn, the turn may not end ``completed`` until the
model has RUN that file successfully (a ``bash``/``powershell`` result with no error whose
command actually *executes* the written file — run by an interpreter or directly, not
merely ``echo``/``cat``-ing it). If it cannot get there within ``attempt_cap``
nudges, the turn ends gracefully as ``recipe_stalled`` rather than deadlocking or claiming
a false success.

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

from zakcode.messages import ToolResultBlock
from zakcode.providers.base import ToolCall

_WRITE_TOOLS = {"write_file", "edit_file"}
_RUN_TOOLS = {"bash", "powershell"}

# Programs that actually EXECUTE a script (vs. merely naming it). Used to require a real
# run before the gate is satisfied, so ``echo``/``cat``/``ls``/``rm <file>`` no longer
# count. Cross-shell, structural names — not a vendor branch.
_INTERPRETERS = {
    "py",
    "python",
    "python2",
    "python3",
    "node",
    "deno",
    "bun",
    "ruby",
    "bash",
    "sh",
    "zsh",
    "pwsh",
    "powershell",
}
#: Package runners that execute via a ``run`` subcommand (e.g. ``uv run app.py``).
_RUNNERS = {"uv", "poetry", "pdm", "hatch", "rye", "pipenv"}
#: Shell tokens that separate one command from the next; matching is per-segment so an
#: interpreter in one segment never blesses a filename merely named in another.
_SEGMENT_SEPARATORS = {"&&", "||", ";", "|", "&", "\n"}

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
        head = _basename_any(seg[0]).lower()
        if head in _INTERPRETERS:
            body = seg[1:]
        elif head in _RUNNERS and len(seg) >= 2 and _basename_any(seg[1]).lower() == "run":
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


def _python_path(call: ToolCall, result: ToolResultBlock) -> str | None:
    """The ``.py`` path a successful write/edit touched, else ``None``."""
    path: str | None = None
    if isinstance(result.data, dict):
        candidate = result.data.get("path")
        if isinstance(candidate, str):
            path = candidate
    if path is None:
        candidate = call.arguments.get("path")
        path = candidate if isinstance(candidate, str) else None
    return path if (path is not None and path.endswith(".py")) else None


def resolve_python_run(path: str) -> str | None:
    """A shell command to run ``path`` with an available interpreter, or None if none.

    Tries ``py`` (the Windows launcher), then ``python3``, then ``python`` via
    ``shutil.which``. Used by the harness-issued verification run (Slice 2b-A) so the
    correct interpreter is chosen deterministically; returns None when none resolves so
    the caller falls back to nudging the model rather than manufacturing a failing run.
    """
    for interpreter in ("py", "python3", "python"):
        if shutil.which(interpreter):
            return f'{interpreter} "{path}"'
    return None


class RecipeCursor:
    """Per-turn 'verify what you wrote before finishing' gate (see the module docstring)."""

    def __init__(
        self, *, enabled: bool, attempt_cap: int = 3, acceptance: str | None = None
    ) -> None:
        self.enabled = enabled
        self.attempt_cap = max(0, attempt_cap)
        self.acceptance = acceptance  # required substring in the run output, or None
        self.wrote_runnable = False  # a .py was created/edited this turn
        self.nudges = 0  # verification attempts spent (nudges + harness runs) toward the cap
        self._targets: set[str] = set()  # basenames of .py files written this turn
        self._abs_targets: list[str] = []  # their paths, in write order (for the harness run)
        self._verified: set[str] = set()  # basenames that have been run successfully
        self.harness_runs = 0  # how many harness-issued verification runs were issued

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
            if result is None or result.is_error:
                continue
            if call.name in _WRITE_TOOLS:
                path = _python_path(call, result)
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

    def needs_verification(self) -> bool:
        """True when the turn should not end yet: a runnable file written, not yet run."""
        return self.enabled and self.wrote_runnable and not (self._targets <= self._verified)

    def can_nudge(self) -> bool:
        """Whether another verification attempt is allowed before giving up (recipe_stalled)."""
        return self.nudges < self.attempt_cap

    def pending_target(self) -> str | None:
        """The most-recently-written runnable .py still needing verification, else None."""
        for path in reversed(self._abs_targets):
            if os.path.basename(path) not in self._verified:
                return path
        return None

    def consume_attempt(self) -> None:
        """Count one verification attempt (e.g. a harness-issued run) toward the cap."""
        self.nudges += 1

    def nudge(self) -> str:
        """Consume one attempt and return the one-step corrective instruction to inject."""
        self.nudges += 1
        message = (
            "You created or edited a Python file but have not run it successfully yet. "
            "Run it now (use `py <file>`) to verify it works; if it errors, fix the file "
            "and run it again. Do not finish until the program runs without error."
        )
        if self.acceptance is not None:
            message += (
                f" The program must print `{self.acceptance}`; if it ran but did not "
                "produce that, fix it and run again."
            )
        return message

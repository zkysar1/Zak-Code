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

import importlib.util
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
# A CLI OPTION FLAG (``-v``, ``--top N``, ``--help``) is a usage/synopsis token, never an
# expected-stdout literal — but it reads like one when a request says e.g. "Print the top 10
# ... Support a `--top N` flag". A leading ``-`` followed by a LETTER is the flag shape; a
# leading ``-`` then a DIGIT is kept, so a genuine negative-number output (``-5``, ``-3.14``)
# is NOT mistaken for a flag. (Generic, like the filename guard — not a hardcoded blocklist.)
_CLI_FLAG_RE = re.compile(r"^--?[A-Za-z]")
# A FORMAT TEMPLATE reads like an expected-output literal but describes the SHAPE of many
# lines, not one exact stdout string: 'prints the top words as "word count" lines'. The tell
# is around the quote, not inside it — a template lead-in right before it ("as", "like", "in
# the form", ...) or a plural/format noun right after it ("lines", "pairs", "rows", "format",
# ...). Either rejects the candidate. Measured 2026-09-05 on coach's local model: the
# wordstats request extracted `word count`, which no run could ever print, so the gate re-ran
# the file to the attempt cap and stalled a fully green turn (ADR-0114).
_TEMPLATE_LEAD_RE = re.compile(
    r"(?:\b(?:as|like|such\s+as|formatted\s+as|of\s+the\s+form|in\s+the\s+form(?:\s+of)?"
    r"|following\s+the\s+pattern|shaped\s+like)|e\.g\.)\s*$",
    re.IGNORECASE,
)
_TEMPLATE_TAIL_RE = re.compile(
    r"^\s*(?:lines?|rows?|pairs?|entries|entry|records?|columns?|fields?|tuples?|blocks?"
    r"|format|style|template|pattern)\b",
    re.IGNORECASE,
)


def extract_acceptance(user_text: str) -> str | None:
    """Conservatively extract a stated expected-output literal from the request.

    Returns the string the program should print, or ``None`` when the request does not
    clearly and unambiguously state one (the common case -> exit-0-only verification).
    High precision by design: any ambiguity (no match, more than one distinct candidate,
    a path/filename-looking literal, a format template, multi-line, or over-long) returns
    ``None`` so a wrong extraction can never trap the turn in an unsatisfiable acceptance
    check.
    """
    candidates: list[str] = []
    for m in _ACCEPT_RE.finditer(user_text):
        group = next((i for i in range(1, _ACCEPT_RE.groups + 1) if m.group(i) is not None), None)
        if group is None:
            continue
        # The text between the cue verb and the opening quote, and what follows the closing
        # quote: a template lead-in or a format noun means the literal is a shape, not an output.
        lead = user_text[m.start() : m.start(group) - 1]
        tail = user_text[m.end(group) + 1 : m.end(group) + 25]
        if _TEMPLATE_LEAD_RE.search(lead) or _TEMPLATE_TAIL_RE.match(tail):
            continue
        candidates.append(m.group(group))
    distinct = list(dict.fromkeys(candidates))
    if len(distinct) != 1:
        return None
    value = distinct[0].strip()
    if not value or len(value) > 200 or "\n" in value:
        return None
    if "/" in value or "\\" in value or _FILENAME_RE.match(value):
        return None
    if _CLI_FLAG_RE.match(value):  # a CLI option flag (e.g. `--top N`, `-v`), not expected stdout
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


def _segments(command: str) -> list[list[str]]:
    """Split a command into execution segments at shell separators (``&&``/``|``/``;``/...).

    Token matching is per-segment so an interpreter (or a test runner) in one segment never
    blesses a token that lives in another.
    """
    segments: list[list[str]] = [[]]
    for tok in _tokenize(command):
        if tok in _SEGMENT_SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(tok)
    return segments


def _executed_targets(command: str, targets: set[str]) -> set[str]:
    """The subset of ``targets`` (basenames) that ``command`` actually *executes*.

    Unlike a naive ``basename in command`` substring test, each target must appear as a
    whole path token in an execution position — run by an interpreter
    (``py``/``python``/``node``/...), by a package runner (``uv run x.py``), by a test runner
    that names it (``pytest test_x.py`` executes that module's tests — the module under test
    is only imported, and is credited through :func:`_runs_test_suite` instead), or directly
    (``./x.py`` / ``.\\x.py``) — within a single command segment. So a command that merely
    *names* the file (``echo``/``cat``/``ls``/``rm`` ``x.py``) does not count, and a longer
    unrelated filename (``aa.py`` for target ``a.py``) no longer false-positives.
    """
    if not command or not targets:
        return set()
    executed: set[str] = set()
    for seg in _segments(command):
        if not seg:
            continue
        head = _interpreter_name(seg[0])
        if head in _INTERPRETERS or head in _TEST_RUNNER_HEADS:
            body = seg[1:]
        elif head in _RUNNERS and len(seg) >= 2 and _interpreter_name(seg[1]) == "run":
            body = seg[2:]
        else:
            body = []
        executed |= {b for tok in body if (b := _basename_any(tok)) in targets}
        # `python -m pkg.mod` executes pkg/mod.py without ever naming the file: the module
        # path's last component is the executed basename (ADR-0114).
        for i, tok in enumerate(body[:-1]):
            if tok.strip().strip("'\"") == "-m":
                module = body[i + 1].strip().strip("'\"")
                if (b := module.rsplit(".", 1)[-1] + ".py") in targets:
                    executed.add(b)
        # Direct execution: a ./x.py or .\x.py token anywhere in the segment.
        for tok in seg:
            stripped = tok.strip().strip("'\"")
            if stripped.startswith(("./", ".\\")) and (b := _basename_any(stripped)) in targets:
                executed.add(b)
    return executed


#: Test-runner heads whose SUCCESSFUL (exit-0) run verifies the written code through its
#: tests. A runner DISCOVERS and IMPORTS the modules under test, so their filenames never
#: appear as execution tokens (``pytest test_x.py`` runs ``test_x.py`` but only imports
#: ``x.py``) — yet a green suite is a STRONGER signal than a bare run that the code works.
#: Recognized structurally (mirroring _executed_targets' per-segment split), multi-language to
#: match the gate's cross-language arming — never a vendor branch.
_TEST_RUNNER_HEADS = {"pytest", "py.test", "jest", "vitest", "mocha", "ava", "rspec"}
#: Modules run as ``python -m <module>`` (or ``py -m unittest``) that execute a test suite.
_TEST_RUNNER_MODULES = {"pytest", "unittest", "nose2"}


def _runs_test_suite(command: str) -> bool:
    """True if any segment of ``command`` invokes a recognized test runner.

    A green run of one satisfies the verify-before-finish gate (a test runner exercises the
    written modules through their tests, importing them rather than naming them as execution
    tokens). Covers bare runners (``pytest``), the ``<launcher> -m <module>`` form
    (``python -m pytest`` / ``py -m unittest``), and the ``<tool> test`` subcommand of common
    toolchains (``npm``/``pnpm``/``yarn`` ``test``, ``deno``/``bun``/``go`` ``test``,
    ``cargo test``, ``node --test``). Per-segment, so ``cd sub && pytest`` is handled.
    """
    if not command:
        return False
    for seg in _segments(command):
        if not seg:
            continue
        head = _interpreter_name(seg[0])
        body = [t.strip().strip("'\"").lower() for t in seg[1:]]
        if head in _TEST_RUNNER_HEADS:
            return True
        # `python -m pytest` / `py -m unittest`: an interpreter head with `-m <runner module>`.
        if head in _INTERPRETERS and "-m" in body:
            i = body.index("-m")
            if i + 1 < len(body) and body[i + 1] in _TEST_RUNNER_MODULES:
                return True
        # `<tool> test` subcommand forms.
        if head in {"npm", "pnpm", "yarn"} and "test" in body:
            return True
        if head in {"deno", "bun", "go"} and body[:1] == ["test"]:
            return True
        if head == "cargo" and "test" in body:
            return True
        if head == "node" and "--test" in body:
            return True
    return False


_USAGE_LINE = re.compile(r"^\s*usage[: ]", re.IGNORECASE)


def _is_usage_refusal(output: str) -> bool:
    """True when a failed run's output leads with a usage/synopsis line.

    An args-required CLI run with no arguments that prints ``Usage: x.sh <id>`` and
    exits nonzero has parsed, executed, and correctly demanded its arguments — the
    strongest verification a harness run can get without inventing arguments. Only
    the FIRST non-empty line (within the first three) counts, so an incidental
    "usage" deep in a real failure never matches.
    """
    for line in output.splitlines()[:3]:
        if line.strip():
            return bool(_USAGE_LINE.match(line))
    return False


#: Python package/pytest plumbing that is never RUN as a script: ``__init__.py`` executes only
#: through an import and ``conftest.py`` only under pytest. Writing one must not arm the gate —
#: a harness run of either proves nothing (``py conftest.py`` exits 0 having done nothing) and
#: cannot verify the module it configures. They are verified the way they run: by a green
#: suite or a sibling module's run (ADR-0114).
_NOT_RUNNABLE_BASENAMES = {"__init__.py", "conftest.py"}
#: A pytest-style test module (``test_x.py`` / ``x_test.py``): run as a bare script it imports
#: nothing correctly (its own directory becomes ``sys.path[0]``) and asserts nothing; the run
#: that verifies it is the test runner's (ADR-0114).
_TEST_MODULE_RE = re.compile(r"^(?:test_.*|.*_test)\.py$", re.IGNORECASE)


def _runnable_path(call: ToolCall, result: ToolResultBlock) -> str | None:
    """The path a successful write/edit touched IF it is a runnable script, else ``None``.

    "Runnable" = an extension in :data:`_INTERPRETER_BY_EXT` (``.py``/``.js``/``.ts``/
    ``.sh``/``.rb``/``.ps1``/...) that is not package plumbing
    (:data:`_NOT_RUNNABLE_BASENAMES`). A write of such a file arms the verify-before-finish gate.
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
    if _basename_any(path).lower() in _NOT_RUNNABLE_BASENAMES:
        return None
    return path if os.path.splitext(path)[1].lower() in _INTERPRETER_BY_EXT else None


def _package_root(directory: str) -> tuple[str, list[str]]:
    """Walk up from ``directory`` while each level is a package (has an ``__init__.py``).

    Returns ``(root, parts)``: the first non-package ancestor — the directory Python must run
    from for the package's absolute imports to resolve — and the package path down from it.
    ``parts`` is empty when ``directory`` is not a package.
    """
    parts: list[str] = []
    current = os.path.abspath(directory)
    while os.path.isfile(os.path.join(current, "__init__.py")):
        parent, name = os.path.split(current)
        if not name or parent == current:
            break
        parts.insert(0, name)
        current = parent
    return current, parts


def _python_exe() -> str | None:
    """The Python the harness runs a ``.py`` with: first PATH candidate, else ``sys.executable``."""
    for exe in _INTERPRETER_BY_EXT[".py"]:
        if shutil.which(exe):
            return exe
    return f'"{sys.executable}"' if sys.executable else None


def _python_run_command(path: str) -> str | None:
    """The package- and test-aware run command for a written ``.py``, or None for a plain script.

    A module INSIDE a package (``pkg/cli.py`` beside ``pkg/__init__.py``) cannot run as
    ``py "pkg/cli.py"``: its own directory becomes ``sys.path[0]`` and its first absolute
    import of ``pkg`` fails — a failure the harness manufactures, then blames on the model
    (measured 2026-09-05: three identical ``ModuleNotFoundError`` runs, then ``recipe_stalled``
    over a working CLI). It runs as ``cd "<root>"; py -m pkg.cli``. A test module runs under
    the test runner (``cd "<root>"; pytest -q "<path>"``). The ``;`` separator is the one both
    harness shells accept (bash, and Windows PowerShell 5 has no ``&&``). Both forms are
    credited — :func:`_executed_targets` reads the ``-m`` module path, :func:`_runs_test_suite`
    the runner (ADR-0114).
    """
    directory, basename = os.path.split(os.path.abspath(path))
    root, parts = _package_root(directory)
    if _TEST_MODULE_RE.match(basename):
        if not parts and os.path.basename(directory).lower() in {"tests", "test"}:
            root = os.path.dirname(directory)  # the package under test lives beside tests/
        if shutil.which("pytest"):
            return f'cd "{root}"; pytest -q "{path}"'
        if sys.executable and importlib.util.find_spec("pytest") is not None:
            return f'cd "{root}"; "{sys.executable}" -m pytest -q "{path}"'
        return None  # no runner available: fall through to the plain script run
    if parts:
        exe = _python_exe()
        if exe is None:
            return None
        module = ".".join([*parts, os.path.splitext(basename)[0]])
        return f'cd "{root}"; {exe} -m {module}'
    return None


def resolve_run_command(path: str) -> str | None:
    """A shell command that runs ``path`` with an available interpreter, or None if none.

    Picks the interpreter from the file's extension (:data:`_INTERPRETER_BY_EXT`) and the
    first candidate present on PATH (``shutil.which``) — e.g. ``py "x.py"``, ``node "x.js"``,
    ``bash "x.sh"``. Used by the harness-issued verification run so the right interpreter is
    chosen deterministically; returns None when none resolves so the caller falls back to
    nudging the model rather than manufacturing a failing run.

    A ``.py`` runs the way Python actually runs it (:func:`_python_run_command`): a module
    inside a package as ``cd "<root>"; py -m pkg.mod``, a test module under the test runner,
    and only a plain script as ``py "x.py"``.

    For ``.py`` there is a guaranteed last resort: the interpreter currently running Zak Code
    (``sys.executable``) can always run a written ``.py``, even when no bare ``py``/``python``
    is on PATH (a Windows venv whose ``Scripts`` dir isn't on PATH, or an embedded/isolated
    interpreter launched by absolute path). ``sys.executable`` is an absolute path; its
    ``.exe`` suffix is normalized off by :func:`_executed_targets`, so the run is still
    credited as executing the file.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        special = _python_run_command(path)
        if special is not None:
            return special
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
        # A green run of a recognized test runner (pytest/unittest/jest/...) verifies the
        # written code THROUGH its tests — the modules are imported, never named as execution
        # tokens — so it satisfies the gate as a whole (see :func:`_runs_test_suite`). This is
        # what stops a create-with-tests turn that runs `pytest` green from falsely stalling as
        # recipe_stalled. Reset by a fresh runnable write (the new code is unverified again).
        self._suite_verified = False

    @property
    def verified(self) -> bool:
        """True once the written runnables are verified this turn — by EITHER path:

        * a recognized test runner ran green (``_suite_verified``), which exercises every
          written module through its tests (they are imported, not named as run tokens); OR
        * EVERY runnable written this turn has been run successfully (per-target — so writing
          two files and running only one does not mark the whole turn verified; the gate keeps
          its 'run what you wrote' promise across multiple files).
        """
        if self._suite_verified:
            return True
        return bool(self._targets) and self._targets <= self._verified

    @property
    def written_paths(self) -> list[str]:
        """Absolute paths of the runnable scripts written this turn, in write order (deduped).

        The quality gate (seam A) reads these to score the ACTUAL work, not just the claimed text.
        """
        seen: dict[str, None] = {}
        for path in self._abs_targets:
            seen.setdefault(path, None)
        return list(seen)

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
                # A failed run verifies nothing — EXCEPT a usage refusal: an
                # args-required script run without arguments that prints its
                # synopsis and exits nonzero has run and refused CORRECTLY.
                # Without this, the gate re-ran such a script to the attempt cap
                # and bounced the finish each time (measured 2026-08-25:
                # drive-fetch.sh <file_id> — three identical Usage exits). An
                # explicit acceptance literal still demands a real green run.
                if (
                    call.name in _RUN_TOOLS
                    and self.wrote_runnable
                    and self.acceptance is None
                    and _is_usage_refusal(result.output or "")
                ):
                    command = call.arguments.get("command")
                    if isinstance(command, str):
                        self._verified |= _executed_targets(command, self._targets)
                continue
            if call.name in _WRITE_TOOLS:
                path = _runnable_path(call, result)
                if path is not None:
                    self.wrote_runnable = True
                    base = os.path.basename(path)
                    self._targets.add(base)
                    self._abs_targets.append(path)
                    self._verified.discard(base)  # a fresh write must be re-verified
                    # ...and it invalidates a prior green test run: the new/edited code has
                    # not been exercised by the suite yet, so re-require verification.
                    self._suite_verified = False
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
                # A green run of a recognized test runner (this is the success path — is_error
                # is False here) verifies the written modules through their tests even though
                # their filenames are imported, not named as run tokens. It satisfies the gate
                # as a whole — but only when no exact-output literal was demanded (a suite run
                # does not demonstrate a specific stdout string, so an acceptance check still
                # requires a direct run that prints it).
                if self.acceptance is None and _runs_test_suite(command):
                    self._suite_verified = True

    def needs_verification(self) -> bool:
        """True when the turn should not end yet: a runnable file written, not yet verified.

        Verification is satisfied by a per-target run OR a green test-runner run — see the
        :attr:`verified` property. (``wrote_runnable`` implies ``_targets`` is non-empty, so
        this is the old ``not (_targets <= _verified)`` plus the green-suite short-circuit.)
        """
        return self.enabled and self.wrote_runnable and not self.verified

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

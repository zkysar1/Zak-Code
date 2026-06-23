"""Claude Code ``statusLine`` support (cosmetic; CC-compatible).

Claude Code lets a workspace configure a *status line* — a short, always-visible
strip the host repaints after each turn — via a ``statusLine`` object in
``.claude/settings.json``::

    {"statusLine": {"type": "command", "command": "my-status.sh"}}

The configured command is run after a turn with a **status JSON on stdin** (the
session id, the active model, the workspace, the running cost…) and its **stdout's
first line** is rendered as the status line. Frameworks use it to surface
context-budget pressure, the active agent, spend, etc.

This module is the GENERIC host side of that contract. It is deliberately:

* **Cosmetic and fail-safe.** A status line is decoration — it must NEVER break or
  slow a turn. Every public entry point swallows its own errors and returns ``None``
  (no line) on a missing/dangerous command, a non-zero exit, a timeout, a spawn
  failure, or any exception. :func:`render_status_line` never raises.
* **Off by default.** Gated by ``Settings.status_line`` (env ``ZAKCODE_STATUS_LINE``)
  or a per-Agent override, mirroring ``settings_hooks`` — a workspace carrying
  another runtime's ``statusLine`` does not silently run a subprocess here.
* **Secure like every other settings.json shell command.** The command is run as an
  argv array (never through a shell), scanned against the shared dangerous-pattern
  blocklist, and spawned with provider API keys scrubbed from its environment.

The stdin JSON matches Claude Code's documented ``statusLine`` input shape — top-level
``hook_event_name`` / ``session_id`` / ``cwd`` / ``transcript_path`` / ``version`` plus
nested ``model`` / ``workspace`` / ``cost`` objects — so a status script written for
Claude Code reads it unmodified (the same faithful-host promise as hooks/skills).

The runner mirrors :mod:`zakcode.hooks` (argv array, stdin JSON, scrubbed env, short
timeout, whole-tree teardown), kept separate because a status line is neither a
lifecycle hook nor a tool gate — it is presentation a *client* renders.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from zakcode._subprocess import new_group_kwargs, resolve_executable, terminate_process_tree

logger = logging.getLogger("zakcode.status_line")

#: Default timeout (seconds) for a status-line command. Short on purpose: the line is
#: cosmetic, so a slow script must not visibly stall the prompt. A command exceeding
#: this is killed and yields no line — never an error.
DEFAULT_STATUS_LINE_TIMEOUT = 5.0


class StatusLineModel(BaseModel):
    """The ``model`` object inside the status JSON (Claude Code shape)."""

    id: str = ""
    display_name: str = Field(default="", serialization_alias="display_name")


class StatusLineWorkspace(BaseModel):
    """The ``workspace`` object inside the status JSON (Claude Code shape).

    ``current_dir`` is where the turn ran; ``project_dir`` is the project root. Both
    are the workspace root here (Zak Code has one workspace per session), matching how
    Claude Code populates them for a single-root session.
    """

    current_dir: str = ""
    project_dir: str = ""


class StatusLineCost(BaseModel):
    """The ``cost`` object inside the status JSON.

    ``total_cost_usd`` / ``total_duration_ms`` are Claude Code's field names; the token
    fields are the cumulative-usage view a status script commonly renders. Cost and
    tokens are the session-cumulative totals (what ``/cost`` shows).
    """

    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class StatusLineInput(BaseModel):
    """The JSON document handed to a ``statusLine`` command on stdin.

    Field names and nesting match Claude Code's documented ``statusLine`` input shape so
    a CC status script reads it unmodified: top-level ``hook_event_name`` (always
    ``"Status"``), ``session_id``, ``cwd``, ``transcript_path``, ``version``, and the
    nested ``model`` / ``workspace`` / ``cost`` objects. Serialized with
    ``by_alias=True`` so the nested ``display_name`` alias is honored on the wire.
    """

    hook_event_name: str = "Status"
    session_id: str = ""
    cwd: str = ""
    transcript_path: str = ""
    version: str = ""
    model: StatusLineModel = Field(default_factory=StatusLineModel)
    workspace: StatusLineWorkspace = Field(default_factory=StatusLineWorkspace)
    cost: StatusLineCost = Field(default_factory=StatusLineCost)


class StatusLineSpec(BaseModel):
    """A configured ``statusLine`` command: the argv to run and how long to wait.

    ``command`` is an argv array (run WITHOUT a shell, like every settings.json shell
    command, so model-influenced text can't be injected); ``drop_env`` lists env-var
    names scrubbed from the child before spawn (provider-key hygiene).
    """

    command: list[str] = Field(..., description="argv array; run WITHOUT a shell.")
    timeout: float = DEFAULT_STATUS_LINE_TIMEOUT
    drop_env: list[str] = Field(default_factory=list)


def _split_command(cmd: str) -> list[str]:
    """Split a shell command string into an argv array (platform-aware).

    Mirrors :func:`zakcode.hooks.settings_loader._split_command`: ``argv[0]`` is resolved
    to a real executable path so a ``bash my-status.sh`` spec spawns the actual Git Bash,
    not the Windows WSL app-exec stub (see ``resolve_executable``).
    """
    argv = shlex.split(cmd, posix=False) if sys.platform == "win32" else shlex.split(cmd)
    if argv:
        argv[0] = resolve_executable(argv[0])
    return argv


def load_status_line_spec(
    workspace_root: Path,
    *,
    permission_mode: str = "ask",
) -> tuple[StatusLineSpec | None, str | None]:
    """Read the ``statusLine`` command from the workspace settings.json files.

    Discovery mirrors the hook loader: ``.claude/settings.json`` then
    ``.claude/settings.local.json`` (the per-machine local override wins, matching how
    Claude Code layers them). ``${CLAUDE_PROJECT_DIR}`` / ``$CLAUDE_PROJECT_DIR`` in the
    command is substituted with the workspace root (forward-slash form for Git Bash), and
    the command is scanned against the shared dangerous-pattern blocklist — a match is
    hard-denied in ``autonomous`` mode (no spec returned) and registered-with-a-warning
    otherwise, exactly like a settings.json hook.

    Returns ``(spec, error)``: the spec is ``None`` when no ``statusLine`` is configured,
    its type is not ``"command"``, the command is empty/unparseable, or it was denied.
    ``error`` is a human-readable reason when something was found but rejected, else
    ``None``. Never raises — a malformed settings file yields ``(None, reason)``.
    """
    from zakcode.permissions import scan_command_danger
    from zakcode.secrets import provider_key_env_names

    scrub_names = provider_key_env_names()
    project_dir = workspace_root.as_posix()

    # Read both files; the LATER one (settings.local.json) wins for a non-hook key like
    # statusLine — so a per-machine local override replaces the shared one.
    candidates = [
        workspace_root / ".claude" / "settings.json",
        workspace_root / ".claude" / "settings.local.json",
    ]
    raw_command: str | None = None
    command_type: str | None = None
    raw_timeout: object = None
    for settings_path in candidates:
        if not settings_path.is_file():
            continue
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return None, f"parse error in {settings_path}: {exc}"
        if not isinstance(data, dict):
            continue
        block = data.get("statusLine")
        if not isinstance(block, dict):
            continue
        # A later file's statusLine fully replaces an earlier one (override semantics).
        command_type = block.get("type") if isinstance(block.get("type"), str) else command_type
        cmd = block.get("command")
        if isinstance(cmd, str):
            raw_command = cmd
        if "timeout" in block:
            raw_timeout = block.get("timeout")

    if raw_command is None or not raw_command.strip():
        return None, None  # no statusLine configured (the common case)
    if command_type is not None and command_type != "command":
        return None, f"unsupported statusLine type: {command_type!r}"

    danger = scan_command_danger(raw_command)
    if danger:
        if permission_mode == "autonomous":
            logger.warning("statusLine command denied (autonomous): %s", danger)
            return None, f"DENIED in autonomous mode: {danger}"
        logger.warning(
            "statusLine command: dangerous pattern (%s), registering with warning", danger
        )
        # Fall through to register with a warning (parity with the hook loader).

    substituted = raw_command.replace("${CLAUDE_PROJECT_DIR}", project_dir).replace(
        "$CLAUDE_PROJECT_DIR", project_dir
    )
    try:
        argv = _split_command(substituted)
    except ValueError as exc:
        return None, f"command parse error: {exc}"
    if not argv:
        return None, "empty command"

    timeout = (
        float(raw_timeout) if isinstance(raw_timeout, (int, float)) else DEFAULT_STATUS_LINE_TIMEOUT
    )
    error = f"WARNING: {danger}" if danger else None
    return StatusLineSpec(command=argv, timeout=timeout, drop_env=scrub_names), error


def build_status_input(
    *,
    session_id: str,
    cwd: str,
    model_id: str,
    cost_usd: float,
    total_tokens: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    duration_ms: int = 0,
    transcript_path: str = "",
    version: str = "",
    display_name: str = "",
) -> StatusLineInput:
    """Assemble the :class:`StatusLineInput` for one status-line invocation.

    The caller (the CLI, after a turn) supplies the session/usage/model/cwd snapshot;
    this packs it into the CC-shaped document. ``display_name`` defaults to ``model_id``
    when not given (Claude Code shows a friendly name there). ``cwd`` populates both the
    top-level field and the nested ``workspace`` (one workspace root per session).
    """
    return StatusLineInput(
        session_id=session_id,
        cwd=cwd,
        transcript_path=transcript_path,
        version=version,
        model=StatusLineModel(id=model_id, display_name=display_name or model_id),
        workspace=StatusLineWorkspace(current_dir=cwd, project_dir=cwd),
        cost=StatusLineCost(
            total_cost_usd=cost_usd,
            total_duration_ms=duration_ms,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


async def render_status_line(spec: StatusLineSpec, status: StatusLineInput) -> str | None:
    """Run the status-line command and return its rendered line, or ``None``.

    Spawns ``spec.command`` as an argv array (never a shell), feeds ``status`` as JSON on
    stdin, scrubs provider keys from the child env, and bounds it with ``spec.timeout`` —
    killing the whole process tree on timeout. On success the command's stdout is reduced
    to its **first non-empty line, trimmed** (the status strip is one line).

    FAIL-SAFE — the whole point of this function: a missing command, a spawn failure, a
    non-zero exit, a timeout, empty/blank stdout, or ANY exception yields ``None`` (no
    status line). It never raises, so a turn can call it without a guard and a broken
    status script can never break or stall the turn. (``asyncio.CancelledError`` is the
    one thing it re-raises — a cancelled turn must unwind, after killing the child.)
    """
    if not spec.command:
        return None
    cwd = status.cwd
    stdin_bytes = status.model_dump_json(by_alias=True).encode("utf-8")
    child_env = _status_env(spec.drop_env, cwd)

    try:
        proc = await asyncio.create_subprocess_exec(
            *spec.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd if cwd and Path(cwd).is_dir() else None,
            env=child_env,
            **new_group_kwargs(),  # own process group so the whole tree is killable
        )
    except (OSError, ValueError) as exc:
        logger.warning("statusLine command %r failed to start: %s", spec.command, exc)
        return None

    try:
        stdout, _stderr = await asyncio.wait_for(
            proc.communicate(stdin_bytes), timeout=spec.timeout
        )
    except (TimeoutError, asyncio.CancelledError) as exc:
        # Kill the whole child tree (not just the parent) on timeout AND on turn
        # cancellation — CancelledError is a BaseException the bare ``except Exception``
        # below would not catch, so without this a cancelled turn could orphan the child.
        await terminate_process_tree(proc)
        if isinstance(exc, asyncio.CancelledError):
            raise
        logger.warning("statusLine command %r timed out after %ss", spec.command, spec.timeout)
        return None
    except Exception as exc:  # noqa: BLE001 — a status line must never propagate a failure
        logger.warning("statusLine command %r errored: %s", spec.command, exc)
        return None

    if proc.returncode != 0:
        # A non-zero exit means "no line", not an error to surface — fail quiet.
        return None
    text = (stdout or b"").decode("utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _status_env(drop: list[str], cwd: str) -> dict[str, str]:
    """Child env for the status command: ``os.environ`` minus *drop*, plus
    ``CLAUDE_PROJECT_DIR`` set to the workspace root (forward-slash form for Git Bash).

    Mirrors :func:`zakcode.hooks._hook_env` — a status script written for Claude Code
    resolves its own paths through ``CLAUDE_PROJECT_DIR`` the same way a hook does, and
    provider API keys are scrubbed so the cosmetic subprocess can't read model credentials.
    """
    import os

    env = dict(os.environ)
    for name in drop:
        env.pop(name, None)
    if cwd:
        env["CLAUDE_PROJECT_DIR"] = cwd.replace("\\", "/")
    return env


__all__ = [
    "DEFAULT_STATUS_LINE_TIMEOUT",
    "StatusLineCost",
    "StatusLineInput",
    "StatusLineModel",
    "StatusLineSpec",
    "StatusLineWorkspace",
    "build_status_input",
    "load_status_line_spec",
    "render_status_line",
]

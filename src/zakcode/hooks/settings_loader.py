"""Load hook specs from workspace ``settings.json`` files (PR-T5).

Discovery order:
1. ``<workspace>/.claude/settings.json``
2. ``<workspace>/.zakcode/settings.json``

Both are read; hooks from both are merged into a single list.

Event-name mapping: Claude Code's ``"Stop"`` event IS the ``TURN_END`` seam.
Unknown or unimplemented events (``StopFailure``, ``UserPromptExpansion``) are
skipped with a warning in the returned errors dict.

Security: every command string is scanned against ``DANGEROUS_PATTERNS`` before
registration. In autonomous mode a match is hard-denied (hook not registered);
in other modes the hook is registered with a warning.

TE-R1: every returned ``HookSpec`` carries ``drop_env`` populated with
:func:`~zakcode.secrets.provider_key_env_names`, so the generic shell runners
scrub provider keys from ALL workspace-sourced hooks — not just TURN_END.
"""

from __future__ import annotations

import json
import logging
import shlex
import sys
from pathlib import Path

from zakcode.hooks import DEFAULT_HOOK_TIMEOUT, HookEvent, HookSpec

logger = logging.getLogger("zakcode.hooks.settings_loader")

#: Claude Code settings.json event names -> HookEvent.  ``"Stop"`` maps to
#: ``TURN_END`` (the seam Mind's stop-hook.sh targets).
_EVENT_MAP: dict[str, HookEvent] = {
    "PreToolUse": HookEvent.PRE_TOOL_USE,
    "PostToolUse": HookEvent.POST_TOOL_USE,
    "PreLLMCall": HookEvent.PRE_LLM_CALL,
    "SessionStart": HookEvent.SESSION_START,
    "SessionEnd": HookEvent.SESSION_END,
    "PreCompact": HookEvent.PRE_COMPACT,
    "OnSkillSelected": HookEvent.ON_SKILL_SELECTED,
    "Stop": HookEvent.TURN_END,
}

#: Events recognised but not yet implemented — skipped with a warning.
_SKIP_EVENTS: set[str] = {"StopFailure", "UserPromptExpansion"}


def _split_command(cmd: str) -> list[str]:
    """Split a shell command string into an argv array (platform-aware)."""
    if sys.platform == "win32":
        return shlex.split(cmd, posix=False)
    return shlex.split(cmd)


def _is_dangerous(command_str: str) -> str | None:
    """Return the reason if *command_str* matches a DANGEROUS_PATTERN, else None."""
    from zakcode.permissions import DANGEROUS_PATTERNS

    for pattern, reason in DANGEROUS_PATTERNS:
        if pattern.search(command_str):
            return reason
    return None


def load_settings_hooks(
    workspace_root: Path,
    *,
    permission_mode: str = "ask",
) -> tuple[list[HookSpec], dict[str, str]]:
    """Load hook specs from workspace settings.json files.

    Args:
        workspace_root: The project workspace root.
        permission_mode: Current permission mode string (e.g. ``"autonomous"``,
            ``"ask"``).  In ``"autonomous"`` mode, dangerous commands are
            hard-denied rather than warned.

    Returns:
        ``(specs, errors)`` where *errors* maps ``"<event>/<index>"`` to a
        human-readable reason the spec was skipped or flagged.
    """
    specs: list[HookSpec] = []
    errors: dict[str, str] = {}

    # Lazy import so the module import is cheap when unused.
    from zakcode.secrets import provider_key_env_names

    scrub_names = provider_key_env_names()

    candidates = [
        workspace_root / ".claude" / "settings.json",
        workspace_root / ".zakcode" / "settings.json",
    ]
    for settings_path in candidates:
        if not settings_path.is_file():
            continue
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors[str(settings_path)] = f"parse error: {exc}"
            continue
        if not isinstance(data, dict):
            continue
        hooks_block = data.get("hooks")
        if not isinstance(hooks_block, dict):
            continue
        for event_name, entries in hooks_block.items():
            if event_name in _SKIP_EVENTS:
                errors[event_name] = f"event not implemented: {event_name}"
                continue
            hook_event = _EVENT_MAP.get(event_name)
            if hook_event is None:
                errors[event_name] = f"unknown event: {event_name}"
                continue
            if not isinstance(entries, list):
                continue
            for i, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                matcher = entry.get("matcher", "*")
                if not isinstance(matcher, str):
                    matcher = "*"
                inner_hooks = entry.get("hooks")
                if not isinstance(inner_hooks, list):
                    continue
                for j, hook_def in enumerate(inner_hooks):
                    if not isinstance(hook_def, dict):
                        continue
                    if hook_def.get("type") != "command":
                        continue
                    command_str = hook_def.get("command")
                    if not isinstance(command_str, str) or not command_str.strip():
                        continue
                    timeout = hook_def.get("timeout", DEFAULT_HOOK_TIMEOUT)
                    if not isinstance(timeout, (int, float)):
                        timeout = DEFAULT_HOOK_TIMEOUT

                    # Security gate: DANGEROUS_PATTERNS scan.
                    danger = _is_dangerous(command_str)
                    key = f"{event_name}/{i}/{j}"
                    if danger:
                        if permission_mode == "autonomous":
                            errors[key] = (
                                f"DENIED in autonomous mode: {danger}"
                            )
                            logger.warning(
                                "settings.json hook %s denied (autonomous): %s",
                                key,
                                danger,
                            )
                            continue  # Do not register.
                        else:
                            errors[key] = f"WARNING: {danger}"
                            logger.warning(
                                "settings.json hook %s: dangerous pattern (%s), "
                                "registering with warning",
                                key,
                                danger,
                            )
                            # Fall through to register with warning.

                    try:
                        argv = _split_command(command_str)
                    except ValueError as exc:
                        errors[key] = f"command parse error: {exc}"
                        continue
                    if not argv:
                        continue

                    specs.append(
                        HookSpec(
                            event=hook_event,
                            command=argv,
                            matcher=matcher,
                            timeout=float(timeout),
                            drop_env=scrub_names,
                        )
                    )

    return specs, errors

"""Per-workspace adoption decisions for Claude-Code compatibility surfaces (folder trust).

The compatibility surfaces are OFF by default at the library layer (see
``docs/INTEGRATIONS.md``): a workspace carrying *another* runtime's ``.claude/`` config must
never change Zak Code's behavior un-opted-in. But hooks are shell commands, and an operator
sitting in a Claude-Code-shaped workspace almost always wants them — Claude Code answers the
same tension with a one-time folder-trust decision. This module is that answer for Zak Code:

- the DECISION LOGIC (:func:`resolve_hooks_adoption`) and the PERSISTENCE
  (``~/.zakcode/workspace-trust.json``) live here in the core;
- an interactive client (the CLI) supplies only the ``ask`` callback and renders the
  outcome — UI in the interface, policy in the core;
- non-interactive hosts (the server, ``chat -p`` one-shots, embedders) never prompt: the
  stored decision or the explicit ``ZAKCODE_SETTINGS_HOOKS`` setting is all they honor, so
  headless behavior stays deterministic.

The store schema is ``{"<resolved workspace path>": {"settings_hooks": "always" | "never"}}``
— keyed per surface so future surfaces (permissions, statusLine, output-styles) can join the
same file without re-asking what was already answered. Reads fail open to "no decision"
(corrupt/missing file ⇒ ask again); writes are best-effort (a read-only home dir degrades to
asking every session, never to a crash).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("zakcode.workspace_trust")

#: The three adoption answers an interactive ``ask`` may return (None = dismissed/EOF).
ADOPT_ALWAYS = "always"
ADOPT_SESSION = "session"
ADOPT_NEVER = "never"


def _store_path() -> Path:
    """The trust store file (computed per call so a test's HOME monkeypatch is honored)."""
    return Path.home() / ".zakcode" / "workspace-trust.json"


def _key(workspace_root: Path | str) -> str:
    return str(Path(workspace_root).resolve())


def _read_store(store_path: Path) -> dict[str, dict[str, str]]:
    try:
        data = json.loads(store_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, dict)
    }  # tolerate foreign keys; never crash on them


def hooks_decision(workspace_root: Path | str, *, store_path: Path | None = None) -> str | None:
    """The remembered hooks decision for this workspace: ``"always"``, ``"never"``, or None.

    Fail-open: a missing, unreadable, or corrupt store reads as "no decision" (ask again) —
    never as a grant.
    """
    store = _read_store(store_path or _store_path())
    value = store.get(_key(workspace_root), {}).get("settings_hooks")
    return value if value in (ADOPT_ALWAYS, ADOPT_NEVER) else None


def remember_hooks_decision(
    workspace_root: Path | str, decision: str, *, store_path: Path | None = None
) -> None:
    """Persist ``"always"`` / ``"never"`` for this workspace. Best-effort: a write failure
    logs and degrades to asking again next session — it never raises into startup."""
    if decision not in (ADOPT_ALWAYS, ADOPT_NEVER):  # "session" is deliberately not stored
        return
    path = store_path or _store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        store = _read_store(path)
        entry = store.setdefault(_key(workspace_root), {})
        entry["settings_hooks"] = decision
        path.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("could not persist workspace trust decision: %s", exc)


@dataclass(frozen=True)
class HooksAdoption:
    """What a host should do about workspace hooks this session."""

    enable: bool  # pass enable_settings_hooks=True to the Agent
    remember: str | None  # "always" | "never" → persist via remember_hooks_decision
    notice: str | None  # one line for the operator; None = stay silent


def _label(summary: dict[str, int]) -> str:
    return ", ".join(f"{name} x{count}" for name, count in summary.items())


def resolve_hooks_adoption(
    *,
    configured: bool | None,
    summary: dict[str, int],
    decision: str | None,
    interactive: bool,
    ask: Callable[[dict[str, int]], str | None],
) -> HooksAdoption:
    """Decide whether to load a workspace's settings.json hooks — the folder-trust policy.

    Args:
        configured: the tri-state ``Settings.settings_hooks`` (env/.env). An explicit
            True/False is the operator's standing answer and short-circuits everything.
        summary: :func:`~zakcode.hooks.settings_loader.summarize_settings_hooks` output —
            ``{}`` means the workspace declares nothing loadable, so there is nothing to ask.
        decision: the remembered per-workspace answer (:func:`hooks_decision`).
        interactive: True only when a human can answer right now (a REPL on a tty). When
            False the unset state resolves to OFF with a pointer, never a hang or a surprise.
        ask: renders the question and returns ``"always"``/``"session"``/``"never"``
            (None = dismissed → off, nothing persisted). Called at most once, and only when
            ``configured`` is unset, ``summary`` is non-empty, ``decision`` is None, and
            ``interactive`` is True.
    """
    if configured is not None:
        # Explicitly set (env or .env): honored globally, silently — same as today.
        return HooksAdoption(enable=bool(configured), remember=None, notice=None)
    if not summary:
        return HooksAdoption(enable=False, remember=None, notice=None)
    label = _label(summary)
    if decision == ADOPT_ALWAYS:
        return HooksAdoption(
            enable=True,
            remember=None,
            notice=f"workspace hooks loaded ({label}) - trusted workspace",
        )
    if decision == ADOPT_NEVER:
        return HooksAdoption(
            enable=False,
            remember=None,
            notice=(
                f"workspace hooks present but off for this workspace ({label}) - "
                "re-decide by editing ~/.zakcode/workspace-trust.json"
            ),
        )
    if not interactive:
        return HooksAdoption(
            enable=False,
            remember=None,
            notice=(
                f"workspace defines hooks ({label}) that are NOT loaded - set "
                "ZAKCODE_SETTINGS_HOOKS=1 for headless runs, or run zakcode chat "
                "here once to decide"
            ),
        )
    answer = ask(summary)
    if answer == ADOPT_ALWAYS:
        return HooksAdoption(
            enable=True,
            remember=ADOPT_ALWAYS,
            notice=f"workspace hooks loaded ({label}) - remembered for this workspace",
        )
    if answer == ADOPT_SESSION:
        return HooksAdoption(
            enable=True,
            remember=None,
            notice=f"workspace hooks loaded ({label}) - this session only",
        )
    if answer == ADOPT_NEVER:
        return HooksAdoption(
            enable=False,
            remember=ADOPT_NEVER,
            notice="workspace hooks stay off for this workspace - remembered",
        )
    return HooksAdoption(enable=False, remember=None, notice="workspace hooks not loaded")

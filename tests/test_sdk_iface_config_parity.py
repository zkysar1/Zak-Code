"""SDK ⇄ interface CONFIG parity — does each interface build the RIGHT agent?

The sibling of ``tests/test_sdk_iface_parity.py``. That file pins TRANSPORT
parity (each interface relays the SDK's event stream faithfully). This file pins
CONFIG parity: each interface constructs the :class:`~zakcode.Agent` with its
INTENDED capability posture, and the posture DELTA between interfaces is exactly
the documented, intentional set — nothing accidentally added or dropped.

THE DELIBERATE ASYMMETRY (not a bug to converge)
------------------------------------------------
``zakcode serve`` (:func:`~zakcode.server.app._default_agent_factory`) builds a
FEATURE-REDUCED agent — **skills + rules only** — because sub-agents / MCP /
plugins / compaction are a separate posture decision, out of scope for the
connection substrate (see ``app.py``). The CLI
(:func:`~zakcode.cli._build_chat_agent`) builds the FULL interactive agent. This
test makes that split EXPLICIT and regression-proof: if someone flips
``enable_mcp`` on in the server factory, drops ``enable_compaction`` from the
CLI, or an Agent-constructor DEFAULT drifts, a named assertion fails.

HOW
---
Spy on ``Agent.__init__`` to capture the kwargs each interface REQUESTS — the
config decision is the interface's job; constructing the agent is the SDK's.
Effective posture resolves each flag against the Agent constructor's real
signature default (:data:`_AGENT_SIG`), so a changed DEFAULT is caught too, not
just an edited call site.

HERMETIC
--------
``Agent.__init__`` is replaced by a recorder (no real agent, no provider /
litellm import); the CLI builder's bare ``SessionStore()`` — which eagerly
mkdirs ``~/.zakcode`` — is redirected to a tmp dir.

NOT COVERED (by design — distinct concerns)
-------------------------------------------
Whether the SHARED, settings-driven posture (permission mode, workspace root,
model routing) matches: both interfaces read the same ``Settings``, so that is
same-by-construction, not a per-interface hardcoded decision this test could
meaningfully pin. And the permission PROMPTER differs on purpose (console vs WS
bridge vs fail-closed headless) — a UI transport, not a capability.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest import mock

from zakcode import Agent
from zakcode.cli import _build_chat_agent
from zakcode.config import Settings
from zakcode.server.app import _default_agent_factory
from zakcode.session.store import Session, SessionStore

# The capability flags whose per-interface posture this test pins. Order-free;
# compared as sets / dicts below.
_FEATURE_FLAGS: tuple[str, ...] = (
    "enable_skills",
    "enable_rules",
    "enable_subagents",
    "enable_mcp",
    "enable_plugins",
    "enable_compaction",
)

# The REAL Agent-constructor signature, captured at import time (before any spy
# patches it), so effective posture can fall back to the true default of a flag
# an interface does not pass — making default-drift a caught failure, not a
# silent posture change.
_AGENT_SIG = inspect.signature(Agent.__init__)

# The intended postures. `serve` is feature-reduced; the CLI is full.
_SERVER_EXPECTED: dict[str, bool] = {
    "enable_skills": True,
    "enable_rules": True,
    "enable_subagents": False,
    "enable_mcp": False,
    "enable_plugins": False,
    "enable_compaction": False,
}
_CLI_EXPECTED: dict[str, bool] = dict.fromkeys(_FEATURE_FLAGS, True)

# The documented, intentional split (used by the delta test so the asymmetry is
# asserted as a contract, not just implied by the two posture dicts).
_INTENDED_CLI_ONLY: set[str] = {
    "enable_subagents",
    "enable_mcp",
    "enable_plugins",
    "enable_compaction",
}
_SHARED_ENABLED: set[str] = {"enable_skills", "enable_rules"}


# ── capture + posture helpers ─────────────────────────────────────────────────────


def _capture_agent_kwargs(build: Callable[[], object]) -> dict[str, Any]:
    """Run ``build`` with ``Agent.__init__`` replaced by a recorder; return its kwargs.

    The real ``__init__`` never runs, so no provider is built and litellm is not
    imported — the interface's *config request* is captured without constructing
    an agent.
    """
    captured: dict[str, Any] = {}

    def _spy(_self: object, *_args: object, **kwargs: object) -> None:
        captured.update(kwargs)

    with mock.patch.object(Agent, "__init__", _spy):
        build()
    return captured


def _posture(kwargs: dict[str, Any]) -> dict[str, bool]:
    """The effective on/off state of every feature flag for one interface.

    A flag the interface passes wins; otherwise the Agent constructor's real
    default applies — so flipping a default flips the posture here too.
    """
    return {
        flag: bool(kwargs[flag] if flag in kwargs else _AGENT_SIG.parameters[flag].default)
        for flag in _FEATURE_FLAGS
    }


def _server_kwargs(tmp_path: Path) -> dict[str, Any]:
    """The kwargs ``zakcode serve``'s default factory passes to build a turn's agent."""
    settings = Settings(default_model="scripted/parity", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "server-sessions")
    factory = _default_agent_factory(settings, store)
    session = Session(cwd=".", model="scripted/parity")
    return _capture_agent_kwargs(lambda: factory(session, None, None))


def _cli_kwargs(tmp_path: Path) -> dict[str, Any]:
    """The kwargs the CLI's ``_build_chat_agent`` passes for an interactive session."""
    overrides: dict[str, Any] = {}
    # Redirect the builder's bare SessionStore() (eager ~/.zakcode mkdir) to tmp.
    # _build_chat_agent imports SessionStore at call time, so patching the module
    # attribute takes effect; the factory calls the real class (bound here) with a
    # tmp base_dir, so there is no recursion.
    real_store = SessionStore

    def _tmp_store(*_a: object, **_k: object) -> SessionStore:
        return real_store(base_dir=tmp_path / "cli-sessions")

    with mock.patch("zakcode.session.store.SessionStore", _tmp_store):
        return _capture_agent_kwargs(lambda: _build_chat_agent(cast(Any, mock.Mock()), overrides))


# ── tests ─────────────────────────────────────────────────────────────────────────


def test_server_agent_is_feature_reduced(tmp_path: Path) -> None:
    """`zakcode serve` builds skills+rules only — the connection-substrate posture."""
    assert _posture(_server_kwargs(tmp_path)) == _SERVER_EXPECTED


def test_cli_agent_is_full_featured(tmp_path: Path) -> None:
    """The CLI builds the full interactive agent (subagents/mcp/plugins/compaction on)."""
    assert _posture(_cli_kwargs(tmp_path)) == _CLI_EXPECTED


def test_config_delta_is_the_documented_intentional_set(tmp_path: Path) -> None:
    """The posture DELTA is exactly the documented split — same base, intended extras.

    This is the config-parity statement: the interfaces agree on ``skills``+``rules``
    and differ ONLY by the four features the CLI adds. A flag drifting into the
    server, or out of the CLI, breaks exactly this assertion.
    """
    server = _posture(_server_kwargs(tmp_path))
    cli = _posture(_cli_kwargs(tmp_path))

    cli_only = {f for f in _FEATURE_FLAGS if cli[f] and not server[f]}
    shared_enabled = {f for f in _FEATURE_FLAGS if cli[f] and server[f]}
    server_only = {f for f in _FEATURE_FLAGS if server[f] and not cli[f]}

    assert cli_only == _INTENDED_CLI_ONLY
    assert shared_enabled == _SHARED_ENABLED
    # The served (customer-facing) agent must never enable a capability the
    # interactive CLI itself does not — that would be a posture inversion.
    assert server_only == set()

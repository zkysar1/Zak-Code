"""SDK ⇄ interface PERMISSION-ESCALATION parity — the bidirectional round-trip.

The third axis, alongside ``test_sdk_iface_parity.py`` (transport) and
``test_sdk_iface_config_parity.py`` (config). Those pin one-way relays; this pins
the ONE place an interface talks BACK to the SDK mid-turn.

THE CLAIM
---------
When a tool call escalates (ASK mode, a write-tier tool), the SDK asks its
:class:`~zakcode.permissions.PermissionPrompter` for a decision. Over WebSocket
that becomes a round-trip: the server emits a ``WSActionRequired`` control frame,
the client answers with a ``WSApproval``, and the server hands the decision back
to the SDK's prompter. Parity means: **given the same client decision, the SDK
produces the SAME event stream it would with a direct in-process prompter** —
i.e. the WS bridge delivers the outcome faithfully, and serializes the request
faithfully on the way out.

THE DESIGN
----------
One escalating scenario (a ``write_file`` under ASK mode → the SDK prompts),
run for each outcome through two layers that share ONE agent builder
(:func:`_build_escalation_agent`, ASK mode so the tool escalates):

* ``sdk`` — the agent with a direct :class:`_ScriptedPrompter` that records the
  request and returns a fixed outcome.
* ``ws``  — the SAME agent behind ``/ws``; the client answers the
  ``action_required`` frame with the SAME outcome and drains the resumed events.

The break point: if the WS bridge mis-serializes the request or mis-delivers the
decision, ``ws`` diverges from ``sdk`` for that outcome.

WHY BOTH allow AND deny
-----------------------
They produce DIFFERENT tool results (allow → the write runs; deny → it is
blocked), so per-outcome parity is a real constraint — a bridge that dropped the
client's decision (always-deny, say) would match one outcome and FAIL the other.
:func:`test_outcome_actually_changes_the_result` pins that the outcome matters.

HERMETIC: ``ScriptedProvider`` (no network); ``TestClient`` drives ``/ws``
in-process; the escalating tool writes into a tmp workspace.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from zakcode import Agent
from zakcode.config import Settings
from zakcode.evals.harness import ScriptedProvider, call_tool, reply
from zakcode.events import (
    AgentDone,
    AgentEvent,
    AgentStatus,
    AgentTaskUpdate,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.permissions import (
    PermissionMode,
    PermissionOutcome,
    PermissionPolicy,
    PermissionPrompter,
    PermissionRequest,
)
from zakcode.server.app import create_app
from zakcode.server.wire import event_from_dict
from zakcode.session.store import Session, SessionStore

_CANONICAL_INPUT = "__canonical permission input__"

# The fixed escalation scenario: one WORKSPACE_WRITE tool call, then a reply.
# ``write_file`` needs write permission, so ASK mode routes it to prompter.confirm
# before it can run — the escalation this axis exists to test.
_SCRIPT = (
    call_tool("write_file", {"path": "guarded.txt", "content": "hi"}, id="w1"),
    reply("Handled."),
)

# The outcomes exercised. Both are "_once" (no session memory to reason about for a
# single escalation). allow → the tool runs; deny → it is blocked.
_OUTCOMES = [PermissionOutcome.ALLOW_ONCE, PermissionOutcome.DENY_ONCE]


# ── normalization: the client-observable, transport-stable projection ─────────────
# (kept identical to the sibling transport-parity file's projection; a local copy
# so this axis is self-contained.)


def _normalize(events: list[AgentEvent]) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    text_buf: list[str] = []

    def flush() -> None:
        if text_buf:
            out.append(("text", "".join(text_buf)))
            text_buf.clear()

    for ev in events:
        if isinstance(ev, AgentTextDelta):
            text_buf.append(ev.text)
            continue
        flush()
        if isinstance(ev, AgentToolCall):
            out.append(("tool_call", ev.id, ev.name, ev.arguments))
        elif isinstance(ev, AgentToolResult):
            out.append(("tool_result", ev.tool_use_id, ev.is_error, ev.output))
        elif isinstance(ev, AgentStatus):
            out.append(("status", ev.message))
        elif isinstance(ev, AgentTaskUpdate):
            out.append(("task_update", ev.plan, ev.finished, ev.total, ev.complete))
        elif isinstance(ev, AgentUsage):
            out.append(("usage",))
        elif isinstance(ev, AgentDone):
            out.append(
                (
                    "done",
                    ev.stop_reason,
                    ev.iterations,
                    ev.degraded,
                    ev.error,
                    ev.routed_category,
                    ev.routed_escalated,
                )
            )
        else:  # pragma: no cover — a new AgentEvent variant must extend this map.
            raise AssertionError("_normalize reached an unhandled AgentEvent variant")
    flush()
    return out


# ── the direct prompter + the one agent builder both layers share ─────────────────


class _ScriptedPrompter:
    """A direct in-process prompter: records each request, returns a fixed outcome.

    Structurally a :class:`~zakcode.permissions.PermissionPrompter` (runtime-checkable
    Protocol) — the ``sdk`` layer's stand-in for the WS bridge.
    """

    def __init__(self, outcome: PermissionOutcome) -> None:
        self.outcome = outcome
        self.requests: list[PermissionRequest] = []

    async def confirm(self, request: PermissionRequest) -> PermissionOutcome:
        self.requests.append(request)
        return self.outcome


def _build_escalation_agent(
    prompter: PermissionPrompter | None,
    *,
    workspace_root: Path,
    session: Session | None = None,
) -> Agent:
    """The SAME agent both layers use. ASK mode makes ``write_file`` escalate to
    ``prompter`` — an injected policy is the agent's full permission authority, so
    the policy's prompter (direct or WS bridge) is exactly what the SDK consults.
    """
    return Agent(
        provider=ScriptedProvider(list(_SCRIPT)),
        session=session,
        permission_policy=PermissionPolicy(PermissionMode.ASK, prompter=prompter),
        default_model="scripted/parity",
        workspace_root=str(workspace_root),
        max_iterations=8,
    )


# ── layer runners ──────────────────────────────────────────────────────────────────


def _run_sdk(
    outcome: PermissionOutcome, workspace_root: Path
) -> tuple[list[AgentEvent], list[PermissionRequest]]:
    """L0 — the direct prompter returns ``outcome``; return (events, requests seen)."""
    prompter = _ScriptedPrompter(outcome)

    async def go() -> list[AgentEvent]:
        agent = _build_escalation_agent(prompter, workspace_root=workspace_root)
        return [ev async for ev in agent.astream_turn(_CANONICAL_INPUT)]

    events = asyncio.run(go())
    return events, prompter.requests


def _run_ws(
    outcome: PermissionOutcome, workspace_root: Path
) -> tuple[list[AgentEvent], list[dict[str, Any]]]:
    """L1 — the WS round-trip: answer the ``action_required`` frame with ``outcome``.

    Returns (the resumed event stream, the ``action_required`` control frames seen).
    ``action_required`` is a control frame (``type``, not ``event``) delivered
    out-of-band; it is handled inline and filtered from the event stream, so the
    remaining events are exactly what the SDK produced — comparable to ``sdk``.
    """
    settings = Settings(default_model="scripted/parity", workspace_root=workspace_root)
    store = SessionStore(base_dir=workspace_root / "sessions")

    def factory(session: Session, model: str | None, prompter: PermissionPrompter | None) -> Agent:  # noqa: ARG001 — model unused; prompter IS the WS bridge (the point of this test)
        return _build_escalation_agent(prompter, workspace_root=workspace_root, session=session)

    app = create_app(settings=settings, store=store, agent_factory=factory)
    client = TestClient(app)
    session = Session(cwd=".", model="scripted/parity")
    store.save(session)

    event_frames: list[dict[str, Any]] = []
    action_frames: list[dict[str, Any]] = []
    with client.websocket_connect(f"/ws/{session.id}") as ws:
        ws.send_json({"type": "input", "message": _CANONICAL_INPUT})
        while True:
            frame = ws.receive_json()
            if frame.get("type") == "action_required":
                action_frames.append(frame)
                ws.send_json({"type": "approval", "outcome": outcome.value})
                continue
            if "event" not in frame:  # an error frame here is a real divergence
                raise AssertionError(f"WS sent an unexpected control frame: {frame}")
            event_frames.append(frame)
            if frame["event"] == "done":
                break
    return [event_from_dict(f) for f in event_frames], action_frames


# ── tests ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("outcome", _OUTCOMES, ids=lambda o: o.value)
def test_ws_escalation_outcome_matches_sdk(outcome: PermissionOutcome, tmp_path: Path) -> None:
    """The WS round-trip delivers the client's decision faithfully.

    For the same outcome, the resumed WS event stream equals the direct-prompter
    stream. ``[..-deny]`` green while ``[..-allow]`` red (or vice-versa) would mean
    the bridge dropped or fixed the decision instead of relaying it.
    """
    sdk_events, _ = _run_sdk(outcome, tmp_path / "sdk")
    ws_events, _ = _run_ws(outcome, tmp_path / "ws")
    assert _normalize(ws_events) == _normalize(sdk_events)


@pytest.mark.parametrize("outcome", _OUTCOMES, ids=lambda o: o.value)
def test_ws_action_required_frame_is_faithful(outcome: PermissionOutcome, tmp_path: Path) -> None:
    """The outbound half: the ``action_required`` frame carries the SDK's request.

    Exactly one escalation happens, and the frame's tool/args/tier match the
    ``PermissionRequest`` the direct prompter recorded for the same run.
    """
    _, sdk_requests = _run_sdk(outcome, tmp_path / "sdk")
    _, ws_actions = _run_ws(outcome, tmp_path / "ws")

    assert len(sdk_requests) == 1
    assert len(ws_actions) == 1
    request = sdk_requests[0]
    frame = ws_actions[0]
    assert frame["tool_name"] == request.tool_name
    assert frame["arguments"] == request.arguments
    assert frame["tier"] == request.tier.name


def test_outcome_actually_changes_the_result(tmp_path: Path) -> None:
    """Guard: allow and deny produce DIFFERENT streams.

    Without this, per-outcome parity could pass trivially if the decision were
    ignored — this pins that the outcome genuinely drives the turn, so the parity
    assertions above are real constraints.
    """
    allow_events, _ = _run_sdk(PermissionOutcome.ALLOW_ONCE, tmp_path / "allow")
    deny_events, _ = _run_sdk(PermissionOutcome.DENY_ONCE, tmp_path / "deny")
    assert _normalize(allow_events) != _normalize(deny_events)

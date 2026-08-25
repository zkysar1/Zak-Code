"""SDK ⇄ interface PERMISSION-ESCALATION parity — the bidirectional round-trip.

The third axis, alongside ``test_sdk_iface_parity.py`` (transport) and
``test_sdk_iface_config_parity.py`` (config). Those pin one-way relays; this pins
the ONE place an interface talks BACK to the SDK mid-turn.

THE CLAIM
---------
When a tool call escalates (ASK mode, a write-tier tool), the SDK asks its
:class:`~zakcode.permissions.PermissionPrompter` for a decision. On the server
that is the say-inbox round-trip: ``SayInboxPrompter`` announces a
``WSActionRequired`` frame on the watch bus and polls the workspace say inbox
for a y/a/n answer — the ONE contract every surface writes. Parity means:
**given the same answer through the inbox, the SDK produces the SAME event
stream it would with a direct in-process prompter** — i.e. the say bridge
delivers the outcome faithfully, and announces the request faithfully on the
way out.

THE DESIGN
----------
One escalating scenario (a ``write_file`` under ASK mode → the SDK prompts),
run for each outcome through two layers that share ONE agent builder
(:func:`_build_escalation_agent`, ASK mode so the tool escalates):

* ``sdk`` — the agent with a direct :class:`_ScriptedPrompter` that records the
  request and returns a fixed outcome.
* ``say`` — the SAME agent with a ``SayInboxPrompter``; the announce callback
  answers the ``action_required`` frame by writing the SAME outcome into the
  workspace say inbox, exactly as a cockpit box / ``zakcode say`` / ``POST /say``
  user would.

The break point: if the say bridge mis-announces the request or mis-delivers the
decision, ``say`` diverges from ``sdk`` for that outcome.

WHY BOTH allow AND deny
-----------------------
They produce DIFFERENT tool results (allow → the write runs; deny → it is
blocked), so per-outcome parity is a real constraint — a bridge that dropped the
client's decision (always-deny, say) would match one outcome and FAIL the other.
:func:`test_outcome_actually_changes_the_result` pins that the outcome matters.

HERMETIC: ``ScriptedProvider`` (no network); the inbox is a tmp-path file; the
escalating tool writes into a tmp workspace.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from zakcode import Agent
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
from zakcode.server.app import SayInboxPrompter
from zakcode.session.say_inbox import say_path, write_say
from zakcode.session.store import Session

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


_ANSWER_TEXT = {PermissionOutcome.ALLOW_ONCE: "y", PermissionOutcome.DENY_ONCE: "n"}


def _run_say(
    outcome: PermissionOutcome, workspace_root: Path
) -> tuple[list[AgentEvent], list[dict[str, Any]]]:
    """L1 — the say-inbox round-trip: answer the announced frame through the inbox.

    The prompter's ``publish`` callback stands in for a watching user: the moment
    the ``action_required`` frame is announced, the answer is written into the
    workspace say inbox (the same file every surface writes), and the prompter's
    poll picks it up.
    """
    workspace_root.mkdir(parents=True, exist_ok=True)
    action_frames: list[dict[str, Any]] = []

    def announce(frame: dict[str, Any]) -> None:
        action_frames.append(frame)
        assert write_say(say_path(workspace_root), _ANSWER_TEXT[outcome])

    prompter = SayInboxPrompter(workspace_root, publish=announce, poll=0.02, timeout=5.0)
    session = Session(cwd=".", model="scripted/parity")

    async def go() -> list[AgentEvent]:
        agent = _build_escalation_agent(prompter, workspace_root=workspace_root, session=session)
        return [ev async for ev in agent.astream_turn(_CANONICAL_INPUT)]

    return asyncio.run(go()), action_frames


# ── tests ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("outcome", _OUTCOMES, ids=lambda o: o.value)
def test_say_escalation_outcome_matches_sdk(outcome: PermissionOutcome, tmp_path: Path) -> None:
    """The say-inbox round-trip delivers the answer faithfully.

    For the same outcome, the say-bridge event stream equals the direct-prompter
    stream. ``[..-deny]`` green while ``[..-allow]`` red (or vice-versa) would mean
    the bridge dropped or fixed the decision instead of relaying it.
    """
    sdk_events, _ = _run_sdk(outcome, tmp_path / "sdk")
    say_events, _ = _run_say(outcome, tmp_path / "say")
    assert _normalize(say_events) == _normalize(sdk_events)


@pytest.mark.parametrize("outcome", _OUTCOMES, ids=lambda o: o.value)
def test_say_action_required_frame_is_faithful(outcome: PermissionOutcome, tmp_path: Path) -> None:
    """The outbound half: the announced frame carries the SDK's request.

    Exactly one escalation happens, and the frame's tool/args/tier match the
    ``PermissionRequest`` the direct prompter recorded for the same run.
    """
    _, sdk_requests = _run_sdk(outcome, tmp_path / "sdk")
    _, say_actions = _run_say(outcome, tmp_path / "say")

    assert len(sdk_requests) == 1
    assert len(say_actions) == 1
    request = sdk_requests[0]
    frame = say_actions[0]
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

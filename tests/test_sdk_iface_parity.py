"""SDK ⇄ interface parity — the break point between the canonical SDK event
stream and every end interface that relays it.

WHY THIS FILE EXISTS
--------------------
guard-4547: interfaces carry NO business logic. The SDK (:class:`zakcode.Agent`)
owns all of it and emits the one canonical stream of
:data:`~zakcode.events.AgentEvent`; every interface (HTTP/SSE, WebSocket, and the
CLI's remote client) is a thin transport that RELAYS that stream. This test pins
that contract with a demonstrable *break point*: it proves the interface
reproduces the SDK's stream, and when it does not, it says WHICH layer broke.

THE DESIGN
----------
A golden scenario is ``(canonical input, deterministic ScriptedProvider script,
expected normalized AgentEvent stream)``. One scenario runs through four layers:

* ``sdk``  — :meth:`Agent.astream_turn` called directly (the canonical stream).
* ``http`` — the SAME agent injected into the ASGI app and driven through
  ``POST /chat/stream``; its SSE frames are parsed back into ``AgentEvent`` via
  the wire adapter (:func:`~zakcode.server.wire.event_from_dict`).
* ``ws``   — the SAME agent driven through the ``/ws/{session_id}`` WebSocket;
  each server→client frame is the same ``event_to_dict`` payload, parsed back
  through the same adapter.
* ``cli``  — the CLI's REMOTE path: the SAME agent reached through the real
  :class:`~zakcode.server.client.ServerClient` (the production ``zakcode chat
  --server`` → sidecar transport) over an in-memory ASGI client. This is the
  only true *interface* in the CLI's path — the CLI's LOCAL mode calls
  :meth:`Agent.astream_turn` directly (no relay), and the terminal
  :class:`~zakcode.cli.render.StreamRenderer` is a lossy human-facing sink whose
  correctness is a separate axis, not event-stream parity. This layer exercises
  the production client-side deserializer that ``http`` (hand-rolled parser)
  bypasses.

All layers run the IDENTICAL ``Agent`` + ``ScriptedProvider`` (built by one
:func:`_build_parity_agent`), so the ONLY variable is the transport. ``layer`` is
a pytest parameter, so a failure LOCALIZES::

    test_layer_matches_golden[text_only-sdk]  PASSED
    test_layer_matches_golden[text_only-http] FAILED   <- HTTP adapter diverged, not the SDK

That green-sdk / red-http split IS the break point between interface and SDK.
:func:`test_interface_matches_sdk_reference` adds a golden-INDEPENDENT guarantee
(the interface must match whatever the SDK currently emits), so pure
interface-fidelity is guarded even when an intentional SDK change makes the
written golden stale.

EXTENDING
---------
Add a scenario to :data:`SCENARIOS`, or a runner to :data:`_LAYERS`. Every
scenario is then checked against every interface against the one golden. The
three end interfaces (http, ws, cli) are covered; the remaining widenings are new
scenarios and the DISTINCT axes this harness deliberately does not test —
rendering-correctness (the lossy CLI ``StreamRenderer``), the permission
escalation round-trip, and config-parity (server-reduced vs CLI-full agent).

HERMETIC
--------
``ScriptedProvider`` never touches a network; ``TestClient`` drives both the
finite ``/chat/stream`` and the ``/ws`` socket in-process (no live server — the
infinite ``/watch`` stream is the one that needs a real socket, not these). The
WS turn runs server-side as a background task and the client drains frames to the
terminal ``done`` event. :func:`_normalize` excludes the volatile fields (the
per-turn decision ``trace``, raw ``usage`` numbers) and collapses consecutive
text deltas, so the contract is robust to a future interface that re-chunks text.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
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
from zakcode.permissions import PermissionPolicy
from zakcode.providers.base import LLMResult
from zakcode.server.app import create_app
from zakcode.server.client import ServerClient
from zakcode.server.wire import event_from_dict
from zakcode.session.store import Session, SessionStore

# The label carried in every scenario's canonical input, so a stray real turn
# (a mis-wired factory that built the production agent) would look obviously
# wrong in a failure diff rather than plausibly right.
_CANONICAL_INPUT = "__canonical parity input__"


# ── normalization: the client-observable, transport-stable projection ─────────────


def _normalize(events: Sequence[AgentEvent]) -> list[tuple[Any, ...]]:
    """Project an ``AgentEvent`` stream to the stable form a client observes.

    Collapses consecutive ``text`` deltas into one string (so a transport that
    re-chunks text still matches) and drops the volatile ``AgentDone.trace`` /
    raw ``usage`` numbers (deterministic here, but excluded on principle so the
    parity contract never depends on timing).
    """
    out: list[tuple[Any, ...]] = []
    text_buf: list[str] = []

    def flush() -> None:
        if text_buf:
            out.append(("text", "".join(text_buf)))
            text_buf.clear()

    for ev in events:
        # isinstance (not ``ev.event ==``) so mypy narrows the discriminated union
        # and each field access below type-checks — the repo's established idiom.
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


# ── golden scenarios (goldens captured from the live SDK, not guessed) ────────────


@dataclass(frozen=True)
class Scenario:
    id: str
    canonical_input: str
    script: tuple[LLMResult, ...]
    expected: list[tuple[Any, ...]]


SCENARIOS: list[Scenario] = [
    Scenario(
        id="text_only",
        canonical_input=_CANONICAL_INPUT,
        script=(reply("Hello - parity holds."),),
        expected=[
            ("text", "Hello - parity holds."),
            ("usage",),
            ("done", "completed", 1, False, "", None, False),
        ],
    ),
    Scenario(
        id="tool_then_reply",
        canonical_input=_CANONICAL_INPUT,
        script=(
            call_tool("write_file", {"path": "parity.txt", "content": "hi"}, id="w1"),
            reply("Wrote parity.txt."),
        ),
        expected=[
            ("tool_call", "w1", "write_file", {"path": "parity.txt", "content": "hi"}),
            ("tool_result", "w1", False, "Wrote 2 bytes to parity.txt"),
            ("text", "Wrote parity.txt."),
            ("usage",),
            ("done", "completed", 2, False, "", None, False),
        ],
    ),
]


# ── one agent builder, used identically under every layer ─────────────────────────


def _build_parity_agent(
    script: Sequence[LLMResult],
    *,
    workspace_root: Path,
    session: Session | None = None,
) -> Agent:
    """The single Agent construction every layer shares.

    Injecting ``provider`` makes the agent hermetic (``_provider_injected`` skips
    the litellm import + availability probe and drives every role from the
    script). ``permission_policy="allow"`` lets the tool scenario's ``write_file``
    run without a prompter. Because this is the ONLY place an agent is built, the
    sole difference between any two layer runs below is the transport.
    """
    return Agent(
        provider=ScriptedProvider(list(script)),
        session=session,
        permission_policy=PermissionPolicy("allow"),
        default_model="scripted/parity",
        workspace_root=str(workspace_root),
        max_iterations=8,
    )


# ── layer runners: canonical input -> list[AgentEvent] ────────────────────────────


def _run_sdk(scenario: Scenario, workspace_root: Path) -> list[AgentEvent]:
    """L0 — the canonical stream: ``Agent.astream_turn`` called directly."""

    async def go() -> list[AgentEvent]:
        agent = _build_parity_agent(scenario.script, workspace_root=workspace_root)
        return [ev async for ev in agent.astream_turn(scenario.canonical_input)]

    # Sync wrapper (asyncio.run in a fresh loop) so the parametrized test stays
    # sync and never nests pytest-asyncio's loop inside TestClient's portal.
    return asyncio.run(go())


def _sse_data_frames(text: str) -> list[dict[str, Any]]:
    """Extract the JSON payloads from a finite SSE stream's ``data:`` lines."""
    frames: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            frames.append(json.loads(line[len("data:") :].strip()))
    return frames


def _run_http(scenario: Scenario, workspace_root: Path) -> list[AgentEvent]:
    """L1 — the SAME agent, driven through the HTTP/SSE interface.

    The factory injects the identical ``_build_parity_agent`` (bound to the
    server's session), so ``/chat/stream`` relays the SDK's stream. The finite
    stream is buffered by ``TestClient``; each ``data:`` frame parses back into an
    ``AgentEvent`` through the same wire adapter a real client uses.
    """
    settings = Settings(default_model="scripted/parity", workspace_root=workspace_root)
    store = SessionStore(base_dir=workspace_root / "sessions")

    def factory(session: Session, model: str | None, prompter: object) -> Agent:  # noqa: ARG001
        return _build_parity_agent(scenario.script, workspace_root=workspace_root, session=session)

    app = create_app(settings=settings, store=store, agent_factory=factory)
    client = TestClient(app)
    resp = client.post("/chat/stream", json={"message": scenario.canonical_input})
    resp.raise_for_status()
    return [event_from_dict(frame) for frame in _sse_data_frames(resp.text)]


def _run_ws(scenario: Scenario, workspace_root: Path) -> list[AgentEvent]:
    """L2 — the SAME agent, driven through the WebSocket interface.

    The ``/ws/{session_id}`` handler LOADS an existing session by path id (unlike
    ``/chat/stream``'s get-or-create), so the session is pre-created in the store.
    The client sends one ``input`` message; the turn runs server-side as a
    background task that relays ``event_to_dict`` over the socket — the same 1:1
    relay as SSE — and the client drains frames to the terminal ``done`` event.

    A non-event control frame (an ``error`` or ``action_required``) is itself a
    divergence from the SDK stream, so it fails loudly rather than being silently
    parsed. The server-supplied ``prompter`` is ignored on purpose: the ``allow``
    policy never escalates, keeping the agent config identical across every layer.
    """
    settings = Settings(default_model="scripted/parity", workspace_root=workspace_root)
    store = SessionStore(base_dir=workspace_root / "sessions")

    def factory(session: Session, model: str | None, prompter: object) -> Agent:  # noqa: ARG001
        return _build_parity_agent(scenario.script, workspace_root=workspace_root, session=session)

    app = create_app(settings=settings, store=store, agent_factory=factory)
    client = TestClient(app)
    # WS loads the session by path id, so it must exist before the handshake.
    session = Session(cwd=".", model="scripted/parity")
    store.save(session)

    frames: list[dict[str, Any]] = []
    with client.websocket_connect(f"/ws/{session.id}") as ws:
        ws.send_json({"type": "input", "message": scenario.canonical_input})
        while True:
            frame = ws.receive_json()
            if "event" not in frame:  # error / action_required: a real divergence
                raise AssertionError(f"WS sent a non-event control frame: {frame}")
            frames.append(frame)
            if frame["event"] == "done":
                break
    return [event_from_dict(frame) for frame in frames]


def _run_cli(scenario: Scenario, workspace_root: Path) -> list[AgentEvent]:
    """L3 — the CLI's REMOTE path: the SAME agent reached through ``ServerClient``.

    ``zakcode chat --server`` drives a turn through
    :meth:`ServerClient.astream_turn`, which POSTs ``/chat/stream`` and
    reconstructs each ``AgentEvent`` from the wire (``event_from_dict``) before the
    terminal renderer ever sees it. That reconstruction is the CLI's real interface
    to a served SDK — and it is DIFFERENT code from the ``http`` layer, whose
    hand-rolled SSE parser bypasses the production client deserializer this layer
    exercises. Driven over ``httpx``'s in-memory ASGI transport (no socket), the
    same idiom the repo's M3 parity criterion (``test_server_client.py``) uses.

    NOT covered here (by design — separate axes, see the module docstring): the
    CLI's LOCAL mode calls :meth:`Agent.astream_turn` directly (no interface to
    test), and the ``StreamRenderer`` is a lossy human-facing sink, not an
    event-stream relay.
    """
    settings = Settings(default_model="scripted/parity", workspace_root=workspace_root)
    store = SessionStore(base_dir=workspace_root / "sessions")

    def factory(session: Session, model: str | None, prompter: object) -> Agent:  # noqa: ARG001
        return _build_parity_agent(scenario.script, workspace_root=workspace_root, session=session)

    app = create_app(settings=settings, store=store, agent_factory=factory)

    async def go() -> list[AgentEvent]:
        transport = httpx.ASGITransport(app=app)
        http = httpx.AsyncClient(transport=transport, base_url="http://parity.testserver")
        try:
            client = ServerClient(http_client=http)
            return [ev async for ev in client.astream_turn(scenario.canonical_input)]
        finally:
            await http.aclose()  # injected client is caller-owned; ServerClient won't close it

    # Sync wrapper (asyncio.run) so the parametrized test stays sync, like _run_sdk.
    return asyncio.run(go())


_LAYERS: dict[str, Callable[[Scenario, Path], list[AgentEvent]]] = {
    "sdk": _run_sdk,
    "http": _run_http,
    "ws": _run_ws,
    "cli": _run_cli,
}

# The interface layers = every layer that is not the canonical SDK reference.
_INTERFACE_LAYERS: list[str] = sorted(name for name in _LAYERS if name != "sdk")


# ── tests ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("layer", sorted(_LAYERS))
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_layer_matches_golden(scenario: Scenario, layer: str, tmp_path: Path) -> None:
    """Every layer reproduces the written golden for the canonical input.

    The break point: ``[…-sdk]`` green while ``[…-http]`` red means the HTTP
    adapter diverged from the SDK, not the SDK from its spec. Both red means the
    SDK (or this golden) changed — update the golden deliberately.
    """
    events = _LAYERS[layer](scenario, tmp_path / layer)
    assert _normalize(events) == scenario.expected


@pytest.mark.parametrize("layer", _INTERFACE_LAYERS)
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_interface_matches_sdk_reference(scenario: Scenario, layer: str, tmp_path: Path) -> None:
    """Golden-independent: each interface matches whatever the SDK emits NOW.

    This never goes stale on an intentional SDK wording change (both sides move
    together); it fires only when an interface stops faithfully relaying the
    SDK — the purest statement of the break point.
    """
    sdk = _normalize(_run_sdk(scenario, tmp_path / "sdk"))
    iface = _normalize(_LAYERS[layer](scenario, tmp_path / layer))
    assert iface == sdk

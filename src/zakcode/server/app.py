"""The FastAPI application — Layer 2, exposing the core engine over HTTP + SSE.

``create_app()`` builds an app that wraps the **same** core the CLI uses in-process:
a turn is run by an :class:`~zakcode.Agent` (or any object with the same surface),
and the server only serializes the result / event stream to HTTP. There is **no new
agent logic here** — this is a transport.

Testability: ``create_app(agent_factory=...)`` injects how an agent is built for a
session, so tests pass a factory backed by a scripted provider and never touch a
network or a model. The default factory builds a real :class:`~zakcode.Agent`.

Endpoints (see ``docs/ARCHITECTURE.md`` — Server API surface):

* ``GET  /health``            — liveness + version
* ``GET  /config``            — resolved non-secret settings (never API keys)
* ``GET  /tools``             — registered tool definitions
* ``GET  /sessions``          — list stored sessions
* ``POST /sessions``          — create a session
* ``GET  /sessions/{id}``     — fetch one session's summary
* ``DELETE /sessions/{id}``   — delete a session
* ``POST /chat``              — run one buffered turn → :class:`ChatResponse`
* ``POST /chat/stream``       — run one turn, streaming ``AgentEvent``s as SSE

The WebSocket channel (bidirectional input + interrupt + permission approval)
is added in a later increment; this module is REST + SSE only.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

import pydantic
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from sse_starlette.sse import EventSourceResponse

from zakcode.agent.loop import TurnResult
from zakcode.config import Settings, load_settings
from zakcode.events import AgentEvent
from zakcode.permissions import PermissionOutcome, PermissionPrompter, PermissionRequest
from zakcode.server.wire import (
    ChatRequest,
    ChatResponse,
    SessionInfo,
    ToolInfo,
    WSActionRequired,
    WSApproval,
    WSInterrupt,
    WSUserInput,
    client_message_from_dict,
    event_to_dict,
)
from zakcode.session.store import Session, SessionNotFound, SessionStore
from zakcode.tools.base import ToolRegistry
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.version import __version__

logger = logging.getLogger("zakcode.server")


class AgentLike(Protocol):
    """The surface the server needs from an agent (so a fake can stand in)."""

    session: Session

    async def arun_turn(self, user_text: str) -> TurnResult: ...
    def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]: ...


#: How the server builds an agent for a given session, optional model override, and
#: an optional permission prompter (the WebSocket channel supplies one so escalations
#: can be approved interactively; REST/SSE pass ``None`` and ``ask`` fails closed).
AgentFactory = Callable[[Session, str | None, PermissionPrompter | None], AgentLike]


def _default_agent_factory(settings: Settings, store: SessionStore) -> AgentFactory:
    """Build the production factory: a real :class:`~zakcode.Agent` per request.

    Bound to ``settings`` so every agent shares the operator's configured posture
    (model, permission mode, workspace root, …) and to ``store`` so a turn persists
    incrementally at message boundaries — the same durability the in-process CLI
    agent gets.

    A per-request ``model`` override swaps **only** the model via
    :meth:`Settings.model_copy`, preserving the rest of the posture; rebuilding
    ``Settings`` from the environment would silently drop ``permission_mode`` /
    ``workspace_root`` and change the security stance of the turn. A ``prompter``
    (from the WebSocket bridge) makes ``ask`` mode interactive; with none, ``ask``
    fails closed (writes/shell denied) — the safe default for headless REST/SSE.
    """
    from zakcode import Agent

    def factory(
        session: Session, model: str | None, prompter: PermissionPrompter | None
    ) -> AgentLike:
        # model_copy (not dump+rebuild) so excluded fields like api_key survive.
        agent_settings: Settings = settings
        if model:
            agent_settings = settings.model_copy(update={"default_model": model})
        return Agent(
            session=session,
            settings=agent_settings,
            session_store=store,
            prompter=prompter,
        )

    return factory


class WebSocketPermissionPrompter:
    """A :class:`~zakcode.permissions.PermissionPrompter` that asks over a WebSocket.

    When the core escalates a tool call, :meth:`confirm` sends a
    :class:`~zakcode.server.wire.WSActionRequired` frame to the client and awaits a
    matching ``approval`` message (delivered by the connection's receive loop via
    ``await_approval``). The core stays UI-agnostic; this is just the transport.
    """

    def __init__(
        self,
        send: Callable[[dict[str, Any]], Awaitable[None]],
        await_approval: Callable[[], Awaitable[PermissionOutcome]],
    ) -> None:
        self._send = send
        self._await_approval = await_approval

    async def confirm(self, request: PermissionRequest) -> PermissionOutcome:
        await self._send(WSActionRequired.from_request(request).model_dump(mode="json"))
        return await self._await_approval()


def create_app(
    *,
    settings: Settings | None = None,
    store: SessionStore | None = None,
    agent_factory: AgentFactory | None = None,
    tool_registry: ToolRegistry | None = None,
) -> FastAPI:
    """Construct the Zak Code HTTP app.

    All collaborators are injectable for testing; production callers pass nothing
    and get a real store + agent factory.
    """
    resolved_settings = settings or load_settings()
    resolved_store = store or SessionStore()
    resolved_factory = agent_factory or _default_agent_factory(resolved_settings, resolved_store)
    resolved_registry = tool_registry or default_registry()

    app = FastAPI(
        title="Zak Code",
        version=__version__,
        summary="Vendor-agnostic agentic coding engine over HTTP.",
    )

    # Session ids with a turn currently in flight. Two overlapping turns on one
    # session would both mutate Session.messages and race on store.save (the store
    # is last-writer-wins, with no cross-request lock), corrupting the transcript —
    # so REST/SSE refuse a second turn with 409 while one is running. In asyncio the
    # check-then-add in each endpoint is atomic (no await between them). The WS
    # channel enforces the same one-turn-at-a-time rule per connection separately.
    inflight: set[str] = set()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _get_or_create_session(session_id: str | None, model: str | None) -> Session:
        """Load an existing session by id (404 if unknown), or create a new one."""
        if session_id is not None:
            try:
                return resolved_store.load(session_id)
            except SessionNotFound as exc:
                raise HTTPException(status_code=404, detail=f"no session {session_id!r}") from exc
        session = Session(
            cwd=str(resolved_settings.workspace_root),
            model=model or resolved_settings.default_model,
        )
        resolved_store.save(session)
        return session

    # ── meta endpoints ─────────────────────────────────────────────────────────

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/config")
    def get_config() -> dict[str, Any]:
        # Provider keys live in env (read by litellm), never in Settings. The only
        # secret-bearing field, ``api_key``, is marked ``exclude=True`` in
        # ``Settings`` so ``model_dump`` already omits it; the explicit pop is
        # belt-and-suspenders in case that field convention is ever changed.
        data = resolved_settings.model_dump(mode="json")
        data.pop("api_key", None)
        return data

    @app.get("/tools")
    def list_tools() -> list[ToolInfo]:
        infos: list[ToolInfo] = []
        for name in resolved_registry.names():
            tool = resolved_registry.get(name)
            if tool is not None:
                infos.append(ToolInfo.from_spec(tool.spec))
        return infos

    # ── session CRUD ───────────────────────────────────────────────────────────

    @app.get("/sessions")
    def list_sessions() -> list[SessionInfo]:
        infos: list[SessionInfo] = []
        for session_id in resolved_store.list():
            try:
                infos.append(SessionInfo.from_session(resolved_store.load(session_id)))
            except Exception:  # noqa: BLE001 — skip an unreadable/corrupt session file
                continue
        return infos

    @app.post("/sessions", status_code=201)
    def create_session() -> SessionInfo:
        session = Session(
            cwd=str(resolved_settings.workspace_root),
            model=resolved_settings.default_model,
        )
        resolved_store.save(session)
        return SessionInfo.from_session(session)

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> SessionInfo:
        try:
            return SessionInfo.from_session(resolved_store.load(session_id))
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}") from exc

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> None:
        if not resolved_store.delete(session_id):
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")

    # ── chat ───────────────────────────────────────────────────────────────────

    def _claim_session(session: Session) -> None:
        """Reserve a session for a turn, or 409 if one is already in flight."""
        if session.id in inflight:
            raise HTTPException(
                status_code=409,
                detail=f"a turn is already running for session {session.id!r}",
            )
        inflight.add(session.id)

    @app.post("/chat")
    async def chat(request: ChatRequest) -> ChatResponse:
        session = _get_or_create_session(request.session_id, request.model)
        _claim_session(session)
        try:
            agent = resolved_factory(session, request.model, None)
            result = await agent.arun_turn(request.message)
            resolved_store.save(agent.session)
            return ChatResponse.from_turn(agent.session.id, result)
        finally:
            inflight.discard(session.id)

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest) -> EventSourceResponse:
        session = _get_or_create_session(request.session_id, request.model)
        _claim_session(session)
        agent = resolved_factory(session, request.model, None)

        async def event_source() -> AsyncIterator[dict[str, str]]:
            try:
                async for event in agent.astream_turn(request.message):
                    yield {"data": json.dumps(event_to_dict(event))}
            finally:
                # Persist whatever state the turn produced, even if the client
                # disconnects mid-stream (EventSourceResponse cancels us), and
                # always release the in-flight reservation.
                resolved_store.save(agent.session)
                inflight.discard(session.id)

        return EventSourceResponse(event_source())

    # ── WebSocket: bidirectional input + interrupt + permission approval ────────

    @app.websocket("/ws/{session_id}")
    async def ws_chat(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        try:
            session = resolved_store.load(session_id)
        except SessionNotFound:
            await websocket.send_json({"type": "error", "detail": f"no session {session_id!r}"})
            await websocket.close()
            return

        send_lock = asyncio.Lock()
        # Holds the future the prompter is currently waiting on (one at a time).
        approval: dict[str, asyncio.Future[PermissionOutcome]] = {}

        async def send(payload: dict[str, Any]) -> None:
            # Serialize sends so a turn's events and an action_required prompt never
            # interleave on the wire.
            async with send_lock:
                await websocket.send_json(payload)

        async def await_approval() -> PermissionOutcome:
            fut: asyncio.Future[PermissionOutcome] = asyncio.get_running_loop().create_future()
            approval["fut"] = fut
            try:
                return await fut
            finally:
                approval.pop("fut", None)

        prompter = WebSocketPermissionPrompter(send, await_approval)
        agent = resolved_factory(session, None, prompter)
        current_turn: asyncio.Task[None] | None = None

        async def run_turn(text: str) -> None:
            try:
                async for event in agent.astream_turn(text):
                    await send(event_to_dict(event))
            except asyncio.CancelledError:
                await send({"event": "status", "message": "interrupted"})
                raise
            except Exception as exc:  # noqa: BLE001 — surface, never crash the socket
                await send({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})
            finally:
                resolved_store.save(agent.session)

        try:
            while True:
                raw = await websocket.receive_json()
                try:
                    msg = client_message_from_dict(raw)
                except pydantic.ValidationError:
                    await send({"type": "error", "detail": "unrecognized message"})
                    continue

                if isinstance(msg, WSUserInput):
                    if current_turn is not None and not current_turn.done():
                        await send({"type": "error", "detail": "a turn is already running"})
                        continue
                    current_turn = asyncio.create_task(run_turn(msg.message))
                elif isinstance(msg, WSInterrupt):
                    if current_turn is not None and not current_turn.done():
                        current_turn.cancel()
                elif isinstance(msg, WSApproval):
                    fut = approval.get("fut")
                    if fut is not None and not fut.done():
                        try:
                            fut.set_result(PermissionOutcome(msg.outcome))
                        except ValueError:
                            fut.set_result(PermissionOutcome.DENY_ONCE)
        except WebSocketDisconnect:
            if current_turn is not None and not current_turn.done():
                current_turn.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await current_turn

    return app


__all__ = ["create_app", "AgentLike", "AgentFactory"]

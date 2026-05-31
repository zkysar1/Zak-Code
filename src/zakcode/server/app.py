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

import json
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse

from zakcode.agent.loop import TurnResult
from zakcode.config import Settings, load_settings
from zakcode.events import AgentEvent
from zakcode.server.wire import (
    ChatRequest,
    ChatResponse,
    SessionInfo,
    ToolInfo,
    event_to_dict,
)
from zakcode.session.store import Session, SessionNotFound, SessionStore
from zakcode.tools.base import ToolRegistry
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.version import __version__


class AgentLike(Protocol):
    """The surface the server needs from an agent (so a fake can stand in)."""

    session: Session

    async def arun_turn(self, user_text: str) -> TurnResult: ...
    def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]: ...


#: How the server builds an agent for a given session + optional model override.
AgentFactory = Callable[[Session, str | None], AgentLike]


def _default_agent_factory(settings: Settings) -> AgentFactory:
    """Build the production factory: a real :class:`~zakcode.Agent` per request.

    Bound to ``settings`` so every agent shares the operator's configured posture
    (model, permission mode, …). A per-request ``model`` override rebuilds settings
    with that model. The server passes no interactive prompter, so ``ask`` mode
    fails closed (writes/shell denied) — the safe default for headless REST/SSE;
    the WebSocket channel adds the interactive approval bridge.
    """
    from zakcode import Agent

    def factory(session: Session, model: str | None) -> AgentLike:
        if model:
            return Agent(session=session, default_model=model)
        return Agent(session=session, settings=settings)

    return factory


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
    resolved_factory = agent_factory or _default_agent_factory(resolved_settings)
    resolved_registry = tool_registry or default_registry()

    app = FastAPI(
        title="Zak Code",
        version=__version__,
        summary="Vendor-agnostic agentic coding engine over HTTP.",
    )

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
        # Settings hold no secrets (provider keys live in env, read by litellm), so
        # the whole model is safe to expose. api_key, if ever set, is the only
        # sensitive field — drop it defensively.
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

    @app.post("/chat")
    async def chat(request: ChatRequest) -> ChatResponse:
        session = _get_or_create_session(request.session_id, request.model)
        agent = resolved_factory(session, request.model)
        result = await agent.arun_turn(request.message)
        resolved_store.save(agent.session)
        return ChatResponse.from_turn(agent.session.id, result)

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest) -> EventSourceResponse:
        session = _get_or_create_session(request.session_id, request.model)
        agent = resolved_factory(session, request.model)

        async def event_source() -> AsyncIterator[dict[str, str]]:
            try:
                async for event in agent.astream_turn(request.message):
                    yield {"data": json.dumps(event_to_dict(event))}
            finally:
                # Persist whatever state the turn produced, even if the client
                # disconnects mid-stream (EventSourceResponse cancels us).
                resolved_store.save(agent.session)

        return EventSourceResponse(event_source())

    return app


__all__ = ["create_app", "AgentLike", "AgentFactory"]

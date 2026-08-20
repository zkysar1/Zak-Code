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
* ``GET  /sessions/{id}/artifacts`` — list downloadable artifacts
* ``POST /sessions/{id}/uploads``   — attach a user file to the session
* ``DELETE /sessions/{id}``   — delete a session
* ``POST /chat``              — run one buffered turn → :class:`ChatResponse`
* ``POST /chat/stream``       — run one turn, streaming ``AgentEvent``s as SSE
* ``POST /complete``          — raw schema-valid completion (no tools / loop / session)
* ``WS   /ws/{session_id}``   — bidirectional: input, interrupt, permission approval

``/chat`` and ``/chat/stream`` refuse a second turn on a session that already has
one in flight (HTTP 409); the WebSocket enforces the same one-turn-at-a-time rule
per connection. Secrets never leave the process: ``GET /config`` dumps settings
with the only secret-bearing field (``api_key``) excluded at the model level.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hmac
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

import pydantic
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from zakcode import __version__
from zakcode.agent.loop import TurnResult
from zakcode.artifacts import (
    ArtifactChangedError,
    ArtifactError,
    ArtifactRef,
    artifact_from_path,
    resolve_artifact_path,
)
from zakcode.config import Settings, load_settings
from zakcode.events import AgentEvent
from zakcode.knowledge import okf_bundle, read_knowledge_bundle
from zakcode.permissions import PermissionOutcome, PermissionPrompter, PermissionRequest
from zakcode.providers.base import Provider, ProviderError
from zakcode.providers.structured import complete_structured, schema_error
from zakcode.secrets import strip_url_credentials
from zakcode.server.event_bus import EventBusRegistry
from zakcode.server.safe_projection import SafeEventProjection
from zakcode.server.wire import (
    ChatRequest,
    ChatResponse,
    CompleteRequest,
    CompleteResponse,
    NudgeRequest,
    SayRequest,
    SessionInfo,
    ToolInfo,
    UploadRequest,
    UploadResponse,
    WatchMarkerRequest,
    WSActionRequired,
    WSApproval,
    WSInterrupt,
    WSUserInput,
    client_message_from_dict,
    event_to_dict,
    events_schema,
)
from zakcode.session.store import (
    Session,
    SessionCorruptError,
    SessionNotFound,
    SessionStore,
    SessionVersionError,
)
from zakcode.tools.base import ToolRegistry
from zakcode.tools.builtins.default_registry import default_registry

logger = logging.getLogger("zakcode.server")

#: How long the WebSocket permission bridge waits for a client's approval before
#: failing closed (deny-once). Bounds a stuck/disconnected client from hanging a turn.
APPROVAL_TIMEOUT_SECONDS = 120.0

#: WebSocket close code for an unauthenticated handshake (RFC 6455 policy violation).
WS_UNAUTHORIZED_CODE = 1008

#: The only HTTP path served without a bearer token when auth is enabled (liveness probes
#: from a load balancer / orchestrator must not need a credential).
_AUTH_EXEMPT_PATHS = frozenset({"/health"})
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_UPLOAD_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[-\w.+/]+(?:;[-\w=.+]+)*);base64,(?P<data>.*)$", re.S
)
_UPLOAD_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WINDOWS_RESERVED_FILENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _extract_bearer(header: str | None) -> str | None:
    """Return the token from an ``Authorization: Bearer <token>`` header, or ``None``.

    Tolerant of casing/whitespace in the scheme; returns ``None`` for a missing header or
    any non-``Bearer`` scheme so the caller treats it as unauthenticated (never raises).
    """
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def _token_matches(provided: str | None, expected: str) -> bool:
    """Constant-time compare of a presented token against the configured one.

    ``hmac.compare_digest`` avoids leaking the token length/content via timing. A missing
    presented token is rejected without comparing. Both sides are compared as UTF-8 *bytes*:
    the str form of ``compare_digest`` raises ``TypeError`` on non-ASCII input, which would
    turn a malformed client token into a 500 instead of a clean rejection.
    """
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _subprotocol_token(header: str | None) -> str | None:
    """Extract a bearer token offered as ``Sec-WebSocket-Protocol: bearer, <token>``.

    Browsers cannot set an ``Authorization`` header on a WebSocket handshake, but they CAN
    offer subprotocols. The client offers exactly two — ``bearer`` and the token — and the
    server echoes ``bearer`` on accept. This keeps the token out of the URL (and therefore
    out of uvicorn's access log) unlike a ``?token=`` query param. Returns ``None`` unless the
    header is precisely ``bearer, <token>``.
    """
    if not header:
        return None
    parts = [p.strip() for p in header.split(",") if p.strip()]
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def _decode_upload_data(data: str) -> bytes:
    """Decode raw base64 or a browser ``data:*;base64,...`` URL with a hard byte cap."""
    payload = data.strip()
    match = _UPLOAD_DATA_URL_RE.match(payload)
    if match:
        payload = match.group("data")
    compact = "".join(payload.split())
    # Reject obviously oversized payloads before allocating decoded bytes. Base64 expands 3
    # bytes to 4 chars; the small padding margin keeps exact-limit files valid.
    if len(compact) > ((_MAX_UPLOAD_BYTES + 2) // 3) * 4 + 8:
        raise ValueError(f"upload is too large (max {_MAX_UPLOAD_BYTES} bytes)")
    try:
        blob = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("'data' must be valid base64 bytes or a base64 data URL") from exc
    if not blob:
        raise ValueError("upload decoded to an empty file")
    if len(blob) > _MAX_UPLOAD_BYTES:
        raise ValueError(f"upload is too large ({len(blob)} bytes; max {_MAX_UPLOAD_BYTES})")
    return blob


def _safe_upload_filename(filename: str) -> str:
    """Return a single safe filename component, preserving ordinary Unicode names."""
    raw_name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = _UPLOAD_FILENAME_RE.sub("_", raw_name).strip(" .")
    if not name:
        name = "upload.bin"
    if len(name) > 180:
        suffix = Path(name).suffix[:32]
        stem = Path(name).stem[: max(1, 180 - len(suffix))]
        name = f"{stem}{suffix}" if suffix else stem
    stem = Path(name).stem.upper()
    if stem in _WINDOWS_RESERVED_FILENAMES:
        name = f"_{name}"
    return name


def _unique_upload_path(workspace_root: Path, session: Session, filename: str) -> Path:
    """Allocate a non-clobbering upload path under ``uploads/<session>/``."""
    root = workspace_root.resolve()
    # The FULL session id (not an 8-char prefix) so two sessions never share an upload dir.
    upload_dir = (root / "uploads" / session.id).resolve()
    try:
        upload_dir.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defensive; path is internally derived
        raise ValueError("upload directory escaped the workspace") from exc
    upload_dir.mkdir(parents=True, exist_ok=True)

    path = upload_dir / filename
    if not path.exists():
        return path
    parsed = Path(filename)
    stem = parsed.stem or "upload"
    suffix = parsed.suffix
    for index in range(2, 1000):
        candidate = upload_dir / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise ValueError("could not allocate a unique upload filename")


def _write_upload_file(target: Path, file_bytes: bytes) -> None:
    """Write an upload without clobbering a concurrently-created file."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(target, flags)
    wrapped = False
    try:
        with os.fdopen(fd, "wb") as handle:
            wrapped = True
            handle.write(file_bytes)
    except Exception:
        if not wrapped:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            target.unlink()
        raise


def _reader_tool_for_artifact(artifact: ArtifactRef) -> str:
    """Best first reader for an uploaded artifact."""
    suffix = Path(artifact.filename).suffix.lower()
    if suffix == ".docx":
        return "read_docx"
    if suffix in {".xlsx", ".xlsm"}:
        return "read_xlsx"
    if artifact.kind == "image":
        return "inspect_image"
    if artifact.kind == "text":
        return "read_file"
    return ""


def _upload_prompt(artifact: ArtifactRef, suggested_tool: str) -> str:
    """Prompt text the browser can place in the composer after a successful upload."""
    quoted_path = f"`{artifact.path}`"
    if suggested_tool:
        return (
            f"Please inspect the uploaded file {quoted_path} with `{suggested_tool}` "
            "and summarize it."
        )
    return f"Please inspect the uploaded file {quoted_path} and tell me what you can do with it."


class AgentLike(Protocol):
    """The surface the server needs from an agent (so a fake can stand in)."""

    session: Session

    async def arun_turn(self, user_text: str) -> TurnResult: ...
    def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]: ...


async def _release_agent(agent: Any) -> None:
    """Tear down a finished per-request/connection agent's loop-owned resources.

    Currently this stops the egress-proxy listener (a no-op unless ``ZAKCODE_EGRESS_PROXY`` is on)
    so it is not leaked for the life of the long-lived server event loop. Guarded so it is safe
    for any ``AgentLike`` (a remote/stub agent with no ``loop`` is simply skipped) and never
    raises. Deliberately does NOT call ``agent.aclose()`` — the SESSION_END lifecycle on the
    server is unchanged.
    """
    loop = getattr(agent, "loop", None)
    aclose = getattr(loop, "aclose", None)
    if aclose is not None:
        with contextlib.suppress(Exception):
            await aclose()


#: How the server builds an agent for a given session, optional model override, and
#: an optional permission prompter (the WebSocket channel supplies one so escalations
#: can be approved interactively; REST/SSE pass ``None`` and ``ask`` fails closed).
AgentFactory = Callable[[Session, str | None, PermissionPrompter | None], AgentLike]


#: How the server builds a RAW provider for ``/complete`` (a bounded, tool-less completion —
#: NOT an agent turn), given an optional per-request model override.
ProviderFactory = Callable[[str | None], Provider]


def _default_provider_factory(settings: Settings) -> ProviderFactory:
    """Build the production ``/complete`` provider factory: a :class:`LiteLLMProvider` DIRECTLY.

    Deliberately NOT wrapped in the text-tool protocol — ``/complete`` is a raw completion (no
    tools) and the wrapper's tool-less branch is a literal passthrough anyway. A per-request
    ``model`` override swaps only the model via :meth:`Settings.model_copy`, preserving the rest
    of the posture (including the excluded ``api_key``).
    """
    from zakcode.providers.litellm_provider import LiteLLMProvider

    def factory(model: str | None) -> Provider:
        resolved = settings.model_copy(update={"default_model": model}) if model else settings
        return LiteLLMProvider(resolved)

    return factory


def _default_agent_factory(settings: Settings, store: SessionStore) -> AgentFactory:
    """Build the production factory: a real :class:`~zakcode.Agent` per request, running a MIND.

    Each request's agent loads the env's MIND from ``settings.workspace_root`` — the operator
    identity (``self.md``), always-on rules, and skills — so ``zakcode serve`` behaves like the
    CLI. The topology is one container per customer env, selected by the workspace root;
    sub-agents / MCP / plugins / compaction are deliberately NOT enabled here (a separate posture
    decision, out of scope for the connection substrate).

    Bound to ``settings`` so every agent shares the operator's configured posture (model,
    permission mode, workspace root, …) and to ``store`` so a turn persists incrementally at
    message boundaries — the same durability the in-process CLI agent gets.

    A per-request ``model`` override swaps **only** the model via :meth:`Settings.model_copy`,
    preserving the rest of the posture; rebuilding ``Settings`` from the environment would
    silently drop ``permission_mode`` / ``workspace_root`` and change the security stance of
    the turn. A ``prompter`` (from the WebSocket bridge) makes ``ask`` mode interactive; with
    none, ``ask`` fails closed (writes/shell denied) — the safe default for headless REST/SSE.

    Cross-session memory is NOT a harness concern (see docs/PERSISTENCE-BOUNDARY.md): a served
    MIND attaches its own recall via the generic hook/tool seams; the factory wires none.
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
            enable_skills=True,
            enable_rules=True,
            lean_rules=agent_settings.lean_rules,
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


# ── PEARL knowledge base (§10.4) + viewer nudge (§Layer-4) ────────────────────
#: Defense-in-depth length cap on a queued viewer nudge. The gateway is the real
#: sanitization + rate-limit trust boundary; the server only caps and queues.
NUDGE_MAX_CHARS = 500

#: Defense-in-depth length cap on a queued user say (the watch/talk unification).
#: Larger than a nudge — a say is a real conversational message, not a suggestion —
#: but still bounded; the gateway is the real sanitization + ownership boundary.
SAY_MAX_CHARS = 2000



def create_app(
    *,
    settings: Settings | None = None,
    store: SessionStore | None = None,
    agent_factory: AgentFactory | None = None,
    provider_factory: ProviderFactory | None = None,
    tool_registry: ToolRegistry | None = None,
) -> FastAPI:
    """Construct the Zak Code HTTP app.

    All collaborators are injectable for testing; production callers pass nothing
    and get a real store + agent factory + provider factory (the latter drives ``/complete``,
    a raw schema-valid completion that bypasses the agent loop).
    """
    resolved_settings = settings or load_settings()
    resolved_store = store or SessionStore()
    resolved_factory = agent_factory or _default_agent_factory(resolved_settings, resolved_store)
    resolved_provider_factory = provider_factory or _default_provider_factory(resolved_settings)
    resolved_registry = tool_registry or default_registry()

    app = FastAPI(
        title="Zak Code",
        version=__version__,
        summary="Vendor-agnostic agentic coding engine over HTTP.",
    )

    # ── auth (opt-in; inert and zero-overhead when no token is configured) ───────
    # When an auth token is set, EVERY HTTP request (except /health) must carry a matching
    # ``Authorization: Bearer`` header. The middleware is only registered when a token
    # exists, so the unauthenticated loopback-dev path is byte-for-byte unchanged. NOTE:
    # http middleware never sees the ``websocket`` ASGI scope, so the WS route authenticates
    # itself in-handler (below). The bundled web client at ``/`` is NOT exempt: enabling auth
    # means an external front-end (holding the token) is the intended interface.
    auth_token = resolved_settings.auth_token
    if auth_token:

        @app.middleware("http")
        async def _require_bearer(request: Request, call_next: Callable) -> Any:
            if request.url.path in _AUTH_EXEMPT_PATHS:
                return await call_next(request)
            presented = _extract_bearer(request.headers.get("authorization"))
            if not _token_matches(presented, auth_token):
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
            return await call_next(request)

    def _check_model(model: str | None) -> None:
        """Reject a per-request model override that is not in the operator allowlist.

        No-op when the allowlist is empty (the default) or no override was supplied.
        """
        allowed = resolved_settings.allowed_models
        if model and allowed and model not in allowed:
            raise HTTPException(status_code=400, detail=f"model {model!r} is not allowed")

    # Session ids with a turn currently in flight. Two overlapping turns on one
    # session would both mutate Session.messages and race on store.save (the store
    # is last-writer-wins, with no cross-request lock), corrupting the transcript —
    # so REST/SSE refuse a second turn with 409 while one is running. In asyncio the
    # check-then-add in each endpoint is atomic (no await between them). The WS channel
    # honors the SAME per-session reservation (in addition to its per-connection guard),
    # so a WS turn cannot overlap another WS client's or a REST turn on one session.
    inflight: set[str] = set()

    # Read-only watch fan-out (P0-3). Every turn tees its AgentEvents into the per-session
    # bus; GET /watch tails that bus, projecting each event to a secret-redacted SafeEvent.
    # The registry is bounded per session (ring buffer) so an abandoned watcher never grows
    # memory. The projection loads secret VALUES + workspace paths once here (deterministic
    # per event thereafter), so it is safe to share across sessions.
    event_bus_registry = EventBusRegistry()
    safe_projection = SafeEventProjection(workspace_root=str(resolved_settings.workspace_root))

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

    def _load_session_or_404(session_id: str) -> Session:
        try:
            return resolved_store.load(session_id)
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}") from exc

    def _session_artifacts(session: Session) -> list[ArtifactRef]:
        """Collect unique artifacts recorded by uploads and tool results."""
        artifacts: dict[str, ArtifactRef] = {
            artifact.id: artifact for artifact in session.uploaded_artifacts
        }
        for message in session.messages:
            for block in message.blocks:
                for artifact in getattr(block, "artifacts", []):
                    artifacts[artifact.id] = artifact
        return list(artifacts.values())

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
        # api_base may carry RFC-3986 userinfo (user:pass@host); mask it so /config never
        # serializes embedded credentials. api_key is already excluded above. (audit3 #7)
        if data.get("api_base"):
            data["api_base"] = strip_url_credentials(data["api_base"])
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
            except (SessionCorruptError, SessionVersionError):
                continue  # a genuinely corrupt / too-new file: skip silently (expected)
            except Exception:  # noqa: BLE001 — an UNEXPECTED read error (perms/IO): skip but log
                logger.warning(
                    "skipping unreadable session %r in listing", session_id, exc_info=True
                )
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
        return SessionInfo.from_session(_load_session_or_404(session_id))

    @app.get("/sessions/{session_id}/artifacts")
    def list_session_artifacts(session_id: str) -> list[ArtifactRef]:
        return _session_artifacts(_load_session_or_404(session_id))

    @app.post("/sessions/{session_id}/uploads", status_code=201)
    def upload_session_file(session_id: str, request: UploadRequest) -> UploadResponse:
        session = _load_session_or_404(session_id)
        if session.id in inflight:
            raise HTTPException(
                status_code=409,
                detail=f"a turn is already running for session {session.id!r}",
            )
        try:
            file_bytes = _decode_upload_data(request.data)
            filename = _safe_upload_filename(request.filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        target: Path | None = None
        try:
            for _attempt in range(1000):
                candidate = _unique_upload_path(resolved_settings.workspace_root, session, filename)
                try:
                    _write_upload_file(candidate, file_bytes)
                except FileExistsError:
                    continue
                target = candidate
                break
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"failed to save upload: {exc}") from exc
        if target is None:
            raise HTTPException(status_code=409, detail="could not allocate upload filename")

        try:
            artifact = artifact_from_path(
                target,
                workspace_root=resolved_settings.workspace_root,
                created_by_tool="upload",
            )
        except ArtifactError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        session.uploaded_artifacts = [
            item for item in session.uploaded_artifacts if item.id != artifact.id
        ]
        session.uploaded_artifacts.append(artifact)
        resolved_store.save(session)
        suggested_tool = _reader_tool_for_artifact(artifact)
        return UploadResponse(
            path=artifact.path,
            bytes=artifact.size,
            artifact=artifact,
            suggested_tool=suggested_tool,
            prompt=_upload_prompt(artifact, suggested_tool),
        )

    @app.get("/sessions/{session_id}/artifacts/{artifact_id}/download")
    def download_session_artifact(session_id: str, artifact_id: str) -> FileResponse:
        session = _load_session_or_404(session_id)
        artifact = next(
            (item for item in _session_artifacts(session) if item.id == artifact_id),
            None,
        )
        if artifact is None:
            raise HTTPException(status_code=404, detail=f"no artifact {artifact_id!r}")
        try:
            path = resolve_artifact_path(
                artifact,
                workspace_root=resolved_settings.workspace_root,
                verify_digest=True,
            )
        except ArtifactChangedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ArtifactError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type=artifact.mime_type, filename=artifact.filename)

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
        _check_model(request.model)
        session = _get_or_create_session(request.session_id, request.model)
        _claim_session(session)
        agent: AgentLike | None = None
        try:
            agent = resolved_factory(session, request.model, None)
            result = await agent.arun_turn(request.message)
            resolved_store.save(agent.session)
            return ChatResponse.from_turn(agent.session.id, result)
        finally:
            await _release_agent(agent)
            inflight.discard(session.id)

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest) -> EventSourceResponse:
        _check_model(request.model)
        session = _get_or_create_session(request.session_id, request.model)
        _claim_session(session)
        # Build the agent INSIDE a guard: the factory can raise (e.g. a bad request.model
        # → provider/deny-pattern construction fails) before the stream generator exists,
        # and its finally would then never run — permanently stranding the reservation and
        # 409-ing the session until restart. Release it on a build failure. (audit2 #5)
        try:
            agent = resolved_factory(session, request.model, None)
        except BaseException:
            inflight.discard(session.id)
            raise

        async def event_source() -> AsyncIterator[dict[str, str]]:
            try:
                async for event in agent.astream_turn(request.message):
                    # Tee into the watch bus (P0-3) BEFORE yielding to the turn-driver. suppress:
                    # the read-only fan-out must NEVER break the turn stream. get_or_create
                    # returns an OPEN bus (a closed one is replaced), so publish cannot raise in
                    # practice — the guard is belt-and-suspenders against a concurrent discard.
                    with contextlib.suppress(Exception):
                        event_bus_registry.get_or_create(session.id).publish(event)
                    yield {"data": json.dumps(event_to_dict(event))}
            finally:
                # Persist whatever state the turn produced, even if the client disconnects
                # mid-stream (EventSourceResponse cancels us). Release the reservation in its
                # OWN finally so a store.save error cannot strand it. (audit2 #5)
                try:
                    resolved_store.save(agent.session)
                finally:
                    await _release_agent(agent)  # stop the egress proxy (no-op when off)
                    inflight.discard(session.id)

        return EventSourceResponse(event_source())

    @app.post("/complete")
    async def complete(request: CompleteRequest) -> CompleteResponse:
        """Raw schema-valid completion — NO tools, NO agent loop, NO session, NO permission gate.

        A thin proxy over a provider for callers that want bounded structured output (e.g. a
        semantic extractor): supply ``prompt`` or ``messages`` (+ optional ``schema``/``model``)
        and get back JSON the server validated. On a malformed request -> 400; on a provider
        error or schema-invalid output (after the bounded repair) -> 502 with a
        ``{error, detail, raw_text}`` body. This endpoint never creates or touches a session.
        """
        _check_model(request.model)
        try:
            messages = request.to_messages()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # A malformed client schema is a CALLER error (400), not a 500 and not a wasted model
        # call — meta-validate it before anything else. (jsonschema raises SchemaError, which is
        # not a ValidationError/ProviderError, so it would otherwise escape as an unhandled 500.)
        if request.schema_ is not None:
            schema_problem = schema_error(request.schema_)
            if schema_problem is not None:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "invalid_schema", "detail": schema_problem, "raw_text": None},
                )

        # Build the provider AND run the completion inside ONE guard so a construction error
        # (e.g. a bad request.model) becomes a clean 502, never an unhandled 500.
        try:
            provider = resolved_provider_factory(request.model)
            result = await complete_structured(
                provider, messages, system=request.system, schema=request.schema_, max_repairs=1
            )
        except ProviderError as exc:
            raise HTTPException(
                status_code=502,
                detail={"error": "provider_error", "detail": str(exc), "raw_text": None},
            ) from exc

        if request.schema_ is not None and not result.valid:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "schema_validation_failed",
                    "detail": "model output did not satisfy the schema after repair",
                    "raw_text": result.text,
                },
            )
        return CompleteResponse(
            data=result.data,
            text=result.text,
            usage=result.usage,
            cost_usd=result.usage.cost_usd,
            repaired=result.repaired,
        )

    # ── WebSocket: bidirectional input + interrupt + permission approval ────────

    @app.websocket("/ws/{session_id}")
    async def ws_chat(websocket: WebSocket, session_id: str) -> None:
        # Authenticate the handshake BEFORE accepting. http middleware never runs for the
        # websocket scope, so the token is checked here. The token comes from the
        # ``Authorization`` header (server-to-server clients, e.g. a reverse proxy) or, for
        # browsers that cannot set that header, the ``Sec-WebSocket-Protocol: bearer, <token>``
        # subprotocol. We deliberately do NOT read a ``?token=`` query param: uvicorn logs the
        # request path+query, so a token there would be persisted in the access log. Reject by
        # closing before accept → the client sees a failed handshake (HTTP 403).
        if auth_token:
            presented = _extract_bearer(websocket.headers.get("authorization"))
            selected_subprotocol: str | None = None
            if presented is None:
                presented = _subprotocol_token(websocket.headers.get("sec-websocket-protocol"))
                if presented is not None:
                    selected_subprotocol = "bearer"  # echo only what the client offered
            if not _token_matches(presented, auth_token):
                await websocket.close(code=WS_UNAUTHORIZED_CODE)
                return
            await websocket.accept(subprotocol=selected_subprotocol)
        else:
            # Auth off: still ECHO an offered ``bearer`` subprotocol. A browser client written
            # for an auth-ON deployment always offers ``bearer, <token>``, and per RFC 6455 it
            # aborts the handshake if it offered subprotocols and the server selected none — so
            # one client can talk to both postures. There is no token to enforce here.
            offered = _subprotocol_token(websocket.headers.get("sec-websocket-protocol"))
            await websocket.accept(subprotocol="bearer" if offered is not None else None)
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
                # Fail CLOSED if the client never answers: a disconnected or unresponsive
                # client must not hang the turn (and the server resources behind it)
                # forever. On timeout we deny once — the model sees the denial and the
                # turn proceeds; the operator can retry.
                return await asyncio.wait_for(fut, timeout=APPROVAL_TIMEOUT_SECONDS)
            except TimeoutError:
                return PermissionOutcome.DENY_ONCE
            finally:
                approval.pop("fut", None)

        prompter = WebSocketPermissionPrompter(send, await_approval)
        agent = resolved_factory(session, None, prompter)
        current_turn: asyncio.Task[None] | None = None

        async def run_turn(text: str) -> None:
            try:
                async for event in agent.astream_turn(text):
                    # Tee into the watch bus (P0-3) — see chat_stream for the suppress rationale.
                    with contextlib.suppress(Exception):
                        event_bus_registry.get_or_create(session.id).publish(event)
                    await send(event_to_dict(event))
            except asyncio.CancelledError:
                await send({"event": "status", "message": "interrupted"})
                raise
            except Exception:  # noqa: BLE001 — surface generically, never crash the socket
                # Log the real detail server-side; send the client a GENERIC frame so a
                # raw exception string (which may echo filesystem paths / internals) never
                # crosses the wire. (RISKS.md: WS info-leak) The socket stays alive.
                logger.exception("turn failed for session %s", session.id)
                await send({"type": "error", "detail": "internal error"})
            finally:
                # Release the per-session reservation in its own finally so a save error
                # cannot strand it (mirrors /chat/stream). (audit2 #4/#5)
                try:
                    resolved_store.save(agent.session)
                finally:
                    inflight.discard(session.id)

        try:
            while True:
                raw = await websocket.receive_json()
                try:
                    msg = client_message_from_dict(raw)
                except pydantic.ValidationError:
                    await send({"type": "error", "detail": "unrecognized message"})
                    continue

                if isinstance(msg, WSUserInput):
                    # Honor the SAME per-session reservation REST uses: reject if a turn is
                    # in flight on this connection OR on this session via any transport
                    # (another WS client or a REST turn) — else two turns race on
                    # Session.messages/store.save and clobber the transcript. (audit2 #4)
                    turn_running = current_turn is not None and not current_turn.done()
                    if turn_running or session.id in inflight:
                        await send({"type": "error", "detail": "a turn is already running"})
                        continue
                    inflight.add(session.id)
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
        finally:
            # The per-connection agent is done — release its egress-proxy listener (no-op unless
            # ZAKCODE_EGRESS_PROXY is on) so it isn't leaked for the life of the server loop.
            await _release_agent(agent)

    # ── read-only watch surface (P0-3) ──────────────────────────────────────────

    @app.get("/watch/{session_id}")
    async def watch(session_id: str, since: int | None = None) -> EventSourceResponse:
        """Read-only SSE tail of a session's events, secret-redacted (P0-3, spec sec 4+7).

        A late-joining, read-only observer (a parent watching a kid's agent, a dashboard)
        streams the SAME AgentEvents the turn-driver sees, projected to allow-listed
        SafeEvent frames. Bearer auth is enforced by the middleware (registered iff a token
        is configured) exactly like every other HTTP route. NO agent/turn is created — this
        only tails the fan-out bus. ``?since=<cursor>`` resumes after a known event (each
        frame's SSE ``id`` is its cursor); omit it to replay the retained buffer then tail.

        The literal id ``current`` is a gateway-facing alias (the PEARL watch UI streams
        ``/watch/current`` without knowing the concrete id) resolved here to the active
        loop session named by the ``.current-session`` marker. 404 if the session does
        not exist, or if ``current`` is requested with no active session.
        """
        if session_id == "current":
            # Resolve the alias BEFORE the existence check so the default watch URL works
            # without the caller knowing the concrete session id. The marker is written by
            # the sidecar-driver each iteration; absent only before the first loop turn.
            resolved = _current_session_id()
            if resolved is None:
                raise HTTPException(status_code=404, detail="no active session to watch")
            session_id = resolved
        _load_session_or_404(session_id)  # existence check; never creates an agent/turn
        bus = event_bus_registry.get_or_create(session_id)

        async def event_source() -> AsyncIterator[dict[str, str]]:
            async for cursor, event in bus.subscribe(since=since):
                safe = safe_projection.project(event)
                if safe is None:
                    continue  # dropped by the whitelist (usage/action_required/unknown type)
                yield {"id": str(cursor), "data": json.dumps(safe.model_dump())}

        # ping=15: a 15s keepalive comment so an idle watch (no turn running yet) holds the
        # connection open through proxies without emitting spurious events.
        return EventSourceResponse(event_source(), ping=15)

    @app.post("/watch/{session_id}/marker", status_code=201)
    def publish_watch_marker(session_id: str, request: WatchMarkerRequest) -> dict[str, Any]:
        """Publish a server-side meta-event to a session's watch bus (P0-3 companion).

        The ``AgentEvent`` stream carries only turn activity; a lifecycle signal a watcher needs —
        today, a driver ``session_rotated`` after the daemon dropped a session — has no other
        channel. The sidecar-driver POSTs one here (via ``ServerClient.publish_watch_marker``) so
        tailing observers see a clean rotation notice and reconnect to ``current``, instead of a
        bare stream close they must guess at. Bearer-gated like every route. Whitelist-by-
        construction still holds: the marker projects through :class:`SafeEventProjection`, so only
        its allow-listed ``SafeSessionRotated`` form ever reaches a browser. The ``current`` alias
        resolves as in ``GET /watch`` so the driver can publish without knowing the concrete id.

        Uses ``get`` (not ``get_or_create``): the marker targets EXISTING watch observers, so a
        session with no live bus (no watchers, or already discarded) is a no-op — publishing into
        the void would only let any bearer holder spawn unbounded buses for arbitrary ids.
        """
        if session_id == "current":
            resolved = _current_session_id()
            if resolved is None:
                raise HTTPException(status_code=404, detail="no active session to watch")
            session_id = resolved
        bus = event_bus_registry.get(session_id)
        if bus is None:
            return {"published": False, "cursor": None}
        cursor = bus.publish(request)
        return {"published": True, "cursor": cursor}

    # ── sidecar read-only surface (P0-4 /workspace/summary, P0-5 /sidecar/health) ─
    # These back the env-server SidecarProxyVerticle (spec P0-7): it proxies the public
    # /sidecar/summary → /workspace/summary and /sidecar/health → this /sidecar/health.
    # Both are auth-required: _AUTH_EXEMPT_PATHS is the EXACT string "/health", so
    # "/sidecar/health" still carries the bearer token (it is NOT the liveness probe).
    # Both READ artifacts a SEPARATE component writes (the agent loop / sidecar-driver:
    # research/journal.md, research/findings*, .current-session) and MUST degrade
    # gracefully when those are absent — this surface is safe to poll before the loop
    # has written anything. Read-only: never spawns an agent or turn.
    _workspace_root = resolved_settings.workspace_root

    def _current_session_id() -> str | None:
        """Active loop session id from the ``.current-session`` marker, or None if absent.

        Written by the sidecar-driver (spec P0-8) each iteration; absent before the first
        loop turn. Never raises — a missing/empty/unreadable marker reads as None.
        """
        try:
            value = (
                (_workspace_root / ".current-session")
                .read_text(encoding="utf-8", errors="replace")
                .strip()
            )
        except OSError:
            return None
        return value or None

    def _count_findings() -> int:
        """Count entries in the agent's research findings list (spec sec 10.2).

        The v0 Tricks loop writes findings alongside ``research/journal.md``. Two shapes
        are accepted so the count holds regardless of which the writer uses: a
        ``research/findings/`` directory (one file per finding) or a flat
        ``research/findings.md`` list (markdown bullets, falling back to non-blank lines).
        Absent → 0. Never raises.
        """
        research = _workspace_root / "research"
        try:
            findings_dir = research / "findings"
            if findings_dir.is_dir():
                return sum(
                    1 for p in findings_dir.iterdir() if p.is_file() and not p.name.startswith(".")
                )
            findings_file = research / "findings.md"
            if findings_file.is_file():
                lines = findings_file.read_text(encoding="utf-8", errors="replace").splitlines()
                bullets = [ln for ln in lines if ln.lstrip()[:2] in ("- ", "* ", "+ ")]
                return len(bullets) if bullets else sum(1 for ln in lines if ln.strip())
        except OSError:
            return 0
        return 0

    @app.get("/workspace/summary")
    def workspace_summary() -> dict[str, Any]:
        """World-view summary for the gateway: research journal head + finding count (P0-4).

        Returns the first 5000 chars of the research journal (empty string if absent),
        the finding count, and the active loop session id. Every field degrades gracefully,
        so the gateway may poll this before the agent loop has produced any research.

        Journal location differs by agent shape: the v0 Tricks loop writes
        ``research/journal.md``; a lean research agent writes ``journal/journal.md``.
        Read whichever exists (``research/`` takes precedence).
        """
        journal = ""
        for journal_path in (
            _workspace_root / "research" / "journal.md",
            _workspace_root / "journal" / "journal.md",
        ):
            try:
                if journal_path.is_file():
                    journal = journal_path.read_text(encoding="utf-8", errors="replace")[:5000]
                    break
            except OSError:
                journal = ""
        return {
            "journal": journal,
            "finding_count": _count_findings(),
            "session_id": _current_session_id(),
        }

    @app.get("/sidecar/health")
    def sidecar_health() -> dict[str, Any]:
        """Sidecar health + active-session discovery for the gateway (P0-5).

        DISTINCT from the unauthenticated ``/health`` liveness probe — this one is
        auth-required and reports the active loop session id from ``.current-session``
        (None before the first loop turn) so the gateway can discover which session to watch.
        """
        return {"status": "ok", "active_session_id": _current_session_id()}

    # ── PEARL viewer nudge (§Layer-4) ─────────────────────────────────────────────
    # Backs the env-server /sidecar/nudge proxy → gateway /nudge → Vinheim NudgeInput.
    @app.post("/nudge")
    def nudge(request: NudgeRequest) -> dict[str, Any]:
        """Queue a viewer suggestion for the driver to fold into the next turn's
        preamble. The suggestion is written to ``<workspace>/.nudge`` (atomic
        temp-write + replace), NEVER sent as a chat message. Refuses with 429 when a
        ``.nudge`` is already pending (single-slot queue) so a burst cannot stack.
        Bearer-gated by the server middleware; the gateway is the sanitization +
        rate-limit trust boundary — here we only length-cap and queue.
        """
        text = (request.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text required")
        text = text[:NUDGE_MAX_CHARS]
        root = Path(resolved_settings.workspace_root)
        target = root / ".nudge"
        if target.exists():
            raise HTTPException(status_code=429, detail="a suggestion is already pending")
        root.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".nudge.{os.getpid()}.tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, target)
        return {"queued": True}

    # ── PEARL user say (watch/talk unification) ───────────────────────────────────
    # Backs the env-server /sidecar/say proxy → gateway /say → the unified session
    # view. Unlike a /nudge suggestion (folded into the preamble), a say IS the next
    # turn's message: the driver replaces its continue cue with it and publishes a
    # user_message watch marker, so every watcher sees the question and then the
    # reply streaming on the same /watch feed — talking is just the session's next turn.
    @app.post("/say")
    def say(request: SayRequest) -> dict[str, Any]:
        """Queue a user message for the driver to deliver as the next turn's message.
        Written to ``<workspace>/.say`` (atomic temp-write + replace). Refuses with 429
        while one is pending (single-slot queue — the sender waits for the agent to
        finish its current thought). Bearer-gated by the server middleware; the gateway
        is the sanitization + rate-limit + ownership trust boundary — here we only
        length-cap and queue.
        """
        text = (request.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text required")
        text = text[:SAY_MAX_CHARS]
        root = Path(resolved_settings.workspace_root)
        target = root / ".say"
        if target.exists():
            raise HTTPException(status_code=429, detail="a message is already pending")
        root.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".say.{os.getpid()}.tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, target)
        return {"queued": True}

    # ── PEARL knowledge base (§10.4) — read-only browse over the pre-projected bundle ──
    # Backs the env-server /sidecar/knowledge/* proxy → gateway /knowledge/* → Vinheim
    # KnowledgeExplorer. Every route reads the already-filtered + redacted bundle the
    # Mind's KnowledgeProjection wrote (§10.3 — filter at the source); the daemon holds
    # no projection logic and fails open to an empty base before the first export.
    @app.get("/knowledge/tree")
    def knowledge_tree() -> dict[str, Any]:
        """The wiki map: node keys, titles, and parent/child edges (no bodies)."""
        bundle = read_knowledge_bundle(Path(resolved_settings.workspace_root))
        index = [
            {
                "key": str(n.get("key") or ""),
                "title": str(n.get("title") or ""),
                "parent": str(n.get("parent") or ""),
                "children": [str(c) for c in (n.get("children") or []) if c],
            }
            for n in bundle["tree"]
            if isinstance(n, dict)
        ]
        return {"nodes": index, "count": len(index)}

    @app.get("/knowledge/node/{key}")
    def knowledge_node(key: str) -> dict[str, Any]:
        """One node (title, summary sampler, full body, parent, child links). 404 if absent.

        ``body`` is the full node article — already redacted upstream by the Mind's
        KnowledgeProjection for a full Mind, or the raw note for a lean-agent workspace
        (g-335-191). Deliberately kept OUT of ``/knowledge/tree`` (the map), which stays
        lightweight; the body is fetched only on a per-node click.
        """
        bundle = read_knowledge_bundle(Path(resolved_settings.workspace_root))
        for n in bundle["tree"]:
            if isinstance(n, dict) and str(n.get("key") or "") == key:
                return {
                    "key": key,
                    "title": str(n.get("title") or ""),
                    "summary": str(n.get("summary") or ""),
                    "body": str(n.get("body") or ""),
                    "parent": str(n.get("parent") or ""),
                    "children": [str(c) for c in (n.get("children") or []) if c],
                }
        raise HTTPException(status_code=404, detail=f"no node {key!r}")

    @app.get("/knowledge/hypotheses")
    def knowledge_hypotheses() -> dict[str, Any]:
        """Projected hypotheses (statement, horizon, status, outcome)."""
        bundle = read_knowledge_bundle(Path(resolved_settings.workspace_root))
        items = [h for h in bundle["hypotheses"] if isinstance(h, dict)]
        return {"hypotheses": items, "count": len(items)}

    @app.get("/knowledge/guardrails")
    def knowledge_guardrails() -> dict[str, Any]:
        """Projected domain guardrails (plain-language rules)."""
        bundle = read_knowledge_bundle(Path(resolved_settings.workspace_root))
        items = [g for g in bundle["guardrails"] if isinstance(g, dict)]
        return {"guardrails": items, "count": len(items)}

    @app.get("/knowledge/export")
    def knowledge_export() -> dict[str, Any]:
        """The whole projected base as one downloadable bundle (PEARL §10.5).

        Emits the OKF transfer-bundle export shape — a portable, human-readable
        wiki (Markdown concept docs + a manifest), NOT the internal JSON dump
        the browse routes serve. See ``okf_bundle``. The browse routes
        (/tree, /node, /hypotheses, /guardrails) are unchanged: they back a live
        UI and speak the viewer shape; only the DOWNLOAD boundary is OKF.
        """
        return okf_bundle(read_knowledge_bundle(Path(resolved_settings.workspace_root)))

    # ── wire-schema contract + web client ────────────────────────────────────────

    @app.get("/schema/events")
    def get_events_schema() -> dict[str, Any]:
        """Publish the AgentEvent JSON Schema (the wire contract the web client renders).

        Generated from the same adapter that serializes every frame, so it cannot drift
        from the actual stream. The bundled web client fetches this to stay in lockstep.
        """
        return events_schema()

    # Serve the bundled thin web client (a pure AgentEvent renderer — no agent logic).
    # Mounted LAST so it never shadows the API routes above; ``html=True`` serves
    # ``index.html`` at ``/``. Absent in some checkouts, so guard the mount.
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

        app.mount("/app", StaticFiles(directory=static_dir, html=True), name="webclient")

    return app


__all__ = ["create_app", "AgentLike", "AgentFactory"]

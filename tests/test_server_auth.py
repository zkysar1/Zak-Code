"""Tests for the server's multi-tenant hardening: bearer auth, the model allowlist,
and the genericized WebSocket error frame.

Hermetic: fake agent/provider factories (no network, no model) + a tmp-dir SessionStore.
The auth middleware is only registered when ``Settings.auth_token`` is set, so the
no-token path proves back-compat (the loopback-dev posture is unchanged).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from zakcode.agent.loop import TurnResult  # noqa: E402
from zakcode.config import Settings  # noqa: E402
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta  # noqa: E402
from zakcode.messages import Message  # noqa: E402
from zakcode.providers.base import Capabilities, LLMResult, Provider  # noqa: E402
from zakcode.server.app import create_app  # noqa: E402
from zakcode.session.store import Session, SessionStore  # noqa: E402
from zakcode.usage import Usage  # noqa: E402

TOKEN = "s3cret-tenant-token"


class _EchoAgent:
    """Minimal AgentLike: mutates the session and emits one text delta + done."""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def arun_turn(self, user_text: str) -> TurnResult:
        self.session.add_message(Message.user(user_text))
        return TurnResult(stop_reason="completed", iterations=1, usage=Usage(total_tokens=1))

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        yield AgentTextDelta(text=f"echo:{user_text}")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage(total_tokens=1))


class _BoomAgent:
    """Raises inside the stream, to exercise the WS error catch-all."""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def arun_turn(self, user_text: str) -> TurnResult:  # pragma: no cover - unused
        return TurnResult(stop_reason="completed", iterations=1)

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        yield AgentTextDelta(text="starting")
        raise RuntimeError("secret path C:/Users/zak/.env leaked here")


class _ScriptedProvider(Provider):
    def __init__(self, text: str = "ok") -> None:
        self._text = text

    async def acomplete(self, messages, *, system=None, tools=None, response_format=None, **kw):
        return LLMResult(text=self._text, usage=Usage(total_tokens=1, cost_usd=0.0))

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


def _app(
    tmp_path: Path,
    *,
    auth_token: str | None = None,
    allowed_models: list[str] | None = None,
    agent_cls: type = _EchoAgent,
) -> tuple[TestClient, SessionStore]:
    settings = Settings(
        default_model="scripted/test",
        workspace_root=tmp_path,
        auth_token=auth_token,
        allowed_models=allowed_models or [],
    )
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings,
        store=store,
        agent_factory=lambda s, m, p: agent_cls(s),
        provider_factory=lambda m: _ScriptedProvider(),
    )
    return TestClient(app, raise_server_exceptions=False), store


def _sid(store: SessionStore) -> str:
    session = Session(cwd=".", model="scripted/test")
    store.save(session)
    return session.id


# ── auth off: back-compat ────────────────────────────────────────────────────────


def test_no_token_routes_work_without_header(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)
    assert client.get("/health").status_code == 200
    assert client.get("/tools").status_code == 200
    assert client.post("/chat", json={"message": "hi"}).status_code == 200


def test_ws_auth_off_echoes_offered_subprotocol(tmp_path: Path) -> None:
    # Auth off: a browser client written for an auth-ON deployment always offers
    # ``bearer, <token>``. Per RFC 6455 it aborts the handshake if the server selects no
    # subprotocol, so the auth-off path must ECHO ``bearer`` to stay connectable by one client.
    client, store = _app(tmp_path)  # no auth_token
    sid = _sid(store)
    with client.websocket_connect(f"/ws/{sid}", subprotocols=["bearer", "ignored"]) as ws:
        assert ws.accepted_subprotocol == "bearer"
        ws.send_json({"type": "input", "message": "hi"})
        assert ws.receive_json()["text"] == "echo:hi"


def test_ws_auth_off_no_subprotocol_offered_accepts_plain(tmp_path: Path) -> None:
    # ...and a client that offers NONE (the bundled web client) still connects with none selected.
    client, store = _app(tmp_path)
    sid = _sid(store)
    with client.websocket_connect(f"/ws/{sid}") as ws:
        assert ws.accepted_subprotocol is None
        ws.send_json({"type": "input", "message": "hi"})
        assert ws.receive_json()["text"] == "echo:hi"


# ── auth on: HTTP ──────────────────────────────────────────────────────────────────


def test_health_is_exempt_when_auth_on(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, auth_token=TOKEN)
    assert client.get("/health").status_code == 200  # no header, still 200


def test_protected_route_401_without_token(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, auth_token=TOKEN)
    r = client.get("/tools")
    assert r.status_code == 401
    assert r.json()["detail"] == "unauthorized"


def test_protected_route_401_with_wrong_token(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, auth_token=TOKEN)
    r = client.get("/tools", headers={"Authorization": "Bearer not-the-token"})
    assert r.status_code == 401


def test_protected_route_200_with_correct_token(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, auth_token=TOKEN)
    r = client.get("/tools", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_chat_requires_token(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, auth_token=TOKEN)
    assert client.post("/chat", json={"message": "hi"}).status_code == 401
    ok = client.post("/chat", json={"message": "hi"}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200


def test_non_bearer_scheme_rejected(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, auth_token=TOKEN)
    r = client.get("/tools", headers={"Authorization": f"Basic {TOKEN}"})
    assert r.status_code == 401


def test_token_matches_non_ascii_returns_false_without_raising() -> None:
    # A non-ASCII presented token must compare False, never raise: str hmac.compare_digest
    # raises TypeError on non-ASCII, so we compare UTF-8 bytes. (HTTP headers can't carry
    # non-ASCII, but the WS ``?token=`` query param can.)
    from zakcode.server.app import _extract_bearer, _token_matches

    assert _token_matches("tøken-with-ünicode", TOKEN) is False
    assert _token_matches(None, TOKEN) is False
    assert _token_matches(TOKEN, TOKEN) is True
    # _extract_bearer: case-insensitive scheme, None for non-bearer / empty / missing.
    assert _extract_bearer("Bearer abc") == "abc"
    assert _extract_bearer("bearer abc") == "abc"
    assert _extract_bearer("Basic abc") is None
    assert _extract_bearer("Bearer   ") is None
    assert _extract_bearer(None) is None


def test_auth_token_never_in_config(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, auth_token=TOKEN)
    r = client.get("/config", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200
    body = r.json()
    assert "auth_token" not in body
    assert TOKEN not in json.dumps(body)


# ── auth on: WebSocket ─────────────────────────────────────────────────────────────


def test_ws_rejected_without_token(tmp_path: Path) -> None:
    client, store = _app(tmp_path, auth_token=TOKEN)
    sid = _sid(store)
    with pytest.raises(WebSocketDisconnect), client.websocket_connect(f"/ws/{sid}"):
        pass


def test_ws_rejected_with_wrong_subprotocol_token(tmp_path: Path) -> None:
    client, store = _app(tmp_path, auth_token=TOKEN)
    sid = _sid(store)
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(f"/ws/{sid}", subprotocols=["bearer", "nope"]),
    ):
        pass


def test_ws_accepts_subprotocol_token(tmp_path: Path) -> None:
    # Browsers authenticate the WS via the Sec-WebSocket-Protocol "bearer, <token>" subprotocol.
    client, store = _app(tmp_path, auth_token=TOKEN)
    sid = _sid(store)
    with client.websocket_connect(f"/ws/{sid}", subprotocols=["bearer", TOKEN]) as ws:
        ws.send_json({"type": "input", "message": "hi"})
        assert ws.receive_json()["text"] == "echo:hi"


def test_ws_accepts_authorization_header(tmp_path: Path) -> None:
    client, store = _app(tmp_path, auth_token=TOKEN)
    sid = _sid(store)
    with client.websocket_connect(f"/ws/{sid}", headers={"Authorization": f"Bearer {TOKEN}"}) as ws:
        ws.send_json({"type": "input", "message": "hi"})
        assert ws.receive_json()["text"] == "echo:hi"


def test_ws_query_token_is_ignored(tmp_path: Path) -> None:
    # The ?token= query param is deliberately NOT honored (it would be persisted in uvicorn's
    # access log). Even a CORRECT token in the URL must fail the handshake. (review: WS log leak)
    client, store = _app(tmp_path, auth_token=TOKEN)
    sid = _sid(store)
    with pytest.raises(WebSocketDisconnect), client.websocket_connect(f"/ws/{sid}?token={TOKEN}"):
        pass


# ── model allowlist ────────────────────────────────────────────────────────────────


def test_complete_disallowed_model_400(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, allowed_models=["ollama_chat/qwen2.5:3b"])
    r = client.post("/complete", json={"prompt": "hi", "model": "openai/gpt-4o"})
    assert r.status_code == 400
    assert "not allowed" in r.json()["detail"]


def test_complete_allowed_model_ok(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, allowed_models=["ollama_chat/qwen2.5:3b"])
    r = client.post("/complete", json={"prompt": "hi", "model": "ollama_chat/qwen2.5:3b"})
    assert r.status_code == 200


def test_complete_no_override_unaffected_by_allowlist(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, allowed_models=["ollama_chat/qwen2.5:3b"])
    r = client.post("/complete", json={"prompt": "hi"})  # uses the server default
    assert r.status_code == 200


def test_chat_disallowed_model_400(tmp_path: Path) -> None:
    client, _ = _app(tmp_path, allowed_models=["scripted/test"])
    r = client.post("/chat", json={"message": "hi", "model": "openai/gpt-4o"})
    assert r.status_code == 400


def test_empty_allowlist_permits_any_model(tmp_path: Path) -> None:
    client, _ = _app(tmp_path)  # no allowlist
    r = client.post("/complete", json={"prompt": "hi", "model": "openai/gpt-4o"})
    assert r.status_code == 200


# ── WS error frame does not leak internals ─────────────────────────────────────────


def test_ws_error_frame_is_generic(tmp_path: Path) -> None:
    client, store = _app(tmp_path, agent_cls=_BoomAgent)
    sid = _sid(store)
    with client.websocket_connect(f"/ws/{sid}") as ws:
        ws.send_json({"type": "input", "message": "go"})
        assert ws.receive_json()["text"] == "starting"
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["detail"] == "internal error"
        # The raw exception text (with the filesystem path) must NOT cross the wire.
        assert ".env" not in json.dumps(err)
        assert "RuntimeError" not in json.dumps(err)

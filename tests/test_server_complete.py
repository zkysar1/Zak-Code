"""Tests for POST /complete — the raw schema-valid completion endpoint (no tools/loop/session).

Hermetic: an injected ``provider_factory`` returns a scripted provider, so nothing touches a
network or a model. ``agent_factory`` is stubbed since /complete never builds an agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from zakcode.config import Settings  # noqa: E402
from zakcode.providers.base import Capabilities, LLMResult, Provider, RequestFailed  # noqa: E402
from zakcode.server.app import create_app  # noqa: E402
from zakcode.session.store import SessionStore  # noqa: E402
from zakcode.usage import Usage  # noqa: E402

_STR_SCHEMA = {
    "type": "object",
    "required": ["a"],
    "additionalProperties": False,
    "properties": {"a": {"type": "string"}},
}

_LODESTAR_SCHEMA = {
    "type": "object",
    "required": ["context", "approach", "outcome", "lesson", "tags"],
    "additionalProperties": False,
    "properties": {
        "context": {"type": "string"},
        "approach": {"type": "array", "items": {"type": "string"}},
        "outcome": {"type": "string"},
        "lesson": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}


class _ScriptedProvider(Provider):
    """Returns canned texts in order (repeats the last); raises ``error`` if given."""

    def __init__(self, texts: list[str] | None = None, *, error: Exception | None = None) -> None:
        self._texts = list(texts or [""])
        self._last = self._texts[0]
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def acomplete(self, messages, *, system=None, tools=None, response_format=None, **kw):
        if self._error is not None:
            raise self._error
        self.calls.append(
            {"system": system, "response_format": response_format, "n": len(messages)}
        )
        if self._texts:
            self._last = self._texts.pop(0)
        return LLMResult(text=self._last, usage=Usage(total_tokens=1, cost_usd=0.0))

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


def _client(
    tmp_path: Path, provider: Provider, models: list[str | None] | None = None
) -> TestClient:
    def provider_factory(model: str | None) -> Provider:
        if models is not None:
            models.append(model)
        return provider

    settings = Settings(default_model="scripted/test", context_window=8192, workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings,
        store=store,
        provider_factory=provider_factory,
        agent_factory=lambda s, m, p: None,  # /complete never builds an agent
    )
    # raise_server_exceptions=False so a regression that 500s shows as status 500 (the
    # "/complete never 500" invariant is asserted directly), not re-raised as an exception.
    return TestClient(app, raise_server_exceptions=False)


def test_complete_no_schema_returns_raw_text(tmp_path: Path) -> None:
    client = _client(tmp_path, _ScriptedProvider(["just the answer"]))
    r = client.post("/complete", json={"prompt": "hi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["data"] is None
    assert body["text"] == "just the answer"
    assert body["repaired"] is False


def test_complete_schema_valid_passthrough(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    provider = _ScriptedProvider(['{"a": "x"}'])
    client = _client(tmp_path, provider)
    r = client.post("/complete", json={"prompt": "hi", "schema": _STR_SCHEMA})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["data"] == {"a": "x"}
    assert body["repaired"] is False
    assert provider.calls[0]["response_format"] is not None  # schema requested upstream


def test_complete_invalid_then_repaired(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    client = _client(tmp_path, _ScriptedProvider(['{"a": 1}', '{"a": "x"}']))
    r = client.post("/complete", json={"prompt": "hi", "schema": _STR_SCHEMA})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["data"] == {"a": "x"} and body["repaired"] is True


def test_complete_unrecoverable_returns_502(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    client = _client(tmp_path, _ScriptedProvider(['{"a": 1}']))  # always invalid
    r = client.post("/complete", json={"prompt": "hi", "schema": _STR_SCHEMA})
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["error"] == "schema_validation_failed"
    assert detail["raw_text"] == '{"a": 1}'


def test_complete_lodestar_schema_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    valid = '{"context":"c","approach":["a1","a2"],"outcome":"o","lesson":"l","tags":["t"]}'
    client = _client(tmp_path, _ScriptedProvider([valid]))
    r = client.post(
        "/complete",
        json={"messages": [{"role": "user", "content": "a trace"}], "schema": _LODESTAR_SCHEMA},
    )
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["lesson"] == "l" and data["approach"] == ["a1", "a2"]


def test_complete_prompt_and_messages_forms_equivalent(tmp_path: Path) -> None:
    r1 = _client(tmp_path, _ScriptedProvider(["ok"])).post("/complete", json={"prompt": "hi"})
    r2 = _client(tmp_path, _ScriptedProvider(["ok"])).post(
        "/complete", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert r1.json()["text"] == "ok" and r2.json()["text"] == "ok"


def test_complete_both_prompt_and_messages_400(tmp_path: Path) -> None:
    client = _client(tmp_path, _ScriptedProvider(["x"]))
    r = client.post(
        "/complete",
        json={"prompt": "hi", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 400


def test_complete_neither_prompt_nor_messages_400(tmp_path: Path) -> None:
    client = _client(tmp_path, _ScriptedProvider(["x"]))
    r = client.post("/complete", json={})
    assert r.status_code == 400


def test_complete_model_override_reaches_factory(tmp_path: Path) -> None:
    models: list[str | None] = []
    client = _client(tmp_path, _ScriptedProvider(["ok"]), models=models)
    client.post("/complete", json={"prompt": "hi", "model": "override/x"})
    assert models == ["override/x"]


def test_complete_provider_error_returns_502(tmp_path: Path) -> None:
    client = _client(tmp_path, _ScriptedProvider(error=RequestFailed("upstream boom")))
    r = client.post("/complete", json={"prompt": "hi"})
    assert r.status_code == 502
    assert r.json()["detail"]["error"] == "provider_error"


def test_complete_creates_no_session(tmp_path: Path) -> None:
    client = _client(tmp_path, _ScriptedProvider(["ok"]))
    client.post("/complete", json={"prompt": "hi"})
    assert client.get("/sessions").json() == []


def test_complete_malformed_schema_returns_400(tmp_path: Path) -> None:
    # A meta-invalid client schema is a 400 (caller error), NOT an unhandled 500. (review2 #1)
    pytest.importorskip("jsonschema")
    client = _client(tmp_path, _ScriptedProvider(['{"a": "x"}']))
    r = client.post("/complete", json={"prompt": "hi", "schema": {"type": "striing"}})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "invalid_schema"


def test_complete_unresolvable_ref_schema_never_500(tmp_path: Path) -> None:
    # An unresolvable $ref slips past meta-validation but must still never 500 (clean 5xx).
    pytest.importorskip("jsonschema")
    client = _client(tmp_path, _ScriptedProvider(['{"a": "x"}']))
    r = client.post("/complete", json={"prompt": "hi", "schema": {"$ref": "#/definitions/Missing"}})
    assert r.status_code != 500

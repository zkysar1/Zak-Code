"""Tests for the PEARL surfaces: POST /nudge (§Layer-4) + GET /knowledge/* (§10.4).

/nudge queues a single viewer suggestion into ``<workspace>/.nudge`` (atomic, single-slot,
length-capped) — never a chat message. The /knowledge/* routes are read-only browses over the
pre-projected ``.knowledge-bundle.json`` the Mind's KnowledgeProjection wrote (§10.3 — filter at
the source); the daemon holds no projection logic and fails open to an empty base before the first
export. All are plain JSON (no streaming), so Starlette's TestClient drives them directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from zakcode.config import Settings
from zakcode.server.app import create_app
from zakcode.session.store import Session, SessionStore


class _FakeAgent:
    """Minimal AgentLike — never invoked here (no turn runs on these routes)."""

    def __init__(self, session: Session) -> None:
        self.session = session


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


def _make_app(workspace: Path) -> FastAPI:
    settings = Settings(default_model="scripted/test", workspace_root=workspace)
    store = SessionStore(base_dir=workspace / "sessions")
    return create_app(settings=settings, store=store, agent_factory=_factory)


def _client(workspace: Path) -> TestClient:
    return TestClient(_make_app(workspace))


def _seed_bundle(workspace: Path) -> None:
    bundle = {
        "counts": {"tree": 2, "hypotheses": 1, "guardrails": 1},
        "tree": [
            {"key": "root", "title": "Root", "summary": "the top", "parent": "", "children": ["leaf"]},
            {"key": "leaf", "title": "Leaf", "summary": "a child", "parent": "root", "children": []},
        ],
        "hypotheses": [{"statement": "H1", "horizon": "short", "status": "active"}],
        "guardrails": [{"rule": "always redact secrets"}],
        "lessons": [{"lesson": "test before ship"}],
    }
    (workspace / ".knowledge-bundle.json").write_text(json.dumps(bundle), encoding="utf-8")


# ── /nudge ────────────────────────────────────────────────────────────────────

def test_nudge_writes_single_slot_file(tmp_path: Path) -> None:
    resp = _client(tmp_path).post("/nudge", json={"text": "try looking at gravity"})
    assert resp.status_code == 200
    assert resp.json() == {"queued": True}
    assert (tmp_path / ".nudge").read_text(encoding="utf-8").strip() == "try looking at gravity"


def test_nudge_empty_text_is_400(tmp_path: Path) -> None:
    assert _client(tmp_path).post("/nudge", json={"text": "   "}).status_code == 400
    assert not (tmp_path / ".nudge").exists()


def test_nudge_second_pending_is_429(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.post("/nudge", json={"text": "first"}).status_code == 200
    assert client.post("/nudge", json={"text": "second"}).status_code == 429
    # the first suggestion is preserved, not overwritten by the burst
    assert (tmp_path / ".nudge").read_text(encoding="utf-8").strip() == "first"


def test_nudge_length_capped(tmp_path: Path) -> None:
    _client(tmp_path).post("/nudge", json={"text": "x" * 5000})
    assert len((tmp_path / ".nudge").read_text(encoding="utf-8").strip()) == 500


# ── /knowledge/* ────────────────────────────────────────────────────────────────

def test_knowledge_tree_returns_map(tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    body = _client(tmp_path).get("/knowledge/tree").json()
    assert body["count"] == 2
    keys = {n["key"] for n in body["nodes"]}
    assert keys == {"root", "leaf"}


def test_knowledge_node_found_and_404(tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    client = _client(tmp_path)
    node = client.get("/knowledge/node/leaf").json()
    assert node["title"] == "Leaf"
    assert node["parent"] == "root"
    assert client.get("/knowledge/node/nope").status_code == 404


def test_knowledge_hypotheses_and_guardrails(tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    client = _client(tmp_path)
    assert client.get("/knowledge/hypotheses").json()["count"] == 1
    assert client.get("/knowledge/guardrails").json()["count"] == 1


def test_knowledge_export_returns_full_bundle(tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    body = _client(tmp_path).get("/knowledge/export").json()
    assert body["counts"]["tree"] == 2
    assert len(body["tree"]) == 2
    assert len(body["lessons"]) == 1


def test_knowledge_fails_open_when_bundle_absent(tmp_path: Path) -> None:
    """Before the Mind's first export the bundle file is absent — every route returns empty, never 500."""
    client = _client(tmp_path)
    assert client.get("/knowledge/tree").json() == {"nodes": [], "count": 0}
    assert client.get("/knowledge/hypotheses").json() == {"hypotheses": [], "count": 0}
    assert client.get("/knowledge/export").json()["tree"] == []


def test_knowledge_fails_open_on_malformed_bundle(tmp_path: Path) -> None:
    (tmp_path / ".knowledge-bundle.json").write_text("{not valid json", encoding="utf-8")
    assert _client(tmp_path).get("/knowledge/tree").json() == {"nodes": [], "count": 0}

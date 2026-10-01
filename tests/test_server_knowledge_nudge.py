"""Tests for the knowledge surfaces: POST /nudge + GET /knowledge/*.

/nudge queues a single viewer suggestion into ``<workspace>/.nudge`` (atomic, single-slot,
length-capped) — never a chat message. The /knowledge/* routes are read-only browses over the
pre-projected ``.knowledge-bundle.json`` the host's KnowledgeProjection wrote (§10.3 — filter at
the source); the daemon holds no projection logic and fails open to an empty base before the first
export. All are plain JSON (no streaming), so Starlette's TestClient drives them directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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
    settings = Settings(
        default_model="scripted/test", context_window=8192, workspace_root=workspace
    )
    store = SessionStore(base_dir=workspace / "sessions")
    return create_app(settings=settings, store=store, agent_factory=_factory)


def _client(workspace: Path) -> TestClient:
    return TestClient(_make_app(workspace))


def _seed_bundle(workspace: Path) -> None:
    bundle = {
        "counts": {"tree": 2, "hypotheses": 1, "guardrails": 1},
        "tree": [
            {
                "key": "root",
                "title": "Root",
                "summary": "the top",
                "parent": "",
                "children": ["leaf"],
            },
            {
                "key": "leaf",
                "title": "Leaf",
                "summary": "a child",
                "parent": "root",
                "children": [],
            },
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


def _seed_node(workspace: Path, extra: dict[str, object]) -> None:
    """A one-node bundle whose row carries ``extra`` beside the viewer fields."""
    row: dict[str, object] = {
        "key": "leaf",
        "title": "Leaf",
        "summary": "a child",
        "body": "text",
        "parent": "",
        "children": [],
    }
    row.update(extra)
    bundle = json.dumps({"tree": [row]})
    (workspace / ".knowledge-bundle.json").write_text(bundle, encoding="utf-8")


def test_knowledge_node_carries_the_projected_handle(tmp_path: Path) -> None:
    """A row's ``handle`` reaches the caller unchanged.

    The projection may give an item an opaque ``handle`` so a front end can refer back to that
    one item (to correct it, say). This route rebuilds the row field by field, so a field it does
    not name never reaches the caller, and the front end cannot offer the action on a live
    workspace. Exact equality pins the whole shape: a dropped or renamed key fails here.
    """
    _seed_node(tmp_path, {"handle": "0123456789abcdef"})
    node = _client(tmp_path).get("/knowledge/node/leaf").json()
    assert node == {
        "key": "leaf",
        "title": "Leaf",
        "summary": "a child",
        "body": "text",
        "parent": "",
        "children": [],
        "handle": "0123456789abcdef",
    }


@pytest.mark.parametrize(
    "extra",
    [{}, {"handle": None}, {"handle": ""}, {"handle": 123}, {"handle": ["0123456789abcdef"]}],
    ids=["absent", "null", "empty", "number", "list"],
)
def test_knowledge_node_omits_an_unusable_handle(tmp_path: Path, extra: dict[str, object]) -> None:
    """No usable handle means no ``handle`` key at all, never ``""``, ``None`` or a coerced value.

    A missing key is how a caller tells "this item cannot be addressed" from an address, so the
    route must not invent one: ``str(None)`` would publish the address ``"None"``.
    """
    _seed_node(tmp_path, extra)
    assert "handle" not in _client(tmp_path).get("/knowledge/node/leaf").json()


def test_knowledge_node_carries_the_unredacted_mark(tmp_path: Path) -> None:
    """A row marked ``unredacted: true`` reaches the caller with the mark.

    The mark says the text shown is the item's stored text, unchanged, so a front end may offer
    a correction. Exact equality pins the whole shape beside the handle it travels with.
    """
    _seed_node(tmp_path, {"handle": "0123456789abcdef", "unredacted": True})
    node = _client(tmp_path).get("/knowledge/node/leaf").json()
    assert node == {
        "key": "leaf",
        "title": "Leaf",
        "summary": "a child",
        "body": "text",
        "parent": "",
        "children": [],
        "handle": "0123456789abcdef",
        "unredacted": True,
    }


@pytest.mark.parametrize(
    "value",
    [False, None, "true", 1, [True]],
    ids=["false", "null", "string", "one", "list"],
)
def test_knowledge_node_omits_a_mark_that_is_not_literally_true(
    tmp_path: Path, value: object
) -> None:
    """Anything but a literal ``true`` means no ``unredacted`` key at all.

    A missing key tells a front end not to offer a correction, so a value that only looks true
    must not become the mark: ``1 == True`` in Python, and ``"true"`` is truthy.
    """
    _seed_node(tmp_path, {"handle": "0123456789abcdef", "unredacted": value})
    assert "unredacted" not in _client(tmp_path).get("/knowledge/node/leaf").json()


# Every field the projection publishes on a node row, each with a value the route can serve.
# The export writes these; keep this row in step with it. The test below derives its
# expectation from this row, so a field added here is checked without extending a list.
_PUBLISHED_NODE_ROW: dict[str, object] = {
    "key": "leaf",
    "title": "Leaf",
    "summary": "a child",
    "body": "text",
    "parent": "root",
    "children": ["leaf-a"],
    "last_updated": "2026-09-30T12:00:00",
    "handle": "0123456789abcdef",
    "unredacted": True,
}

# Row fields this route does not serve, each with its reason. A stale entry fails the test
# below, so the list stays true.
_NODE_FIELDS_NOT_SERVED: dict[str, str] = {
    "last_updated": "dropped here today, as on /knowledge/tree; carrying it is a separate change",
}


def test_knowledge_node_serves_every_published_row_field(tmp_path: Path) -> None:
    """Each field of a fully published row reaches the caller unchanged, except those named.

    The route rebuilds the row field by field, so a field it does not name is dropped without
    any error. Deriving the expected keys from the row itself means the next field is caught
    as soon as it is added to ``_PUBLISHED_NODE_ROW``.
    """
    bundle = json.dumps({"tree": [_PUBLISHED_NODE_ROW]})
    (tmp_path / ".knowledge-bundle.json").write_text(bundle, encoding="utf-8")
    node = _client(tmp_path).get("/knowledge/node/leaf").json()
    served = {f for f in _PUBLISHED_NODE_ROW if f not in _NODE_FIELDS_NOT_SERVED}
    assert served  # an empty row must not pass vacuously
    assert set(_NODE_FIELDS_NOT_SERVED) <= set(_PUBLISHED_NODE_ROW)
    assert set(node) == served
    for field in sorted(served):
        assert node[field] == _PUBLISHED_NODE_ROW[field], field


def test_knowledge_hypotheses_and_guardrails(tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    client = _client(tmp_path)
    assert client.get("/knowledge/hypotheses").json()["count"] == 1
    assert client.get("/knowledge/guardrails").json()["count"] == 1


def test_knowledge_export_returns_full_bundle(tmp_path: Path) -> None:
    """/export is the OKF transfer bundle, NOT the internal viewer JSON.

    CONTRACT CHANGE: this route used to return
    ``_read_knowledge_bundle`` verbatim — a database dump. §10.5 requires a
    "portable, human-readable wiki (Markdown nodes + a manifest)". The browse
    routes still speak the viewer shape (they back a live UI); only this
    download boundary changed. The assertions below are the same coverage
    intent as before — every seeded section reaches the caller — re-expressed
    against the new shape.
    """
    _seed_bundle(tmp_path)
    body = _client(tmp_path).get("/knowledge/export").json()
    assert body["bundle"]["format"] == "okf-transfer-bundle"
    assert body["bundle"]["counts"] == {
        "nodes": 2,
        "hypotheses": 1,
        "guardrails": 1,
        "lessons": 1,
    }
    # Paths are deterministic: slug of the record's own heading, or a
    # positional fallback when the record carries no recognizable heading
    # (the seeded lesson is {"lesson": ...}, an unmodelled key -> "Lesson 1").
    assert set(body["files"]) == {
        "index.md",
        "nodes/root.md",
        "nodes/leaf.md",
        "hypotheses/h1.md",
        "guardrails/always-redact-secrets.md",
        "lessons/lesson-1.md",
    }


def test_knowledge_fails_open_when_bundle_absent(tmp_path: Path) -> None:
    """Before the first export the bundle is absent — every route returns empty, never 500."""
    client = _client(tmp_path)
    assert client.get("/knowledge/tree").json() == {"nodes": [], "count": 0}
    assert client.get("/knowledge/hypotheses").json() == {"hypotheses": [], "count": 0}
    # /export still fails open — a valid, EMPTY OKF bundle carrying only its
    # index, never a 500 and never a bundle with no manifest.
    empty = client.get("/knowledge/export").json()
    assert empty["bundle"]["counts"] == {}
    assert list(empty["files"]) == ["index.md"]


#: Two ways a bundle is unreadable: broken JSON, and bytes that are not UTF-8.
@pytest.mark.parametrize(
    "broken", [b"{not valid json", b'{"tree": []}\xff'], ids=["not-json", "not-utf8"]
)
def test_knowledge_fails_open_on_malformed_bundle(tmp_path: Path, broken: bytes) -> None:
    (tmp_path / ".knowledge-bundle.json").write_bytes(broken)
    assert _client(tmp_path).get("/knowledge/tree").json() == {"nodes": [], "count": 0}

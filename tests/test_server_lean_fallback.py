"""Tests for the lean-research-agent fallbacks on the read-only surface.

A lean research agent writes raw markdown notes to ``<workspace>/knowledge/tree/*.md``
and a running journal to ``<workspace>/journal/journal.md`` — it never runs the Mind's
KnowledgeProjection (no ``.knowledge-bundle.json``) and never uses the v0 Tricks
``research/journal.md`` path. Without a fallback the wiki (/knowledge/tree) and the
World view (/workspace/summary) both read empty even though the agent has produced
knowledge. These tests lock in the raw-file fallbacks:

  * /knowledge/tree + /knowledge/node/* fall open to raw ``knowledge/tree/*.md`` when
    no projected bundle (or an empty projected tree) is present.
  * A projected bundle, when present, still takes precedence (the redacted path is
    unchanged for a full Mind).
  * /workspace/summary reads ``journal/journal.md`` when ``research/journal.md`` is
    absent, with ``research/`` winning when both exist.
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


def _seed_raw_tree(workspace: Path, notes: dict[str, str]) -> None:
    tree = workspace / "knowledge" / "tree"
    tree.mkdir(parents=True, exist_ok=True)
    for key, body in notes.items():
        (tree / f"{key}.md").write_text(body, encoding="utf-8")


# ── /knowledge/* raw fallback ───────────────────────────────────────────────────


def test_knowledge_tree_falls_open_to_raw_notes(tmp_path: Path) -> None:
    _seed_raw_tree(
        tmp_path,
        {
            "gravity": "# Gravity\n\nBends spacetime.\n",
            "photons": "# Photons\n\nMassless.\n",
        },
    )
    body = _client(tmp_path).get("/knowledge/tree").json()
    assert body["count"] == 2
    keys = {n["key"] for n in body["nodes"]}
    assert keys == {"gravity", "photons"}
    titles = {n["title"] for n in body["nodes"]}
    assert titles == {"Gravity", "Photons"}


def test_knowledge_node_serves_raw_note_body(tmp_path: Path) -> None:
    _seed_raw_tree(tmp_path, {"gravity": "# Gravity\n\nBends spacetime.\n"})
    node = _client(tmp_path).get("/knowledge/node/gravity").json()
    assert node["key"] == "gravity"
    assert node["title"] == "Gravity"
    assert "Bends spacetime" in node["summary"]


def test_knowledge_node_carries_full_body_distinct_from_summary(tmp_path: Path) -> None:
    # g-335-191: a multi-paragraph raw note exposes a short sampler ``summary`` for the
    # map AND the full article as ``body`` so click-through shows the whole note.
    long_note = "# Gravity\n\nFirst paragraph sampler.\n\n" + "Deep detail. " * 200
    _seed_raw_tree(tmp_path, {"gravity": long_note})
    node = _client(tmp_path).get("/knowledge/node/gravity").json()
    assert node["summary"] == "First paragraph sampler."
    assert "Deep detail." in node["body"]
    assert node["body"] != node["summary"]
    assert len(node["body"]) > len(node["summary"])


def test_knowledge_tree_map_omits_body(tmp_path: Path) -> None:
    # g-335-191 verification check: the listing endpoint stays lightweight — no body.
    _seed_raw_tree(tmp_path, {"gravity": "# Gravity\n\nBends spacetime.\n\n" + "x " * 100})
    body = _client(tmp_path).get("/knowledge/tree").json()
    assert body["count"] == 1
    assert "body" not in body["nodes"][0]


def test_raw_note_title_falls_back_to_filename_without_heading(tmp_path: Path) -> None:
    _seed_raw_tree(tmp_path, {"no-heading": "just body text, no markdown heading\n"})
    node = _client(tmp_path).get("/knowledge/node/no-heading").json()
    assert node["title"] == "no-heading"


def test_projected_bundle_takes_precedence_over_raw_notes(tmp_path: Path) -> None:
    # A full Mind's redacted bundle must win — the raw path is a lean-agent-only fallback.
    _seed_raw_tree(tmp_path, {"raw-only": "# Raw Only\n\nshould be hidden\n"})
    bundle = {
        "counts": {"tree": 1},
        "tree": [
            {
                "key": "projected",
                "title": "Projected",
                "summary": "s",
                "parent": "",
                "children": [],
            },
        ],
        "hypotheses": [],
        "guardrails": [],
        "lessons": [],
    }
    (tmp_path / ".knowledge-bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    body = _client(tmp_path).get("/knowledge/tree").json()
    keys = {n["key"] for n in body["nodes"]}
    assert keys == {"projected"}
    assert "raw-only" not in keys


def test_knowledge_tree_still_empty_when_no_bundle_and_no_raw(tmp_path: Path) -> None:
    # Regression guard for the original fail-open contract (no dirs at all).
    assert _client(tmp_path).get("/knowledge/tree").json() == {"nodes": [], "count": 0}


# ── real framework store fallback (PEARL mind-api sidecar layout) ────────────────


def _seed_jsonl(workspace: Path, name: str, records: list[dict[str, object]]) -> None:
    kn = workspace / "knowledge"
    kn.mkdir(parents=True, exist_ok=True)
    (kn / name).write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_hypotheses_fall_open_to_real_pipeline_store(tmp_path: Path) -> None:
    # PEARL sidecar: the mind-api daemon (AYOAI_WORLD=<workspace>/knowledge) names
    # its pipeline store ``pipeline.jsonl``, and the note-hypothesis.sh tool writes
    # ``prediction`` + ``stage`` — the reader must surface it, not only the legacy
    # lean ``hypotheses.jsonl`` name.
    _seed_jsonl(
        tmp_path,
        "pipeline.jsonl",
        [
            {
                "id": "2026-07-17_black-holes",
                "title": "Does info escape a black hole?",
                "prediction": "Does information that falls into a black hole ever come back out?",
                "stage": "discovered",
                "horizon": "short",
                "category": "black-holes",
            }
        ],
    )
    body = _client(tmp_path).get("/knowledge/hypotheses").json()
    assert body["count"] == 1
    h = body["hypotheses"][0]
    # prediction wins as the statement; stage maps onto status.
    assert h["statement"].startswith("Does information that falls into a black hole")
    assert h["status"] == "discovered"
    assert h["horizon"] == "short"


def test_guardrails_fall_open_to_real_store(tmp_path: Path) -> None:
    # note-guardrail.sh writes {rule, category, trigger_condition, source} to
    # <workspace>/knowledge/guardrails.jsonl via the real guardrails-add.sh daemon.
    _seed_jsonl(
        tmp_path,
        "guardrails.jsonl",
        [
            {
                "id": "guard-001",
                "rule": "Never trust a single popular-science source for a physics claim.",
                "category": "black-holes",
                "trigger_condition": "when researching physics topics",
                "source": "autonomous-research",
            }
        ],
    )
    body = _client(tmp_path).get("/knowledge/guardrails").json()
    assert body["count"] == 1
    assert body["guardrails"][0]["rule"].startswith("Never trust a single popular-science")


# ── /workspace/summary journal fallback ─────────────────────────────────────────


def test_workspace_summary_falls_back_to_journal_dir(tmp_path: Path) -> None:
    (tmp_path / "journal").mkdir()
    journal = tmp_path / "journal" / "journal.md"
    journal.write_text("## Lean Journal\nturn one\n", encoding="utf-8")
    body = _client(tmp_path).get("/workspace/summary").json()
    assert body["journal"].startswith("## Lean Journal")


def test_research_journal_takes_precedence_over_journal_dir(tmp_path: Path) -> None:
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "journal.md").write_text("RESEARCH wins\n", encoding="utf-8")
    (tmp_path / "journal").mkdir()
    (tmp_path / "journal" / "journal.md").write_text("lean loses\n", encoding="utf-8")
    body = _client(tmp_path).get("/workspace/summary").json()
    assert body["journal"].startswith("RESEARCH wins")

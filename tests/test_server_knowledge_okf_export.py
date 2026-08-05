"""GET /knowledge/export conforms to the OKF transfer-bundle export shape.

PEARL-SDK-ACCESS-ARCHITECTURE §10.5 / g-335-45. The download boundary must hand
back "a portable, human-readable wiki (Markdown nodes + a manifest), not a
database dump". The contract it targets is the framework's own
``core/config/conventions/transfer-bundle-export-shape.md``, whose invariants
are numbered 1-7; each is pinned below by number so a future edit that breaks
one fails against the invariant it broke, not against an opaque snapshot.

The two invariants worth the most here are 3 and 4:

  * **3 (required `type` discriminator)** is the ONLY field a consumer may
    assume exists. If a producer ever emits a doc without it, every consumer's
    routing silently falls through to its default — a failure that looks like
    "the import worked, the content just went to the wrong place".
  * **4 (unknown keys PRESERVED)** is the load-bearing forward-compatibility
    rule. It is also the easiest to regress: the natural way to write a
    projector is to enumerate the fields you know, which drops everything else.
    The tests below plant unmodelled fields specifically to catch that.

Companion: ``test_server_knowledge_nudge.py`` covers the browse routes, which
deliberately still speak the internal viewer shape.
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
    def __init__(self, session: Session) -> None:
        self.session = session


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


def _make_app(workspace: Path) -> FastAPI:
    settings = Settings(default_model="scripted/test", workspace_root=workspace)
    store = SessionStore(base_dir=workspace / "sessions")
    return create_app(settings=settings, store=store, agent_factory=_factory)


def _export(workspace: Path, bundle: dict | None = None) -> dict:
    if bundle is not None:
        (workspace / ".knowledge-bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    return TestClient(_make_app(workspace)).get("/knowledge/export").json()


def _frontmatter(doc: str) -> dict[str, str]:
    """Parse the frontmatter block as raw key -> literal text.

    Deliberately NOT a YAML library: the point is to prove the emitted text is
    well-formed on its own terms (opens and closes with ---, one key per line),
    and to read back the literal so the JSON-scalar encoding is observable.
    """
    lines = doc.split("\n")
    assert lines[0] == "---", f"doc must open with a frontmatter fence: {lines[:2]}"
    end = lines.index("---", 1)
    out: dict[str, str] = {}
    for ln in lines[1:end]:
        k, _, v = ln.partition(": ")
        out[k] = v
    return out


RICH = {
    "tree": [
        {
            "key": "biosignatures",
            "title": "Biosignatures",
            "summary": "What life leaves behind.",
            "parent": "",
            "children": ["oxygen-anomaly", "never-projected"],
            # Unmodelled by the producer — invariant 4 says it must survive.
            "confidence": 0.82,
            "capability_level": "EXPLORE",
        },
        {"key": "oxygen-anomaly", "title": "Oxygen anomaly", "summary": "O2 out of equilibrium."},
    ],
    "hypotheses": [
        {
            "statement": "Methane spikes correlate with season",
            "horizon": "short",
            "status": "active",
            "outcome": "",
            "surprise_level": 7,
        }
    ],
    "guardrails": [{"rule": "Cross-check every claim against >=2 sources", "category": "method"}],
    "lessons": [{"title": "Read the raw spectra", "content": "Summaries hide the noise floor."}],
}


# ── invariant 1: bundle = the unit of distribution ──────────────────────────


def test_bundle_is_a_path_to_content_map_with_a_manifest(tmp_path: Path) -> None:
    body = _export(tmp_path, RICH)
    assert body["bundle"]["format"] == "okf-transfer-bundle"
    assert body["bundle"]["version"] == 1
    assert body["bundle"]["file_count"] == len(body["files"])
    # Every entry is a writable relative path -> text. Nothing absolute, no
    # traversal: write `files` to a directory verbatim and you have the bundle.
    for path, content in body["files"].items():
        assert not path.startswith("/") and ".." not in path, path
        assert isinstance(content, str) and content.endswith("\n")


# ── invariant 2: concept = one .md with YAML frontmatter ────────────────────


def test_every_concept_is_one_markdown_doc_with_frontmatter(tmp_path: Path) -> None:
    body = _export(tmp_path, RICH)
    assert body["files"], "a seeded bundle must produce documents"
    for path, doc in body["files"].items():
        assert path.endswith(".md"), path
        fm = _frontmatter(doc)
        assert fm, f"{path} has an empty frontmatter block"


# ── invariant 3: exactly one REQUIRED key — the type discriminator ──────────


def test_every_doc_carries_a_non_empty_type_discriminator(tmp_path: Path) -> None:
    body = _export(tmp_path, RICH)
    for path, doc in body["files"].items():
        fm = _frontmatter(doc)
        assert "type" in fm, f"{path} is missing the required type discriminator"
        assert json.loads(fm["type"]), f"{path} has an empty type"


def test_type_is_the_first_frontmatter_key(tmp_path: Path) -> None:
    # A consumer that streams frontmatter can route before parsing the rest.
    body = _export(tmp_path, RICH)
    for path, doc in body["files"].items():
        assert doc.split("\n")[1].startswith("type:"), path


def test_types_route_by_section(tmp_path: Path) -> None:
    files = _export(tmp_path, RICH)["files"]
    got = {p: json.loads(_frontmatter(d)["type"]) for p, d in files.items()}
    # "node", not "concept" — matches the Mind's already-shipped OKF writer
    # (knowledge-export.py write_okf_bundle). One declared format must have ONE
    # spelling of its required discriminator, or a consumer routes half the
    # bundle to its default. See the _OKF_BODY_FIELDS comment in app.py.
    assert got["nodes/biosignatures.md"] == "node"
    assert got["hypotheses/methane-spikes-correlate-with-season.md"] == "hypothesis"
    assert got["guardrails/cross-check-every-claim-against-2-sources.md"] == "guardrail"
    assert got["lessons/read-the-raw-spectra.md"] == "lesson"
    assert got["index.md"] == "index"


# ── invariant 4: unknown keys are PRESERVED, not dropped ────────────────────


def test_unmodelled_fields_survive_into_frontmatter(tmp_path: Path) -> None:
    files = _export(tmp_path, RICH)["files"]
    node = _frontmatter(files["nodes/biosignatures.md"])
    assert json.loads(node["confidence"]) == 0.82
    assert json.loads(node["capability_level"]) == "EXPLORE"
    hyp = _frontmatter(files["hypotheses/methane-spikes-correlate-with-season.md"])
    assert json.loads(hyp["surprise_level"]) == 7
    guard = _frontmatter(files["guardrails/cross-check-every-claim-against-2-sources.md"])
    assert json.loads(guard["category"]) == "method"


def test_a_wholly_unknown_field_shape_is_preserved_not_dropped(tmp_path: Path) -> None:
    # A composite value the producer models nowhere. Dropping it would be a
    # silent loss at the export boundary; the contract says carry it through.
    body = _export(
        tmp_path,
        {"tree": [{"key": "n", "title": "N", "provenance": {"source": "arxiv", "ids": [1, 2]}}]},
    )
    fm = _frontmatter(body["files"]["nodes/n.md"])
    assert "provenance" in fm
    assert json.loads(json.loads(fm["provenance"])) == {"source": "arxiv", "ids": [1, 2]}


def test_frontmatter_values_are_escaped_not_hand_quoted(tmp_path: Path) -> None:
    # Colons, quotes and newlines are exactly what breaks a hand-rolled quoter.
    nasty = 'He said: "it\'s 3:30" \n next line -- and a #hash'
    body = _export(tmp_path, {"tree": [{"key": "k", "title": "T", "note": nasty}]})
    fm = _frontmatter(body["files"]["nodes/k.md"])
    assert json.loads(fm["note"]) == nasty
    # The raw literal must be a single line — a bare newline would terminate
    # the frontmatter block early and corrupt every key after it.
    assert "\n" not in fm["note"]


# ── invariant 6: links are bundle-relative and MAY dangle ───────────────────


def test_child_links_are_bundle_relative_and_dangling_is_allowed(tmp_path: Path) -> None:
    doc = _export(tmp_path, RICH)["files"]["nodes/biosignatures.md"]
    assert "](./oxygen-anomaly.md)" in doc
    # 'never-projected' was filtered out upstream by KnowledgeProjection. The
    # link still renders — a frontier marker, not a validation error. If this
    # ever starts failing because links are pruned, invariant 6 regressed.
    assert "](./never-projected.md)" in doc
    assert "nodes/never-projected.md" not in _export(tmp_path, RICH)["files"]


# ── invariant 7: optional progressive-disclosure index ──────────────────────


def test_index_lists_every_document(tmp_path: Path) -> None:
    body = _export(tmp_path, RICH)
    index = body["files"]["index.md"]
    for path in body["files"]:
        if path == "index.md":
            continue
        assert f"](./{path})" in index, f"{path} missing from the index"


# ── robustness: the export boundary must not lose or collide records ────────


def test_records_that_slug_identically_do_not_overwrite_each_other(tmp_path: Path) -> None:
    body = _export(
        tmp_path,
        {"guardrails": [{"rule": "Same rule!"}, {"rule": "same  rule?"}, {"rule": "SAME RULE"}]},
    )
    guard_files = [p for p in body["files"] if p.startswith("guardrails/")]
    assert len(guard_files) == 3, f"a collision silently dropped a record: {guard_files}"
    assert len(set(guard_files)) == 3


def test_non_dict_rows_are_skipped_not_fatal(tmp_path: Path) -> None:
    body = _export(tmp_path, {"tree": ["a bare string", None, {"key": "ok", "title": "OK"}]})
    assert set(body["files"]) == {"index.md", "nodes/ok.md"}


def test_hostile_keys_cannot_escape_the_bundle_directory(tmp_path: Path) -> None:
    body = _export(tmp_path, {"tree": [{"key": "../../etc/passwd", "title": "x"}]})
    path = next(p for p in body["files"] if p != "index.md")
    assert ".." not in path and not path.startswith("/")
    assert path.startswith("nodes/")


def test_empty_base_still_produces_a_valid_bundle(tmp_path: Path) -> None:
    body = _export(tmp_path, {})
    assert body["bundle"]["format"] == "okf-transfer-bundle"
    assert list(body["files"]) == ["index.md"]
    assert json.loads(_frontmatter(body["files"]["index.md"])["type"]) == "index"


# ── the PROJECTED lesson shape (g-115-4606) ────────────────────────────────
#
# Every lesson fixture above is hand-authored as {title, content}. The Mind's
# KnowledgeProjection builds each lesson as exactly {title, lesson}
# (knowledge_projection.py -> bundle.lessons), and .knowledge-bundle.json is the
# only thing this path ever reads — so the suite was green against a shape the
# real producer never emits, while every real lesson rendered as a blank page.
# A fixture that does not match its producer is not coverage; these pin the
# producer's actual shape.

#: Exactly what KnowledgeProjection writes — two keys, no content/text/summary.
PROJECTED_LESSON = {
    "lessons": [{"title": "Read the raw spectra", "lesson": "Summaries hide the noise floor."}]
}


def _body(doc: str) -> str:
    """Everything after the frontmatter block and the `# heading` line."""
    assert doc.startswith("---\n"), doc[:40]
    after_fm = doc.split("\n---\n", 1)[1]
    lines = [ln for ln in after_fm.splitlines() if not ln.startswith("# ")]
    return "\n".join(lines).strip()


def test_projected_lesson_renders_a_non_empty_body(tmp_path: Path) -> None:
    # Half 1 of the fix: the body expression must reach `lesson`. Without it the
    # document is a heading over nothing.
    files = _export(tmp_path, PROJECTED_LESSON)["files"]
    doc = files["lessons/read-the-raw-spectra.md"]
    assert _body(doc) == "Summaries hide the noise floor."


def test_projected_lesson_prose_is_not_parked_in_frontmatter(tmp_path: Path) -> None:
    # Half 2: `lesson` must be MODELLED, or invariant 4's preserve-unknown-keys
    # rule faithfully carries the whole prose into a YAML scalar. Conforming as a
    # record, unreadable as a document.
    files = _export(tmp_path, PROJECTED_LESSON)["files"]
    fm = _frontmatter(files["lessons/read-the-raw-spectra.md"])
    assert "lesson" not in fm, f"prose leaked into frontmatter: {fm}"
    assert json.loads(fm["type"]) == "lesson"


def test_hand_authored_lesson_shapes_still_render(tmp_path: Path) -> None:
    # The fix must not regress the legacy/hand-authored keys the map still lists.
    for key in ("content", "text", "summary"):
        files = _export(tmp_path, {"lessons": [{"title": "T", key: "prose via " + key}]})["files"]
        assert _body(files["lessons/t.md"]) == "prose via " + key, key


def test_lesson_key_wins_over_a_stray_sibling_prose_field(tmp_path: Path) -> None:
    # Ordering is load-bearing, not cosmetic. Once `lesson` is modelled it no
    # longer falls through to frontmatter, so if the body expression preferred a
    # stray `summary` the lesson prose would vanish from BOTH places — silent
    # loss at the export boundary. This pins the tuple order that prevents it.
    files = _export(
        tmp_path,
        {"lessons": [{"title": "T", "lesson": "the real lesson", "summary": "a stray sampler"}]},
    )["files"]
    assert _body(files["lessons/t.md"]) == "the real lesson"

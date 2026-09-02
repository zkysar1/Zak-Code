"""Tests for /knowledge/program — the bundle's shared-purpose projection.

The object-shape twin of ``test_server_knowledge_self.py``. The same three
properties are pinned, for the same reasons, and the second is again the one a
reader is most likely to break by accident:

  * EMPTINESS IS THE SIGNAL, SO EMPTY IS NOT A 404. ``{}`` means "nothing
    published"; a populated object means published. Collapsing the two would tell
    a caller the route is absent when the real answer is "nothing published yet"
    (guard-5493). ``published`` states which case it is so the caller never has to
    infer it from truthiness.

    THIS MATTERS MORE HERE THAN IT DID FOR ``self``. ``project_program`` fails
    CLOSED — it publishes only the region between the literal
    ``<!-- public:begin -->`` / ``<!-- public:end -->`` markers in
    ``world/program.md``, and ``{}`` when no marker is present. self.md has an
    enforced section structure that makes a structural cut safe; program.md does
    not. So ``{}`` is the EXPECTED steady state for an unmarked world, not a
    failure, and a 404 here would misreport the normal case as a broken route.

  * ``program`` IS AN OBJECT, AND THE COERCION THAT KEEPS IT ONE IS
    ORDER-DEPENDENT. ``_KNOWLEDGE_SECTIONS`` drives a loop coercing every member
    to a LIST, so a dict section needs its own assignment — and that assignment
    must sit AFTER the loop, so it wins.

    MEASURED, not assumed. Three edits, three outcomes (mutation-tested
    2026-09-02 against this file, mirroring the self suite's table):

      A. add ``"program"`` to ``_KNOWLEDGE_SECTIONS``  -> STILL PASSES. The loop
         flattens it to ``[]`` and the later assignment overwrites that. Harmless
         alone; this file does NOT claim to catch it, because it does not.
      B. delete the ``out["program"]`` assignment       -> CAUGHT.
      C. hoist that assignment ABOVE the loop, with ``"program"`` registered
                                                       -> CAUGHT.

    So the property actually pinned is "the dict assignment exists and runs
    last", not "program is absent from the tuple". A and C differ only in ORDER.

  * THE CONSUMER HOLDS NO PROJECTION LOGIC. The marker cut is made at the source
    by the Mind's KnowledgeProjection (PEARL §10.3). This route serves what it is
    given, verbatim — a second redactor here would diverge from the real one.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from zakcode.config import Settings
from zakcode.knowledge import read_knowledge_bundle
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


def _seed_bundle(workspace: Path, payload: dict) -> None:
    (workspace / ".knowledge-bundle.json").write_text(json.dumps(payload), encoding="utf-8")


_PUBLISHED = {
    "purpose": "We are building a world members want to live in.",
    "published_region": "public:begin..public:end",
    "last_updated": "2026-09-02",
}


def test_published_program_is_served_verbatim(tmp_path: Path) -> None:
    _seed_bundle(tmp_path, {"program": _PUBLISHED, "tree": []})
    body = _client(tmp_path).get("/knowledge/program").json()
    # Verbatim: the consumer re-projects nothing (PEARL §10.3, filter-at-source).
    assert body["program"] == _PUBLISHED
    assert body["published"] is True


def test_absent_bundle_publishes_nothing_and_does_not_404(tmp_path: Path) -> None:
    # No bundle at all — the loop simply has not exported yet.
    resp = _client(tmp_path).get("/knowledge/program")
    assert resp.status_code == 200, "empty must not be reported as a missing route"
    assert resp.json() == {"program": {}, "published": False}


def test_bundle_without_a_program_key_publishes_nothing(tmp_path: Path) -> None:
    """The REAL steady state at the time this route shipped.

    Every bundle in the fleet was written before the `program` key existed, so
    this is not a hypothetical edge case — it is what every world returns until a
    producer writes the key AND that world carries the markers.
    """
    _seed_bundle(tmp_path, {"tree": [], "guardrails": [], "self": {}})
    assert _client(tmp_path).get("/knowledge/program").json() == {
        "program": {},
        "published": False,
    }


def test_wrong_shaped_program_degrades_to_unpublished_rather_than_500(tmp_path: Path) -> None:
    # A corrupt field must not 500 a browse — same posture as the list sections.
    for bad in ([1, 2], "a-string", 7, None):
        _seed_bundle(tmp_path, {"program": bad, "tree": []})
        resp = _client(tmp_path).get("/knowledge/program")
        assert resp.status_code == 200, f"{bad!r} should degrade, not raise"
        assert resp.json() == {"program": {}, "published": False}


def test_object_shape_survives_the_list_coercion(tmp_path: Path) -> None:
    """ANTI-VACUITY. Reads the bundle DIRECTLY, below the route.

    Every other test here would still pass if `program` were coerced to a list and
    the route papered over it with `isinstance(..., dict) else {}` — they would all
    just report unpublished, which is indistinguishable from an unmarked world.
    That is a REAL risk here rather than a theoretical one, because unpublished IS
    the expected answer for every world today: a route stuck permanently empty
    would look exactly like correct behaviour. This asserts the object reaches the
    reader's output intact.
    """
    _seed_bundle(tmp_path, {"program": _PUBLISHED, "tree": []})
    bundle = read_knowledge_bundle(tmp_path)
    assert isinstance(bundle["program"], dict), "program was flattened out of object shape"
    assert bundle["program"] == _PUBLISHED
    # Positive control: the list sections are still lists, so a green result above
    # cannot be explained by the coercion having stopped running altogether.
    assert isinstance(bundle["tree"], list)


def test_program_and_self_are_independent(tmp_path: Path) -> None:
    """Neither object section may clobber the other.

    Both are assigned below the same coercion loop, so a copy-paste that reused
    the wrong key would make one shadow the other while every single-key test
    above still passed.
    """
    self_payload = {"purpose": "identity, not purpose"}
    _seed_bundle(tmp_path, {"program": _PUBLISHED, "self": self_payload, "tree": []})
    client = _client(tmp_path)
    assert client.get("/knowledge/program").json() == {
        "program": _PUBLISHED,
        "published": True,
    }
    assert client.get("/knowledge/self").json() == {
        "self": self_payload,
        "published": True,
    }


def test_empty_bundle_carries_program_so_callers_never_keyerror(tmp_path: Path) -> None:
    # read_knowledge_bundle's fail-open path must carry every key a route reads.
    bundle = read_knowledge_bundle(tmp_path / "nonexistent")
    assert bundle["program"] == {}

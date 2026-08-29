"""Tests for /knowledge/self — the bundle's agent-identity projection.

Three things are pinned here, and the second is the one a reader is most likely
to break by accident:

  * EMPTINESS IS THE SIGNAL, SO EMPTY IS NOT A 404. ``{}`` means "no identity
    published"; a populated object means published. Collapsing the two would tell
    a caller the route is absent when the real answer is "nothing published yet"
    (guard-5493). ``published`` states which case it is so the caller never has to
    infer it from truthiness.

  * ``self`` IS AN OBJECT, AND THE COERCION THAT KEEPS IT ONE IS ORDER-DEPENDENT.
    ``_KNOWLEDGE_SECTIONS`` drives a loop coercing every member to a LIST, so a
    dict section needs its own assignment — and that assignment sits AFTER the
    loop deliberately, so it wins.

    MEASURED, because the obvious claim here is wrong. Three edits, three
    outcomes (mutation-tested 2026-08-29):

      A. add ``"self"`` to ``_KNOWLEDGE_SECTIONS``  -> STILL PASSES. The loop
         flattens it to ``[]`` and the later assignment overwrites that. Harmless
         on its own; this file does NOT claim to catch it, because it does not.
      B. delete the ``out["self"]`` assignment       -> CAUGHT (2 fail).
      C. hoist that assignment ABOVE the loop, with ``"self"`` registered
                                                     -> CAUGHT (2 fail).

    So the property actually pinned is "the dict assignment exists and runs last",
    not "self is absent from the tuple". A and C differ only in ORDER, which is
    why the order is worth stating rather than leaving to look incidental.

  * THE CONSUMER HOLDS NO PROJECTION LOGIC. The cut is made at the source by the
    Mind's KnowledgeProjection (PEARL §10.3). This route serves what it is given,
    verbatim — a second redactor here would diverge from the real one.
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
    (workspace / ".knowledge-bundle.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


_PUBLISHED = {
    "purpose": "The agent maintains a shared world model.",
    "created": "2026-01-02",
    "last_updated": "2026-08-29",
}


def test_published_identity_is_served_verbatim(tmp_path: Path) -> None:
    _seed_bundle(tmp_path, {"self": _PUBLISHED, "tree": []})
    body = _client(tmp_path).get("/knowledge/self").json()
    # Verbatim: the consumer re-projects nothing (PEARL §10.3, filter-at-source).
    assert body["self"] == _PUBLISHED
    assert body["published"] is True


def test_absent_bundle_publishes_nothing_and_does_not_404(tmp_path: Path) -> None:
    # No bundle at all — the loop simply has not exported yet.
    resp = _client(tmp_path).get("/knowledge/self")
    assert resp.status_code == 200, "empty must not be reported as a missing route"
    assert resp.json() == {"self": {}, "published": False}


def test_bundle_without_a_self_key_publishes_nothing(tmp_path: Path) -> None:
    # A bundle written before `self` existed: absent key, not a malformed one.
    _seed_bundle(tmp_path, {"tree": [], "guardrails": []})
    assert _client(tmp_path).get("/knowledge/self").json() == {
        "self": {},
        "published": False,
    }


def test_wrong_shaped_self_degrades_to_unpublished_rather_than_500(tmp_path: Path) -> None:
    # A corrupt field must not 500 a browse — same posture as the list sections.
    for bad in ([1, 2], "a-string", 7, None):
        _seed_bundle(tmp_path, {"self": bad, "tree": []})
        resp = _client(tmp_path).get("/knowledge/self")
        assert resp.status_code == 200, f"{bad!r} should degrade, not raise"
        assert resp.json() == {"self": {}, "published": False}


def test_object_shape_survives_the_list_coercion(tmp_path: Path) -> None:
    """ANTI-VACUITY. Reads the bundle DIRECTLY, below the route.

    Every other test here would still pass if `self` were coerced to a list and
    the route papered over it with `isinstance(..., dict) else {}` — they would
    all just report unpublished, which is indistinguishable from an empty world.
    This asserts the object reaches the reader's output intact.

    Mutation-proven against edits B and C in the module docstring; edit A does not
    break it and is not claimed to. Do not restate that claim without re-running
    the mutations — the first version of this docstring asserted A and was wrong.
    """
    _seed_bundle(tmp_path, {"self": _PUBLISHED, "tree": []})
    bundle = read_knowledge_bundle(tmp_path)
    assert isinstance(bundle["self"], dict), "self was flattened out of object shape"
    assert bundle["self"] == _PUBLISHED
    # Positive control: the list sections are still lists, so a green result above
    # cannot be explained by the coercion having stopped running altogether.
    assert isinstance(bundle["tree"], list)


def test_empty_bundle_carries_self_so_callers_never_keyerror(tmp_path: Path) -> None:
    # read_knowledge_bundle's fail-open path must carry every key a route reads.
    bundle = read_knowledge_bundle(tmp_path / "nonexistent")
    assert bundle["self"] == {}

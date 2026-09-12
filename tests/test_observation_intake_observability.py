"""Tests for the perception-intake observability on /sidecar/health (g-373-03).

The goal these pin: "today success is indistinguishable from failure." The vessel's
bridge counted a 4xx as a delivered frame, its counters had no callers, the receiver's
``superseded`` flag was never read, and nothing anywhere recorded how STALE a frame was
when it landed (§16.3 — a stale frame delivered at the next spawn's first iteration must
be visible).

The half pinned HERE is the receiving end: what /sidecar/health reports about intake.
The sending end (status-class split, per-send log line) is pinned in the environment
server's TestPerceptionBridgeVerticle.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from zakcode.config import Settings
from zakcode.server.app import OBSERVATION_MAX_CHARS, create_app
from zakcode.session.store import Session, SessionStore


class _FakeAgent:
    """Minimal AgentLike — never invoked here (no turn runs on these routes)."""

    def __init__(self, session: Session) -> None:
        self.session = session


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


def _client(workspace: Path) -> TestClient:
    settings = Settings(
        default_model="scripted/test", context_window=8192, workspace_root=workspace
    )
    store = SessionStore(base_dir=workspace / "sessions")
    app: FastAPI = create_app(settings=settings, store=store, agent_factory=_factory)
    return TestClient(app)


def _envelope(**over: object) -> dict[str, object]:
    body: dict[str, object] = {
        "envelopeVersion": 1,
        "externalClientRef": "char-42",
        "observedAt": "2026-09-06T21:00:00Z",
        "observation": {"nearbyPerception": {"units": ["a", "b"]}},
        "droppedSlices": [],
    }
    body.update(over)
    return body


def _intake(client: TestClient) -> dict[str, object]:
    resp = client.get("/sidecar/health")
    assert resp.status_code == 200, resp.text
    return resp.json()["observation_intake"]


# ── the guard-3169 property ───────────────────────────────────────────────────


def test_every_counter_reads_zero_while_idle(tmp_path: Path) -> None:
    """A counter that appears only after the first failure cannot tell "healthy" from
    "nobody called it" (guard-3169). Every field is present, and reads an explicit zero,
    before a single frame has arrived — so a bridge that has gone SILENT is
    distinguishable from one that is delivering."""
    intake = _intake(_client(tmp_path))
    for key in (
        "accepted",
        "superseded",
        "refused_bad_version",
        "refused_missing_ref",
        "refused_too_large",
    ):
        assert key in intake, f"{key} must be present before the first frame"
        assert intake[key] == 0, f"{key} reads zero, not absent"
    assert intake["last_frame_age_seconds"] is None
    assert intake["last_observed_at"] is None


# ── deliveries ────────────────────────────────────────────────────────────────


def test_accepted_frame_is_counted_and_aged(tmp_path: Path) -> None:
    client = _client(tmp_path)
    sent_at_ms = int(time.time() * 1000) - 5000  # five seconds stale
    resp = client.post("/observe", json=_envelope(observedAt=str(sent_at_ms)))
    assert resp.status_code == 200
    intake = _intake(client)
    assert intake["accepted"] == 1
    assert intake["superseded"] == 0
    age = intake["last_frame_age_seconds"]
    assert age is not None
    assert 4.0 <= age <= 60.0, f"a five-second-old frame should age to ~5s, got {age}"


def test_superseded_delivery_is_counted_separately(tmp_path: Path) -> None:
    """A superseded frame is still DELIVERED — it counts as accepted AND as superseded.
    A sustained rise in the second is the mind failing to keep up with its vessel, which
    no send/drop split can show."""
    client = _client(tmp_path)
    client.post("/observe", json=_envelope())
    client.post("/observe", json=_envelope())  # overwrites the first, still unread
    intake = _intake(client)
    assert intake["accepted"] == 2
    assert intake["superseded"] == 1, "only the second overwrote an unread frame"


# ── refusals, split by reason ─────────────────────────────────────────────────


def test_bad_version_is_refused_and_counted_not_accepted(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post("/observe", json=_envelope(envelopeVersion=99))
    assert resp.status_code == 400
    intake = _intake(client)
    assert intake["refused_bad_version"] == 1
    assert intake["accepted"] == 0, "a refusal is never an acceptance"


def test_missing_ref_is_refused_and_counted(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post("/observe", json=_envelope(externalClientRef="   "))
    assert resp.status_code == 400
    intake = _intake(client)
    assert intake["refused_missing_ref"] == 1
    assert intake["accepted"] == 0


def test_oversize_is_refused_and_counted(tmp_path: Path) -> None:
    client = _client(tmp_path)
    huge = {"bigPerception": {"blob": "x" * (OBSERVATION_MAX_CHARS + 100)}}
    resp = client.post("/observe", json=_envelope(observation=huge))
    assert resp.status_code == 413
    intake = _intake(client)
    assert intake["refused_too_large"] == 1
    assert intake["accepted"] == 0


# ── frame age parses BOTH shapes this contract actually carries ───────────────


def test_frame_age_handles_iso8601(tmp_path: Path) -> None:
    """This receiver's own tests and the discovery ledger speak ISO-8601."""
    client = _client(tmp_path)
    client.post("/observe", json=_envelope(observedAt="2026-09-06T21:00:00Z"))
    age = _intake(client)["last_frame_age_seconds"]
    assert age is not None
    assert isinstance(age, float)


def test_frame_age_handles_epoch_millis(tmp_path: Path) -> None:
    """The VESSEL puts System.currentTimeMillis() into this field. Parsing only ISO and
    guessing at this shape would make the age silently wrong for every real send."""
    client = _client(tmp_path)
    client.post("/observe", json=_envelope(observedAt=str(int(time.time() * 1000))))
    age = _intake(client)["last_frame_age_seconds"]
    assert age is not None
    assert -5.0 <= age <= 5.0, f"a just-now millis frame should age to ~0, got {age}"


def test_unparseable_observed_at_ages_to_none_not_a_fabricated_number(
    tmp_path: Path,
) -> None:
    """An age nobody can compute must read None. A fabricated 0 would say "fresh" about
    a frame whose age is unknown, which is the exact confusion this goal removes."""
    client = _client(tmp_path)
    resp = client.post("/observe", json=_envelope(observedAt="not-a-timestamp"))
    assert resp.status_code == 200, "an unreadable clock is not a reason to drop a frame"
    intake = _intake(client)
    assert intake["accepted"] == 1
    assert intake["last_frame_age_seconds"] is None
    assert intake["last_observed_at"] == "not-a-timestamp"

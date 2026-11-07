"""What ``POST /observe`` stages for a frame, and what it attempts when no hook is declared.

Waking a host for a CHANGE or a FAREWELL is the workspace's ``ObservationReceived`` hook
(``test_observe_hook.py``). What these pin is the wire and the no-hook case:

* ``kind`` and ``changedSlices`` SURVIVE the wire. Pydantic's default is ``extra="ignore"``,
  so the ``kind`` discriminator the vessel stamps on every envelope was SILENTLY DISCARDED by
  ``ObserveRequest`` and the frame still returned 200. Nothing was broken in a way anything
  could see.
* With no hook declared nothing is attempted, whatever the kind: ``wake`` answers
  ``not-attempted`` (never ``dropped``) and the frame stays staged. Both wake counters on the
  health surface are present from the first request, and stay at zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zakcode.config import Settings
from zakcode.server.app import create_app
from zakcode.session.observation_inbox import KIND_CHANGE, KIND_FAREWELL, KIND_HEARTBEAT
from zakcode.session.store import Session, SessionStore


class _FakeAgent:
    """Minimal AgentLike — never invoked here (no turn runs on this route)."""

    def __init__(self, session: Session) -> None:
        self.session = session


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


def _client(workspace: Path) -> TestClient:
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=workspace,
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


def _staged(workspace: Path) -> dict[str, object]:
    return json.loads((workspace / ".observation").read_text(encoding="utf-8"))


def _intake(client: TestClient) -> dict[str, object]:
    resp = client.get("/sidecar/health")
    assert resp.status_code == 200, resp.text
    return resp.json()["observation_intake"]


# ── the silent-drop defect ───────────────────────────────────────────────────────────


def test_kind_and_changed_slices_survive_the_wire(tmp_path: Path) -> None:
    """Undeclared fields are DISCARDED by Pydantic, not refused — so pin that they arrive."""
    resp = _client(tmp_path).post(
        "/observe",
        json=_envelope(kind=KIND_CHANGE, changedSlices=["nearbyPerception"]),
    )
    assert resp.status_code == 200
    staged = _staged(tmp_path)
    assert staged["kind"] == KIND_CHANGE
    assert staged["changedSlices"] == ["nearbyPerception"]


def test_an_unstamped_envelope_stages_an_empty_kind(tmp_path: Path) -> None:
    """Absent is distinguishable from "heartbeat": the vessel said nothing, we record that."""
    resp = _client(tmp_path).post("/observe", json=_envelope())
    assert resp.status_code == 200
    staged = _staged(tmp_path)
    assert staged["kind"] == ""
    assert staged["changedSlices"] == []


# ── no hook declared: nothing attempted ─────────────────────────────────────────────


@pytest.mark.parametrize("kind", [KIND_CHANGE, KIND_FAREWELL, KIND_HEARTBEAT])
def test_with_no_hook_nothing_is_attempted_and_the_frame_stays(tmp_path: Path, kind: str) -> None:
    """A failed wake must not read as an UN-ATTEMPTED one, nor the reverse.

    This is why the field is three states and not a boolean. With no hook there is no
    hand-over to fail: `dropped` here would report a defect on every round of a healthy
    workspace and the field would be ignored inside a day.
    """
    body = _client(tmp_path).post("/observe", json=_envelope(kind=kind)).json()
    assert body["wake"] == "not-attempted"
    assert body["accepted"] is True
    assert _staged(tmp_path)["kind"] == kind, "not waking is not dropping"


def test_both_wake_counters_are_present_from_the_first_request_and_stay_zero(
    tmp_path: Path,
) -> None:
    """Present and zero before the first frame, so a bridge that was never called stays
    distinguishable from one delivering cleanly; and a frame that attempts no wake moves neither."""
    client = _client(tmp_path)

    idle = _intake(client)
    assert idle["wake_dropped"] == 0, "present and zero before the first frame, not absent"
    assert idle["wake_delivered"] == 0

    client.post("/observe", json=_envelope(kind=KIND_CHANGE))
    after = _intake(client)
    assert after["wake_dropped"] == 0
    assert after["wake_delivered"] == 0

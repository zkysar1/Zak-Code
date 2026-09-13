"""The vessel-to-mind channel must be visible in serve.log AT ALL (g-373-40).

Sibling of ``test_observation_intake_observability.py``. That file pins what
``/sidecar/health`` REPORTS about intake; this one pins whether anything reaches the
file an operator actually tails.

The gap these close, measured 2026-09-12: zakcode configured NO stdlib logging
anywhere — zero ``basicConfig``/``dictConfig`` across ``src/zakcode``. Every
``logger.info`` therefore fell through to the root logger's last-resort handler, which
emits at WARNING and to stderr, so the ``perception-intake`` line was discarded before
it could reach stdout. And the three REFUSAL paths logged nothing at all. Together that
made the channel invisible in BOTH directions: an operator tailing serve.log could not
distinguish "no vessel is sending" from "every frame is being rejected".

That is the worst shape for this channel specifically, because P4 makes every failure on
it a SILENT DROP by design — producer cooperation is optional, so the receiver's log is
the only witness. Per guard-5501, a diagnostic's silence is not evidence until you prove
the diagnostic can fire, which is what the refusal tests below do for each reason.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zakcode.cli import LOG_LEVEL_ENV, _configure_logging
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


# ── the accept direction ──────────────────────────────────────────────────────


def test_an_accepted_frame_writes_an_info_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The success line exists in the source; what was missing is it ever being EMITTED.

    Asserted at INFO against the server's own logger, which is the level the CLI now
    configures — a success line that only appears at DEBUG would not reach serve.log.
    """
    client = _client(tmp_path)
    with caplog.at_level(logging.INFO, logger="zakcode.server"):
        resp = client.post("/observe", json=_envelope())
    assert resp.status_code == 200
    assert any("perception-intake" in r.message for r in caplog.records), (
        "an accepted frame must leave a perception-intake line: "
        f"{[r.message for r in caplog.records]}"
    )


# ── the refusal direction — the one that matters ──────────────────────────────
#
# Each case proves the diagnostic CAN fire for that specific reason (guard-5501).
# A config that only shows successes does not close this goal: under P4 a refusal is
# indistinguishable from silence from outside, so the refusal line IS the signal.


@pytest.mark.parametrize(
    ("body", "expected_status", "reason_token"),
    [
        (_envelope(envelopeVersion=99), 400, "reason=bad_version"),
        (_envelope(externalClientRef="   "), 400, "reason=missing_ref"),
        (
            _envelope(
                observation={"nearbyPerception": {"blob": "x" * (OBSERVATION_MAX_CHARS + 64)}}
            ),
            413,
            "reason=too_large",
        ),
    ],
    ids=["bad_version", "missing_ref", "too_large"],
)
def test_a_refused_frame_writes_a_warning_line(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    body: dict[str, object],
    expected_status: int,
    reason_token: str,
) -> None:
    client = _client(tmp_path)
    with caplog.at_level(logging.WARNING, logger="zakcode.server"):
        resp = client.post("/observe", json=body)
    assert resp.status_code == expected_status, resp.text
    refusals = [r.getMessage() for r in caplog.records if "REFUSED" in r.getMessage()]
    assert refusals, f"a {expected_status} must leave a REFUSED line, got: {caplog.records}"
    assert any(reason_token in m for m in refusals), (
        f"the refusal line must name WHY so an operator can act on it; "
        f"wanted {reason_token!r}, got {refusals}"
    )


# ── the CLI logging config ────────────────────────────────────────────────────


@pytest.fixture
def _restore_root_logging():
    """Snapshot and restore the root logger around a real ``_configure_logging()`` call.

    ``basicConfig(force=True)`` REMOVES every existing root handler — including the one
    pytest's logging plugin installs to back ``caplog``. Without this fixture these three
    tests would silently disarm caplog for every test that ran after them in the same
    session, which would surface as unrelated flaky failures rather than as a fault here.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        yield
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_cli_configures_stdlib_logging_to_stdout(
    monkeypatch: pytest.MonkeyPatch, _restore_root_logging: None
) -> None:
    """The census that measured zero basicConfig across src/zakcode must now find one.

    Pinned behaviourally rather than by grepping for the call: what matters is that a
    root handler exists afterwards and writes to stdout, because the deployment's
    serve.log is a stdout redirect. A handler on stderr would pass a grep and still
    leave serve.log empty.
    """
    import sys

    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    _configure_logging()
    root = logging.getLogger()
    assert root.handlers, "basicConfig must leave a root handler"
    streams = [getattr(h, "stream", None) for h in root.handlers]
    assert sys.stdout in streams, f"the handler must write to stdout, got {streams}"
    assert root.level == logging.INFO, "default level must be INFO without an opt-in"


def test_log_level_is_env_controlled(
    monkeypatch: pytest.MonkeyPatch, _restore_root_logging: None
) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "debug")
    _configure_logging()
    assert logging.getLogger().level == logging.DEBUG, (
        "level must come from the env var (case-insensitive)"
    )


def test_an_unrecognised_level_falls_back_to_info_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch, _restore_root_logging: None
) -> None:
    """A typo in an env var must not stop the server from starting."""
    monkeypatch.setenv(LOG_LEVEL_ENV, "NOT_A_LEVEL")
    _configure_logging()
    assert logging.getLogger().level == logging.INFO

"""A CHANGE envelope reaching ``POST /observe`` WAKES a sleeping mind (g-373-10 leg b).

The mind half of this already existed and was inert. ``perception-received`` was a valid
signal name (``session.py`` VALID_SIGNALS), was declared in the framework's session
manifest, and was polled as a BLOCKER-class wake by ``interruptible-sleep.sh`` — with no
writer anywhere. A vessel could perceive a change while its mind slept through it.

Two defects had to be fixed together, and the first is why nine armed perception rounds
never revealed the second: Pydantic's default is ``extra="ignore"``, so the ``kind``
discriminator the vessel stamps on every envelope was SILENTLY DISCARDED by
``ObserveRequest`` and the frame still returned 200. Nothing was broken in a way anything
could see.

What these pin, in order of how easy each is to regress:

* ``kind`` and ``changedSlices`` SURVIVE the wire — the silent-drop defect.
* Only a change wakes. A heartbeat is a timer-driven full picture; an unstamped envelope
  is not a change either. Waking on either converts every quiescent sleep into a busy-poll.
* The frame is on disk BEFORE the signal — asserted from inside the fake setter, the only
  moment the two orders are distinguishable.
* Only an AUTONOMOUS agent wakes. `reader` and `assistant` run no perpetual loop, so
  there is no sleeping reader for the marker to reach — and an UNREADABLE mode is not
  autonomous either, so it does not wake.
* Fail-open: a non-seed workspace, or one with no resident agent, still stages its frame.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from zakcode.config import Settings
from zakcode.server.app import create_app
from zakcode.session.framework_signal import (
    AUTONOMOUS_MODE,
    MODE_GET_SCRIPT,
    SIGNAL_SET_SCRIPT,
    framework_session_dir,
)
from zakcode.session.observation_inbox import (
    KIND_CHANGE,
    KIND_HEARTBEAT,
    PERCEPTION_RECEIVED_SIGNAL,
)
from zakcode.session.store import Session, SessionStore

AGENT = "alpha"


class _FakeAgent:
    """Minimal AgentLike — never invoked here (no turn runs on this route)."""

    def __init__(self, session: Session) -> None:
        self.session = session


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


def _client(workspace: Path, *, agent: str | None = AGENT) -> TestClient:
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=workspace,
        run_stop_agent=agent,
    )
    store = SessionStore(base_dir=workspace / "sessions")
    app: FastAPI = create_app(settings=settings, store=store, agent_factory=_factory)
    return TestClient(app)


def _plant_signal_setter(root: Path, body: str) -> Path:
    """Plant a stand-in for the framework's ``session-signal-set.sh`` at the real path."""
    script = root / SIGNAL_SET_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(body, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _real_setter_body() -> str:
    """A faithful stand-in: touches the marker the real script touches, nothing else."""
    return (
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'dir="agents/${AYOAI_AGENT}/session"\n'
        'mkdir -p "$dir"\n'
        'touch "$dir/$1"\n'
    )


def _real_mode_getter_body() -> str:
    """A faithful stand-in for ``session-mode-get.sh``, including its absent-file default.

    Mirrors the real script rather than echoing a fixed answer, so the tests exercise the
    contract that matters: mode file present -> its trimmed contents; mode file ABSENT ->
    ``reader``, which is a real answer and not an error.
    """
    return (
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'f="agents/${AYOAI_AGENT}/session/agent-mode"\n'
        'if [ -f "$f" ]; then tr -d "[:space:]" < "$f"; echo; else echo "reader"; fi\n'
    )


def _set_mode(root: Path, mode: str, agent: str = AGENT) -> None:
    """Write the framework's ``agent-mode`` file the way /start would."""
    session = framework_session_dir(root, agent)
    session.mkdir(parents=True, exist_ok=True)
    (session / "agent-mode").write_text(mode, encoding="utf-8")


def _plant_seed(
    root: Path, *, mode: str | None = AUTONOMOUS_MODE, setter_body: str | None = None
) -> None:
    """Plant both framework scripts a wake needs, and put the agent in ``mode``.

    ``mode=None`` plants the scripts but writes no mode file, so the getter returns its
    ``reader`` default — the shape an un-started agent presents.
    """
    _plant_signal_setter(root, setter_body or _real_setter_body())
    _plant_mode_getter(root, _real_mode_getter_body())
    if mode is not None:
        _set_mode(root, mode)


def _plant_mode_getter(root: Path, body: str) -> Path:
    """Plant a stand-in for the framework's ``session-mode-get.sh`` at the real path."""
    script = root / MODE_GET_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(body, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _frame_first_setter_body() -> str:
    """The ORDER assertion, made where it is decidable — from inside the setter.

    A wake that overtakes its own payload wakes the loop to an empty inbox. At signal time
    the staged frame must already exist, so the check belongs here rather than in a caller
    that can only ever observe the end state.
    """
    return (
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        '[ -f ".observation" ] '
        '|| { echo "frame absent at signal time" >&2; exit 9; }\n'
        'dir="agents/${AYOAI_AGENT}/session"\n'
        'mkdir -p "$dir"\n'
        'touch "$dir/$1"\n'
    )


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


def _marker(workspace: Path, agent: str = AGENT) -> Path:
    return framework_session_dir(workspace, agent) / PERCEPTION_RECEIVED_SIGNAL


def _staged(workspace: Path) -> dict[str, object]:
    return json.loads((workspace / ".observation").read_text(encoding="utf-8"))


# ── the silent-drop defect ───────────────────────────────────────────────────────────


def test_kind_and_changed_slices_survive_the_wire(tmp_path: Path) -> None:
    """Undeclared fields are DISCARDED by Pydantic, not refused — so pin that they arrive."""
    resp = _client(tmp_path, agent=None).post(
        "/observe",
        json=_envelope(kind=KIND_CHANGE, changedSlices=["nearbyPerception"]),
    )
    assert resp.status_code == 200
    staged = _staged(tmp_path)
    assert staged["kind"] == KIND_CHANGE
    assert staged["changedSlices"] == ["nearbyPerception"]


def test_an_unstamped_envelope_stages_an_empty_kind(tmp_path: Path) -> None:
    """Absent is distinguishable from "heartbeat": the vessel said nothing, we record that."""
    resp = _client(tmp_path, agent=None).post("/observe", json=_envelope())
    assert resp.status_code == 200
    staged = _staged(tmp_path)
    assert staged["kind"] == ""
    assert staged["changedSlices"] == []


# ── which envelopes wake ─────────────────────────────────────────────────────────────


def test_change_envelope_raises_the_wake(tmp_path: Path) -> None:
    _plant_seed(tmp_path)
    resp = _client(tmp_path).post("/observe", json=_envelope(kind=KIND_CHANGE))
    assert resp.status_code == 200
    assert _marker(tmp_path).exists()


def test_heartbeat_envelope_does_not_wake(tmp_path: Path) -> None:
    """A heartbeat arrives on a timer and says nothing new — waking on it is a busy-poll."""
    _plant_seed(tmp_path)
    resp = _client(tmp_path).post("/observe", json=_envelope(kind=KIND_HEARTBEAT))
    assert resp.status_code == 200
    assert not _marker(tmp_path).exists()
    # The frame still staged: not waking is not the same as not perceiving.
    assert _staged(tmp_path)["kind"] == KIND_HEARTBEAT


def test_unstamped_envelope_does_not_wake(tmp_path: Path) -> None:
    """An older vessel that stamps no kind must not be read as announcing a change."""
    _plant_seed(tmp_path)
    resp = _client(tmp_path).post("/observe", json=_envelope())
    assert resp.status_code == 200
    assert not _marker(tmp_path).exists()


def test_an_unknown_kind_does_not_wake(tmp_path: Path) -> None:
    """The gate is equality with "change", not "not a heartbeat" — fail-safe on new values."""
    _plant_seed(tmp_path)
    resp = _client(tmp_path).post("/observe", json=_envelope(kind="snapshot"))
    assert resp.status_code == 200
    assert not _marker(tmp_path).exists()


# ── ordering, placement, and the fail-open paths ─────────────────────────────────────


def test_the_frame_is_on_disk_before_the_wake(tmp_path: Path) -> None:
    _plant_seed(tmp_path, setter_body=_frame_first_setter_body())
    resp = _client(tmp_path).post("/observe", json=_envelope(kind=KIND_CHANGE))
    assert resp.status_code == 200
    assert _marker(tmp_path).exists(), "setter refused: the wake preceded its own payload"


def test_order_assertion_actually_fires(tmp_path: Path) -> None:
    """The guard above is only worth having if its setter CAN refuse. Prove it does."""
    _plant_signal_setter(tmp_path, _frame_first_setter_body())
    # Same setter, invoked with no staged frame beside it — the condition it checks.
    from zakcode.session.framework_signal import set_framework_signal

    assert set_framework_signal(tmp_path, AGENT, PERCEPTION_RECEIVED_SIGNAL) is False
    assert not _marker(tmp_path).exists()


def test_wake_lands_agent_level_not_per_session(tmp_path: Path) -> None:
    """``sessions/<SID>/`` holds same-named files the loop reads differently."""
    _plant_seed(tmp_path)
    _client(tmp_path).post("/observe", json=_envelope(kind=KIND_CHANGE))
    assert _marker(tmp_path).exists()
    strays = list((tmp_path / "agents" / AGENT).glob(f"sessions/*/{PERCEPTION_RECEIVED_SIGNAL}"))
    assert strays == []


def test_no_resident_agent_writes_no_signal_and_still_stages(tmp_path: Path) -> None:
    """No address = no signal. A non-seed workspace is untouched, never degraded.

    Seeded with NO mode file on purpose: the mode is never consulted without an agent (the
    `and agent` guard short-circuits first), so writing one here would create the very
    ``agents/`` dir this test asserts the observe path never touches.
    """
    _plant_seed(tmp_path, mode=None)
    resp = _client(tmp_path, agent=None).post("/observe", json=_envelope(kind=KIND_CHANGE))
    assert resp.status_code == 200
    assert _staged(tmp_path)["kind"] == KIND_CHANGE
    assert not (tmp_path / "agents").exists()


def test_absent_setter_still_stages_the_frame(tmp_path: Path) -> None:
    """Fail-open: losing the early wake costs latency; losing the frame costs the event."""
    resp = _client(tmp_path).post("/observe", json=_envelope(kind=KIND_CHANGE))
    assert resp.status_code == 200
    assert _staged(tmp_path)["kind"] == KIND_CHANGE
    assert not _marker(tmp_path).exists()


def test_a_failing_setter_does_not_fail_the_frame(tmp_path: Path) -> None:
    _plant_seed(tmp_path, setter_body="#!/usr/bin/env bash\nexit 7\n")
    resp = _client(tmp_path).post("/observe", json=_envelope(kind=KIND_CHANGE))
    assert resp.status_code == 200
    assert _staged(tmp_path)["kind"] == KIND_CHANGE
    assert not _marker(tmp_path).exists()


def test_a_second_change_before_the_mind_reads_is_still_one_wake(tmp_path: Path) -> None:
    """The marker is a LEVEL the loop consumes, not a counter — idempotent by design."""
    _plant_seed(tmp_path)
    client = _client(tmp_path)
    client.post("/observe", json=_envelope(kind=KIND_CHANGE))
    resp = client.post("/observe", json=_envelope(kind=KIND_CHANGE))
    assert resp.status_code == 200
    assert _marker(tmp_path).exists()


# ── autonomous-only: which MODES wake (goal outcome 1) ───────────────────────────────


def test_assistant_mode_never_writes_the_signal(tmp_path: Path) -> None:
    """Outcome 1. Assistant mode runs no loop, so there is nothing for a marker to wake."""
    _plant_seed(tmp_path, mode="assistant")
    resp = _client(tmp_path).post("/observe", json=_envelope(kind=KIND_CHANGE))
    assert resp.status_code == 200
    assert not _marker(tmp_path).exists()
    # Perception itself is unaffected — only the WAKE is gated on mode.
    assert _staged(tmp_path)["kind"] == KIND_CHANGE


def test_reader_mode_never_writes_the_signal(tmp_path: Path) -> None:
    _plant_seed(tmp_path, mode="reader")
    resp = _client(tmp_path).post("/observe", json=_envelope(kind=KIND_CHANGE))
    assert resp.status_code == 200
    assert not _marker(tmp_path).exists()
    assert _staged(tmp_path)["kind"] == KIND_CHANGE


def test_an_unstarted_agent_does_not_wake(tmp_path: Path) -> None:
    """No mode file at all: the framework's reader answers ``reader``, so no wake."""
    _plant_seed(tmp_path, mode=None)
    resp = _client(tmp_path).post("/observe", json=_envelope(kind=KIND_CHANGE))
    assert resp.status_code == 200
    assert not _marker(tmp_path).exists()


def test_an_unreadable_mode_does_not_wake(tmp_path: Path) -> None:
    """Mode getter absent entirely — cannot ask, so do not wake (fail-safe direction)."""
    _plant_signal_setter(tmp_path, _real_setter_body())  # setter present, getter is NOT
    _set_mode(tmp_path, AUTONOMOUS_MODE)  # and the mode would have said yes
    resp = _client(tmp_path).post("/observe", json=_envelope(kind=KIND_CHANGE))
    assert resp.status_code == 200
    assert not _marker(tmp_path).exists()
    assert _staged(tmp_path)["kind"] == KIND_CHANGE


def test_a_failing_mode_getter_does_not_wake_or_fail_the_frame(tmp_path: Path) -> None:
    _plant_signal_setter(tmp_path, _real_setter_body())
    _plant_mode_getter(tmp_path, "#!/usr/bin/env bash\nexit 5\n")
    _set_mode(tmp_path, AUTONOMOUS_MODE)
    resp = _client(tmp_path).post("/observe", json=_envelope(kind=KIND_CHANGE))
    assert resp.status_code == 200
    assert not _marker(tmp_path).exists()
    assert _staged(tmp_path)["kind"] == KIND_CHANGE


def test_framework_agent_mode_separates_unreadable_from_reader(tmp_path: Path) -> None:
    """None and "reader" must never collapse: "cannot ask" is not "the answer is no loop".

    Both currently decline the wake, so nothing downstream distinguishes them today — which
    is exactly why it is pinned here. A future caller that wants to LOG or ALERT on an
    unreadable seed needs the distinction to still exist.
    """
    from zakcode.session.framework_signal import framework_agent_mode

    assert framework_agent_mode(tmp_path, AGENT) is None  # no getter planted at all
    _plant_mode_getter(tmp_path, _real_mode_getter_body())
    assert framework_agent_mode(tmp_path, AGENT) == "reader"  # getter's absent-file default
    _set_mode(tmp_path, AUTONOMOUS_MODE)
    assert framework_agent_mode(tmp_path, AGENT) == AUTONOMOUS_MODE

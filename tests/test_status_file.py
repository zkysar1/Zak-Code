"""Tests for the live process status file writer and ``zakcode status`` CLI command.

Covers: each state transition, session binding, the ``working`` state, concurrent tool
counting, first-output write behavior, throttled writes, atomic write, daemon refresher,
7-day startup prune, error swallowing (never raises), privacy (no model/tool text leaks),
the CLI command (live/stale/exited, --json, --all, -w, duration formatting, detail strings),
atexit/restart paths, and ``leave_turn`` exception handling.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from zakcode.session.status_file import (
    SCHEMA_VERSION,
    STALE_THRESHOLD_SECONDS,
    PlanPosition,
    PlanSource,
    StatusFileWriter,
    _atomic_write,
    _prune_stale_files,
    read_status_files,
    status_dir,
)

# ── helpers ──────────────────────────────────────────────────────────────────

SID = "test-session-1"


def _make_writer(tmp_path: Path, session_id: str = SID) -> StatusFileWriter:
    """Create and start a fresh writer for testing."""
    w = StatusFileWriter()
    w.start(
        session_id=session_id,
        workspace=str(tmp_path),
        model="test-model",
        build="test-build-abc123",
    )
    return w


def _read(w: StatusFileWriter) -> dict[str, Any]:
    """Read the status file the writer is maintaining."""
    assert w._path is not None
    return json.loads(w._path.read_text(encoding="utf-8"))


# ── atomic write ─────────────────────────────────────────────────────────────


class TestAtomicWrite:
    def test_writes_json_atomically(self, tmp_path: Path) -> None:
        target = tmp_path / "status" / "s1.json"
        data = {"state": "idle", "v": 1}
        _atomic_write(target, data)
        assert target.exists()
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded == data

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "s.json"
        _atomic_write(target, {"a": 1})
        _atomic_write(target, {"a": 2})
        assert json.loads(target.read_text(encoding="utf-8"))["a"] == 2

    def test_no_temp_file_left_on_success(self, tmp_path: Path) -> None:
        target = tmp_path / "s.json"
        _atomic_write(target, {"x": 1})
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "s.json"


# ── prune ────────────────────────────────────────────────────────────────────


class TestPrune:
    def test_prunes_files_older_than_7_days(self, tmp_path: Path) -> None:
        old = tmp_path / "old-session.json"
        old.write_text("{}", encoding="utf-8")
        old_mtime = time.time() - 8 * 24 * 60 * 60
        os.utime(old, (old_mtime, old_mtime))

        recent = tmp_path / "recent-session.json"
        recent.write_text("{}", encoding="utf-8")

        _prune_stale_files(tmp_path)

        assert not old.exists()
        assert recent.exists()

    def test_leaves_non_json_files_alone(self, tmp_path: Path) -> None:
        txt = tmp_path / "notes.txt"
        txt.write_text("keep me", encoding="utf-8")
        old_mtime = time.time() - 8 * 24 * 60 * 60
        os.utime(txt, (old_mtime, old_mtime))

        _prune_stale_files(tmp_path)
        assert txt.exists()

    def test_handles_missing_directory(self, tmp_path: Path) -> None:
        _prune_stale_files(tmp_path / "nonexistent")


# ── writer lifecycle ─────────────────────────────────────────────────────────


class TestWriterLifecycle:
    def test_start_writes_initial_idle(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            data = _read(w)
            assert data["v"] == SCHEMA_VERSION
            assert data["state"] == "idle"
            assert data["session"] == SID
            assert data["workspace"] == str(tmp_path)
            assert data["model"] == "test-model"
            assert data["build"] == "test-build-abc123"
            assert data["pid"] == os.getpid()
            assert data["turn"] is None
            assert data["call"] is None
            assert data["tool"] is None
            assert data["wakeup"] is None
            assert data["last_turn"] is None
        finally:
            w._stop_event.set()

    def test_start_is_idempotent(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.start(session_id="other", workspace="/other", model="m2", build="b2")
            data = _read(w)
            assert data["session"] == SID  # first call wins
        finally:
            w._stop_event.set()

    def test_not_started_transitions_are_noop(self) -> None:
        w = StatusFileWriter()
        # None of these should raise.
        w.set_idle(SID)
        w.set_turn_start(SID)
        w.set_model_call_start(SID)
        w.on_thinking_delta(SID, "hello")
        w.on_text_delta(SID, "hello")
        w.on_tool_call_delta(SID, "hello")
        w.set_model_call_end(SID)
        w.set_tool_start(SID, "Bash")
        w.set_tool_end(SID)
        w.set_wakeup(SID, 1234567890.0)
        w.set_restarting()
        w.leave_turn(SID, "error")
        assert w._path is None

    def test_a_start_that_fails_leaves_the_writer_unstarted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The writer never raises: with no usable status directory it simply never starts,
        # and every later transition is a no-op.
        def no_home() -> Path:
            raise RuntimeError("could not determine the home directory")

        monkeypatch.setattr("zakcode.session.status_file.status_dir", no_home)
        w = StatusFileWriter()
        w.start(session_id=SID, workspace="/w", model="m", build="b")
        w.set_turn_start(SID)
        assert w._started is False
        assert w._path is None

    def test_start_with_wakeup(self, tmp_path: Path) -> None:
        w = StatusFileWriter()
        epoch = 1719849600.0
        w.start(
            session_id="wake-test",
            workspace=str(tmp_path),
            model="m",
            build="b",
            wakeup_due_at=epoch,
        )
        try:
            data = _read(w)
            assert data["wakeup"] is not None
            assert "due_at" in data["wakeup"]
        finally:
            w._stop_event.set()

    @pytest.mark.parametrize("left", ["idle", "restarting", "exited", "side_call"])
    def test_a_resumed_session_says_how_its_last_turn_ended(
        self, tmp_path: Path, left: str
    ) -> None:
        # A build restart resumes the session in a fresh process, which writes a new file at
        # the same path. The file the previous process left with no turn open says how the
        # last turn ended, an interrupted one included, so the new file starts with exactly
        # that. No turn is open at an idle prompt, after a restart or exit between turns, or
        # in a side call made at the prompt (a compaction) that the process never left.
        old = _make_writer(tmp_path)
        old.set_turn_start(SID)
        old.leave_turn(SID, "interrupted")
        if left == "restarting":
            old.set_restarting()
        elif left == "exited":
            old.set_exited()
        elif left == "side_call":
            old._side_call_start(SID, "compaction", "")
        old._stop_event.set()
        before = _read(old)
        assert before["state"] == ("model_call" if left == "side_call" else left)
        assert before["turn"] is None
        assert before["last_turn"]["stop_reason"] == "interrupted"

        new = _make_writer(tmp_path)
        try:
            assert _read(new)["state"] == "idle"
            assert _read(new)["last_turn"] == before["last_turn"]
            # The next turn's end replaces the carried value as usual.
            new.set_turn_start(SID)
            new.set_idle(SID, stop_reason="completed")
            assert _read(new)["last_turn"]["stop_reason"] == "completed"
        finally:
            new._stop_event.set()

    @pytest.mark.parametrize("restarted", [False, True])
    def test_a_file_left_mid_turn_carries_nothing(self, tmp_path: Path, restarted: bool) -> None:
        # The previous process died, or restarted, mid-turn. Its file still holds the turn
        # before, but that is not how the last turn ended, so the new file starts without a
        # last turn. A restart keeps the open turn in the file, which is how it shows.
        old = _make_writer(tmp_path)
        old.set_turn_start(SID)
        old.set_idle(SID, stop_reason="completed")
        old.set_turn_start(SID)
        if restarted:
            old.set_restarting()
        old._stop_event.set()
        assert _read(old)["state"] == ("restarting" if restarted else "working")
        assert _read(old)["turn"] is not None
        assert _read(old)["last_turn"]["stop_reason"] == "completed"

        new = _make_writer(tmp_path)
        try:
            assert _read(new)["last_turn"] is None
        finally:
            new._stop_event.set()

    def test_an_unreadable_previous_file_carries_nothing(self, tmp_path: Path) -> None:
        path = status_dir() / f"{SID}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        new = _make_writer(tmp_path)
        try:
            assert new._started is True
            assert _read(new)["last_turn"] is None
        finally:
            new._stop_event.set()


# ── session binding ──────────────────────────────────────────────────────────


class TestSessionBinding:
    def test_wrong_session_id_is_noop(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            wrong = "other-session"
            w.set_turn_start(wrong)
            data = _read(w)
            assert data["state"] == "idle"  # unchanged
            assert data["turn"] is None
        finally:
            w._stop_event.set()

    def test_correct_session_id_transitions(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            data = _read(w)
            assert data["state"] == "working"
            assert data["turn"] is not None
        finally:
            w._stop_event.set()

    def test_model_call_wrong_session_is_noop(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.set_model_call_start("other-session")
            data = _read(w)
            assert data["state"] == "working"  # not model_call
        finally:
            w._stop_event.set()

    def test_tool_wrong_session_is_noop(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.set_tool_start("other-session", "Bash")
            data = _read(w)
            assert data["state"] == "working"
            assert data["tool"] is None
        finally:
            w._stop_event.set()

    def test_wakeup_from_another_session_is_noop(self, tmp_path: Path) -> None:
        # A nested run's wake-up slot must not write the top-level session's wake-up.
        w = _make_writer(tmp_path)
        try:
            w.set_wakeup("other-session", 1719849600.0)
            w.set_wakeup(None, 1719849600.0)
            assert _read(w)["wakeup"] is None
            w.set_wakeup(SID, 1719849600.0)  # positive control: the owner's own arm lands
            assert _read(w)["wakeup"] == {"due_at": "2024-07-01T16:00:00Z"}
        finally:
            w._stop_event.set()


# ── state transitions ────────────────────────────────────────────────────────


class TestTransitions:
    @pytest.fixture(autouse=True)
    def _writer(self, tmp_path: Path) -> Any:
        self.w = _make_writer(tmp_path)
        yield
        self.w._stop_event.set()

    def test_turn_start_goes_to_working(self) -> None:
        self.w.set_turn_start(SID)
        data = _read(self.w)
        assert data["state"] == "working"
        assert data["turn"] is not None
        assert data["turn"]["model_calls"] == 0
        assert data["turn"]["tool_calls"] == 0
        assert "started_at" in data["turn"]

    def test_model_call_start_and_end(self) -> None:
        self.w.set_turn_start(SID)
        self.w.set_model_call_start(SID, model="test-provider-v2")
        data = _read(self.w)
        assert data["state"] == "model_call"
        assert data["call"] is not None
        assert data["call"]["phase"] == "waiting"
        assert data["call"]["model"] == "test-provider-v2"
        assert data["call"]["first_output_at"] is None
        assert data["turn"]["model_calls"] == 1
        # Top-level model updated.
        assert data["model"] == "test-provider-v2"

        self.w.set_model_call_end(SID)
        data = _read(self.w)
        assert data["state"] == "working"  # not idle
        assert data["call"] is None

    def test_model_call_start_without_model_keeps_existing(self) -> None:
        self.w.set_turn_start(SID)
        self.w.set_model_call_start(SID)
        data = _read(self.w)
        assert data["call"]["model"] == "test-model"
        assert data["model"] == "test-model"

    def test_streaming_deltas(self) -> None:
        self.w.set_turn_start(SID)
        self.w.set_model_call_start(SID)

        # Force throttle reset so writes go through.
        self.w._last_progress_write = 0.0

        self.w.on_thinking_delta(SID, "think")
        data = _read(self.w)
        assert data["call"]["phase"] == "thinking"
        assert data["call"]["thinking_chars"] == 5
        assert data["call"]["first_output_at"] is not None

        self.w._last_progress_write = 0.0
        self.w.on_text_delta(SID, "hello world")
        data = _read(self.w)
        assert data["call"]["phase"] == "text"
        assert data["call"]["text_chars"] == 11

        self.w._last_progress_write = 0.0
        self.w.on_tool_call_delta(SID, '{"arg":"val"}')
        data = _read(self.w)
        assert data["call"]["phase"] == "tool_call"
        assert data["call"]["tool_call_chars"] == 13

    def test_tool_start_and_end(self) -> None:
        self.w.set_turn_start(SID)
        self.w.set_tool_start(SID, "Bash")
        data = _read(self.w)
        assert data["state"] == "tool"
        assert data["tool"]["name"] == "Bash"
        assert data["tool"]["running"] == 1
        assert data["turn"]["tool_calls"] == 1

        self.w.set_tool_end(SID)
        data = _read(self.w)
        assert data["state"] == "working"  # not idle
        assert data["tool"] is None

    def test_idle_with_stop_reason(self) -> None:
        self.w.set_turn_start(SID)
        self.w.set_idle(SID, stop_reason="end_turn")
        data = _read(self.w)
        assert data["state"] == "idle"
        assert data["last_turn"] is not None
        assert data["last_turn"]["stop_reason"] == "end_turn"
        assert data["turn"] is None

    def test_wakeup_arm_and_clear(self) -> None:
        epoch = 1719849600.0
        self.w.set_wakeup(SID, epoch)
        data = _read(self.w)
        assert data["wakeup"] == {"due_at": "2024-07-01T16:00:00Z"}

        self.w.set_wakeup(SID, None)
        data = _read(self.w)
        assert data["wakeup"] is None

    def test_a_wakeup_time_that_is_not_a_time_reads_as_none(self) -> None:
        # The writer never raises into the wake-up slot: an out-of-range time records no
        # wake-up rather than failing the arm.
        self.w.set_wakeup(SID, 1719849600.0)
        self.w.set_wakeup(SID, float("inf"))
        assert _read(self.w)["wakeup"] is None

    def test_restarting(self) -> None:
        self.w.set_restarting()
        data = _read(self.w)
        assert data["state"] == "restarting"

    def test_exited(self) -> None:
        self.w.set_exited()
        data = _read(self.w)
        assert data["state"] == "exited"
        assert data["call"] is None
        assert data["tool"] is None
        assert data["turn"] is None

    def test_multiple_tool_calls_increment_counter(self) -> None:
        self.w.set_turn_start(SID)
        self.w.set_tool_start(SID, "Read")
        self.w.set_tool_end(SID)
        self.w.set_tool_start(SID, "Write")
        self.w.set_tool_end(SID)
        data = _read(self.w)
        assert data["turn"]["tool_calls"] == 2


# ── concurrent tools ─────────────────────────────────────────────────────────


class TestConcurrentTools:
    def test_concurrent_tool_count(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.set_tool_start(SID, "Read")
            data = _read(w)
            assert data["tool"]["running"] == 1

            w.set_tool_start(SID, "Grep")
            data = _read(w)
            assert data["tool"]["running"] == 2
            assert data["tool"]["name"] == "Grep"

            w.set_tool_end(SID)
            data = _read(w)
            assert data["state"] == "tool"
            assert data["tool"]["running"] == 1

            w.set_tool_end(SID)
            data = _read(w)
            assert data["state"] == "working"
            assert data["tool"] is None
        finally:
            w._stop_event.set()

    def test_tool_count_never_goes_negative(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.set_tool_end(SID)  # end without start
            data = _read(w)
            assert data["state"] == "working"
        finally:
            w._stop_event.set()


# ── leave_turn ───────────────────────────────────────────────────────────────


class TestLeaveTurn:
    def test_leave_turn_goes_to_idle(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.set_model_call_start(SID)
            w.leave_turn(SID, "interrupted")
            data = _read(w)
            assert data["state"] == "idle"
            assert data["call"] is None
            assert data["tool"] is None
            assert data["turn"] is None
            assert data["last_turn"]["stop_reason"] == "interrupted"
        finally:
            w._stop_event.set()

    def test_leave_turn_error_reason(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.leave_turn(SID, "error")
            data = _read(w)
            assert data["state"] == "idle"
            assert data["last_turn"]["stop_reason"] == "error"
        finally:
            w._stop_event.set()

    def test_leave_turn_wrong_session_is_noop(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.leave_turn("other-session", "error")
            data = _read(w)
            assert data["state"] == "working"  # unchanged
        finally:
            w._stop_event.set()

    def test_leave_turn_after_the_turn_ended_changes_nothing(self, tmp_path: Path) -> None:
        # A turn that already ended keeps the stop reason it ended with, and a stray
        # leave_turn writes nothing at all.
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.set_idle(SID, stop_reason="end_turn")
            w._data["since"] = "2000-01-01T00:00:00Z"  # a stray transition would overwrite this
            w.leave_turn(SID, "error")
            assert w._data["since"] == "2000-01-01T00:00:00Z"
            assert w._data["last_turn"]["stop_reason"] == "end_turn"
            assert _read(w)["last_turn"]["stop_reason"] == "end_turn"
        finally:
            w._stop_event.set()


# ── side calls ───────────────────────────────────────────────────────────────


class TestSideCall:
    """ADR-0276: a model call outside the conversation shows as a ``model_call`` naming its kind,
    and the file goes back to what is still open when it ends."""

    @pytest.fixture(autouse=True)
    def _writer(self, tmp_path: Path) -> Any:
        self.w = _make_writer(tmp_path)
        yield
        self.w._stop_event.set()

    def test_a_side_call_in_a_turn_names_its_kind_and_ends_back_in_working(self) -> None:
        self.w.set_turn_start(SID)
        with self.w.side_call(SID, "summarizer", model="summary-model"):
            during = _read(self.w)
        after = _read(self.w)
        assert during["state"] == "model_call"
        assert during["call"]["kind"] == "summarizer"
        assert during["call"]["model"] == "summary-model"
        assert during["call"]["phase"] == "waiting"
        # Not one of the conversation's calls, and not a change of the conversation's model.
        assert during["turn"]["model_calls"] == 0
        assert during["model"] == "test-model"
        assert after["state"] == "working"
        assert after["call"] is None
        assert after["turn"]["model_calls"] == 0

    def test_a_side_call_without_a_model_names_the_session_s_model(self) -> None:
        self.w.set_turn_start(SID)
        with self.w.side_call(SID, "critic"):
            assert _read(self.w)["call"]["model"] == "test-model"

    def test_a_side_call_inside_a_tool_goes_back_to_the_tool(self) -> None:
        # The plan critique runs inside the plan tool; the tool is still running after it.
        self.w.set_turn_start(SID)
        self.w.set_tool_start(SID, "update_plan")
        with self.w.side_call(SID, "plan_critique"):
            during = _read(self.w)
        after = _read(self.w)
        assert during["state"] == "model_call"
        assert during["tool"]["name"] == "update_plan"
        assert after["state"] == "tool"
        assert after["tool"]["name"] == "update_plan"
        assert after["call"] is None

    def test_a_side_call_at_the_prompt_goes_back_to_idle(self) -> None:
        # /compact compacts with no turn open: the file returns to idle, and the last turn's
        # ending is kept.
        self.w.set_turn_start(SID)
        self.w.set_idle(SID, stop_reason="end_turn")
        with self.w.side_call(SID, "summarizer"):
            assert _read(self.w)["state"] == "model_call"
        after = _read(self.w)
        assert after["state"] == "idle"
        assert after["turn"] is None
        assert after["last_turn"]["stop_reason"] == "end_turn"

    def test_a_side_call_from_another_session_never_touches_the_file(self) -> None:
        self.w.set_turn_start(SID)
        self.w._data["since"] = "2000-01-01T00:00:00Z"  # any transition would overwrite this
        before = _read(self.w)
        with self.w.side_call("other-session", "critic", model="judge-model"):
            assert _read(self.w) == before
            assert self.w._data["since"] == "2000-01-01T00:00:00Z"
        assert _read(self.w) == before
        assert self.w._data["since"] == "2000-01-01T00:00:00Z"
        with self.w.side_call(SID, "critic"):  # positive control: the owner's own side call lands
            assert _read(self.w)["call"]["kind"] == "critic"

    def test_a_side_call_whose_body_raises_still_ends_and_the_error_propagates(self) -> None:
        self.w.set_turn_start(SID)
        with pytest.raises(RuntimeError, match="judge down"), self.w.side_call(SID, "critic"):
            raise RuntimeError("judge down")
        after = _read(self.w)
        assert after["state"] == "working"
        assert after["call"] is None

    def test_overlapping_side_calls_hold_the_file_until_the_last_one_ends(self) -> None:
        # A deliberation asks for its samples at once. The first one back must not take the file
        # off the side call while another is still out.
        self.w.set_turn_start(SID)
        self.w.set_tool_start(SID, "deep_think")
        first = self.w.side_call(SID, "deep_think")
        second = self.w.side_call(SID, "deep_think")
        first.__enter__()
        second.__enter__()
        first.__exit__(None, None, None)
        assert _read(self.w)["state"] == "model_call"
        assert _read(self.w)["call"]["kind"] == "deep_think"
        second.__exit__(None, None, None)
        after = _read(self.w)
        assert after["state"] == "tool"
        assert after["call"] is None


# ── plan position ────────────────────────────────────────────────────────────


class _Plan:
    """A stand-in plan source whose answer a test changes between transitions."""

    def __init__(self, position: PlanPosition | None) -> None:
        self.position = position
        self.reads = 0

    def __call__(self) -> PlanPosition | None:
        self.reads += 1
        return self.position


class TestPlanPosition:
    """ADR-0277: the file carries where the plan stands, read through the source the turn hands
    over, at each transition after which the plan can have moved."""

    @pytest.fixture(autouse=True)
    def _writer(self, tmp_path: Path) -> Any:
        self.w = _make_writer(tmp_path)
        yield
        self.w._stop_event.set()

    def test_the_file_starts_with_no_plan(self) -> None:
        assert _read(self.w)["plan"] is None

    def test_a_turn_start_reads_the_plan_and_writes_its_time_in_the_file_s_form(self) -> None:
        self.w.set_turn_start(SID, plan=_Plan(("2.1", "2026-10-03T03:19:48+00:00")))
        assert _read(self.w)["plan"] == {"active": "2.1", "moved_at": "2026-10-03T03:19:48Z"}

    def test_the_plan_is_read_again_after_each_transition_that_can_move_it(self) -> None:
        plan = _Plan(("1", "2026-10-03T03:00:00+00:00"))
        self.w.set_turn_start(SID, plan=plan)
        plan.position = ("2", "2026-10-03T03:05:00+00:00")
        self.w.set_tool_start(SID, "update_plan")
        assert _read(self.w)["plan"]["active"] == "1"  # nothing has run yet at a tool's start
        self.w.set_tool_end(SID)
        assert _read(self.w)["plan"] == {"active": "2", "moved_at": "2026-10-03T03:05:00Z"}
        plan.position = ("3", "2026-10-03T03:06:00+00:00")
        self.w.set_model_call_start(SID)
        assert _read(self.w)["plan"]["active"] == "3"
        self.w.set_model_call_end(SID)
        plan.position = (None, "2026-10-03T03:07:00+00:00")  # the conclusion closed the last step
        self.w.set_idle(SID, stop_reason="end_turn")
        assert _read(self.w)["plan"] == {"active": None, "moved_at": "2026-10-03T03:07:00Z"}

    def test_a_turn_that_unwinds_writes_the_plan_as_it_was_left(self) -> None:
        plan = _Plan(("1", "2026-10-03T03:00:00+00:00"))
        self.w.set_turn_start(SID, plan=plan)
        plan.position = ("2", "2026-10-03T03:01:00+00:00")
        self.w.leave_turn(SID, "interrupted")
        assert _read(self.w)["plan"]["active"] == "2"

    def test_a_session_with_no_plan_writes_none(self) -> None:
        self.w.set_turn_start(SID, plan=_Plan(None))
        self.w.set_model_call_start(SID)
        assert _read(self.w)["plan"] is None

    def test_a_move_time_that_is_missing_or_unreadable_is_left_out(self) -> None:
        self.w.set_turn_start(SID, plan=_Plan(("1", "")))
        assert _read(self.w)["plan"] == {"active": "1", "moved_at": None}
        self.w.set_turn_start(SID, plan=_Plan(("1", "not a time")))
        assert _read(self.w)["plan"] == {"active": "1", "moved_at": None}

    def test_a_failing_plan_source_costs_the_plan_and_nothing_else(self) -> None:
        def unreadable() -> PlanPosition | None:
            raise RuntimeError("plan unreadable")

        self.w.set_turn_start(SID, plan=unreadable)
        data = _read(self.w)
        assert data["state"] == "working"
        assert data["plan"] is None

    def test_a_turn_start_without_a_source_keeps_the_one_held(self) -> None:
        plan = _Plan(("1", "2026-10-03T03:00:00+00:00"))
        self.w.set_turn_start(SID, plan=plan)
        self.w.set_idle(SID, stop_reason="end_turn")
        plan.position = ("2", "2026-10-03T03:10:00+00:00")
        self.w.set_turn_start(SID)
        assert _read(self.w)["plan"]["active"] == "2"

    def test_another_session_s_turn_never_hands_its_plan_over(self) -> None:
        mine = _Plan(("1", "2026-10-03T03:00:00+00:00"))
        theirs = _Plan(("9", "2026-10-03T03:30:00+00:00"))
        self.w.set_turn_start(SID, plan=mine)
        self.w.set_turn_start("other-session", plan=theirs)
        self.w.set_model_call_start(SID)
        assert _read(self.w)["plan"]["active"] == "1"
        assert theirs.reads == 0

    def test_a_write_alone_never_reads_the_plan(self) -> None:
        # The refresher thread writes between transitions while the loop may be changing the
        # plan, so only a transition, on the loop's own thread, may read it.
        plan = _Plan(("1", "2026-10-03T03:00:00+00:00"))
        self.w.set_turn_start(SID, plan=plan)
        reads = plan.reads
        self.w._write()
        assert plan.reads == reads


# ── first output write ───────────────────────────────────────────────────────


class TestFirstOutputWrite:
    def test_first_output_writes_immediately(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.set_model_call_start(SID)
            # Set throttle so it would normally skip.
            w._last_progress_write = time.monotonic()

            w.on_thinking_delta(SID, "first")
            data = _read(w)
            # Should write despite throttle since it is the first output.
            assert data["call"]["first_output_at"] is not None
            assert data["call"]["thinking_chars"] == 5
        finally:
            w._stop_event.set()

    def test_phase_change_writes_immediately(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.set_model_call_start(SID)
            w._last_progress_write = 0.0
            w.on_thinking_delta(SID, "think")

            # Now set throttle high so only phase changes write.
            w._last_progress_write = time.monotonic()
            w.on_text_delta(SID, "hello")
            data = _read(w)
            assert data["call"]["phase"] == "text"
            assert data["call"]["text_chars"] == 5
        finally:
            w._stop_event.set()


# ── throttled write ──────────────────────────────────────────────────────────


class TestThrottledWrite:
    def test_skips_write_within_interval(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.set_model_call_start(SID)

            # First delta writes (first output, so immediate via was_first path).
            w._last_progress_write = 0.0
            w.on_text_delta(SID, "a")
            first_data = _read(w)

            # Simulate recent throttled write so the next one is within interval.
            w._last_progress_write = time.monotonic()

            # Second delta: same phase, within interval -- throttled, no disk write.
            w.on_text_delta(SID, "b")
            second_data = _read(w)
            assert second_data["call"]["text_chars"] == first_data["call"]["text_chars"]
        finally:
            w._stop_event.set()


# ── error swallowing ─────────────────────────────────────────────────────────


class TestErrorSwallowing:
    def test_write_to_unwritable_path_does_not_raise(self, tmp_path: Path) -> None:
        w = StatusFileWriter()
        w._started = True
        w._session_id = SID
        w._path = Path("/proc/nonexistent/status.json")
        w._data = {"state": "idle"}
        w._lock = __import__("threading").Lock()
        w._write()

    def test_transitions_swallow_write_errors(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w._path = Path("/proc/nonexistent/status.json")
            w.set_idle(SID)
            w.set_turn_start(SID)
            w.set_model_call_start(SID)
            w.set_model_call_end(SID)
            w.set_tool_start(SID, "test")
            w.set_tool_end(SID)
            w.leave_turn(SID, "error")
        finally:
            w._stop_event.set()


# ── privacy ──────────────────────────────────────────────────────────────────


class TestPrivacy:
    """The status file must never contain model text, thinking text, prompts,
    user input, tool arguments, tool output, or file contents.
    """

    def test_no_private_content_in_file(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path, session_id="privacy-test")
        try:
            w.set_turn_start("privacy-test")
            w.set_model_call_start("privacy-test")

            markers = []
            for label, method in [
                ("THINKING_SECRET_xK9mQ", w.on_thinking_delta),
                ("TEXT_SECRET_pL3nR", w.on_text_delta),
                ("TOOLCALL_SECRET_vW7jT", w.on_tool_call_delta),
            ]:
                markers.append(label)
                w._last_progress_write = 0.0
                method("privacy-test", label)

            w.set_model_call_end("privacy-test")
            w.set_tool_start("privacy-test", "Bash")
            w.set_tool_end("privacy-test")
            w.set_idle("privacy-test", stop_reason="end_turn")

            assert w._path is not None
            raw = w._path.read_bytes()
            for marker in markers:
                assert marker.encode() not in raw, (
                    f"Private marker {marker!r} leaked into the status file"
                )

            data = json.loads(raw)
            flat = json.dumps(data)
            for marker in markers:
                assert marker not in flat
        finally:
            w._stop_event.set()


# ── reader helper ────────────────────────────────────────────────────────────


class TestReadStatusFiles:
    def test_reads_valid_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path))
        sdir = tmp_path / "status"
        sdir.mkdir()
        (sdir / "s1.json").write_text(json.dumps({"session": "s1", "state": "idle"}))
        (sdir / "s2.json").write_text(json.dumps({"session": "s2", "state": "exited"}))

        entries = read_status_files()
        assert len(entries) == 2
        sessions = {e["session"] for e in entries}
        assert sessions == {"s1", "s2"}

    def test_skips_corrupt_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path))
        sdir = tmp_path / "status"
        sdir.mkdir()
        (sdir / "good.json").write_text(json.dumps({"session": "good"}))
        (sdir / "bad.json").write_text("not json {{{{")

        entries = read_status_files()
        assert len(entries) == 1
        assert entries[0]["session"] == "good"

    def test_empty_when_no_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path))
        assert read_status_files() == []


# ── atexit handler ───────────────────────────────────────────────────────────


class TestAtexit:
    def test_on_exit_writes_exited_state(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.set_model_call_start(SID)
            w._on_exit()
            data = _read(w)
            assert data["state"] == "exited"
        finally:
            w._stop_event.set()


# ── daemon refresher ─────────────────────────────────────────────────────────


class TestRefresher:
    def test_refresher_updates_timestamp(self, tmp_path: Path) -> None:
        w = StatusFileWriter()
        with patch.object(
            type(w),
            "_refresh_loop",
            wraps=w._refresh_loop,
        ):
            w.start(
                session_id="refresher-test",
                workspace=str(tmp_path),
                model="m",
                build="b",
            )
            try:
                _read(w)["updated_at"]
                with w._lock:
                    w._data["updated_at"] = "manual-update"
                w._write()
                data = _read(w)
                assert data["updated_at"] == "manual-update"
            finally:
                w._stop_event.set()
                if w._refresher:
                    w._refresher.join(timeout=2.0)


# ── CLI command ──────────────────────────────────────────────────────────────


def _write_status(sdir: Path, session_id: str, state: str, updated_at: str, **extra: Any) -> None:
    """Write a synthetic status file."""
    sdir.mkdir(parents=True, exist_ok=True)
    data = {
        "v": 1,
        "pid": 12345,
        "session": session_id,
        "state": state,
        "updated_at": updated_at,
        "since": extra.pop("since", updated_at),
        "model": extra.get("model", "test-model"),
        "workspace": extra.get("workspace", "/tmp/ws"),
        "build": "b",
        "turn": extra.get("turn"),
        "call": extra.get("call"),
        "tool": extra.get("tool"),
        "wakeup": extra.get("wakeup"),
        "last_turn": extra.get("last_turn"),
    }
    data.update({k: v for k, v in extra.items() if k not in data})
    (sdir / f"{session_id}.json").write_text(json.dumps(data) + "\n")


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _old_iso() -> str:
    """A timestamp older than STALE_THRESHOLD_SECONDS."""
    t = time.time() - STALE_THRESHOLD_SECONDS - 60
    return datetime.fromtimestamp(t, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_at(t: float) -> str:
    """The status file's stamp for epoch seconds ``t``."""
    return datetime.fromtimestamp(t, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class TestStatusCLI:
    """Test the ``zakcode status`` CLI command via Typer's CliRunner."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.home = tmp_path / ".zakcode"
        self.sdir = self.home / "status"
        monkeypatch.setenv("ZAKCODE_HOME", str(self.home))
        from typer.testing import CliRunner

        from zakcode.cli import app

        self.runner = CliRunner()
        self.app = app

    def test_no_status_files(self) -> None:
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "No status files" in result.output

    def test_live_session_shown(self) -> None:
        _write_status(self.sdir, "live-sess-1", "idle", _now_iso())
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "live-ses" in result.output  # 8 chars

    def test_stale_session_hidden_by_default(self) -> None:
        _write_status(self.sdir, "stale-sess-1", "idle", _old_iso())
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "stale-se" not in result.output
        assert "No live sessions" in result.output

    def test_stale_session_shown_with_all(self) -> None:
        _write_status(self.sdir, "stale-sess-1", "idle", _old_iso())
        result = self.runner.invoke(self.app, ["status", "--all"])
        assert result.exit_code == 0
        assert "stale-se" in result.output
        assert "[stale]" in result.output

    def test_exited_session_hidden_by_default(self) -> None:
        _write_status(self.sdir, "exited-sess-1", "exited", _now_iso())
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "exited-s" not in result.output

    def test_exited_session_shown_with_all(self) -> None:
        _write_status(self.sdir, "exited-sess-1", "exited", _now_iso())
        result = self.runner.invoke(self.app, ["status", "--all"])
        assert result.exit_code == 0
        assert "exited-s" in result.output
        assert "[exited]" in result.output

    def test_json_output(self) -> None:
        _write_status(self.sdir, "json-sess-1", "idle", _now_iso(), model="claude-opus")
        result = self.runner.invoke(self.app, ["status", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["session"] == "json-sess-1"
        assert data[0]["_liveness"] == "live"

    def test_json_lists_stale_and_exited_sessions_without_all(self) -> None:
        # A script must tell "no session" ([]) from "the process is gone": the JSON view
        # lists every session with its liveness, --all or not.
        _write_status(self.sdir, "live-sess-1", "idle", _now_iso())
        _write_status(self.sdir, "stale-sess-1", "idle", _old_iso())
        _write_status(self.sdir, "exited-sess-1", "exited", _now_iso())
        result = self.runner.invoke(self.app, ["status", "--json"])
        assert result.exit_code == 0
        liveness = {row["session"]: row["_liveness"] for row in json.loads(result.output)}
        assert liveness == {
            "live-sess-1": "live",
            "stale-sess-1": "stale",
            "exited-sess-1": "exited",
        }
        # The human view still hides them without --all.
        human = self.runner.invoke(self.app, ["status"]).output
        assert "live-ses" in human
        assert "stale-se" not in human
        assert "exited-s" not in human

    def test_json_empty(self) -> None:
        result = self.runner.invoke(self.app, ["status", "--json"])
        assert result.exit_code == 0
        assert json.loads(result.output) == []

    def test_workspace_filter_exact(self, tmp_path: Path) -> None:
        _write_status(self.sdir, "ws-a", "idle", _now_iso(), workspace="/home/a/project")
        _write_status(self.sdir, "ws-b", "idle", _now_iso(), workspace="/home/b/other")
        result = self.runner.invoke(self.app, ["status", "-w", "/home/a/project"])
        assert result.exit_code == 0
        assert "ws-a" in result.output
        assert "ws-b" not in result.output

    def test_workspace_filter_substring_no_match(self, tmp_path: Path) -> None:
        _write_status(self.sdir, "ws-a", "idle", _now_iso(), workspace="/home/a/project")
        result = self.runner.invoke(self.app, ["status", "-w", "project"])
        assert result.exit_code == 0
        # Exact match only now -- "project" != "/home/a/project".
        assert "ws-a" not in result.output

    def test_workspace_filter_matches_a_relative_spelling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The same directory still matches when the file and the operator spell it
        # differently: here the file holds an unnormalized path, the operator a relative one.
        project = tmp_path / "project"
        project.mkdir()
        spelled = str(project / ".." / "project")
        _write_status(self.sdir, "ws-rel", "idle", _now_iso(), workspace=spelled)
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(self.app, ["status", "-w", "project"])
        assert result.exit_code == 0
        assert "ws-rel" in result.output

    def test_model_call_detail_says_how_long_the_first_output_took(self) -> None:
        # Time to first output separates a slow prompt read from slow generation.
        started = time.time() - 200
        _write_status(
            self.sdir,
            "first-out",
            "model_call",
            _now_iso(),
            since=_iso_at(started),
            call={
                "phase": "thinking",
                "model": "m",
                "started_at": _iso_at(started),
                "first_output_at": _iso_at(started + 90),
                "thinking_chars": 7,
            },
        )
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "thinking m (7 chars, first output after 1m)" in result.output

    def test_idle_detail_names_the_wakeup_and_how_the_last_turn_ended(self) -> None:
        _write_status(
            self.sdir,
            "idle-wake",
            "idle",
            _now_iso(),
            wakeup={"due_at": _iso_at(time.time() + 930)},
            last_turn={"ended_at": _now_iso(), "stop_reason": "completed"},
        )
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "wake-up in 15m, last turn completed" in result.output

    def test_an_overdue_wakeup_is_called_out(self) -> None:
        # An idle session whose wake-up is past due is the one to look at.
        _write_status(
            self.sdir,
            "idle-late",
            "idle",
            _now_iso(),
            wakeup={"due_at": _iso_at(time.time() - 930)},
        )
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "wake-up overdue by 15m" in result.output

    def test_model_call_detail(self) -> None:
        _write_status(
            self.sdir,
            "mc-sess-1",
            "model_call",
            _now_iso(),
            call={
                "phase": "thinking",
                "model": "opus-4",
                "started_at": _now_iso(),
                "first_output_at": _now_iso(),
                "thinking_chars": 100,
                "text_chars": 0,
                "tool_call_chars": 0,
            },
        )
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "model_call" in result.output
        assert "thinking" in result.output

    @staticmethod
    def _side_call(kind: str) -> dict[str, Any]:
        return {
            "phase": "waiting",
            "model": "small-27b",
            "kind": kind,
            "started_at": _now_iso(),
            "first_output_at": None,
            "thinking_chars": 0,
            "text_chars": 0,
            "tool_call_chars": 0,
        }

    def test_a_side_call_says_what_it_is_doing(self) -> None:
        # ADR-0276: a compaction names itself; before, it read as a turn between steps.
        _write_status(
            self.sdir, "side-ses-1", "model_call", _now_iso(), call=self._side_call("summarizer")
        )
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "compacting the conversation (small-27b)" in result.output
        assert "waiting for" not in result.output

    @pytest.mark.parametrize(
        "kind",
        [
            "summarizer",
            "critic",
            "plan_critique",
            "quality_gate",
            "difficulty_classifier",
            "deep_think",
        ],
    )
    def test_every_side_call_the_loop_marks_has_its_own_words(self, kind: str) -> None:
        from zakcode.cli import _state_detail

        detail = _state_detail({"state": "model_call", "call": self._side_call(kind)}, time.time())
        assert detail.endswith(" (small-27b)")
        assert not detail.startswith("side call")

    def test_a_side_call_of_an_unknown_kind_reads_by_its_name(self) -> None:
        _write_status(
            self.sdir, "side-ses-2", "model_call", _now_iso(), call=self._side_call("fact_check")
        )
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "side call fact_check (small-27b)" in result.output

    def test_tool_detail(self) -> None:
        _write_status(
            self.sdir,
            "tool-ses-1",
            "tool",
            _now_iso(),
            tool={"name": "Read", "started_at": _now_iso(), "running": 1},
        )
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "tool" in result.output
        assert "Read" in result.output

    def test_tool_concurrent_detail(self) -> None:
        _write_status(
            self.sdir,
            "tool-conc",
            "tool",
            _now_iso(),
            tool={"name": "Read", "started_at": _now_iso(), "running": 3},
        )
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "+2 concurrent" in result.output

    def test_working_detail(self) -> None:
        _write_status(
            self.sdir,
            "work-ses-1",
            "working",
            _now_iso(),
            turn={"started_at": _now_iso(), "model_calls": 2, "tool_calls": 5},
        )
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "working" in result.output
        assert "2 calls" in result.output
        assert "5 tools" in result.output

    def test_pid_shown(self) -> None:
        _write_status(self.sdir, "pid-test1", "idle", _now_iso())
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "pid 12345" in result.output

    def test_duration_shown(self) -> None:
        # Set since to 90 seconds ago.
        t = time.time() - 90
        since = datetime.fromtimestamp(t, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_status(self.sdir, "dur-test1", "idle", _now_iso(), since=since)
        result = self.runner.invoke(self.app, ["status"])
        assert result.exit_code == 0
        assert "for 1m" in result.output


# ── duration formatting ──────────────────────────────────────────────────────


class TestDurationFormatting:
    def test_seconds(self) -> None:
        from zakcode.cli import _format_duration

        assert _format_duration(42) == "42s"

    def test_minutes(self) -> None:
        from zakcode.cli import _format_duration

        assert _format_duration(660) == "11m"

    def test_hours_and_minutes(self) -> None:
        from zakcode.cli import _format_duration

        assert _format_duration(3600 * 3 + 300) == "3h05m"

    def test_exact_hours(self) -> None:
        from zakcode.cli import _format_duration

        assert _format_duration(7200) == "2h"

    def test_zero(self) -> None:
        from zakcode.cli import _format_duration

        assert _format_duration(0) == "0s"

    def test_negative_clamped(self) -> None:
        from zakcode.cli import _format_duration

        assert _format_duration(-5) == "0s"


# ── restart ──────────────────────────────────────────────────────────────────


class TestRestart:
    def test_restarting_state_written(self, tmp_path: Path) -> None:
        w = _make_writer(tmp_path)
        try:
            w.set_restarting()
            data = _read(w)
            assert data["state"] == "restarting"
            assert "since" in data
        finally:
            w._stop_event.set()

    def test_a_failed_exec_goes_back_to_idle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The process keeps serving on the old build, so the file must stop saying
        # "restarting" (after a successful exec the fresh process's first write replaces it).
        import io
        from types import SimpleNamespace

        from rich.console import Console

        from zakcode import cli

        w = _make_writer(tmp_path)
        try:
            agent = SimpleNamespace(
                session=SimpleNamespace(id=SID, build="old-build"),
                loop=SimpleNamespace(store=None, restart_continuation=None, restart_boundary=None),
            )
            monkeypatch.setattr("zakcode.session.status_file.get_writer", lambda: w)
            monkeypatch.setattr(cli, "install_changed", lambda: ("old-build", "new-build"))
            monkeypatch.setattr(cli, "prepare_restart", lambda: 0)
            monkeypatch.setattr(cli.sys, "argv", ["zakcode", "cli"])
            for name in (
                "ZAKCODE_RESTARTED_INTO",
                "ZAKCODE_RESTART_CONTINUATION",
                "ZAKCODE_RESTART_BOUNDARY",
            ):
                monkeypatch.setenv(name, "x")  # restored at teardown, whatever the restart sets
            seen_at_exec: list[str] = []

            def exec_fails(path: str, argv: list[str]) -> None:
                seen_at_exec.append(_read(w)["state"])  # what an observer sees at the exec
                raise OSError("exec denied")

            monkeypatch.setattr(cli.os, "execv", exec_fails)
            console = Console(theme=cli.ZAK_THEME, file=io.StringIO(), force_terminal=False)
            cli._restart_into_new_build(console, agent)
            assert seen_at_exec == ["restarting"]
            assert _read(w)["state"] == "idle"
        finally:
            w._stop_event.set()


# ── integration tests through real AgentLoop ─────────────────────────────────
#
# These prove that each status hook in loop.py actually fires during a real turn
# by running a hermetic AgentLoop with a ScriptedProvider and verifying the
# sequence of states the status file passes through.


class _RecordingWriter(StatusFileWriter):
    """Wrapper that records every state transition for later assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.transitions: list[str] = []

    def set_turn_start(self, session_id: str, *, plan: PlanSource | None = None) -> None:
        super().set_turn_start(session_id, plan=plan)
        if self._owns(session_id):
            self.transitions.append("turn_start:working")

    def set_model_call_start(self, session_id: str, *, model: str = "") -> None:
        super().set_model_call_start(session_id, model=model)
        if self._owns(session_id):
            self.transitions.append(f"model_call_start:{model or 'default'}")

    def set_model_call_end(self, session_id: str) -> None:
        super().set_model_call_end(session_id)
        if self._owns(session_id):
            self.transitions.append("model_call_end:working")

    def set_tool_start(self, session_id: str, name: str) -> None:
        super().set_tool_start(session_id, name)
        if self._owns(session_id):
            self.transitions.append(f"tool_start:{name}")

    def set_tool_end(self, session_id: str) -> None:
        super().set_tool_end(session_id)
        if self._owns(session_id):
            self.transitions.append("tool_end")

    def set_idle(self, session_id: str, *, stop_reason: str | None = None) -> None:
        super().set_idle(session_id, stop_reason=stop_reason)
        if self._owns(session_id):
            self.transitions.append(f"idle:{stop_reason or 'none'}")

    def leave_turn(self, session_id: str, reason: str) -> None:
        super().leave_turn(session_id, reason)
        if self._owns(session_id):
            self.transitions.append(f"leave_turn:{reason}")

    def on_thinking_delta(self, session_id: str, text: str) -> None:
        super().on_thinking_delta(session_id, text)
        # Not tracked in transitions to avoid noise from throttled writes.

    def on_text_delta(self, session_id: str, text: str) -> None:
        super().on_text_delta(session_id, text)

    def on_tool_call_delta(self, session_id: str, text: str) -> None:
        super().on_tool_call_delta(session_id, text)


def _integration_writer(tmp_path: Path, session_id: str) -> _RecordingWriter:
    """Create and start a recording writer for integration tests."""
    w = _RecordingWriter()
    w.start(
        session_id=session_id,
        workspace=str(tmp_path),
        model="integration-model",
        build="test",
    )
    return w


class TestIntegrationBuffered:
    """Integration: buffered turn (``arun_turn``) through real AgentLoop."""

    def test_buffered_turn_with_tool_sees_correct_states(self, tmp_path: Path) -> None:
        """Point 10a: working -> model_call -> working -> tool -> working -> idle."""
        import asyncio

        from zakcode.agent.loop import AgentLoop
        from zakcode.config import PermissionTier, load_settings
        from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
        from zakcode.session.store import Session
        from zakcode.tools.base import (
            ConcurrencyClass,
            Tool,
            ToolRegistry,
            ToolResult,
            ToolSpec,
        )
        from zakcode.usage import Usage

        _usage = Usage(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cost_usd=0,
        )

        class _Provider(Provider):
            def __init__(self) -> None:
                self.calls = 0

            async def acomplete(self, messages, *, system=None, tools=None, **kw):
                self.calls += 1
                if self.calls == 1:
                    return LLMResult(
                        text="",
                        tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
                        finish_reason="tool_calls",
                        usage=_usage,
                    )
                return LLMResult(
                    text="done",
                    finish_reason="stop",
                    usage=_usage,
                )

            def count_tokens(self, messages, *, system=None):
                return 0

            def capabilities(self):
                return Capabilities(supports_tools=True, context_window=8192)

            def model_id(self):
                return "test-buffered"

        class _EchoTool(Tool):
            spec = ToolSpec(
                name="echo",
                description="Echo.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
                required_permission=PermissionTier.READ_ONLY,
                concurrency=ConcurrencyClass.READ_ONLY_SAFE,
            )

            async def execute(self, args, ctx):
                return ToolResult.ok("ok")

        reg = ToolRegistry()
        reg.register(_EchoTool())
        session = Session(cwd=str(tmp_path), model="test-buffered")
        settings = load_settings(workspace_root=tmp_path)

        rw = _integration_writer(tmp_path, session.id)
        try:
            with patch("zakcode.agent.loop._status_writer", return_value=rw):
                loop = AgentLoop(_Provider(), reg, session, settings=settings)
                asyncio.run(loop.arun_turn("hello"))

            # Expected: turn_start -> model_call_start -> model_call_end ->
            #           tool_start -> tool_end -> model_call_start -> model_call_end -> idle
            assert "turn_start:working" in rw.transitions
            assert "model_call_start:test-buffered" in rw.transitions
            assert "model_call_end:working" in rw.transitions
            assert "tool_start:echo" in rw.transitions
            assert "tool_end" in rw.transitions
            assert any(t.startswith("idle:") for t in rw.transitions)

            # Verify ordering: turn_start comes before model_call_start.
            ts_idx = rw.transitions.index("turn_start:working")
            mc_idx = rw.transitions.index("model_call_start:test-buffered")
            assert ts_idx < mc_idx

            # Verify file state at end.
            data = _read(rw)
            assert data["state"] == "idle"
        finally:
            rw._stop_event.set()

    def test_second_session_id_causes_no_writes(self, tmp_path: Path) -> None:
        """Point 10c: a second AgentLoop with a different session.id causes no writes."""
        import asyncio

        from zakcode.agent.loop import AgentLoop
        from zakcode.config import load_settings
        from zakcode.providers.base import Capabilities, LLMResult, Provider
        from zakcode.session.store import Session
        from zakcode.tools.base import ToolRegistry
        from zakcode.usage import Usage

        class _Provider(Provider):
            async def acomplete(self, messages, *, system=None, tools=None, **kw):
                return LLMResult(
                    text="done",
                    finish_reason="stop",
                    usage=Usage(
                        prompt_tokens=1,
                        completion_tokens=1,
                        total_tokens=2,
                        cost_usd=0,
                    ),
                )

            def count_tokens(self, messages, *, system=None):
                return 0

            def capabilities(self):
                return Capabilities(context_window=8192)

        # Start writer with one session id.
        rw = _integration_writer(tmp_path, "parent-session")
        try:
            # Run loop with a DIFFERENT session id.
            child_session = Session(cwd=str(tmp_path), model="m")
            assert child_session.id != "parent-session"
            settings = load_settings(workspace_root=tmp_path)

            with patch("zakcode.agent.loop._status_writer", return_value=rw):
                loop = AgentLoop(_Provider(), ToolRegistry(), child_session, settings=settings)
                asyncio.run(loop.arun_turn("hello"))

            # No transitions should have been recorded (all gated by session id).
            assert rw.transitions == []
            # The file should still be in the initial idle state.
            data = _read(rw)
            assert data["state"] == "idle"
        finally:
            rw._stop_event.set()


class TestIntegrationStreaming:
    """Integration: streaming turn (``astream_turn``) through real AgentLoop."""

    def test_streaming_turn_with_tool_sees_correct_states(
        self,
        tmp_path: Path,
    ) -> None:
        """Point 10b: streaming turn sees the same state sequence."""
        import asyncio

        from zakcode.agent.loop import AgentLoop
        from zakcode.config import PermissionTier, load_settings
        from zakcode.providers.base import (
            Capabilities,
            LLMResult,
            Provider,
            StreamDone,
            StreamTextDelta,
            StreamToolCallDelta,
            StreamUsage,
        )
        from zakcode.session.store import Session
        from zakcode.tools.base import (
            ConcurrencyClass,
            Tool,
            ToolRegistry,
            ToolResult,
            ToolSpec,
        )
        from zakcode.usage import Usage

        class _StreamProvider(Provider):
            def __init__(self) -> None:
                self.calls = 0

            async def acomplete(self, messages, *, system=None, tools=None, **kw):
                return LLMResult()

            def count_tokens(self, messages, *, system=None):
                return 0

            def capabilities(self):
                return Capabilities(supports_tools=True, context_window=8192)

            def model_id(self):
                return "test-stream"

            async def astream(self, messages, *, system=None, tools=None, **kw):
                self.calls += 1
                usage = Usage(
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                    cost_usd=0,
                )
                if self.calls == 1:
                    yield StreamToolCallDelta(index=0, id="c1", name="echo")
                    yield StreamToolCallDelta(
                        index=0,
                        arguments_delta='{"text": "hi"}',
                    )
                    yield StreamUsage(usage=usage)
                    yield StreamDone(finish_reason="tool_calls")
                else:
                    yield StreamTextDelta(text="done")
                    yield StreamUsage(usage=usage)
                    yield StreamDone(finish_reason="stop")

        class _EchoTool(Tool):
            spec = ToolSpec(
                name="echo",
                description="Echo.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
                required_permission=PermissionTier.READ_ONLY,
                concurrency=ConcurrencyClass.READ_ONLY_SAFE,
            )

            async def execute(self, args, ctx):
                return ToolResult.ok("ok")

        reg = ToolRegistry()
        reg.register(_EchoTool())
        session = Session(cwd=str(tmp_path), model="test-stream")
        settings = load_settings(workspace_root=tmp_path)

        rw = _integration_writer(tmp_path, session.id)
        try:
            with patch("zakcode.agent.loop._status_writer", return_value=rw):
                loop = AgentLoop(_StreamProvider(), reg, session, settings=settings)

                async def run():
                    return [ev async for ev in loop.astream_turn("hello")]

                asyncio.run(run())

            assert "turn_start:working" in rw.transitions
            assert "model_call_start:test-stream" in rw.transitions
            assert "model_call_end:working" in rw.transitions
            assert "tool_start:echo" in rw.transitions
            assert "tool_end" in rw.transitions
            assert any(t.startswith("idle:") for t in rw.transitions)

            data = _read(rw)
            assert data["state"] == "idle"
        finally:
            rw._stop_event.set()

    def test_cancelled_mid_turn_leaves_idle_via_leave_turn(self, tmp_path: Path) -> None:
        """Point 10d: cancelling mid-tool leaves the file idle via leave_turn."""
        import asyncio

        from zakcode.agent.loop import AgentLoop
        from zakcode.config import PermissionTier, load_settings
        from zakcode.providers.base import (
            Capabilities,
            LLMResult,
            Provider,
            StreamDone,
            StreamToolCallDelta,
            StreamUsage,
        )
        from zakcode.session.store import Session
        from zakcode.tools.base import (
            ConcurrencyClass,
            Tool,
            ToolRegistry,
            ToolSpec,
        )
        from zakcode.usage import Usage

        class _Provider(Provider):
            async def acomplete(self, messages, *, system=None, tools=None, **kw):
                return LLMResult()

            def count_tokens(self, messages, *, system=None):
                return 0

            def capabilities(self):
                return Capabilities(supports_tools=True, context_window=8192)

            async def astream(self, messages, *, system=None, tools=None, **kw):
                usage = Usage(
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                    cost_usd=0,
                )
                yield StreamToolCallDelta(index=0, id="c1", name="stall")
                yield StreamToolCallDelta(index=0, arguments_delta="{}")
                yield StreamUsage(usage=usage)
                yield StreamDone(finish_reason="tool_calls")

        class _StallTool(Tool):
            spec = ToolSpec(
                name="stall",
                description="Stall.",
                parameters={"type": "object", "properties": {}},
                required_permission=PermissionTier.READ_ONLY,
                concurrency=ConcurrencyClass.READ_ONLY_SAFE,
            )

            async def execute(self, args, ctx):
                raise asyncio.CancelledError

        reg = ToolRegistry()
        reg.register(_StallTool())
        session = Session(cwd=str(tmp_path), model="m")
        settings = load_settings(workspace_root=tmp_path)

        rw = _integration_writer(tmp_path, session.id)
        try:
            with patch("zakcode.agent.loop._status_writer", return_value=rw):
                loop = AgentLoop(_Provider(), reg, session, settings=settings)

                async def run():
                    try:
                        async for _ in loop.astream_turn("hello"):
                            pass
                    except asyncio.CancelledError:
                        pass

                asyncio.run(run())

            # Should have leave_turn with "interrupted" because CancelledError.
            assert "leave_turn:interrupted" in rw.transitions
            data = _read(rw)
            assert data["state"] == "idle"
        finally:
            rw._stop_event.set()

    def test_concurrent_read_only_tools_show_running_gt_1(self, tmp_path: Path) -> None:
        """Point 10e: concurrent read-only tools show running > 1."""
        import asyncio

        from zakcode.agent.loop import AgentLoop
        from zakcode.config import PermissionTier, load_settings
        from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
        from zakcode.session.store import Session
        from zakcode.tools.base import (
            ConcurrencyClass,
            Tool,
            ToolRegistry,
            ToolResult,
            ToolSpec,
        )
        from zakcode.usage import Usage

        _usage = Usage(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cost_usd=0,
        )
        max_concurrent_seen = 0

        class _ConcurrentWriter(_RecordingWriter):
            """Track the peak running count."""

            def set_tool_start(self, session_id, name):
                nonlocal max_concurrent_seen
                super().set_tool_start(session_id, name)
                if self._owns(session_id) and self._running_tools > max_concurrent_seen:
                    max_concurrent_seen = self._running_tools

        class _Provider(Provider):
            def __init__(self):
                self.calls = 0

            async def acomplete(self, messages, *, system=None, tools=None, **kw):
                self.calls += 1
                if self.calls == 1:
                    return LLMResult(
                        text="",
                        tool_calls=[
                            ToolCall(id="c1", name="slow", arguments={"n": "1"}),
                            ToolCall(id="c2", name="slow", arguments={"n": "2"}),
                        ],
                        finish_reason="tool_calls",
                        usage=_usage,
                    )
                return LLMResult(
                    text="done",
                    finish_reason="stop",
                    usage=_usage,
                )

            def count_tokens(self, messages, *, system=None):
                return 0

            def capabilities(self):
                return Capabilities(supports_tools=True, context_window=8192)

        class _SlowTool(Tool):
            spec = ToolSpec(
                name="slow",
                description="Slow.",
                parameters={
                    "type": "object",
                    "properties": {"n": {"type": "string"}},
                },
                required_permission=PermissionTier.READ_ONLY,
                concurrency=ConcurrencyClass.READ_ONLY_SAFE,
            )

            async def execute(self, args, ctx):
                await asyncio.sleep(0.05)
                return ToolResult.ok("ok")

        reg = ToolRegistry()
        reg.register(_SlowTool())
        session = Session(cwd=str(tmp_path), model="m")
        settings = load_settings(workspace_root=tmp_path)

        rw = _ConcurrentWriter()
        rw.start(
            session_id=session.id,
            workspace=str(tmp_path),
            model="m",
            build="test",
        )
        try:
            with patch("zakcode.agent.loop._status_writer", return_value=rw):
                loop = AgentLoop(_Provider(), reg, session, settings=settings)
                asyncio.run(loop.arun_turn("hello"))

            # Two read-only tools dispatched via asyncio.gather should overlap.
            assert max_concurrent_seen >= 2
        finally:
            rw._stop_event.set()


# ── integration: turns that do not end the ordinary way ──────────────────────


def _answer_loop(tmp_path: Path) -> Any:
    """A hermetic AgentLoop whose provider answers at once, on both turn paths."""
    from zakcode.agent.loop import AgentLoop
    from zakcode.config import load_settings
    from zakcode.providers.base import (
        Capabilities,
        LLMResult,
        Provider,
        StreamDone,
        StreamTextDelta,
        StreamUsage,
    )
    from zakcode.session.store import Session
    from zakcode.tools.base import ToolRegistry
    from zakcode.usage import Usage

    usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0)

    class _Provider(Provider):
        async def acomplete(self, messages, *, system=None, tools=None, **kw):
            return LLMResult(text="done", finish_reason="stop", usage=usage)

        async def astream(self, messages, *, system=None, tools=None, **kw):
            yield StreamTextDelta(text="done")
            yield StreamUsage(usage=usage)
            yield StreamDone(finish_reason="stop")

        def count_tokens(self, messages, *, system=None):
            return 0

        def capabilities(self):
            return Capabilities(supports_tools=True, context_window=8192)

        def model_id(self):
            return "test-answer"

    session = Session(cwd=str(tmp_path), model="test-answer")
    settings = load_settings(workspace_root=tmp_path)
    return AgentLoop(_Provider(), ToolRegistry(), session, settings=settings)


class TestIntegrationUnwind:
    """Integration: what the file says after a turn that did not simply end."""

    def test_a_streamed_turn_run_inside_a_callers_except_block_keeps_its_stop_reason(
        self, tmp_path: Path
    ) -> None:
        # Inside a turn's finally, sys.exc_info() also reports an exception the CALLER is
        # handling. A turn that consulted it there would record "error" for a turn that
        # ended well; this caller is handling an unrelated failure while it streams. A
        # plain loop, as the CLI's renderer consumes: an async comprehension runs in its
        # own frame and hides the caller's exception, so it would not exercise this.
        import asyncio

        from zakcode.events import AgentDone

        loop = _answer_loop(tmp_path)
        rw = _integration_writer(tmp_path, loop.session.id)
        try:
            with patch("zakcode.agent.loop._status_writer", return_value=rw):

                async def run() -> list[Any]:
                    try:
                        raise ValueError("the caller's own, unrelated failure")
                    except ValueError:
                        events = []
                        async for ev in loop.astream_turn("hello"):
                            events.append(ev)
                        return events

                events = asyncio.run(run())
            done = events[-1]
            assert isinstance(done, AgentDone)
            assert done.stop_reason != "error"
            assert not [t for t in rw.transitions if t.startswith("leave_turn:")]
            assert _read(rw)["last_turn"]["stop_reason"] == done.stop_reason
        finally:
            rw._stop_event.set()

    def test_a_buffered_turn_run_inside_a_callers_except_block_is_not_left_as_an_error(
        self, tmp_path: Path
    ) -> None:
        import asyncio

        loop = _answer_loop(tmp_path)
        rw = _integration_writer(tmp_path, loop.session.id)
        try:
            with patch("zakcode.agent.loop._status_writer", return_value=rw):

                async def run() -> Any:
                    try:
                        raise ValueError("the caller's own, unrelated failure")
                    except ValueError:
                        return await loop.arun_turn("hello")

                result = asyncio.run(run())
            assert not [t for t in rw.transitions if t.startswith("leave_turn:")]
            assert _read(rw)["last_turn"]["stop_reason"] == result.stop_reason
        finally:
            rw._stop_event.set()

    @pytest.mark.parametrize("streamed", [True, False], ids=["streamed", "buffered"])
    def test_an_error_inside_the_turn_leaves_the_file_idle_with_error(
        self, tmp_path: Path, streamed: bool
    ) -> None:
        import asyncio

        loop = _answer_loop(tmp_path)
        rw = _integration_writer(tmp_path, loop.session.id)
        try:
            with (
                patch("zakcode.agent.loop._status_writer", return_value=rw),
                patch.object(loop, "_grant_iteration", side_effect=RuntimeError("boom")),
            ):

                async def run() -> None:
                    if streamed:
                        async for _ in loop.astream_turn("hello"):
                            pass
                    else:
                        await loop.arun_turn("hello")

                with pytest.raises(RuntimeError, match="boom"):
                    asyncio.run(run())
            assert "leave_turn:error" in rw.transitions
            data = _read(rw)
            assert data["state"] == "idle"
            assert data["turn"] is None
            assert data["last_turn"]["stop_reason"] == "error"
        finally:
            rw._stop_event.set()

    def test_a_stream_closed_mid_turn_leaves_the_file_idle_as_interrupted(
        self, tmp_path: Path
    ) -> None:
        # A consumer that stops reading and closes the stream raises GeneratorExit at a
        # yield inside the turn: an interruption, and never a file left "working".
        import asyncio

        from zakcode.events import AgentTextDelta

        loop = _answer_loop(tmp_path)
        rw = _integration_writer(tmp_path, loop.session.id)
        try:
            with patch("zakcode.agent.loop._status_writer", return_value=rw):

                async def run() -> None:
                    stream = loop.astream_turn("hello")
                    async for ev in stream:
                        if isinstance(ev, AgentTextDelta):
                            break
                    await stream.aclose()

                asyncio.run(run())
            assert "leave_turn:interrupted" in rw.transitions
            data = _read(rw)
            assert data["state"] == "idle"
            assert data["last_turn"]["stop_reason"] == "interrupted"
        finally:
            rw._stop_event.set()


# ── the writer never raises into the loop; a failed call is over ─────────────


class TestNeverRaises:
    def test_no_transition_raises_even_when_its_own_work_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A transition raising inside the loop would end the turn: per streamed token,
        # or from a turn's except clause, where it would REPLACE the exception the turn
        # is leaving by. Break the writer's own clock and every transition still returns.
        w = _make_writer(tmp_path)
        try:
            w.set_turn_start(SID)
            w.set_model_call_start(SID)

            def broken_clock() -> str:
                raise RuntimeError("clock unavailable")

            monkeypatch.setattr("zakcode.session.status_file._now_iso", broken_clock)
            w.on_text_delta(SID, "first output")
            w.on_thinking_delta(SID, "x")
            w.on_tool_call_delta(SID, "x")
            w.set_model_call_end(SID)
            w.set_tool_start(SID, "Bash")
            w.set_tool_end(SID)
            w.set_wakeup(SID, 1719849600.0)
            w.leave_turn(SID, "error")
            w.set_idle(SID, stop_reason="completed")
            w.set_turn_start(SID)
            w.set_restarting()
            w.set_exited()
        finally:
            w._stop_event.set()


class TestAFailedModelCallIsOver:
    """A failed call ends the model call: a retry sleep is not a wait on the provider."""

    @pytest.mark.parametrize("streamed", [True, False], ids=["streamed", "buffered"])
    def test_a_provider_error_ends_the_model_call_before_the_turn_ends(
        self, tmp_path: Path, streamed: bool
    ) -> None:
        import asyncio

        from zakcode.providers.base import ProviderError

        loop = _answer_loop(tmp_path)

        async def fail_stream(messages: Any, **kw: Any) -> Any:
            raise ProviderError("the pod is down")
            yield  # an async generator, as astream is

        async def fail_complete(messages: Any, **kw: Any) -> Any:
            raise ProviderError("the pod is down")

        loop.provider.astream = fail_stream
        loop.provider.acomplete = fail_complete
        rw = _integration_writer(tmp_path, loop.session.id)
        try:
            with patch("zakcode.agent.loop._status_writer", return_value=rw):

                async def run() -> None:
                    if streamed:
                        async for _ in loop.astream_turn("hello"):
                            pass
                    else:
                        await loop.arun_turn("hello")

                asyncio.run(run())
            start = rw.transitions.index("model_call_start:test-answer")
            idle = next(i for i, t in enumerate(rw.transitions) if t.startswith("idle:"))
            assert "model_call_end:working" in rw.transitions[start:idle]
            assert _read(rw)["state"] == "idle"
        finally:
            rw._stop_event.set()

"""A live process status file: one JSON file per running session that an outside
observer (a monitor script, a person over ssh) can read to learn what the process
is doing right now, since when, and whether it is making progress.

Privacy is a hard rule: no model text, thinking text, prompts, user input, tool
arguments, tool output, or file contents ever appear in this file.  Only enums,
numbers, timestamps, the tool name, the model name, and paths.

The writer is fully fire-and-forget: every public function swallows every error and
debug-logs it.  A status write must NEVER raise into the agent loop or slow a turn.

Liveness is judged by staleness alone — never by pid probes, which are unsafe on
Windows (``os.kill(pid, 0)`` terminates the target).  A daemon thread refreshes
``updated_at`` every 30 seconds while the process lives; an observer treats a file
not updated for 120 seconds as stale (the process is gone or wedged).
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zakcode.config import zakcode_home

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

#: Status file schema version.
SCHEMA_VERSION = 1

#: How often the daemon thread rewrites ``updated_at`` (seconds).
REFRESH_INTERVAL_SECONDS = 30.0

#: A file not updated within this window is considered stale by readers.
STALE_THRESHOLD_SECONDS = 120.0

#: Minimum interval between throttled progress writes during streaming (seconds).
STREAM_PROGRESS_INTERVAL_SECONDS = 5.0

#: Status files older than this are pruned on startup (seconds).
PRUNE_AGE_SECONDS = 7 * 24 * 60 * 60  # 7 days

#: Valid process states.
STATES = frozenset({"idle", "working", "model_call", "tool", "restarting", "exited"})


def _now_iso() -> str:
    """UTC ISO-8601 timestamp (no microseconds — human-readable)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def status_dir() -> Path:
    """The directory holding per-session status files."""
    return zakcode_home() / "status"


def _wakeup_field(due_at_epoch: float | None) -> dict[str, str] | None:
    """The ``wakeup`` field for a wake-up due at ``due_at_epoch``, or ``None`` for none.

    A value that is not a valid time also reads as ``None``: the writer never raises.
    """
    if due_at_epoch is None:
        return None
    try:
        dt = datetime.fromtimestamp(due_at_epoch, tz=UTC)
    except (OverflowError, OSError, TypeError, ValueError):
        return None
    return {"due_at": dt.strftime("%Y-%m-%dT%H:%M:%SZ")}


# ── atomic write helper ─────────────────────────────────────────────────────


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as JSON to ``path`` atomically: temp file + os.replace.

    On Windows ``os.replace`` can fail while a reader has the file open (the
    target cannot be replaced while another handle holds it); retry once after
    a brief sleep, then skip the write.  Never raises.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.debug("status: failed to write temp file %s: %s", tmp, exc)
        return
    for attempt in range(2):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            if attempt == 0:
                time.sleep(0.05)
    # Both attempts failed — clean up the temp file.
    with contextlib.suppress(OSError):
        tmp.unlink()
    logger.debug("status: os.replace failed for %s (skipped)", path)


# ── startup prune ────────────────────────────────────────────────────────────


def _prune_stale_files(directory: Path) -> None:
    """Best-effort deletion of status files not updated for 7 days."""
    try:
        if not directory.is_dir():
            return
        now = time.time()
        for entry in directory.iterdir():
            if entry.suffix != ".json":
                continue
            try:
                if now - entry.stat().st_mtime > PRUNE_AGE_SECONDS:
                    entry.unlink()
            except OSError:
                pass
    except OSError:
        pass


# ── the writer ───────────────────────────────────────────────────────────────


class StatusFileWriter:
    """Maintains a single ``<status_dir>/<session_id>.json`` for this process.

    Construction alone writes nothing.  Call :meth:`start` once the session is
    known; after that, call the transition methods as the process state changes.
    The writer swallows every error and is fully re-entrant from any thread.

    Session binding: :meth:`start` records the owning ``session_id``; every
    transition method takes a ``session_id`` argument and is a no-op when it
    differs from the owning session.  This prevents a nested sub-agent loop
    (which creates its own ``Session``) from clobbering the top-level status.

    The daemon refresher thread (matching the BusyLease pattern in
    ``say_inbox.py``) keeps ``updated_at`` fresh so an observer can judge
    liveness by staleness alone — no pid probes.
    """

    def __init__(self) -> None:
        self._path: Path | None = None
        self._lock = threading.Lock()
        # Serializes file writes. The loop and the refresher thread both write, through one
        # temp file name, so two overlapping writes could otherwise rename a half-written
        # temp file into place, or land the older of two snapshots last.
        self._write_lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self._refresher: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_progress_write: float = 0.0
        self._started = False
        self._session_id: str | None = None
        self._running_tools: int = 0

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(
        self,
        *,
        session_id: str,
        workspace: str,
        model: str,
        build: str | None,
        wakeup_due_at: float | None = None,
    ) -> None:
        """Begin writing the status file.  Called once per process lifetime.

        Prunes old status files, writes the initial ``idle`` state, starts the
        daemon refresher thread, and registers an atexit handler that writes
        ``exited``.  If ``wakeup_due_at`` is provided (epoch seconds), the
        initial state includes the pending wake-up.  If any of that fails the
        writer stays unstarted, so every later call is a no-op.
        """
        if self._started:
            return
        try:
            self._started = True
            self._session_id = session_id
            sdir = status_dir()
            _prune_stale_files(sdir)
            self._path = sdir / f"{session_id}.json"
            now_iso = _now_iso()
            self._data = {
                "v": SCHEMA_VERSION,
                "pid": os.getpid(),
                "build": build or "",
                "session": session_id,
                "workspace": workspace,
                "model": model,
                "state": "idle",
                "since": now_iso,
                "updated_at": now_iso,
                "turn": None,
                "call": None,
                "tool": None,
                "wakeup": _wakeup_field(wakeup_due_at),
                "last_turn": None,
            }
            self._write()

            # Daemon refresher: rewrite updated_at every REFRESH_INTERVAL_SECONDS.
            self._stop_event.clear()
            self._refresher = threading.Thread(
                target=self._refresh_loop, daemon=True, name="status-refresher"
            )
            self._refresher.start()
            atexit.register(self._on_exit)
        except Exception:  # noqa: BLE001 — the writer must never raise
            logger.debug("status: start failed; no status file for this process", exc_info=True)
            self._started = False
            self._path = None

    def _refresh_loop(self) -> None:
        """Daemon thread: touch ``updated_at`` periodically so an observer can
        detect liveness by staleness alone."""
        while not self._stop_event.wait(REFRESH_INTERVAL_SECONDS):
            with self._lock:
                if self._data.get("state") == "exited":
                    break
                self._data["updated_at"] = _now_iso()
            self._write()

    def _on_exit(self) -> None:
        """atexit handler: write the ``exited`` state."""
        self.set_exited()

    # ── internal write ───────────────────────────────────────────────────

    def _write(self) -> None:
        """Persist the current data dict.  Never raises."""
        if self._path is None:
            return
        try:
            with self._write_lock:
                _atomic_write(self._path, self._data)
        except Exception:  # noqa: BLE001 — the writer must never raise
            logger.debug("status: write failed for %s", self._path, exc_info=True)

    def _owns(self, session_id: str) -> bool:
        """True when ``session_id`` matches the bound session."""
        return self._started and session_id == self._session_id

    # ── state transitions ────────────────────────────────────────────────

    def set_idle(self, session_id: str, *, stop_reason: str | None = None) -> None:
        """Transition to ``idle``.  If ``stop_reason`` is given, fill ``last_turn``."""
        with self._lock:
            if not self._owns(session_id):
                return
            now_iso = _now_iso()
            self._data["state"] = "idle"
            self._data["since"] = now_iso
            self._data["updated_at"] = now_iso
            self._data["call"] = None
            self._data["tool"] = None
            self._running_tools = 0
            if stop_reason is not None:
                self._data["last_turn"] = {"ended_at": now_iso, "stop_reason": stop_reason}
            self._data["turn"] = None
        self._write()

    def set_turn_start(self, session_id: str) -> None:
        """A new turn is starting — state goes to ``working``."""
        with self._lock:
            if not self._owns(session_id):
                return
            now_iso = _now_iso()
            self._data["state"] = "working"
            self._data["since"] = now_iso
            self._data["updated_at"] = now_iso
            self._data["turn"] = {
                "started_at": now_iso,
                "model_calls": 0,
                "tool_calls": 0,
            }
            self._data["call"] = None
            self._data["tool"] = None
            self._running_tools = 0
        self._write()

    def leave_turn(self, session_id: str, reason: str) -> None:
        """The turn unwound by an exception — record ``reason`` and go ``idle``.

        ``reason`` is ``"interrupted"`` or ``"error"``. Called when an exception unwinds
        ``arun_turn`` / ``astream_turn``, so the file never stays in ``working`` /
        ``tool`` / ``model_call`` after the turn is gone. A no-op when no turn is open:
        a turn that already ended keeps the stop reason it ended with.
        """
        with self._lock:
            if not self._owns(session_id) or self._data.get("turn") is None:
                return
            now_iso = _now_iso()
            self._data["state"] = "idle"
            self._data["since"] = now_iso
            self._data["updated_at"] = now_iso
            self._data["call"] = None
            self._data["tool"] = None
            self._running_tools = 0
            self._data["last_turn"] = {"ended_at": now_iso, "stop_reason": reason}
            self._data["turn"] = None
        self._write()

    def set_model_call_start(self, session_id: str, *, model: str = "") -> None:
        """The loop is entering a model call (waiting for the provider).

        If ``model`` is provided, the top-level ``model`` field is updated too
        (tracks provider failover without a separate call).
        """
        with self._lock:
            if not self._owns(session_id):
                return
            now_iso = _now_iso()
            self._data["state"] = "model_call"
            self._data["since"] = now_iso
            self._data["updated_at"] = now_iso
            self._data["call"] = {
                "started_at": now_iso,
                "first_output_at": None,
                "phase": "waiting",
                "model": model or self._data.get("model", ""),
                "thinking_chars": 0,
                "text_chars": 0,
                "tool_call_chars": 0,
            }
            self._data["tool"] = None
            if model:
                self._data["model"] = model
            if self._data.get("turn") is not None:
                self._data["turn"]["model_calls"] = self._data["turn"].get("model_calls", 0) + 1
        self._write()

    def on_thinking_delta(self, session_id: str, text: str) -> None:
        """A thinking delta arrived during a model call."""
        with self._lock:
            if not self._owns(session_id) or self._data.get("call") is None:
                return
            call = self._data["call"]
            was_first = call["first_output_at"] is None
            if was_first:
                call["first_output_at"] = _now_iso()
            prev_phase = call["phase"]
            call["phase"] = "thinking"
            call["thinking_chars"] = call.get("thinking_chars", 0) + len(text)
        # Write immediately on first output or phase change; throttle counter-only.
        if was_first or prev_phase != "thinking":
            self._write()
        else:
            self._throttled_write()

    def on_text_delta(self, session_id: str, text: str) -> None:
        """A text delta arrived during a model call."""
        with self._lock:
            if not self._owns(session_id) or self._data.get("call") is None:
                return
            call = self._data["call"]
            was_first = call["first_output_at"] is None
            if was_first:
                call["first_output_at"] = _now_iso()
            prev_phase = call["phase"]
            call["phase"] = "text"
            call["text_chars"] = call.get("text_chars", 0) + len(text)
        if was_first or prev_phase != "text":
            self._write()
        else:
            self._throttled_write()

    def on_tool_call_delta(self, session_id: str, text: str) -> None:
        """A tool-call argument delta arrived during a model call."""
        with self._lock:
            if not self._owns(session_id) or self._data.get("call") is None:
                return
            call = self._data["call"]
            was_first = call["first_output_at"] is None
            if was_first:
                call["first_output_at"] = _now_iso()
            prev_phase = call["phase"]
            call["phase"] = "tool_call"
            call["tool_call_chars"] = call.get("tool_call_chars", 0) + len(text)
        if was_first or prev_phase != "tool_call":
            self._write()
        else:
            self._throttled_write()

    def set_model_call_end(self, session_id: str) -> None:
        """The model call finished — state goes to ``working`` (turn still open)."""
        with self._lock:
            if not self._owns(session_id):
                return
            now_iso = _now_iso()
            self._data["state"] = "working"
            self._data["since"] = now_iso
            self._data["updated_at"] = now_iso
            self._data["call"] = None
        self._write()

    def set_tool_start(self, session_id: str, name: str) -> None:
        """A tool is starting execution.  Concurrent tools are counted."""
        with self._lock:
            if not self._owns(session_id):
                return
            now_iso = _now_iso()
            self._running_tools += 1
            self._data["state"] = "tool"
            self._data["since"] = now_iso
            self._data["updated_at"] = now_iso
            self._data["tool"] = {
                "name": name,
                "started_at": now_iso,
                "running": self._running_tools,
            }
            self._data["call"] = None
            if self._data.get("turn") is not None:
                self._data["turn"]["tool_calls"] = self._data["turn"].get("tool_calls", 0) + 1
        self._write()

    def set_tool_end(self, session_id: str) -> None:
        """A tool finished execution.  When no tools remain, go to ``working``."""
        with self._lock:
            if not self._owns(session_id):
                return
            now_iso = _now_iso()
            self._running_tools = max(0, self._running_tools - 1)
            if self._running_tools == 0:
                self._data["state"] = "working"
                self._data["tool"] = None
            else:
                if self._data.get("tool") is not None:
                    self._data["tool"]["running"] = self._running_tools
            self._data["since"] = now_iso
            self._data["updated_at"] = now_iso
        self._write()

    def set_wakeup(self, session_id: str | None, due_at_epoch: float | None) -> None:
        """Update the pending wake-up: ``due_at_epoch`` as ISO-8601, or clear it.

        Bound to the owning session like every transition: a nested run's wake-up slot
        must not write the top-level session's wake-up.
        """
        with self._lock:
            if session_id is None or not self._owns(session_id):
                return
            self._data["wakeup"] = _wakeup_field(due_at_epoch)
            self._data["updated_at"] = _now_iso()
        self._write()

    def set_restarting(self) -> None:
        """The process is about to ``os.execv`` into a new build."""
        with self._lock:
            if not self._started:
                return
            now_iso = _now_iso()
            self._data["state"] = "restarting"
            self._data["since"] = now_iso
            self._data["updated_at"] = now_iso
        self._write()

    def set_exited(self) -> None:
        """Normal interpreter exit — write the final state."""
        self._stop_event.set()
        with self._lock:
            if not self._started:
                return
            now_iso = _now_iso()
            self._data["state"] = "exited"
            self._data["since"] = now_iso
            self._data["updated_at"] = now_iso
            self._data["call"] = None
            self._data["tool"] = None
            self._data["turn"] = None
            self._running_tools = 0
        self._write()

    # ── throttled write for streaming progress ───────────────────────────

    def _throttled_write(self) -> None:
        """Write at most once every STREAM_PROGRESS_INTERVAL_SECONDS during streaming."""
        now = time.monotonic()
        if now - self._last_progress_write < STREAM_PROGRESS_INTERVAL_SECONDS:
            return
        self._last_progress_write = now
        self._write()


# ── module-level singleton ───────────────────────────────────────────────────
# The writer is a process singleton: one process, one session, one status file.

_writer = StatusFileWriter()


def get_writer() -> StatusFileWriter:
    """Return the process-global status file writer."""
    return _writer


# ── reader helper (for ``zakcode status``) ───────────────────────────────────


def read_status_files() -> list[dict[str, Any]]:
    """Read all status files from the status directory.  Never raises."""
    sdir = status_dir()
    results: list[dict[str, Any]] = []
    try:
        if not sdir.is_dir():
            return results
        for entry in sorted(sdir.iterdir()):
            if entry.suffix != ".json":
                continue
            try:
                raw = entry.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, dict):
                    results.append(data)
            except (OSError, json.JSONDecodeError):
                pass
    except OSError:
        pass
    return results

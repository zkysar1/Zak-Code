"""The say inbox — ONE contract for handing a running agent its next user message.

A "say" is a single pending user message stored at ``<workspace>/.say``. Every
producer and every consumer in the system speaks this file, so "send a message to
the agent" means exactly one thing regardless of transport or interface:

- ``POST /say`` on ``zakcode webapp`` writes it (the web/watch surface's talk seam).
- ``zakcode say`` and the cockpit's say box write it (terminal surfaces).
- The serve driver consumes it between autonomous turns.
- ``zakcode cli`` consumes it between interactive turns (cockpit/say-inbox mode).
- A line typed at a session's OWN REPL while its turn runs does NOT pass through this
  file (ADR-0078): it is handed to that session's agent in-process. The file is the door
  for producers OUTSIDE the process — several sessions can share one workspace, and the
  slot cannot tell which of them a keystroke was meant for.

Semantics (shared by all of the above):

- **Single slot.** One message may be pending. Writing while one is pending is
  refused — the sender waits for the agent to finish its current thought. This is
  the queue discipline, not a failure.
- **Atomic write.** Temp-file + ``os.replace`` so a concurrent reader never sees a
  half-written message.
- **Exactly-once delivery.** Reading consumes (read then delete). A consumer whose
  turn then fails may re-queue via :func:`requeue_say` (newer message wins).
- **Fail-open reads.** Any OS error while reading yields "no say pending".

Two consumers may share a workspace (a runner whose whole night is one turn, and a
cockpit chat pane between its turns): **the turn in flight owns the inbox** — see the
busy marker below (ADR-0060). Between turns, whoever polls first wins.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

#: The inbox filename under the workspace root.
SAY_FILENAME = ".say"


def say_path(workspace_root: str | os.PathLike[str]) -> Path:
    """The say-inbox file for a workspace."""
    return Path(workspace_root) / SAY_FILENAME


def write_say(path: Path, text: str) -> bool:
    """Queue ``text`` as the pending say. Returns False (unwritten) while one is pending.

    Atomic temp-write + replace; creates the parent directory if missing.
    """
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return True


def read_say(path: Path) -> str | None:
    """Consume the pending say, if any: read then DELETE (exactly-once). Fail-open."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:  # includes FileNotFoundError — no say pending
        return None
    with contextlib.suppress(OSError):
        path.unlink()
    return text or None


def requeue_say(path: Path, text: str) -> None:
    """Best-effort re-queue of a consumed say after a failed turn.

    Skipped when a newer say already occupies the slot (the newer message wins).
    """
    with contextlib.suppress(OSError):
        write_say(path, text)


def say_pending(path: Path) -> bool:
    """True while a message sits in the inbox unconsumed."""
    return path.exists()


# ── the interrupt file: .say's sibling control signal ──────────────────────────────
# Where a say is a MESSAGE ("here is your next input"), an interrupt is a CONTROL
# signal ("stop the turn you are running"). Same one-contract discipline: any surface
# (Esc in the cockpit say box, `zakcode interrupt`, a future web stop button) writes
# the file; the running chat consumes it mid-turn and stops through the exact same
# path as a keyboard Ctrl-C. Unlike the say slot it is idempotent — writing twice is
# still one stop — and a stale file is cleared (not obeyed) by an idle or starting
# chat, so a leftover signal can never kill a future turn.

#: The interrupt filename under the workspace root.
INTERRUPT_FILENAME = ".interrupt"


def interrupt_path(workspace_root: str | os.PathLike[str]) -> Path:
    """The interrupt-signal file for a workspace."""
    return Path(workspace_root) / INTERRUPT_FILENAME


def request_interrupt(path: Path) -> None:
    """Ask the workspace's running agent to stop its current turn. Idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text("stop\n", encoding="utf-8")
    os.replace(tmp, path)


def take_interrupt(path: Path) -> bool:
    """Consume a pending interrupt request, if any. Fail-open (False on any error)."""
    try:
        path.unlink()
    except OSError:  # includes FileNotFoundError — nothing pending
        return False
    return True


# ── the busy marker: the turn in flight owns the inbox (ADR-0060) ────────────────────
# Two consumers can legitimately share one workspace — a runner whose whole night is
# one turn, and a cockpit chat pane polling the inbox every 0.3 s between ITS turns —
# and the single slot then goes to whoever reads first, which is always the idle one.
# Measured 2026-08-28 (coach on zc-03): every operator say of a morning reached the
# cockpit pane, none the runner they were steering; one of them was a control command
# that flipped the runner's shared mode file from under it. The marker settles it: a
# main-loop turn claims ``<workspace>/.busy`` for its length and refreshes it while it
# runs; idle consumers stand back while a FRESH marker names another process. The
# holder's own mid-turn poll (ADR-0051) is unaffected, so a say written while the runner
# works lands in the runner at its next iteration boundary. Staleness, not pid
# liveness, is the liveness test: a pid probe is not portable (``os.kill(pid, 0)``
# TERMINATES the target on Windows), and a crashed holder's marker simply ages out.

#: The busy-marker filename under the workspace root.
BUSY_FILENAME = ".busy"
#: A marker older than this names nobody: its holder crashed or hung without releasing.
BUSY_STALE_SECONDS = 120.0
#: How often a holder touches its marker while a turn runs (well inside the stale window).
BUSY_REFRESH_SECONDS = 30.0


def busy_path(workspace_root: str | os.PathLike[str]) -> Path:
    """The busy-marker file for a workspace."""
    return Path(workspace_root) / BUSY_FILENAME


def _read_busy(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _busy_is_fresh(path: Path) -> bool:
    try:
        return time.time() - path.stat().st_mtime < BUSY_STALE_SECONDS
    except OSError:
        return False


def _busy_is_ours(path: Path) -> bool:
    marker = _read_busy(path)
    return marker is not None and marker.get("pid") == os.getpid()


def busy_elsewhere(path: Path) -> bool:
    """True while a fresh marker names a process other than this one. Fail-open (False)."""
    if not _busy_is_fresh(path):
        return False
    marker = _read_busy(path)
    return marker is not None and marker.get("pid") != os.getpid()


def claim_busy(path: Path, session_id: str) -> bool:
    """Mark this process mid-turn on the workspace.

    False (unclaimed) while another process's fresh marker stands — that turn owns the
    inbox and this one runs without a claim. Never raises: the marker is a routing hint,
    not a gate on the turn.
    """
    if busy_elsewhere(path):
        return False
    payload = json.dumps({"pid": os.getpid(), "session": session_id, "since": time.time()})
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        return False
    return True


def refresh_busy(path: Path) -> None:
    """Keep our marker fresh across a long model call. No-op unless the marker is ours."""
    if _busy_is_ours(path):
        with contextlib.suppress(OSError):
            os.utime(path, None)


def release_busy(path: Path) -> None:
    """Drop our marker at turn end. No-op unless the marker is ours."""
    if _busy_is_ours(path):
        with contextlib.suppress(OSError):
            path.unlink()


class BusyLease:
    """Holds the busy marker for exactly one turn: claim, refresh on a timer, release.

    ``acquire`` claims the marker (or records that another process holds it) and starts
    the refresh task only when held; ``release`` cancels the task and drops the marker.
    Both are idempotent and never raise.
    """

    def __init__(self, path: Path, session_id: str) -> None:
        self.path = path
        self.session_id = session_id
        self.held = False
        self._task: asyncio.Task[None] | None = None

    async def acquire(self) -> None:
        self.held = claim_busy(self.path, self.session_id)
        if self.held:
            self._task = asyncio.create_task(self._keep_fresh())

    async def _keep_fresh(self) -> None:
        while True:
            await asyncio.sleep(BUSY_REFRESH_SECONDS)
            refresh_busy(self.path)

    async def release(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self.held:
            release_busy(self.path)
            self.held = False

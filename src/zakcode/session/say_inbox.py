"""The say inbox — ONE contract for handing a running agent its next user message.

A "say" is a single pending user message stored at ``<workspace>/.say``. Every
producer and every consumer in the system speaks this file, so "send a message to
the agent" means exactly one thing regardless of transport or interface:

- ``POST /say`` on ``zakcode serve`` writes it (the web/watch surface's talk seam).
- ``zakcode say`` and the cockpit's say box write it (terminal surfaces).
- The serve driver consumes it between autonomous turns.
- ``zakcode chat`` consumes it between interactive turns (cockpit/say-inbox mode).

Semantics (shared by all of the above):

- **Single slot.** One message may be pending. Writing while one is pending is
  refused — the sender waits for the agent to finish its current thought. This is
  the queue discipline, not a failure.
- **Atomic write.** Temp-file + ``os.replace`` so a concurrent reader never sees a
  half-written message.
- **Exactly-once delivery.** Reading consumes (read then delete). A consumer whose
  turn then fails may re-queue via :func:`requeue_say` (newer message wins).
- **Fail-open reads.** Any OS error while reading yields "no say pending".

Run at most ONE consumer per workspace (a chat in inbox mode OR a serve driver) —
two consumers race for the single slot and one silently wins.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

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

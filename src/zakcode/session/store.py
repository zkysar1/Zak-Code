"""Durable session state and on-disk persistence.

A :class:`Session` captures the full conversation history, accumulated token
usage, and the metadata needed to resume work across CLI invocations. The
:class:`SessionStore` persists sessions as one versioned JSON document per
session id, writing atomically so a crash mid-save never corrupts an existing
session file.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from zakcode.messages import Message
from zakcode.usage import Usage


def _new_id() -> str:
    """Return a fresh random session id (uuid4 hex)."""
    return uuid.uuid4().hex


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


class Session(BaseModel):
    """A persistable conversation session.

    Holds the ordered message history plus per-call usage records. The
    ``version`` field tags the on-disk schema so future migrations can detect
    older documents.
    """

    version: int = 1
    id: str = Field(default_factory=_new_id)
    cwd: str
    model: str
    messages: list[Message] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)
    usages: list[Usage] = Field(default_factory=list)

    def add_message(self, msg: Message) -> None:
        """Append ``msg`` to the conversation history."""
        self.messages.append(msg)

    def add_usage(self, usage: Usage) -> None:
        """Record a single LLM-call ``usage`` entry."""
        self.usages.append(usage)

    def cumulative_usage(self) -> Usage:
        """Return the sum of all recorded usage entries."""
        total = Usage()
        for usage in self.usages:
            total = total + usage
        return total


class SessionStore:
    """Reads and writes :class:`Session` documents on disk.

    Each session is stored as ``<base_dir>/<id>.json``. ``base_dir`` defaults to
    ``~/.zakcode/sessions`` and is created on construction.
    """

    def __init__(self, base_dir: str | os.PathLike[str] | None = None) -> None:
        if base_dir is None:
            base_dir = Path.home() / ".zakcode" / "sessions"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        """Return the JSON file path for ``session_id``."""
        return self.base_dir / f"{session_id}.json"

    def save(self, session: Session) -> Path:
        """Persist ``session`` atomically and return its file path.

        Writes to a temporary file in the same directory, then ``os.replace``\\ s
        it onto the final path so readers never observe a partial write.
        """
        path = self._path_for(session.id)
        payload = session.model_dump_json()
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return path

    def load(self, session_id: str) -> Session:
        """Load and return the session stored under ``session_id``."""
        path = self._path_for(session_id)
        payload = path.read_text(encoding="utf-8")
        return Session.model_validate_json(payload)

    def list(self) -> list[str]:
        """Return the ids of all persisted sessions, sorted."""
        return sorted(p.stem for p in self.base_dir.glob("*.json"))

    def resume(self, session_id: str) -> Session:
        """Load a previously saved session so work can continue."""
        return self.load(session_id)

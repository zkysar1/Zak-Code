"""Durable session state and on-disk persistence.

A :class:`Session` captures the full conversation history, accumulated token
usage, and the metadata needed to resume work across CLI invocations. The
:class:`SessionStore` persists sessions as one versioned JSON document per
session id, writing atomically so a crash mid-save never corrupts an existing
session file.

Robustness policy (corruption + version drift)
----------------------------------------------
* :meth:`SessionStore.load` of an unknown id raises :class:`SessionNotFound`.
* :meth:`SessionStore.load` of a file that is not valid JSON, or whose JSON does
  not match the session schema, raises :class:`SessionCorruptError`. A failed
  load NEVER deletes or rewrites the file -- the bytes on disk are left exactly
  as they were found so an operator can inspect or recover them.
* Schema version drift is keyed off :data:`CURRENT_SCHEMA_VERSION`:
    - A document whose ``version`` is **less than or equal to** the current
      version loads if it is structurally valid. The schema is append-only, so
      older documents are forward-compatible (missing fields take defaults).
    - A document whose ``version`` is **greater than** the current version (a
      newer/unknown writer) is rejected with :class:`SessionVersionError`,
      because this build cannot safely interpret a schema it does not yet know.
      The file is left untouched.
* :meth:`SessionStore.save` is atomic: the document is rendered fully in memory,
  written to a uniquely-named temp file in the same directory, then
  ``os.replace``-d over the target. If anything fails before the replace, the
  temp file is removed and any pre-existing session file is preserved.
* :meth:`SessionStore.list` returns only the ids of plain ``*.json`` files and
  silently skips in-progress temp files, directories, and unrelated entries.
"""

from __future__ import annotations

import contextlib
import json
import ntpath
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from zakcode.messages import Message
from zakcode.usage import Usage

#: Highest on-disk schema version this build can write and fully understand.
CURRENT_SCHEMA_VERSION = 1

#: Suffix marking an in-progress temp file written during an atomic save.
_TMP_SUFFIX = ".tmp"


def _new_id() -> str:
    """Return a fresh random session id (uuid4 hex)."""
    return uuid.uuid4().hex


def _is_safe_session_id(session_id: object) -> bool:
    """Whether ``session_id`` is a safe single filename component (cannot escape base_dir).

    Ids the store mints are uuid4 hex; this also accepts other simple tokens but REJECTS
    anything that could traverse out of the sessions directory — path separators, ``..``,
    a ``:`` (Windows drive / NTFS stream), an absolute or UNC path (checked POSIX- AND
    Windows-style on every OS), a NUL byte, or an empty/blank value. ``session_id`` reaches
    here straight from an HTTP request body (``ChatRequest.session_id``), so this is a
    network-facing trust boundary, not a mere sanity check. (audit3 #1)
    """
    if not isinstance(session_id, str) or not session_id.strip():
        return False
    if session_id in (".", ".."):
        return False
    if any(ch in session_id for ch in ("/", "\\", ":", "\x00")):
        return False
    return not (os.path.isabs(session_id) or ntpath.isabs(session_id))


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


class SessionError(Exception):
    """Base class for all session-store load/persist errors."""


class SessionNotFound(SessionError):
    """Raised when a session id has no corresponding file on disk."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"No session found with id {session_id!r}")


class SessionCorruptError(SessionError):
    """Raised when a session file exists but cannot be parsed or validated.

    Loading is non-destructive: the underlying file is left exactly as found.
    """

    def __init__(self, session_id: str, reason: str) -> None:
        self.session_id = session_id
        self.reason = reason
        super().__init__(f"Session {session_id!r} is corrupt: {reason}")


class SessionVersionError(SessionError):
    """Raised when a session document is newer than this build understands."""

    def __init__(self, session_id: str, found: int, supported: int) -> None:
        self.session_id = session_id
        self.found = found
        self.supported = supported
        super().__init__(
            f"Session {session_id!r} has schema version {found}, but this build "
            f"supports at most {supported}; upgrade Zak Code to read it"
        )


class Session(BaseModel):
    """A persistable conversation session.

    Holds the ordered message history plus per-call usage records. The
    ``version`` field tags the on-disk schema so future migrations can detect
    older documents.
    """

    version: int = CURRENT_SCHEMA_VERSION
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
    ``~/.zakcode/sessions`` and is created on construction. Session ids are validated as
    safe single filename components (:func:`_is_safe_session_id`) on save/load/delete, so a
    request-supplied id can never traverse out of ``base_dir`` — load/delete treat an unsafe
    id as not-found, and save rejects it.
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

        The document is rendered fully in memory first, written to a uniquely
        named temporary file in the same directory, and then ``os.replace``\\ d
        onto the final path so readers never observe a partial write. If
        rendering or writing fails partway, the temp file is cleaned up and any
        previously stored session file is left intact -- no partial/garbage file
        survives a failed save.
        """
        if not _is_safe_session_id(session.id):
            raise ValueError(f"unsafe session id {session.id!r} (would escape the store dir)")
        path = self._path_for(session.id)
        # Render before touching the filesystem: if serialization raises, the
        # existing on-disk file (if any) is never disturbed.
        payload = session.model_dump_json()

        # A unique temp name keeps concurrent-ish saves of the same id from
        # clobbering each other's temp file mid-write, and avoids leaving a
        # predictable artifact behind on failure.
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}{_TMP_SUFFIX}")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            # Best-effort cleanup of the partial temp file; never touch the real
            # file. Swallow cleanup errors so the original failure propagates.
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            raise
        return path

    def load(self, session_id: str) -> Session:
        """Load and return the session stored under ``session_id``.

        Raises:
            SessionNotFound: no file exists for ``session_id``.
            SessionVersionError: the document declares a newer schema version
                than this build supports (the file is left untouched).
            SessionCorruptError: the file is not valid UTF-8 JSON, is not a JSON
                object, or fails schema validation (the file is left untouched).
        """
        # An id that isn't a safe single component cannot name a real session and must not
        # be allowed to traverse out of base_dir — treat it as not-found. (audit3 #1)
        if not _is_safe_session_id(session_id):
            raise SessionNotFound(session_id)
        path = self._path_for(session_id)
        try:
            payload = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SessionNotFound(session_id) from exc

        # Parse JSON ourselves first so we can inspect the declared version
        # before handing the document to the model validator, and so non-JSON or
        # non-object content yields a precise, non-crashing error.
        try:
            raw = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SessionCorruptError(session_id, f"invalid JSON ({exc})") from exc

        if not isinstance(raw, dict):
            raise SessionCorruptError(
                session_id, f"expected a JSON object, found {type(raw).__name__}"
            )

        version = raw.get("version", CURRENT_SCHEMA_VERSION)
        # bool is a subclass of int; reject it explicitly so `true`/`false`
        # versions are flagged as corrupt rather than coerced to 1/0.
        if isinstance(version, bool) or not isinstance(version, int):
            raise SessionCorruptError(session_id, f"non-integer schema version {version!r}")
        if version > CURRENT_SCHEMA_VERSION:
            raise SessionVersionError(session_id, version, CURRENT_SCHEMA_VERSION)

        try:
            return Session.model_validate(raw)
        except ValidationError as exc:
            raise SessionCorruptError(session_id, f"schema mismatch ({exc})") from exc

    def list(self) -> list[str]:
        """Return the ids of all persisted sessions, sorted.

        Only plain ``*.json`` files are considered. In-progress ``*.tmp`` save
        artifacts, directories, and any other entries are skipped so junk in the
        directory never crashes the listing.
        """
        ids: list[str] = []
        for p in self.base_dir.glob("*.json"):
            if p.name.endswith(_TMP_SUFFIX) or not p.is_file():
                continue
            ids.append(p.stem)
        return sorted(ids)

    def resume(self, session_id: str) -> Session:
        """Load a previously saved session so work can continue."""
        return self.load(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete the session stored under ``session_id``.

        Returns ``True`` if a session file was removed, ``False`` if none existed.
        Never raises for a missing session, so a double-delete is harmless.
        """
        if not _is_safe_session_id(session_id):
            return False  # an unsafe id names no session here — nothing to delete
        try:
            self._path_for(session_id).unlink()
            return True
        except FileNotFoundError:
            return False

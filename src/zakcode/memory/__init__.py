"""Cross-session memory — a vendor-agnostic substrate, not a learning loop.

Zak Code keeps no memory between sessions on its own; this package adds the
*storage and retrieval* primitives a memory layer (or a self-learning framework
like Claude-Mind — see ``docs/INTEGRATIONS``) builds policy on top of. The split is
deliberate:

* :class:`MemoryProvider` is the abstract contract — ``add`` / ``search`` /
  ``recent`` / ``delete`` / ``count``. It says nothing about *what* to remember or
  *when*; that routing is policy, owned by the layer above (a ``remember`` tool the
  model calls, a session-end writer, or a framework's encode pass).
* :class:`~zakcode.memory.sqlite_store.SqliteMemoryProvider` is the batteries-included
  default: a local SQLite database (FTS5 full-text search when available, a LIKE
  fallback otherwise) whose path is configurable, so the store can live wherever an
  integrating framework wants it (``ZAKCODE_MEMORY_DB_PATH`` / per-agent paths).
* :class:`MemoryRecallHook` is the bridge to the agent loop: a ``PRE_LLM_CALL``
  context hook (M-Phase-2) that recalls memories relevant to the current turn and
  returns them as injected, *fenced-as-untrusted* background context. It caches per
  turn so it queries once regardless of how many iterations the turn runs.

Because recalled text is folded into the prompt as untrusted data (the loop fences
+ defangs it), memory is never treated as instructions — only as background the
model may use.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class MemoryRecord(BaseModel):
    """One stored memory.

    ``kind`` and ``tags`` are free-form classification the policy layer assigns
    (e.g. ``"preference"``, ``"fact"``, ``"lesson"``); the substrate does not
    interpret them beyond indexing ``tags`` for search. ``score`` is populated by
    :meth:`MemoryProvider.search` (relevance; higher = better) and is ``None`` on a
    stored or recency-listed record.
    """

    id: str = Field(default_factory=_new_id)
    text: str
    kind: str = "note"
    tags: list[str] = Field(default_factory=list)
    source: str = ""
    created_at: str = Field(default_factory=_now_iso)
    score: float | None = None


class MemoryProvider(ABC):
    """Abstract cross-session memory store (storage + retrieval only)."""

    @abstractmethod
    def add(
        self,
        text: str,
        *,
        kind: str = "note",
        tags: list[str] | None = None,
        source: str = "",
    ) -> MemoryRecord:
        """Persist a memory and return the stored :class:`MemoryRecord`."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        """Return up to ``limit`` memories most relevant to ``query`` (best first)."""
        raise NotImplementedError

    @abstractmethod
    def recent(self, *, limit: int = 10) -> list[MemoryRecord]:
        """Return up to ``limit`` most recently added memories (newest first)."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory by id. Returns ``True`` if one was removed."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Total number of stored memories."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources (default no-op)."""
        return None


class MemoryRecallHook:
    """A ``PRE_LLM_CALL`` context hook that injects memories relevant to the turn.

    Searches the provider for the turn's ``user_text`` and renders the hits as a
    short block. The result is cached per ``user_text`` so a multi-iteration turn
    queries the store only once. Returns ``None`` (inject nothing) when there are no
    relevant memories. The loop fences/defangs whatever is returned, so recalled
    content is presented to the model as untrusted background, never instructions.
    """

    def __init__(self, provider: MemoryProvider, *, limit: int = 5) -> None:
        self._provider = provider
        self._limit = limit
        self._cache: dict[str, str] = {}

    def __call__(self, payload: object) -> str | None:
        # Duck-typed on LLMContextPayload to avoid importing the hooks layer here.
        user_text = getattr(payload, "user_text", "") or ""
        if not user_text.strip():
            return None
        if user_text not in self._cache:
            try:
                records = self._provider.search(user_text, limit=self._limit)
            except Exception:  # noqa: BLE001 — recall must never break a turn
                records = []
            self._cache[user_text] = self._render(records)
        return self._cache[user_text] or None

    @staticmethod
    def _render(records: list[MemoryRecord]) -> str:
        if not records:
            return ""
        # Defense-in-depth: scrub credential-shaped text before re-injecting recalled
        # memories into the prompt — covers entries written outside Zak Code's
        # already-scrubbing ``remember`` tool (docs/GUARDRAILS.md §6).
        from zakcode.secrets import redact_secrets

        lines = ["Possibly-relevant memories from past sessions:"]
        for record in records:
            label = f" ({record.kind})" if record.kind and record.kind != "note" else ""
            text, _ = redact_secrets(record.text)
            lines.append(f"-{label} {text}")
        return "\n".join(lines)


__all__ = ["MemoryRecord", "MemoryProvider", "MemoryRecallHook"]

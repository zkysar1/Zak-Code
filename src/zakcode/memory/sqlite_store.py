"""A local SQLite cross-session memory store (FTS5 full-text, LIKE fallback).

This is the batteries-included :class:`~zakcode.memory.MemoryProvider`. It keeps
memories in one SQLite file at a configurable path (so an integrating framework can
relocate it per-agent), using an FTS5 virtual table for ranked full-text search when
the runtime's ``sqlite3`` was built with FTS5, and degrading transparently to a
plain table with substring (``LIKE``) matching otherwise. Either way the public
behavior is the same; only ranking quality differs (bm25 relevance vs recency).

The store is deliberately dumb: it persists and retrieves, and never decides what is
worth remembering — that policy lives above it (see :mod:`zakcode.memory`).
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from zakcode.memory import MemoryProvider, MemoryRecord, _new_id, _now_iso

#: Where the default store lives when no path is configured.
DEFAULT_DB_PATH = Path.home() / ".zakcode" / "memory.db"

#: Cap on how many query tokens feed an FTS/LIKE search (keeps the query bounded).
_MAX_QUERY_TOKENS = 24

_COLUMNS = "mem_id, text, kind, tags, source, created_at"


def _tokenize(query: str) -> list[str]:
    """Lowercase alphanumeric/underscore tokens from a free-text query (bounded)."""
    return re.findall(r"[a-z0-9_]+", query.lower())[:_MAX_QUERY_TOKENS]


class SqliteMemoryProvider(MemoryProvider):
    """A :class:`MemoryProvider` backed by a single SQLite database file."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the loop is single-threaded async, but a caller may
        # construct on one thread and use on another; writes are still serialized by
        # the NEVER_PARALLEL ``remember`` tool.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._fts = self._init_schema()

    # ── schema ───────────────────────────────────────────────────────────────

    def _init_schema(self) -> bool:
        """Create the store, preferring an FTS5 table. Returns whether FTS5 is used.

        Only ``text`` and ``tags`` are indexed (searchable); the rest are UNINDEXED
        FTS5 columns kept purely for retrieval.
        """
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories USING fts5("
                "mem_id UNINDEXED, text, kind UNINDEXED, tags, "
                "source UNINDEXED, created_at UNINDEXED, tokenize='unicode61')"
            )
            self._conn.commit()
            # Confirm it really is the FTS5 table (a prior run may have made a plain one).
            return self._table_is_fts()
        except sqlite3.OperationalError:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS memories ("
                "mem_id TEXT PRIMARY KEY, text TEXT, kind TEXT, tags TEXT, "
                "source TEXT, created_at TEXT)"
            )
            self._conn.commit()
            return False

    def _table_is_fts(self) -> bool:
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        return bool(row) and "fts5" in (row[0] or "").lower()

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_tags(raw: str) -> list[str]:
        """Parse the stored tags column back to a list (JSON; tolerates legacy rows).

        Tags are stored as a JSON array so multi-word tags round-trip losslessly;
        rows written before that (space-joined) are still read correctly.
        """
        raw = raw or ""
        if raw.strip().startswith("["):
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                return [t for t in parsed if isinstance(t, str)]
        return [t for t in raw.split() if t]

    @classmethod
    def _row_to_record(cls, row: tuple, *, score: float | None = None) -> MemoryRecord:
        mem_id, text, kind, tags, source, created_at = row
        tag_list = cls._decode_tags(tags)
        return MemoryRecord(
            id=mem_id,
            text=text,
            kind=kind or "note",
            tags=tag_list,
            source=source or "",
            created_at=created_at or "",
            score=score,
        )

    # ── MemoryProvider interface ────────────────────────────────────────────────

    def add(
        self,
        text: str,
        *,
        kind: str = "note",
        tags: list[str] | None = None,
        source: str = "",
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=_new_id(),
            text=text,
            kind=kind or "note",
            tags=tags or [],
            source=source,
            created_at=_now_iso(),
        )
        self._conn.execute(
            f"INSERT INTO memories ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.text,
                record.kind,
                json.dumps(record.tags),
                record.source,
                record.created_at,
            ),
        )
        self._conn.commit()
        return record

    def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        if self._fts:
            return self._search_fts(tokens, limit)
        return self._search_like(tokens, limit)

    def _search_fts(self, tokens: list[str], limit: int) -> list[MemoryRecord]:
        # Quote each token so FTS5 treats it as a literal term (no operator injection);
        # OR them so any match contributes, ranked by bm25 (the `rank` column; more
        # negative = more relevant, so ascending order is best-first). Relevance score
        # is reported as the negated rank (higher = better) for the public record.
        match = " OR ".join(f'"{t}"' for t in tokens)
        try:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS}, rank FROM memories WHERE memories MATCH ? "
                f"ORDER BY rank LIMIT ?",
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return self._search_like(tokens, limit)
        return [self._row_to_record(r[:-1], score=-float(r[-1])) for r in rows]

    def _search_like(self, tokens: list[str], limit: int) -> list[MemoryRecord]:
        clauses = " OR ".join(["text LIKE ? OR tags LIKE ?"] * len(tokens))
        params: list[object] = []
        for tok in tokens:
            like = f"%{tok}%"
            params.extend([like, like])
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM memories WHERE {clauses} "
            f"ORDER BY created_at DESC, rowid DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def recent(self, *, limit: int = 10) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM memories ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def update(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        kind: str | None = None,
        tags: list[str] | None = None,
    ) -> MemoryRecord | None:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM memories WHERE mem_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        current = self._row_to_record(row)
        # None = leave the field as-is; a provided value replaces it (tags replaces the list).
        new_text = current.text if text is None else text
        new_kind = current.kind if kind is None else (kind or "note")
        new_tags = current.tags if tags is None else tags
        self._conn.execute(
            "UPDATE memories SET text = ?, kind = ?, tags = ? WHERE mem_id = ?",
            (new_text, new_kind, json.dumps(new_tags), memory_id),
        )
        self._conn.commit()
        return MemoryRecord(
            id=current.id,
            text=new_text,
            kind=new_kind,
            tags=new_tags,
            source=current.source,
            created_at=current.created_at,
        )

    def delete(self, memory_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM memories WHERE mem_id = ?", (memory_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def count(self) -> int:
        row = self._conn.execute("SELECT count(*) FROM memories").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()


__all__ = ["SqliteMemoryProvider", "DEFAULT_DB_PATH"]

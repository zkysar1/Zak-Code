"""Deterministic failure-lesson writer (research R1).

When a turn RECOVERS from trouble, record ONE terse, fact-derived lesson to the cross-session
memory store, so a future turn that hits the same symptom can recall what worked. The HARNESS
authors the lesson from signals it already holds — the recipe gate's failed-then-succeeded run
command, or the StuckTracker's repeatedly-failing call — with NO model call, so it stays
small-model-safe and vendor-agnostic. Lessons are de-duplicated (normalized + Jaccard overlap)
so the store does not bloat, and the whole thing is a clean no-op when cross-session memory is
not enabled.
"""

from __future__ import annotations

import re

from zakcode.agent.recipe import RecipeCursor
from zakcode.agent.stuck import StuckTracker
from zakcode.memory import _STOPWORDS, MemoryProvider, _tokens
from zakcode.secrets import redact_secrets

#: ``kind`` stored on harness-authored lessons (distinct from a model's ``remember`` notes).
LESSON_KIND = "lesson"

#: Two normalized lessons whose distinctive-word sets overlap at least this much are duplicates.
_DEDUP_JACCARD = 0.8

#: Volatile fragments collapsed when normalizing a symptom for dedup, so a temp path / a line
#: number / a hash do not defeat duplicate detection. Order matters (paths before bare numbers).
_VOLATILE_RE = re.compile(
    r"""(?xi)
    (?:[a-z]:[\\/][^\s"']*)        # windows absolute path
    | (?:\.{0,2}[\\/][^\s"']+)     # posix / relative path
    | (?:line\s+\d+)               # line numbers
    | (?:0x[0-9a-f]+)              # hex literal
    | (?:[0-9a-f-]{8,})            # long hex / hashes / uuids
    | (?:\d+)                      # bare numbers
    """
)


def _normalize(text: str) -> str:
    """Collapse volatile bits (paths, line numbers, hashes, numbers) and lowercase, so
    near-identical symptoms dedup to the same key."""
    return _VOLATILE_RE.sub("·", text).lower()


def _derive(
    stuck: StuckTracker, cursor: RecipeCursor, *, stop_reason: str
) -> tuple[str, list[str]] | None:
    """The single lesson (text, tags) to record for a recovered turn, or ``None`` if none.

    Gate B (preferred): the recipe gate's create-and-run recovery — a written file failed its
    first run, then a run succeeded; the lesson cites the command that worked. Gate A: the stuck
    recovery ladder fired and the turn then completed; the lesson records the repeatedly-failing
    call as a known-trouble pattern.
    """
    recovered = cursor.recovered_command
    if recovered is not None and cursor.verified:
        cmd, _ = redact_secrets(recovered.strip())
        text = (
            "When a newly written file failed its first run, running "
            f"`{cmd}` succeeded — use that command to run it."
        )
        return text, ["recovery", "run"]

    if stuck.took_action and stop_reason == "completed":
        sigs = stuck.error_signatures()
        if sigs:
            name, args = sigs[0]
            shape = _normalize(args).strip()[:80]
            detail = f" (arguments like `{shape}`)" if shape and shape not in ("{}", "·") else ""
            text = (
                f"Earlier this session, repeated `{name}` calls kept failing{detail} before the "
                f"task completed — if `{name}` keeps failing, investigate the cause first."
            )
            return text, [name, "stuck-recovery"]
    return None


class LessonWriter:
    """Per-turn (stateless) writer of deterministic recovery lessons; no-op without a store."""

    def __init__(self, memory: MemoryProvider | None, *, source: str) -> None:
        self._memory = memory
        self._source = source

    def maybe_write(self, stuck: StuckTracker, cursor: RecipeCursor, *, stop_reason: str) -> bool:
        """Write at most ONE deterministic lesson for a recovered turn; return whether it wrote.

        Returns ``False`` (no-op) when memory is disabled, no recovery occurred, or an equivalent
        lesson already exists. Pure assembly + a single store call — never a model call.
        """
        if self._memory is None:
            return False
        derived = _derive(stuck, cursor, stop_reason=stop_reason)
        if derived is None:
            return False
        text, raw_tags = derived
        scrubbed, _ = redact_secrets(text)
        tags = [redact_secrets(t)[0] for t in raw_tags]
        if self._is_duplicate(scrubbed, tags):
            return False
        self._memory.add(scrubbed, kind=LESSON_KIND, tags=tags, source=self._source)
        return True

    def _is_duplicate(self, text: str, tags: list[str]) -> bool:
        """True when an equivalent lesson already exists.

        A candidate duplicates an existing lesson only when it has the SAME identity (kind +
        tag set — the tag set carries the distinguishing payload, e.g. the failing tool name)
        AND a near-identical body (normalized-exact OR distinctive-word Jaccard >= the floor).
        The tag gate is essential: the templates are mostly boilerplate, so without it two
        stuck-recovery lessons that differ only in tool name would collide above the Jaccard
        floor and the second would be wrongly dropped (review2 #2). Searches on the RAW text
        (FTS returns nothing for a token-less query), then compares in Python.
        """
        if self._memory is None:
            return False
        try:
            hits = self._memory.search(text, limit=10)
        except Exception:  # noqa: BLE001 — dedup is best-effort; a search failure never blocks a write
            return False
        tagset = set(tags)
        norm = _normalize(text)
        norm_tokens = {t for t in _tokens(norm) if t not in _STOPWORDS}
        for hit in hits:
            if hit.kind != LESSON_KIND or set(hit.tags) != tagset:
                continue  # different lesson identity -> never a duplicate
            other = _normalize(hit.text)
            if other == norm:
                return True
            other_tokens = {t for t in _tokens(other) if t not in _STOPWORDS}
            if norm_tokens and other_tokens:
                jaccard = len(norm_tokens & other_tokens) / len(norm_tokens | other_tokens)
                if jaccard >= _DEDUP_JACCARD:
                    return True
        return False


__all__ = ["LESSON_KIND", "LessonWriter"]

"""A lightweight, structured per-turn decision trace (observability).

``TurnResult.stop_reason`` answers *what* ended a turn; this answers *why* — the ordered story of
the decisions the loop made: how it routed, every gate / recovery intervention it fired, and the
final stop with its detail. It is the DECISION layer, complementing the session transcript (which
records what the *model* did). Attached to :class:`~zakcode.agent.loop.TurnResult` /
``AgentDone`` for programmatic inspection, and — when ``ZAKCODE_TRACE_DIR`` (``Settings.trace_dir``)
is set — written to a JSONL file per turn, so debugging a turn never again requires hand-building
a transcript dumper.

Deliberately a flat, append-only event list keyed by a free-form ``kind`` so new decision types
(notably a future best-of-N / critic-ensemble **fan-out**, where each small-model sub-call is one
``"fanout"`` event) are observable without any schema change here.
"""

from __future__ import annotations

import contextlib
from typing import Any

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """One decision the loop made during a turn.

    ``kind`` is a short tag (``"route"``, ``"intervention"``, ``"fanout"``, ``"stop"``); ``detail``
    is a one-line human summary; ``data`` carries the structured specifics (category, model,
    counts, reason, …). Intentionally schema-light so the vocabulary can grow without migrations.
    """

    kind: str
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class TurnTrace(BaseModel):
    """The ordered decision trace for one turn (empty on a clean, uneventful turn)."""

    events: list[TraceEvent] = Field(default_factory=list)

    def note(self, kind: str, detail: str = "", /, **data: Any) -> None:
        """Append a decision event. Best-effort — observability must never break a turn, so a bad
        payload is dropped rather than raised.

        ``kind`` and ``detail`` are positional-only (the ``/``) so a structured field may itself
        be named ``kind`` — e.g. an ``"intervention"`` event whose payload tags the specific gate
        with ``note("intervention", "...", kind="doom_loop")``; the leading ``kind`` is the EVENT
        kind, the keyword ``kind`` lands in ``data``. Without positional-only that call would bind
        ``kind`` twice and raise ``TypeError``."""
        # Tracing is non-essential; a malformed payload is swallowed rather than allowed to
        # abort the loop (the SIM105-preferred form of a try/except/pass guard).
        with contextlib.suppress(Exception):
            self.events.append(TraceEvent(kind=kind, detail=detail, data=data))

    def of_kind(self, kind: str) -> list[TraceEvent]:
        """All events of one ``kind`` (handy for tests and summaries)."""
        return [e for e in self.events if e.kind == kind]

    def to_jsonl(self) -> str:
        """The trace as newline-delimited JSON — one event per line."""
        return "\n".join(e.model_dump_json() for e in self.events)


__all__ = ["TraceEvent", "TurnTrace"]

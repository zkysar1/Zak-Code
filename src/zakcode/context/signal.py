"""Used/useful signal logging for the context gatherer (step 3).

The training-data half of the plan: every turn, record WHAT context the gatherer offered and WHICH
offered refs the model actually USED (referenced in its response or acted on via a tool call). That
labelled stream -- offered ``(ref, score)`` -> ``used?`` -- is what a relevance classifier (step 4)
fine-tunes on; the data loop is the real ML pipeline, the training is downstream and easy once the
labels exist.

A single :class:`SignalLogger` registers as a ``TURN_END`` hook: it reads the gatherer's last offer
plus the just-finished turn's tool calls (from the session). Append-only JSONL, best-effort -- a
logging failure is swallowed and never affects the turn.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from zakcode.hooks import TurnEndPayload
from zakcode.messages import Message
from zakcode.session.store import Session

from .gatherer import ContextGatherer

logger = logging.getLogger(__name__)


class OfferedItem(BaseModel):
    """One offered candidate and whether the model used it this turn."""

    ref: str
    score: float
    source: str
    used: bool = False


class SignalRecord(BaseModel):
    """One turn's relevance training signal: what was offered and what was used."""

    stop_reason: str = ""
    task: str = ""
    offered: list[OfferedItem] = Field(default_factory=list)
    n_offered: int = 0
    n_used: int = 0


def _tool_input_blob(messages: list[Message]) -> str:
    """Every string value from every assistant tool call's input -- 'what the model acted on'."""
    parts: list[str] = []
    for msg in messages:
        if msg.role != "assistant":
            continue
        for tu in msg.tool_uses:
            parts.extend(v for v in tu.input.values() if isinstance(v, str))
    return "\n".join(parts)


class SignalLogger:
    """Logs the (offered context -> used?) signal once per turn as append-only JSONL.

    Register as an observe-only ``TURN_END`` hook (``register_turn_end_observer``) so it fires on
    EVERY turn end. It reads the gatherer's last offer and the just-finished turn's tool calls (from
    the session); ``used`` is true when an offered ref appears in a tool call's arguments or the
    final assistant message. NOTE: this is a reference-based proxy -- it undercounts context the
    model consumed purely in-context (already injected, never re-fetched), and on a multi-iteration
    turn it reflects the final iteration's offer. Best-effort: a logging failure is swallowed.
    """

    def __init__(self, gatherer: ContextGatherer, session: Session, log_path: str | Path) -> None:
        self._gatherer = gatherer
        self._session = session
        self._path = Path(log_path)

    def on_turn_end(self, payload: TurnEndPayload) -> None:
        offer = self._gatherer.last_offer
        if not offer:
            return
        try:
            turn_msgs = self._session.messages[self._gatherer.last_message_count :]
            blob = _tool_input_blob(turn_msgs) + "\n" + payload.last_assistant_message
            items = [
                OfferedItem(ref=c.ref, score=c.cheap_score, source=c.source, used=c.ref in blob)
                for c in offer
            ]
            rec = SignalRecord(
                stop_reason=payload.stop_reason,
                task=self._gatherer.last_task[:500],
                offered=items,
                n_offered=len(items),
                n_used=sum(1 for it in items if it.used),
            )
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(rec.model_dump_json() + "\n")
        except Exception:  # noqa: BLE001 - logging must never affect the turn
            logger.debug("signal logging failed", exc_info=True)

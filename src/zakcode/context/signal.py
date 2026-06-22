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

import inspect
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from zakcode.hooks import TurnEndPayload
from zakcode.messages import Message
from zakcode.session.store import Session

from .gatherer import ContextGatherer
from .used import UsedDetector, reference_used

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
    the session); ``used`` is decided by a pluggable ``UsedDetector``: the default reference proxy
    (an offered ref appears in a tool-call arg or the assistant text), or ``ModelUsedDetector`` -- a
    cheap-model judge that ALSO catches IN-CONTEXT use (the model consuming injected content without
    naming the file, which the proxy structurally misses). Caveat: on a multi-iteration turn it
    reflects the final iteration's offer. Best-effort: a logging failure is swallowed.
    """

    def __init__(
        self,
        gatherer: ContextGatherer,
        session: Session,
        log_path: str | Path,
        *,
        used_detector: UsedDetector = reference_used,
    ) -> None:
        self._gatherer = gatherer
        self._session = session
        self._path = Path(log_path)
        self._used_detector = used_detector

    async def on_turn_end(self, payload: TurnEndPayload) -> None:
        offer = self._gatherer.last_offer
        if not offer:
            return
        try:
            turn_msgs = self._session.messages[self._gatherer.last_message_count :]
            evidence = _tool_input_blob(turn_msgs) + "\n" + payload.last_assistant_message
            task = self._gatherer.last_task
            try:
                maybe = self._used_detector(task, offer, evidence)
                used_refs = await maybe if inspect.isawaitable(maybe) else maybe
            except Exception:  # noqa: BLE001 - a detector failure falls back to the proxy
                logger.debug("used-detector failed; using the reference proxy", exc_info=True)
                used_refs = reference_used(task, offer, evidence)
            items = [
                OfferedItem(
                    ref=c.ref, score=c.cheap_score, source=c.source, used=c.ref in used_refs
                )
                for c in offer
            ]
            rec = SignalRecord(
                stop_reason=payload.stop_reason,
                task=task[:500],
                offered=items,
                n_offered=len(items),
                n_used=sum(1 for it in items if it.used),
            )
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(rec.model_dump_json() + "\n")
        except Exception:  # noqa: BLE001 - logging must never affect the turn
            logger.debug("signal logging failed", exc_info=True)

"""Deterministic context-gatherer: a ``PRE_LLM_CALL`` context hook that runs EVERY turn.

The harness fires registered context hooks before every model completion
(``HookManager.gather_context`` -> the loop folds the returned text in as an
ephemeral, fenced, untrusted tail message). This module is the *deterministic*
gatherer that hangs on that seam: each turn it

1. collects candidate context from a set of deterministic ``Collector``s,
2. ranks them by relevance through a swappable ``RelevanceClassifier`` seam, and
3. renders the top, budget-bounded slice as the text to inject.

The whole point is determinism: the model never decides *whether* to gather --
the gather happens on every turn, unconditionally. The only genuinely cognitive
part, *relevance ranking*, is isolated behind the ``RelevanceClassifier``
Protocol so it can graduate from a zero-model heuristic (this module) to a
small-model classifier to a fine-tuned model with **no change to the gatherer or
the harness**.

Fail-safe: a collector or classifier that raises is swallowed -- a bad context
source degrades gracefully to *less* context, never to a broken turn (same
discipline as the quality-engine judge).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast, runtime_checkable

from zakcode.hooks import LLMContextPayload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """One piece of candidate context a collector surfaced."""

    source: str  # collector kind, e.g. "file" | "history" | "world"
    ref: str  # a stable handle: a path, a turn id, ...
    text: str  # the actual chunk to (maybe) inject
    cheap_score: float = 0.0  # deterministic prior in [0, 1]: recency, name-overlap, ...
    meta: dict[str, str] = field(default_factory=dict)


#: A deterministic candidate source. A pure function of the turn payload; no model.
Collector = Callable[[LLMContextPayload], Sequence[Candidate]]


@runtime_checkable
class RelevanceClassifier(Protocol):
    """The single swap-point. Given the task text and the candidates, return them
    ranked most-relevant-first. Implementations graduate over time -- a zero-model
    heuristic (step 1), a small-model classifier (step 2), a fine-tuned model
    (step 4) -- and the gatherer only ever sees this Protocol."""

    def __call__(
        self, task: str, candidates: Sequence[Candidate], /
    ) -> list[Candidate] | Awaitable[list[Candidate]]: ...


def heuristic_classifier(task: str, candidates: Sequence[Candidate]) -> list[Candidate]:
    """Step-1 backend: ZERO models. Rank by the collectors' deterministic prior."""
    return sorted(candidates, key=lambda c: c.cheap_score, reverse=True)


def _fit_budget(ranked: Sequence[Candidate], *, budget_chars: int, top_k: int) -> list[Candidate]:
    """Take from the ranked list until the char budget or top_k is hit (whichever first)."""
    picked: list[Candidate] = []
    used = 0
    for c in ranked:
        if len(picked) >= top_k:
            break
        size = len(c.text)
        if size == 0:
            continue
        if picked and used + size > budget_chars:
            break
        picked.append(c)
        used += size
    return picked


def _render(picked: Sequence[Candidate]) -> str:
    return "\n\n".join(f"[{c.source}: {c.ref}]\n{c.text.strip()}" for c in picked)


class ContextGatherer:
    """A deterministic ``PRE_LLM_CALL`` context hook. Register once; runs every turn::

        agent.hook_manager.register_context(ContextGatherer(collectors, classifier))

    It is intentionally boring: gather -> rank -> budget -> render. All the
    intelligence that will ever live here is behind ``classifier``.
    """

    def __init__(
        self,
        collectors: Sequence[Collector],
        classifier: RelevanceClassifier = heuristic_classifier,
        *,
        budget_chars: int = 8000,
        top_k: int = 15,
    ) -> None:
        self._collectors = list(collectors)
        self._classify = classifier
        self._budget_chars = budget_chars
        self._top_k = top_k
        # Recorded each call so a TURN_END signal logger can see what was last offered (step 3).
        self.last_offer: list[Candidate] = []
        self.last_message_count: int = 0
        self.last_task: str = ""

    def _collect(self, payload: LLMContextPayload) -> list[Candidate]:
        out: list[Candidate] = []
        for col in self._collectors:
            try:
                out.extend(col(payload))
            except Exception:  # noqa: BLE001 - a bad collector must never trap the turn
                logger.debug(
                    "context collector %r failed; skipping",
                    getattr(col, "__name__", col),
                    exc_info=True,
                )
        return out

    async def __call__(self, payload: LLMContextPayload) -> str | None:
        self.last_message_count = payload.message_count
        self.last_task = payload.user_text
        self.last_offer = []
        # Collectors do blocking file I/O; run them off the event loop so the gather (which fires
        # before every model call) can't stall the loop or concurrent sessions.
        candidates = await asyncio.to_thread(self._collect, payload)
        if not candidates:
            return None
        try:
            maybe = self._classify(payload.user_text, candidates)
            if inspect.isawaitable(maybe):
                ranked = await cast("Awaitable[list[Candidate]]", maybe)
            else:
                ranked = cast("list[Candidate]", maybe)
        except Exception:  # noqa: BLE001 - a bad classifier degrades to the cheap order, never a broken turn
            logger.debug("relevance classifier failed; falling back to heuristic", exc_info=True)
            ranked = heuristic_classifier(payload.user_text, candidates)
        picked = _fit_budget(ranked, budget_chars=self._budget_chars, top_k=self._top_k)
        self.last_offer = picked
        if not picked:
            return None
        return _render(picked) or None

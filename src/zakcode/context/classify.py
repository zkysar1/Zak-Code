"""Small-model relevance classifier for the context gatherer (step 2).

A ``RelevanceClassifier`` (the seam in :mod:`zakcode.context.gatherer`) backed by ONE cheap model
call per turn: it scores every candidate's relevance to the task in a single batched request and
ranks by the scores. One call -- not N-per-candidate -- because this runs on EVERY turn; the
per-attempt fan-out bet is for generation quality, not for an always-on ranking.

Fail-soft: any failure (provider error, unparseable output, no scores) falls back to the
candidates' deterministic heuristic prior, so a model hiccup degrades to the step-1 behaviour --
never a broken or blocked turn. json_object mode + local validation (the judge-primitive pattern).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from zakcode.messages import Message
from zakcode.providers.base import Provider
from zakcode.providers.structured import coerce_structured, make_response_format
from zakcode.usage import Usage

from .gatherer import Candidate, heuristic_classifier

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You rank background context by RELEVANCE to a task. You are given a TASK and a numbered list "
    "of CANDIDATE context snippets, each with a `ref`. For each candidate, score how useful it is "
    "for accomplishing the task: 1.0 = essential, 0.0 = irrelevant. Reply with ONLY a JSON object: "
    '{"scores": [{"ref": "<the candidate ref>", "score": <number 0.0-1.0>}, ...]} covering every '
    "candidate. No prose, no markdown fence."
)
_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"ref": {"type": "string"}, "score": {"type": "number"}},
                "required": ["ref", "score"],
            },
        }
    },
    "required": ["scores"],
}

_MAX_CANDIDATES = 40  # cap candidates put to the model so the prompt stays bounded
_MAX_SNIPPET = 400  # per-candidate snippet chars in the ranking prompt
_MAX_TASK = 2000


def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + " …"


async def score_relevance(
    provider: Provider,
    task: str,
    candidates: Sequence[Candidate],
    *,
    temperature: float = 0.0,
) -> tuple[dict[str, float], Usage]:
    """Score each candidate's relevance to ``task`` in one batched call -> ``(scores, usage)``.

    FAIL-SOFT: any error returns ``({}, Usage())`` so the caller can fall back to the heuristic.
    """
    items = list(candidates)[:_MAX_CANDIDATES]
    if not items:
        return {}, Usage()
    listing = "\n\n".join(
        f"[{i + 1}] ref={c.ref}\n{_clip(c.text, _MAX_SNIPPET)}" for i, c in enumerate(items)
    )
    payload = f"<task>\n{_clip(task, _MAX_TASK)}\n</task>\n\n<candidates>\n{listing}\n</candidates>"
    try:
        result = await provider.acomplete(
            [Message.user(payload)],
            system=_SYSTEM,
            response_format=make_response_format(None),
            temperature=temperature,
        )
        data = coerce_structured(result.text, schema=_SCHEMA)
        scores: dict[str, float] = {}
        for row in data.get("scores", []) or []:
            if not isinstance(row, dict):
                continue
            ref = str(row.get("ref", ""))
            try:
                scores[ref] = max(0.0, min(1.0, float(row.get("score", 0.0))))
            except (TypeError, ValueError):
                continue
        return scores, result.usage
    except Exception:  # noqa: BLE001 - fail-soft: a scoring failure → empty, caller falls back
        logger.debug("relevance scoring failed; caller will fall back to heuristic", exc_info=True)
        return {}, Usage()


class SmallModelClassifier:
    """A ``RelevanceClassifier`` backed by one cheap model call per turn (see ``score_relevance``).

    Ranks candidates by the model's relevance scores; a candidate the model did not score (or every
    candidate, on failure) keeps a downweighted heuristic prior -- so this is strictly a quality
    *upgrade* over step 1 that can never do worse than the heuristic.

    ``on_usage`` (optional) is called with each call's :class:`~zakcode.usage.Usage` so a caller can
    account the spend on its session / budget; omitted, the small cost is simply not tracked.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        temperature: float = 0.0,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._on_usage = on_usage

    async def __call__(self, task: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        scores, usage = await score_relevance(
            self._provider, task, candidates, temperature=self._temperature
        )
        if self._on_usage is not None and usage.total_tokens:
            try:
                self._on_usage(usage)
            except Exception:  # noqa: BLE001 - accounting must never trap the turn
                logger.debug("on_usage callback failed", exc_info=True)
        if not scores:
            return heuristic_classifier(task, candidates)
        # Model score wins; an unscored candidate keeps a downweighted heuristic prior so it sorts
        # below anything the model actually rated.
        return sorted(
            candidates, key=lambda c: scores.get(c.ref, c.cheap_score * 0.5), reverse=True
        )

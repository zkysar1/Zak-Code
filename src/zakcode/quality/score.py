"""Rubric scoring: an ABSOLUTE [0,1] quality score from weighted dimensions (the Pennywise input).

Pairwise selection (:func:`~zakcode.quality.judge.best_of`) answers *"which is better"* but never
*"is this good ENOUGH to ship"* — a gate needs an absolute score. :func:`score_rubric` has the judge
score an artifact on each named dimension [0,1] in one call, then :func:`aggregate_scores` combines
them MULTIPLICATIVELY with an IAUS-style compensation: a zero on any dimension VETOES the whole
score, but a compensation factor (stronger with more dimensions) counters the multiplicative bias so
adding a dimension doesn't unfairly sink a good artifact. Weights temper each dimension's penalty.

Same contract as the judges: vendor-agnostic, json_object mode (sidestepping the Groq json_schema
trap), validated locally, and fail-safe — but the scorer abstains LOW (a 0), because for a SHIP gate
an artifact whose quality can't be verified should keep iterating, not ship unchecked.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from zakcode.messages import Message
from zakcode.providers.base import Provider
from zakcode.providers.structured import coerce_structured, make_response_format
from zakcode.usage import Usage


def aggregate_scores(scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Combine per-dimension scores [0,1] into one [0,1] score, MULTIPLICATIVELY with compensation.

    Each score clamps to [0,1]; a weight ``w`` tempers that dimension's penalty (``1-s``): ``w=1``
    full weight, ``w=0`` ignores the dimension (``s→1``), ``w>1`` amplifies the penalty. A dimension
    whose WEIGHTED score reaches 0 VETOES (→ 0) — a full-weight zero vetoes; ``w<1`` softens even a
    zero. The compensation factor (``1 - 1/n``, stronger with more
    dimensions) counters the bias that multiplying many sub-1 scores collapses toward 0, so a
    dimension a candidate handles well doesn't punish it. Empty → 0.
    """
    if not scores:
        return 0.0
    weights = weights or {}
    mod = 1.0 - 1.0 / len(scores)  # compensation factor: stronger with more dimensions
    product = 1.0
    for name, raw in scores.items():
        s = max(0.0, min(1.0, raw))
        w = weights.get(name, 1.0)
        s = max(0.0, 1.0 - w * (1.0 - s))  # weight tempers the penalty (1 - s)
        product *= s + (1.0 - s) * mod * s  # multiply, + IAUS make-up to compensate
    return product


class ScoreCard(BaseModel):
    """Per-dimension scores [0,1], the aggregate ``overall``, and a one-line rationale."""

    scores: dict[str, float] = Field(default_factory=dict)
    overall: float = 0.0
    notes: str = ""


_SCORE_SYSTEM = (
    "You are a STRICT, calibrated quality scorer. You are given DIMENSIONS (each with what "
    "it means) and an ARTIFACT. Score the artifact on EACH dimension from 0.0 to 1.0: 1.0 ONLY for "
    "genuinely complete, correct work on that dimension; 0.0 if it is absent or broken; partial "
    "credit in between. Judge each dimension on its own merits. Reply with ONLY a JSON object: "
    '{"scores": {"<dimension>": <0..1>, ...}, "notes": "<one short sentence>"}. No prose, no '
    "markdown fence."
)
_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "object", "additionalProperties": {"type": "number"}},
        "notes": {"type": "string"},
    },
    "required": ["scores"],
    "additionalProperties": False,
}
_MAX_CHARS = 4000


def _clip(text: str) -> str:
    text = text or ""
    return text if len(text) <= _MAX_CHARS else text[:_MAX_CHARS] + "\n…[truncated]"


def _coerce_unit(value: object) -> float:
    """A model-provided score → a clamped [0,1] float; unparseable → 0.0."""
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


async def score_rubric(
    provider: Provider,
    *,
    artifact: str,
    dimensions: dict[str, str],
    weights: dict[str, float] | None = None,
    temperature: float = 0.0,
) -> tuple[ScoreCard, Usage]:
    """Score ``artifact`` on each of ``dimensions`` (name → what to assess) in ONE judge call, then
    aggregate (:func:`aggregate_scores`). Returns ``(scorecard, usage)``; ``scorecard.overall`` is
    the aggregate [0,1]. Only the requested dimensions are scored — a missing one is 0 (and so
    vetoes). FAIL-SAFE, abstaining LOW: any error (provider failure, unparseable output) yields
    ``overall=0.0``, so a SHIP gate keeps iterating (bounded by the caller's budget) rather than
    shipping an artifact whose quality couldn't be verified.
    """
    if not dimensions:
        return ScoreCard(), Usage()
    dim_list = "\n".join(f"- {name}: {desc}" for name, desc in dimensions.items())
    payload = (
        f"<dimensions>\n{dim_list}\n</dimensions>\n\n<artifact>\n{_clip(artifact)}\n</artifact>"
    )
    try:
        result = await provider.acomplete(
            [Message.user(payload)],
            system=_SCORE_SYSTEM,
            response_format=make_response_format(None),
            temperature=temperature,
        )
        data = coerce_structured(result.text, schema=_SCORE_SCHEMA)
        raw = data.get("scores")
        raw = raw if isinstance(raw, dict) else {}
        scores = {name: _coerce_unit(raw.get(name)) for name in dimensions}
        overall = aggregate_scores(scores, weights)
        return ScoreCard(
            scores=scores, overall=overall, notes=str(data.get("notes", ""))
        ), result.usage
    except Exception:  # noqa: BLE001 — fail-safe: abstain LOW so the gate keeps iterating, not ships
        return ScoreCard(notes="scoring failed"), Usage()


__all__ = ["ScoreCard", "aggregate_scores", "score_rubric"]

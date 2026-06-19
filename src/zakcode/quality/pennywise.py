"""Pennywise: the score-and-ship gate. Score an artifact, then decide ship-vs-iterate within budget.

The quality engine's cost governor. It pairs the rubric scorer (:func:`~zakcode.quality.score.
score_rubric`) with a cost gate: SHIP when the artifact is good enough (``overall >= threshold``) OR
when there's no budget left to iterate; otherwise ITERATE. The small-model bet is to spend tokens on
iteration only while they buy quality AND the budget allows — quality from bounded churn, never
unbounded. The gate is the thing that keeps "fan out and iterate" from running the meter to zero.

Vendor- and accounting-agnostic: the CALLER owns the budget and passes ``can_afford_iteration`` (a
bool), so this module never reaches into a session or a wallet — it just scores and decides.
"""

from __future__ import annotations

from pydantic import BaseModel

from zakcode.providers.base import Provider
from zakcode.quality.score import ScoreCard, score_rubric
from zakcode.usage import Usage


class ShipDecision(BaseModel):
    """Whether to ship the artifact as-is, with the aggregate score and the one-line why."""

    ship: bool
    overall: float
    reason: str
    scorecard: ScoreCard


async def pennywise(
    provider: Provider,
    *,
    artifact: str,
    dimensions: dict[str, str],
    threshold: float = 0.8,
    can_afford_iteration: bool = True,
    weights: dict[str, float] | None = None,
) -> tuple[ShipDecision, Usage]:
    """Score ``artifact`` (:func:`~zakcode.quality.score.score_rubric`) and decide SHIP vs ITERATE.

    Ship when ``overall >= threshold`` (good enough) OR when ``not can_afford_iteration`` (out of
    budget — ship the best we have rather than stall); otherwise iterate. The caller owns the budget
    check and passes ``can_afford_iteration``, keeping this vendor- and accounting-agnostic. Returns
    ``(decision, usage)`` — the usage is the scorer's, for the caller to account.
    """
    card, usage = await score_rubric(
        provider, artifact=artifact, dimensions=dimensions, weights=weights
    )
    if card.overall >= threshold:
        return (
            ShipDecision(
                ship=True,
                overall=card.overall,
                reason=f"overall {card.overall:.2f} >= threshold {threshold:.2f} — ship",
                scorecard=card,
            ),
            usage,
        )
    if not can_afford_iteration:
        return (
            ShipDecision(
                ship=True,
                overall=card.overall,
                reason=f"overall {card.overall:.2f} < {threshold:.2f}: no budget — ship",
                scorecard=card,
            ),
            usage,
        )
    return (
        ShipDecision(
            ship=False,
            overall=card.overall,
            reason=f"overall {card.overall:.2f} < threshold {threshold:.2f} — iterate",
            scorecard=card,
        ),
        usage,
    )


__all__ = ["ShipDecision", "pennywise"]

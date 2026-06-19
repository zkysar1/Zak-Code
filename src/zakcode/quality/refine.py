"""Bounded refine loop: generate → score & gate (Pennywise) → revise the weak dimensions → re-score.

The consumer that closes the quality engine's `… → iterate` move. It composes the pieces already
built: produce an artifact, score it on a rubric and gate it (:func:`~zakcode.quality.pennywise.
pennywise`), and if it falls short, REVISE it using the scorecard's weakest dimensions as a targeted
brief — repeating until it ships (overall ≥ threshold), runs out of rounds, or runs out of budget.
The small-model bet in loop form: buy quality with a few cheap, FEEDBACK-DIRECTED iterations instead
of one big-model shot.

Pure orchestration — generation is an injected thunk, scoring is the rubric judge — so it stays
vendor-agnostic and fail-safe (the scorer abstains LOW, so an unscoreable round just keeps
iterating, bounded by ``max_rounds`` and the budget).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from zakcode.providers.base import Provider
from zakcode.quality.pennywise import pennywise
from zakcode.quality.score import ScoreCard
from zakcode.usage import Usage


class RefineResult(BaseModel):
    """The BEST artifact the loop produced, its scorecard, whether it shipped, and rounds run."""

    artifact: str
    scorecard: ScoreCard
    shipped: bool  # True iff the returned artifact met the threshold (vs ran out of rounds/budget)
    rounds: int


def weak_dimensions(card: ScoreCard, threshold: float) -> str:
    """A revision brief from ``card``'s below-threshold dimensions, weakest first."""
    weak = sorted((s, name) for name, s in card.scores.items() if s < threshold)
    if not weak:
        return "Improve the overall quality; address any remaining gaps."
    lines = "\n".join(f"- {name} (scored {s:.2f})" for s, name in weak)
    return f"Revise to raise these weak dimensions:\n{lines}"


async def refine(
    provider: Provider,
    *,
    dimensions: dict[str, str],
    generate: Callable[[str | None], Awaitable[tuple[str, Usage]]],
    threshold: float = 0.8,
    max_rounds: int = 3,
    can_afford: Callable[[], bool] | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[RefineResult, Usage]:
    """Generate an artifact, score + gate it (:func:`~zakcode.quality.pennywise.pennywise`), and if
    it falls short REVISE it with the scorecard's weak dimensions — until it ships, hits
    ``max_rounds``, or ``can_afford()`` goes False.

    ``generate(feedback)`` produces the artifact: ``feedback`` is ``None`` on the first round, else
    the weak-dimension brief (:func:`weak_dimensions`). The loop tracks the BEST artifact by
    ``overall`` across rounds, so a regression in a later round never loses earlier good work, and
    returns that best one. ``shipped`` is True iff the returned artifact actually met ``threshold``
    (vs the loop running out of rounds/budget and shipping the best it had). Returns
    ``(RefineResult, total_usage)``. ``max_rounds`` clamps to ≥ 1.
    """
    max_rounds = max(1, max_rounds)
    feedback: str | None = None
    best: tuple[str, ScoreCard] | None = None
    usage = Usage()
    rounds = 0
    for r in range(max_rounds):
        rounds = r + 1
        artifact, gen_usage = await generate(feedback)
        usage = usage + gen_usage
        # The last round can't iterate (no round left); nor can we if the budget callback says so.
        affordable = r < max_rounds - 1 and (can_afford is None or can_afford())
        decision, score_usage = await pennywise(
            provider,
            artifact=artifact,
            dimensions=dimensions,
            threshold=threshold,
            can_afford_iteration=affordable,
            weights=weights,
        )
        usage = usage + score_usage
        if best is None or decision.overall > best[1].overall:
            best = (artifact, decision.scorecard)
        if decision.ship:
            break
        feedback = weak_dimensions(decision.scorecard, threshold)

    assert best is not None  # the loop runs at least once (max_rounds ≥ 1)
    artifact, card = best
    return (
        RefineResult(
            artifact=artifact, scorecard=card, shipped=card.overall >= threshold, rounds=rounds
        ),
        usage,
    )


__all__ = ["RefineResult", "refine", "weak_dimensions"]

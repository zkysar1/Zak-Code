"""Judged decomposition: score or pick the best of N candidate plans for a goal (quality × HTN).

Zak Code already HAS the hierarchical task network (:mod:`zakcode.tasks`); this does NOT decompose —
it JUDGES a decomposition. :func:`score_plan` gives one plan an absolute quality on a decomposition
rubric (coverage / granularity / ordering / soundness); :func:`judge_plan` picks the best of N by a
pairwise tournament (pairwise selection, more reliable than absolute scoring). This fights the
well-documented "collapses to a single-task plan" failure. Both compose existing primitives
(:func:`~zakcode.quality.score.score_rubric`, :func:`~zakcode.quality.judge.best_of`) and are
TEXT-based, so they stay decoupled from the HTN data structure — the caller serializes the plan
(e.g. ``TaskNetwork.render()``).
"""

from __future__ import annotations

from zakcode.providers.base import Provider
from zakcode.quality.judge import best_of
from zakcode.quality.score import ScoreCard, score_rubric
from zakcode.usage import Usage

#: The default decomposition rubric — what makes a plan GOOD, independent of domain.
PLAN_RUBRIC: dict[str, str] = {
    "coverage": "Do the steps together fully achieve the goal, with nothing important left out?",
    "granularity": "Is each step right-sized — one focused run, not trivial and not a mega-task?",
    "ordering": "Are dependencies respected — no step needs a later step's result?",
    "soundness": "Is every step correct and necessary — none wrong, redundant, or dead-end?",
}


async def score_plan(
    provider: Provider,
    *,
    goal: str,
    plan: str,
    rubric: dict[str, str] | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[ScoreCard, Usage]:
    """Score one candidate ``plan`` for ``goal`` on a decomposition ``rubric`` (default
    :data:`PLAN_RUBRIC`) via :func:`~zakcode.quality.score.score_rubric`. The absolute "how good is
    this decomposition?" — as useful for an anti-over-planning gate (a one-liner that already scores
    high needs no plan) as for quality. Returns ``(scorecard, usage)``.
    """
    artifact = f"<goal>\n{goal}\n</goal>\n\n<plan>\n{plan}\n</plan>"
    return await score_rubric(
        provider, artifact=artifact, dimensions=rubric or PLAN_RUBRIC, weights=weights
    )


async def judge_plan(
    provider: Provider,
    *,
    goal: str,
    candidates: list[str],
    votes: int = 1,
) -> tuple[int, Usage]:
    """The INDEX of the best of N candidate decompositions of ``goal`` by a pairwise tournament
    (:func:`~zakcode.quality.judge.best_of`) — pairwise selection, more reliable than absolute
    scoring. ``votes`` polls an N-judge panel per comparison. ``≤ 1`` candidate returns ``0``.
    Returns ``(best_index, total_usage)``.
    """
    criteria = (
        "Which plan better decomposes the GOAL into a sound, complete, well-ordered, right-sized "
        f"set of steps?\n\n<goal>\n{goal}\n</goal>"
    )
    return await best_of(provider, criteria=criteria, candidates=candidates, votes=votes)


__all__ = ["PLAN_RUBRIC", "judge_plan", "score_plan"]

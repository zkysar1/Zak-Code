"""The quality engine: small-model fan-out for quality (judges, scoring, refinement, decomposition).

The bet: a fixed small-model ceiling is beaten by STRUCTURE — decompose → fan out → judge/score →
iterate — not by a bigger model. This package holds those primitives; the agent loop composes them.

Increment 1: :mod:`~zakcode.quality.judge` — LLM-as-judge primitives (binary verdict, N-judge
majority vote, pairwise comparison, pairwise tournament selection).
Increment 2: :mod:`~zakcode.quality.bestof` — :func:`best_of_n`, fan out N attempts at a generation
and judge-select the best.
Selection quality: :mod:`~zakcode.quality.select` — :func:`select_best`, oracle-FILTER then
judge-RANK (the measured fix: oracle for "works", judge for "good").
Increment 3a: :func:`vote_pairwise` — N-judge majority for sharper selection.
Increment 3b: :mod:`~zakcode.quality.score` (:func:`score_rubric`, absolute weighted score) +
:mod:`~zakcode.quality.pennywise` (:func:`pennywise`, the score-and-ship gate).
"""

from zakcode.quality.bestof import best_of_n
from zakcode.quality.judge import (
    BinaryVerdict,
    PairwiseVerdict,
    best_of,
    binary_judge,
    pairwise_judge,
    vote_binary,
    vote_pairwise,
)
from zakcode.quality.pennywise import ShipDecision, pennywise
from zakcode.quality.score import ScoreCard, aggregate_scores, score_rubric
from zakcode.quality.select import select_best

__all__ = [
    "BinaryVerdict",
    "PairwiseVerdict",
    "ScoreCard",
    "ShipDecision",
    "aggregate_scores",
    "best_of",
    "best_of_n",
    "binary_judge",
    "pairwise_judge",
    "pennywise",
    "score_rubric",
    "select_best",
    "vote_binary",
    "vote_pairwise",
]

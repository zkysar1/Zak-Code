"""The quality engine: small-model fan-out for quality (judges, scoring, refinement, decomposition).

The bet: a fixed small-model ceiling is beaten by STRUCTURE — decompose → fan out → judge/score →
iterate — not by a bigger model. This package holds those primitives; the agent loop composes them.

Increment 1: :mod:`~zakcode.quality.judge` — LLM-as-judge primitives (binary verdict, N-judge
majority vote, pairwise comparison, pairwise tournament selection).
Increment 2: :mod:`~zakcode.quality.bestof` — :func:`best_of_n`, fan out N attempts at a generation
and judge-select the best.
Selection quality: :mod:`~zakcode.quality.select` — :func:`select_best`, oracle-FILTER then
judge-RANK (the measured fix: oracle for "works", judge for "good").
"""

from zakcode.quality.bestof import best_of_n
from zakcode.quality.judge import (
    BinaryVerdict,
    PairwiseVerdict,
    best_of,
    binary_judge,
    pairwise_judge,
    vote_binary,
)
from zakcode.quality.select import select_best

__all__ = [
    "BinaryVerdict",
    "PairwiseVerdict",
    "best_of",
    "best_of_n",
    "binary_judge",
    "pairwise_judge",
    "select_best",
    "vote_binary",
]

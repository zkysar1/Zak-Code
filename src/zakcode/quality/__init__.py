"""The quality engine: small-model fan-out for quality (judges, scoring, refinement, decomposition).

The bet: a fixed small-model ceiling is beaten by STRUCTURE — decompose → fan out → judge/score →
iterate — not by a bigger model. This package holds those primitives; the agent loop composes them.

Increment 1 (here): :mod:`~zakcode.quality.judge` — LLM-as-judge primitives (binary verdict,
N-judge majority vote, pairwise comparison, pairwise tournament selection).
"""

from zakcode.quality.judge import (
    BinaryVerdict,
    PairwiseVerdict,
    best_of,
    binary_judge,
    pairwise_judge,
    vote_binary,
)

__all__ = [
    "BinaryVerdict",
    "PairwiseVerdict",
    "best_of",
    "binary_judge",
    "pairwise_judge",
    "vote_binary",
]

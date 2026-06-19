"""The quality engine: small-model fan-out for quality (judges, selection, scoring, ship gate).

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
Increment 4: :mod:`~zakcode.quality.refine` — :func:`refine`, the generate→score→revise loop.
Increment 5: :mod:`~zakcode.quality.plan` (judged decomposition) + :mod:`~zakcode.quality.hooks`
(:func:`set_judge_hook`, the nudge seam).
"""

from zakcode.quality.bestof import best_of_n
from zakcode.quality.hooks import JudgeHook, apply_judge_hook, get_judge_hook, set_judge_hook
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
from zakcode.quality.plan import PLAN_RUBRIC, judge_plan, score_plan
from zakcode.quality.refine import RefineResult, refine, weak_dimensions
from zakcode.quality.score import ScoreCard, aggregate_scores, score_rubric
from zakcode.quality.select import select_best

__all__ = [
    "BinaryVerdict",
    "JudgeHook",
    "PLAN_RUBRIC",
    "PairwiseVerdict",
    "RefineResult",
    "ScoreCard",
    "ShipDecision",
    "aggregate_scores",
    "apply_judge_hook",
    "best_of",
    "best_of_n",
    "binary_judge",
    "get_judge_hook",
    "judge_plan",
    "pairwise_judge",
    "pennywise",
    "refine",
    "score_plan",
    "score_rubric",
    "select_best",
    "set_judge_hook",
    "vote_binary",
    "vote_pairwise",
    "weak_dimensions",
]

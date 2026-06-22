"""Deterministic, every-turn context gathering for the agent loop.

Register the gatherer once and it runs before every model call -- the model
never decides *whether* to gather. Relevance ranking is isolated behind the
``RelevanceClassifier`` seam, so it graduates from a heuristic (here) to a
small-model classifier to a fine-tuned model with no caller change::

    from zakcode.context import default_gatherer
    agent.hook_manager.register_context(default_gatherer())
"""

from __future__ import annotations

from collections.abc import Sequence

from .classify import SmallModelClassifier, score_relevance
from .collectors import mentioned_files_collector, recent_files_collector
from .gatherer import (
    Candidate,
    Collector,
    ContextGatherer,
    RelevanceClassifier,
    heuristic_classifier,
)
from .signal import SignalLogger, SignalRecord
from .train import RelevanceModel, TrainedClassifier, train_relevance
from .used import ModelUsedDetector, UsedDetector, judge_used, reference_used

__all__ = [
    "Candidate",
    "Collector",
    "ContextGatherer",
    "ModelUsedDetector",
    "RelevanceClassifier",
    "RelevanceModel",
    "SignalLogger",
    "SignalRecord",
    "SmallModelClassifier",
    "TrainedClassifier",
    "UsedDetector",
    "default_gatherer",
    "heuristic_classifier",
    "judge_used",
    "mentioned_files_collector",
    "recent_files_collector",
    "reference_used",
    "score_relevance",
    "train_relevance",
]


def default_gatherer(
    classifier: RelevanceClassifier = heuristic_classifier,
    *,
    collectors: Sequence[Collector] | None = None,
    budget_chars: int = 8000,
    top_k: int = 15,
) -> ContextGatherer:
    """A ready-to-register gatherer with sensible default collectors.

    Zero models, sane defaults, budget-bounded -- the enable-and-forget path::

        agent.hook_manager.register_context(default_gatherer())

    Pass ``classifier`` to swap in a smarter ranker (step 2+); pass ``collectors``
    to override the candidate sources.
    """
    cols: Sequence[Collector] = (
        list(collectors)
        if collectors is not None
        else [mentioned_files_collector, recent_files_collector]
    )
    return ContextGatherer(cols, classifier, budget_chars=budget_chars, top_k=top_k)

"""Train-your-own relevance ranker for the context gatherer (step 4).

The model end of the data loop. ``train_relevance`` fits a tiny logistic-regression ranker on the
step-3 signal log (offered ``(task, ref, score)`` -> ``used?``), and ``TrainedClassifier`` -- which
implements the ``RelevanceClassifier`` seam -- scores candidates with it. **No ML dependencies**: a
pure-Python SGD over a handful of features (bias, the heuristic prior, task<->ref name overlap), so
the harness stays lean and a heavier fine-tuner is a later swap-in behind the SAME seam.

The point is the loop: enable gathering + ``context_signal_log`` -> collect labels ->
``train_relevance`` -> ``TrainedClassifier`` ranks. Trained on YOUR traces, so it beats an
off-the-shelf prior on YOUR workloads. Fail-soft: an empty or garbage log yields the default model,
which just echoes the heuristic prior -- never raises.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from .gatherer import Candidate

logger = logging.getLogger(__name__)

_FEATURE_NAMES = ["bias", "cheap_score", "name_overlap"]


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9_]+", (text or "").lower()) if len(t) > 2}


def _name_overlap(task: str, ref: str) -> float:
    t, r = _tokens(task), _tokens(ref)
    return len(t & r) / len(t) if t else 0.0


def _features(task: str, ref: str, cheap_score: float) -> list[float]:
    return [1.0, cheap_score, _name_overlap(task, ref)]


def _sigmoid(z: float) -> float:
    if z < -700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _dot(w: Sequence[float], x: Sequence[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x, strict=True))


class RelevanceModel(BaseModel):
    """A trained logistic-regression ranker: ``score = sigmoid(weights . features)``."""

    weights: list[float] = Field(default_factory=lambda: [0.0, 1.0, 0.0])  # prior: cheap_score wins
    feature_names: list[str] = Field(default_factory=lambda: list(_FEATURE_NAMES))
    n_examples: int = 0

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RelevanceModel:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _read_examples(log_path: str | Path) -> list[tuple[list[float], float]]:
    rows: list[tuple[list[float], float]] = []
    try:
        text = Path(log_path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        task = str(rec.get("task", ""))
        for item in rec.get("offered", []) or []:
            if not isinstance(item, dict):
                continue
            x = _features(task, str(item.get("ref", "")), float(item.get("score", 0.0) or 0.0))
            rows.append((x, 1.0 if item.get("used") else 0.0))
    return rows


def train_relevance(log_path: str | Path, *, epochs: int = 300, lr: float = 0.2) -> RelevanceModel:
    """Fit a logistic-regression ranker on a step-3 signal log. Pure-Python SGD, no ML deps.

    An empty or unlabelled log returns the default (prior-echoing) model -- never raises.
    """
    rows = _read_examples(log_path)
    if not rows:
        return RelevanceModel()
    n = len(_FEATURE_NAMES)
    w = [0.0] * n
    w[1] = 1.0  # warm-start from the heuristic prior (cheap_score)
    for _ in range(epochs):
        for x, y in rows:
            err = _sigmoid(_dot(w, x)) - y
            for j in range(n):
                w[j] -= lr * err * x[j]
    return RelevanceModel(weights=w, n_examples=len(rows))


class TrainedClassifier:
    """A ``RelevanceClassifier`` backed by a trained :class:`RelevanceModel`.

    Ranks candidates by the learned ``sigmoid(weights . features)``. Sync (no model call), so it
    drops into the gatherer exactly like the heuristic -- but tuned on your own traces.
    """

    def __init__(self, model: RelevanceModel) -> None:
        # Guard against a feature-set change: a model trained against a different _FEATURE_NAMES
        # would silently mis-score (and `_dot` is now strict). Fall back to the prior loudly.
        if len(model.weights) == len(_FEATURE_NAMES):
            self._w = model.weights
        else:
            logger.warning(
                "trained model has %d weights but %d features; using the default prior",
                len(model.weights),
                len(_FEATURE_NAMES),
            )
            self._w = list(RelevanceModel().weights)

    def _score(self, task: str, c: Candidate) -> float:
        return _sigmoid(_dot(self._w, _features(task, c.ref, c.cheap_score)))

    def __call__(self, task: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        return sorted(candidates, key=lambda c: self._score(task, c), reverse=True)

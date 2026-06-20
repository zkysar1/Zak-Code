"""Tests for rubric scoring (quality engine, increment 3b).

``aggregate_scores`` is pure IAUS math — tested deterministically. ``score_rubric`` runs with
a scripted provider returning canned score JSON, covering aggregation, the missing-dimension veto,
and fail-low-on-garbage.
"""

from __future__ import annotations

import json
from typing import Any

from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.quality import aggregate_scores, score_rubric
from zakcode.usage import Usage


class _Provider(Provider):
    """Returns a FIXED score-JSON text for every acomplete call."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self._text, usage=Usage(total_tokens=2, cost_usd=0.001))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def model_id(self) -> str:
        return "scorer/test"


# ── aggregate_scores (pure IAUS math) ────────────────────────────────────────


def test_aggregate_empty_is_zero() -> None:
    assert aggregate_scores({}) == 0.0


def test_aggregate_single_dimension_is_the_score() -> None:
    assert aggregate_scores({"a": 1.0}) == 1.0
    assert aggregate_scores({"a": 0.5}) == 0.5


def test_aggregate_zero_on_any_dimension_vetoes() -> None:
    assert aggregate_scores({"a": 1.0, "b": 0.0}) == 0.0


def test_aggregate_compensates_for_dimension_count() -> None:
    # Two 0.5s: a naive product would be 0.25; the compensation factor lifts it.
    v = aggregate_scores({"a": 0.5, "b": 0.5})
    assert v > 0.25 and abs(v - 0.390625) < 1e-9


def test_aggregate_clamps_out_of_range() -> None:
    assert aggregate_scores({"a": 1.5}) == 1.0  # clamped high
    assert aggregate_scores({"a": -1.0, "b": 1.0}) == 0.0  # clamped low → veto


def test_aggregate_non_finite_vetoes_not_inflates() -> None:
    # NaN/Infinity from a model must abstain LOW (0 → veto), NOT become 1.0 (min(1.0, nan) is 1.0).
    assert aggregate_scores({"a": float("nan"), "b": 1.0}) == 0.0
    assert aggregate_scores({"a": float("inf"), "b": 1.0}) == 0.0


def test_aggregate_negative_weight_stays_in_unit() -> None:
    # A hostile negative weight must not push a dimension (or the aggregate) above 1.0.
    assert aggregate_scores({"a": 0.0}, weights={"a": -1.0}) <= 1.0


def test_aggregate_weight_zero_ignores_a_dimension() -> None:
    # b would veto (0.0), but weight 0 removes it → a alone (1.0).
    assert aggregate_scores({"a": 1.0, "b": 0.0}, weights={"b": 0.0}) == 1.0


# ── score_rubric (judged dimensions + aggregate) ─────────────────────────────


async def test_score_rubric_aggregates_judged_dimensions() -> None:
    prov = _Provider(json.dumps({"scores": {"correct": 1.0, "complete": 0.5}, "notes": "ok"}))
    card, u = await score_rubric(
        prov, artifact="x", dimensions={"correct": "is it correct", "complete": "is it complete"}
    )
    assert card.scores == {"correct": 1.0, "complete": 0.5}
    assert abs(card.overall - aggregate_scores({"correct": 1.0, "complete": 0.5})) < 1e-9
    assert card.notes == "ok" and u.total_tokens == 2


async def test_score_rubric_missing_dimension_scores_zero() -> None:
    prov = _Provider(json.dumps({"scores": {"correct": 1.0}}))  # 'complete' omitted
    card, _ = await score_rubric(prov, artifact="x", dimensions={"correct": "c", "complete": "c"})
    assert card.scores["complete"] == 0.0 and card.overall == 0.0  # missing → 0 → vetoes


async def test_score_rubric_fails_low_on_non_json() -> None:
    card, u = await score_rubric(_Provider("not json at all"), artifact="x", dimensions={"a": "a"})
    assert card.overall == 0.0 and u.total_tokens == 0  # abstain LOW, no spend reported


async def test_score_rubric_non_finite_score_vetoes() -> None:
    # json.loads accepts the NaN token; the scorer must coerce it to 0 (veto), not a perfect 1.0.
    prov = _Provider('{"scores": {"correct": NaN, "complete": 1.0}}')
    card, _ = await score_rubric(prov, artifact="x", dimensions={"correct": "c", "complete": "c"})
    assert card.overall == 0.0  # NaN must not inflate the dimension into a passing score

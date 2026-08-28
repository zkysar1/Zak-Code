"""Tests for the bounded refine loop (quality engine, increment 4).

A scripted scorer (cycling score-JSON per call) and a generate thunk that records the feedback it
receives drive the loop deterministically: ship-first-round, revise-then-ship, exhaust-rounds,
best-across-rounds (regression-proof), and the budget cutoff. Plus the weak-dimensions brief helper.
"""

from __future__ import annotations

import json
from typing import Any

from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.quality import refine, weak_dimensions
from zakcode.quality.score import ScoreCard
from zakcode.usage import Usage


class _Provider(Provider):
    """Cycles a list of score-JSON texts (one per acomplete call), clamping at the last."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls = 0

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return LLMResult(text=text, usage=Usage(total_tokens=2, cost_usd=0.001))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)

    def model_id(self) -> str:
        return "scorer/test"


def _scores(card: dict[str, float]) -> str:
    return json.dumps({"scores": card})


def _gen(artifacts: list[str]):
    """A generate thunk: records each feedback, returns artifacts[i] (clamped) per call."""
    state: dict[str, Any] = {"i": 0, "feedbacks": []}

    async def g(feedback: str | None) -> tuple[str, Usage]:
        state["feedbacks"].append(feedback)
        art = artifacts[min(state["i"], len(artifacts) - 1)]
        state["i"] += 1
        return art, Usage(total_tokens=1)

    return g, state


async def test_refine_ships_first_round_when_good() -> None:
    prov = _Provider([_scores({"a": 1.0})])  # overall 1.0 >= 0.8
    gen, state = _gen(["art0"])
    result, usage = await refine(
        prov, dimensions={"a": "a"}, generate=gen, threshold=0.8, max_rounds=3
    )
    assert result.shipped is True and result.rounds == 1 and result.artifact == "art0"
    assert state["feedbacks"] == [None]  # the first round gets no feedback
    assert usage.total_tokens == 1 + 2  # one generate + one score


async def test_refine_revises_then_ships() -> None:
    prov = _Provider([_scores({"a": 0.5}), _scores({"a": 1.0})])  # bad, then good
    gen, state = _gen(["bad", "good"])
    result, _ = await refine(prov, dimensions={"a": "a"}, generate=gen, threshold=0.8, max_rounds=3)
    assert result.rounds == 2 and result.artifact == "good" and result.shipped is True
    assert state["feedbacks"][0] is None
    assert (
        state["feedbacks"][1] is not None and "a" in state["feedbacks"][1]
    )  # weak dim in the brief


async def test_refine_exhausts_rounds_and_ships_best() -> None:
    prov = _Provider([_scores({"a": 0.5})])  # always 0.5 < 0.8
    gen, _ = _gen(["a0", "a1"])
    result, _ = await refine(prov, dimensions={"a": "a"}, generate=gen, threshold=0.8, max_rounds=2)
    assert result.rounds == 2 and result.shipped is False  # never met the threshold
    assert abs(result.scorecard.overall - 0.5) < 1e-9


async def test_refine_returns_best_even_if_last_round_regresses() -> None:
    prov = _Provider([_scores({"a": 0.5}), _scores({"a": 0.7}), _scores({"a": 0.6})])
    gen, _ = _gen(["art0", "art1", "art2"])
    result, _ = await refine(prov, dimensions={"a": "a"}, generate=gen, threshold=0.8, max_rounds=3)
    assert (
        result.rounds == 3 and result.artifact == "art1"
    )  # best (0.7), not the regressed last (0.6)
    assert abs(result.scorecard.overall - 0.7) < 1e-9 and result.shipped is False


async def test_refine_stops_when_cannot_afford() -> None:
    prov = _Provider([_scores({"a": 0.5})])  # 0.5 < 0.8
    gen, _ = _gen(["art0", "art1"])
    result, _ = await refine(
        prov,
        dimensions={"a": "a"},
        generate=gen,
        threshold=0.8,
        max_rounds=5,
        can_afford=lambda: False,
    )
    assert result.rounds == 1 and result.shipped is False  # budget gone → ship best after round 0


def test_weak_dimensions_lists_below_threshold_weakest_first() -> None:
    card = ScoreCard(scores={"good": 0.9, "weak": 0.3, "mid": 0.6}, overall=0.5)
    brief = weak_dimensions(card, threshold=0.8)
    assert "weak" in brief and "mid" in brief and "good" not in brief  # only below-threshold dims
    assert brief.index("weak") < brief.index("mid")  # weakest first


def test_weak_dimensions_all_pass_returns_generic() -> None:
    card = ScoreCard(scores={"a": 0.9, "b": 0.95}, overall=0.85)
    assert "Improve the overall" in weak_dimensions(card, threshold=0.8)

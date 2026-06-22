"""Tests for judged decomposition (quality engine, increment 5).

A scripted provider returns canned score / pairwise JSON, exercising score_plan (the decomposition
rubric) and judge_plan (pairwise best-of-N plan selection) without a network.
"""

from __future__ import annotations

import json
from typing import Any

from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.quality import PLAN_RUBRIC, judge_plan, score_plan
from zakcode.usage import Usage


class _Provider(Provider):
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self._text, usage=Usage(total_tokens=2))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def model_id(self) -> str:
        return "judge/test"


class _RankedJudge(Provider):
    """A CONTENT-AWARE pairwise judge: prefers whichever candidate ranks EARLIER in ``order``.
    Parses candidate_a / candidate_b from the prompt, so it stays CONSISTENT under best_of's
    a/b swap (the position-bias debias) -- a fixed-verdict mock would tie every pair."""

    def __init__(self, order: list[str]) -> None:
        self._rank = {name: i for i, name in enumerate(order)}
        self.calls = 0

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        self.calls += 1
        payload = messages[-1].blocks[0].text
        a = payload.split("<candidate_a>\n", 1)[1].split("\n</candidate_a>", 1)[0]
        b = payload.split("<candidate_b>\n", 1)[1].split("\n</candidate_b>", 1)[0]
        winner = "a" if self._rank[a] < self._rank[b] else "b"
        return LLMResult(text=json.dumps({"winner": winner}), usage=Usage(total_tokens=2))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def model_id(self) -> str:
        return "judge/ranked"


async def test_score_plan_scores_on_the_decomposition_rubric() -> None:
    prov = _Provider(json.dumps({"scores": {dim: 1.0 for dim in PLAN_RUBRIC}}))
    card, u = await score_plan(prov, goal="build X", plan="1. a\n2. b")
    assert set(card.scores) == set(PLAN_RUBRIC) and card.overall > 0.9
    assert u.total_tokens == 2


async def test_judge_plan_picks_best_via_pairwise() -> None:
    # A content-aware judge ranks p2 best → it wins every pair (consistently both ways) → index 2.
    prov = _RankedJudge(order=["p2", "p0", "p1"])
    best, _ = await judge_plan(prov, goal="g", candidates=["p0", "p1", "p2"])
    assert best == 2


async def test_judge_plan_single_candidate_makes_no_calls() -> None:
    prov = _Provider(json.dumps({"winner": "a"}))
    best, u = await judge_plan(prov, goal="g", candidates=["only"])
    assert best == 0 and prov.calls == 0 and u.total_tokens == 0


def test_plan_rubric_dimensions() -> None:
    assert set(PLAN_RUBRIC) == {"coverage", "granularity", "ordering", "soundness"}

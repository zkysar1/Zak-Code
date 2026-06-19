"""Tests for the Pennywise score-and-ship gate (quality engine, increment 3b).

A scripted scorer drives the three branches: good-enough → ship, below-threshold → iterate, and
below-threshold-but-broke → ship (don't stall when there's no budget to improve).
"""

from __future__ import annotations

import json
from typing import Any

from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.quality import pennywise
from zakcode.usage import Usage


class _Provider(Provider):
    def __init__(self, text: str) -> None:
        self._text = text

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        return LLMResult(text=self._text, usage=Usage(total_tokens=2, cost_usd=0.001))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def model_id(self) -> str:
        return "scorer/test"


def _scores(card: dict[str, float]) -> str:
    return json.dumps({"scores": card})


async def test_pennywise_ships_when_good_enough() -> None:
    d, u = await pennywise(
        _Provider(_scores({"a": 1.0})), artifact="x", dimensions={"a": "a"}, threshold=0.8
    )
    assert d.ship is True and d.overall == 1.0 and "ship" in d.reason
    assert u.total_tokens == 2  # the scorer's spend is returned


async def test_pennywise_iterates_when_below_threshold() -> None:
    d, _ = await pennywise(
        _Provider(_scores({"a": 0.5})), artifact="x", dimensions={"a": "a"}, threshold=0.8
    )
    assert d.ship is False and "iterate" in d.reason


async def test_pennywise_ships_when_out_of_budget() -> None:
    # Below threshold, but no budget to iterate → ship the best we have rather than stall.
    d, _ = await pennywise(
        _Provider(_scores({"a": 0.5})),
        artifact="x",
        dimensions={"a": "a"},
        threshold=0.8,
        can_afford_iteration=False,
    )
    assert d.ship is True and "no budget" in d.reason

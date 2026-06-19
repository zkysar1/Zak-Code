"""Tests for the oracle-grounded hybrid selector (quality engine, selection quality).

Hermetic: a scripted judge returns canned pairwise verdicts; the oracle is a plain async
``index -> bool``. Covers oracle-filter-to-one (no judging), filter-then-rank, no-oracle, the
none-pass fallback, oracle fail-safety, and the single-candidate short-circuit.
"""

from __future__ import annotations

import json
from typing import Any

from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.quality import select_best
from zakcode.usage import Usage


class _Judge(Provider):
    """Scripted judge: a FIXED pairwise verdict per acomplete call; counts calls."""

    def __init__(self, winner: str = "a") -> None:
        self._text = json.dumps({"winner": winner})
        self.calls = 0

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self._text, usage=Usage(total_tokens=1))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def model_id(self) -> str:
        return "judge/test"


def _oracle(passes: list[bool]):
    async def o(i: int) -> bool:
        return passes[i]

    return o


async def test_oracle_filters_to_single_passing_no_judging() -> None:
    judge = _Judge()
    # Only candidate 2 works → selected with NO judge call (one survivor). The exact 04 fix: the
    # passing candidate is kept regardless of what the judge would have guessed from the source.
    idx, usage = await select_best(
        judge, criteria="c", candidates=["A", "B", "C"], oracle=_oracle([False, False, True])
    )
    assert idx == 2
    assert judge.calls == 0
    assert usage.total_tokens == 0


async def test_oracle_filters_then_judge_ranks_survivors() -> None:
    judge = _Judge(winner="a")  # among survivors, the earlier wins
    # A and C pass; judge ranks [A, C] → "a" → A (original index 0).
    idx, _ = await select_best(
        judge, criteria="c", candidates=["A", "B", "C"], oracle=_oracle([True, False, True])
    )
    assert idx == 0
    assert judge.calls == 1  # two survivors → one pairwise


async def test_no_oracle_judge_ranks_all() -> None:
    judge = _Judge(winner="b")  # later wins → last candidate
    idx, _ = await select_best(judge, criteria="c", candidates=["A", "B", "C"])
    assert idx == 2
    assert judge.calls == 3  # round-robin of 3


async def test_none_pass_falls_back_to_judging_all() -> None:
    judge = _Judge(winner="b")
    idx, _ = await select_best(
        judge, criteria="c", candidates=["A", "B", "C"], oracle=_oracle([False, False, False])
    )
    assert idx == 2  # nothing passed → judge ranks all → C
    assert judge.calls == 3


async def test_oracle_failure_counts_as_not_passing() -> None:
    judge = _Judge()

    async def flaky(i: int) -> bool:
        if i == 2:
            return True
        raise RuntimeError("oracle blew up")  # A, B error → treated as not passing

    idx, _ = await select_best(judge, criteria="c", candidates=["A", "B", "C"], oracle=flaky)
    assert idx == 2  # C is the only candidate that passed → selected
    assert judge.calls == 0


async def test_single_candidate_no_calls() -> None:
    judge = _Judge()
    idx, usage = await select_best(
        judge, criteria="c", candidates=["only"], oracle=_oracle([False])
    )
    assert idx == 0 and judge.calls == 0 and usage.total_tokens == 0

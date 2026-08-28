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
        return Capabilities(context_window=8192)

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
        return LLMResult(text=json.dumps({"winner": winner}), usage=Usage(total_tokens=1))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)

    def model_id(self) -> str:
        return "judge/ranked"


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
    judge = _RankedJudge(order=["A", "C", "B"])  # among survivors, A is best
    # A and C pass; judge ranks [A, C] → A (index 0). The debias judges both ways → 2 calls.
    idx, _ = await select_best(
        judge, criteria="c", candidates=["A", "B", "C"], oracle=_oracle([True, False, True])
    )
    assert idx == 0
    assert judge.calls == 2  # two survivors → one pair x 2 orderings


async def test_no_oracle_judge_ranks_all() -> None:
    judge = _RankedJudge(order=["C", "A", "B"])  # C best → last candidate
    idx, _ = await select_best(judge, criteria="c", candidates=["A", "B", "C"])
    assert idx == 2
    assert judge.calls == 6  # round-robin of 3 x 2 orderings


async def test_none_pass_falls_back_to_judging_all() -> None:
    judge = _RankedJudge(order=["C", "A", "B"])
    idx, _ = await select_best(
        judge, criteria="c", candidates=["A", "B", "C"], oracle=_oracle([False, False, False])
    )
    assert idx == 2  # nothing passed → judge ranks all → C
    assert judge.calls == 6


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

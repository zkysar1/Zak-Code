"""Tests for the best-of-N orchestrator (quality engine, increment 2).

Hermetic: a scripted judge provider returns canned pairwise verdicts, and a diverse generator thunk
yields a distinct candidate per call (counter-based, no await suspension → deterministic under the
concurrent fan-out). Covers selection, n=1 (no judging), dropping failed attempts, all-fail, and
usage accumulation.
"""

from __future__ import annotations

import json
from typing import Any

from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.quality import best_of_n
from zakcode.usage import Usage


class _Judge(Provider):
    """A scripted judge: returns a FIXED pairwise verdict JSON for every acomplete call."""

    def __init__(self, winner: str = "b") -> None:
        self._text = json.dumps({"winner": winner})
        self.calls = 0

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self._text, usage=Usage(total_tokens=1, cost_usd=0.0005))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def model_id(self) -> str:
        return "judge/test"


class _RankedJudge(Provider):
    """A CONTENT-AWARE pairwise judge: prefers whichever candidate ranks EARLIER in ``order``.
    Parses candidate_a / candidate_b from the prompt, so it stays CONSISTENT when best_of swaps
    a/b (the position-bias debias) -- a fixed-verdict mock would tie every pair."""

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
        return LLMResult(
            text=json.dumps({"winner": winner}), usage=Usage(total_tokens=1, cost_usd=0.0005)
        )

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def model_id(self) -> str:
        return "judge/ranked"


def _gen(candidates: list[str], *, fail_on: int | None = None):
    """A diverse generator thunk: returns ``candidates[i]`` (+ usage) on the i-th call; the call at
    index ``fail_on`` raises (a flaky attempt)."""
    state = {"i": 0}

    async def gen() -> tuple[str, Usage]:
        i = state["i"]
        state["i"] += 1
        if i == fail_on:
            raise RuntimeError("attempt failed")
        return candidates[i], Usage(total_tokens=5, cost_usd=0.002)

    return gen


def _gen_always_fail():
    async def gen() -> tuple[str, Usage]:
        raise RuntimeError("always fails")

    return gen


async def test_best_of_n_selects_the_winner() -> None:
    # A content-aware judge ranks C best → C wins the round-robin (idx 2). The position-bias debias
    # judges each pair BOTH ways, so 3 pairs → 6 judge calls.
    judge = _RankedJudge(order=["C", "A", "B"])
    best, idx, usage = await best_of_n(judge, criteria="c", generate=_gen(["A", "B", "C"]), n=3)
    assert best == "C" and idx == 2
    assert judge.calls == 6  # 3 pairs x 2 orderings (debias)
    assert usage.total_tokens == 3 * 5 + 6 * 1  # 3 generations + 6 judge calls


async def test_best_of_n_n1_generates_once_without_judging() -> None:
    judge = _Judge()
    best, idx, usage = await best_of_n(judge, criteria="c", generate=_gen(["only"]), n=1)
    assert best == "only" and idx == 0
    assert judge.calls == 0  # a single candidate needs no selection
    assert usage.total_tokens == 5


async def test_best_of_n_drops_a_failed_attempt() -> None:
    # 3 attempts; the middle one raises → survivors [A, C] are judged. A content-aware judge ranks A
    # best → A (survivor 0). The debias judges the pair both ways → 2 judge calls.
    judge = _RankedJudge(order=["A", "C", "B"])
    best, idx, usage = await best_of_n(
        judge, criteria="c", generate=_gen(["A", "B", "C"], fail_on=1), n=3
    )
    assert best == "A" and idx == 0  # A is survivor 0
    assert judge.calls == 2  # 2 survivors → 1 pair x 2 orderings
    assert usage.total_tokens == 2 * 5 + 2 * 1  # 2 successful generations + 2 judge calls


async def test_best_of_n_all_attempts_fail_returns_empty() -> None:
    judge = _Judge()
    best, idx, usage = await best_of_n(judge, criteria="c", generate=_gen_always_fail(), n=3)
    assert best == "" and idx == 0
    assert judge.calls == 0  # nothing survived to judge
    assert usage.total_tokens == 0

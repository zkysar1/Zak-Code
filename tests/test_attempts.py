"""Tests for best_attempt — seam B's orchestration core.

Pure orchestration (run + verify injected), so it's exercised with fakes: first-verified-wins, lazy
stop, the all-fail case, and fail-safety on a raising run or verify.
"""

from __future__ import annotations

from zakcode.quality import best_attempt


async def test_returns_first_verified_and_is_lazy() -> None:
    runs: list[int] = []

    async def run(i: int) -> str:
        runs.append(i)
        return f"attempt-{i}"

    async def verify(h: str) -> bool:
        return h == "attempt-1"  # only the 2nd verifies

    winner = await best_attempt(3, run=run, verify=verify)
    assert winner == "attempt-1"
    assert runs == [0, 1]  # LAZY: stopped at the first verified, never ran attempt 2


async def test_none_verify_returns_none() -> None:
    async def run(i: int) -> int:
        return i

    async def verify(h: int) -> bool:
        return False

    assert await best_attempt(3, run=run, verify=verify) is None


async def test_run_failure_is_dropped() -> None:
    async def run(i: int) -> int:
        if i == 0:
            raise RuntimeError("flaky attempt")
        return i

    async def verify(h: int) -> bool:
        return h == 1

    assert await best_attempt(3, run=run, verify=verify) == 1  # 0 raised → dropped; 1 verified


async def test_verify_failure_counts_as_not_passing() -> None:
    async def run(i: int) -> int:
        return i

    async def verify(h: int) -> bool:
        if h == 0:
            raise RuntimeError("verify blew up")
        return h == 1

    assert await best_attempt(3, run=run, verify=verify) == 1  # 0's verify raised → not passing


async def test_zero_n_runs_nothing() -> None:
    async def run(i: int) -> int:
        return i

    async def verify(h: int) -> bool:
        return True

    assert await best_attempt(0, run=run, verify=verify) is None

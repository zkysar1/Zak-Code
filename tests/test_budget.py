"""Tests for the shared IterationBudget (M4-1).

The budget is a pure value object, so these are plain synchronous unit tests:
consume/try_consume accounting, reserve/refund for the delegation path, the child
cap, and depth-aware child derivation.
"""

from __future__ import annotations

import pytest

from zakcode.agent.budget import (
    DEFAULT_MAX_CHILDREN,
    DEFAULT_MAX_DEPTH,
    BudgetExhausted,
    ChildLimitExceeded,
    IterationBudget,
)


def test_initial_state() -> None:
    b = IterationBudget(50)
    assert b.total == 50
    assert b.consumed == 0
    assert b.remaining == 50
    assert b.children_spawned == 0
    assert b.max_children == DEFAULT_MAX_CHILDREN
    assert b.max_depth == DEFAULT_MAX_DEPTH


def test_try_consume_deducts_and_reports() -> None:
    b = IterationBudget(3)
    assert b.try_consume() is True
    assert b.remaining == 2
    assert b.try_consume(2) is True
    assert b.remaining == 0
    # Nothing left: a further consume fails and leaves the budget untouched.
    assert b.try_consume() is False
    assert b.consumed == 3


def test_try_consume_more_than_remaining_is_atomic() -> None:
    b = IterationBudget(5)
    b.try_consume(3)
    # Asking for more than remains does not partially consume.
    assert b.try_consume(5) is False
    assert b.remaining == 2


def test_consume_raises_when_exhausted() -> None:
    b = IterationBudget(1)
    b.consume()
    with pytest.raises(BudgetExhausted):
        b.consume()


def test_reserve_grants_up_to_remaining() -> None:
    b = IterationBudget(10)
    assert b.reserve(4) == 4
    assert b.remaining == 6
    # A reservation larger than the pool is clamped to what's left.
    assert b.reserve(100) == 6
    assert b.remaining == 0


def test_reserve_then_refund_returns_to_pool() -> None:
    b = IterationBudget(10)
    granted = b.reserve(8)
    assert granted == 8
    assert b.remaining == 2
    # A child used 3 of its 8 reserved → refund the unused 5.
    b.refund(granted - 3)
    assert b.remaining == 7
    assert b.consumed == 3


def test_refund_is_clamped_at_zero() -> None:
    b = IterationBudget(10)
    b.consume(2)
    b.refund(100)  # over-refund is a no-op past zero, not an error
    assert b.consumed == 0
    assert b.remaining == 10


def test_concurrent_siblings_cannot_both_overrun_via_reservation() -> None:
    # Two siblings reserve from a shared pool of 10; the second sees the first's
    # reservation already deducted, so their combined ceilings never exceed 10.
    b = IterationBudget(10)
    first = b.reserve(8)
    second = b.reserve(8)
    assert first == 8
    assert second == 2  # only 2 left after the first reservation
    assert first + second <= b.total


def test_child_cap_enforced() -> None:
    b = IterationBudget(100, max_children=2)
    assert b.can_spawn_child() is True
    b.register_child()
    b.register_child()
    assert b.can_spawn_child() is False
    with pytest.raises(ChildLimitExceeded):
        b.register_child()
    assert b.children_spawned == 2


def test_child_budget_disallows_grandchildren_at_depth_limit() -> None:
    b = IterationBudget(20, max_children=8, max_depth=1)
    b.reserve(5)  # parent used some; child should see the remaining pool
    child = b.child_budget(depth=1)
    assert child.total == b.remaining == 15
    # At depth == max_depth the child may not spawn its own children.
    assert child.max_children == 0
    assert child.can_spawn_child() is False


def test_child_budget_allows_grandchildren_below_depth_limit() -> None:
    b = IterationBudget(20, max_children=8, max_depth=2)
    child = b.child_budget(depth=1)
    # depth (1) < max_depth (2): grandchildren still allowed.
    assert child.max_children == 8
    assert child.can_spawn_child() is True


@pytest.mark.parametrize("bad", [-1])
def test_construction_rejects_negative(bad: int) -> None:
    with pytest.raises(ValueError):
        IterationBudget(bad)
    with pytest.raises(ValueError):
        IterationBudget(10, max_children=bad)
    with pytest.raises(ValueError):
        IterationBudget(10, max_depth=bad)


def test_negative_operations_rejected() -> None:
    b = IterationBudget(10)
    with pytest.raises(ValueError):
        b.try_consume(-1)
    with pytest.raises(ValueError):
        b.reserve(-1)
    with pytest.raises(ValueError):
        b.refund(-1)

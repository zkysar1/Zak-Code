"""Tests for the shared IterationBudget (M4).

The budget is a pure value object, so these are plain synchronous unit tests:
consume/try_consume accounting and the child cap. (Nesting depth is enforced
structurally by the spawner, not modeled here.)
"""

from __future__ import annotations

import pytest

from zakcode.agent.budget import (
    DEFAULT_MAX_CHILDREN,
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


def test_shared_pool_is_bounded_across_many_consumers() -> None:
    # The total is the only guarantee: many independent consumers drawing lazily
    # can never collectively exceed it.
    b = IterationBudget(10)
    drawn = 0
    while b.try_consume():
        drawn += 1
    assert drawn == 10
    assert b.remaining == 0


def test_child_cap_enforced() -> None:
    b = IterationBudget(100, max_children=2)
    assert b.can_spawn_child() is True
    b.register_child()
    b.register_child()
    assert b.can_spawn_child() is False
    with pytest.raises(ChildLimitExceeded):
        b.register_child()
    assert b.children_spawned == 2


@pytest.mark.parametrize("bad", [-1])
def test_construction_rejects_negative(bad: int) -> None:
    with pytest.raises(ValueError):
        IterationBudget(bad)
    with pytest.raises(ValueError):
        IterationBudget(10, max_children=bad)


def test_negative_consume_rejected() -> None:
    b = IterationBudget(10)
    with pytest.raises(ValueError):
        b.try_consume(-1)

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


def test_zero_total_is_unlimited() -> None:
    # total=0 = no iteration ceiling (the Settings default since max_iterations went
    # unlimited): consume never refuses, however much is drawn.
    b = IterationBudget(0)
    assert b.unlimited
    for _ in range(10_000):
        assert b.try_consume(1)
    assert b.consumed == 10_000


def test_reset_starts_a_fresh_turn_tree() -> None:
    # The budget is "for a turn and its sub-agents" — reset() is what makes that
    # true across an Agent's lifetime. Without it, one exhausted turn wedged every
    # later turn into an instant max_iterations stop at 0 iterations (2026-08-25
    # field report).
    b = IterationBudget(2, max_cost_usd=1.0, max_tokens=100)
    b.try_consume(2)
    b.add_usage(0.9, 90)
    b.register_child()
    assert not b.try_consume(1)  # drained
    b.reset()
    assert b.consumed == 0
    assert b.cost_spent == 0.0
    assert b.tokens_spent == 0
    assert b.children_spawned == 0
    assert b.try_consume(1)  # the next turn gets a full pool


def test_unpriced_call_makes_a_cost_ceiling_unenforceable() -> None:
    # A model absent from litellm's price map yields cost_usd 0.0 because nothing is KNOWN,
    # not because nothing was spent. Folded in naively that 0.0 is indistinguishable from a
    # free call, so _cost_spent sits at $0.00 forever and max_cost_usd can never trip.
    b = IterationBudget(10, max_cost_usd=5.0)
    b.add_usage(0.0, 1_000_000, cost_unpriced=True)

    assert b.unpriced_calls == 1
    assert b.cost_spent == 0.0  # still never guessed
    assert b.cost_ceiling_unenforceable() is True
    # ...and it must NOT masquerade as a crossed ceiling: nothing was spent, so reporting
    # "budget_exhausted" here would claim a measurement that never happened.
    assert b.over_budget() is False
    assert b.cost_exhausted() is False


def test_unpriced_call_is_inert_without_a_cost_ceiling() -> None:
    # No max_cost_usd = nothing to enforce, so an unpriceable call is not a problem to report.
    b = IterationBudget(10)
    b.add_usage(0.0, 1_000, cost_unpriced=True)
    assert b.unpriced_calls == 1
    assert b.cost_ceiling_unenforceable() is False


def test_priced_calls_leave_the_ceiling_enforceable() -> None:
    b = IterationBudget(10, max_cost_usd=5.0)
    b.add_usage(1.5, 100)
    b.add_usage(2.0, 100)
    assert b.unpriced_calls == 0
    assert b.cost_ceiling_unenforceable() is False
    assert b.cost_spent == pytest.approx(3.5)


def test_token_ceiling_still_bounds_an_unpriced_lane() -> None:
    # The cost ceiling is the only casualty: token counts are always reported, so max_tokens
    # remains a real bound on exactly the lane whose price is unknown.
    b = IterationBudget(10, max_cost_usd=5.0, max_tokens=1_000)
    b.add_usage(0.0, 1_500, cost_unpriced=True)
    assert b.tokens_exhausted() is True
    assert b.over_budget() is True


def test_reset_clears_the_unpriced_count() -> None:
    # reset() is per-turn; a prior turn's unpriceable call must not wedge every later turn
    # into an instant unenforceable-ceiling stop (the max_iterations wedge, one field over).
    b = IterationBudget(10, max_cost_usd=5.0)
    b.add_usage(0.0, 10, cost_unpriced=True)
    assert b.cost_ceiling_unenforceable() is True
    b.reset()
    assert b.unpriced_calls == 0
    assert b.cost_ceiling_unenforceable() is False

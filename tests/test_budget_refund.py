"""Tests for IterationBudget.refund (Tier-2 shared-budget refunds)."""

from __future__ import annotations

import pytest

from zakcode.agent.budget import IterationBudget


def test_refund_returns_unit_to_pool() -> None:
    b = IterationBudget(5)
    assert b.try_consume(3) is True
    assert b.consumed == 3 and b.remaining == 2
    assert b.refund(1) == 1
    assert b.consumed == 2 and b.remaining == 3


def test_refund_capped_at_consumed_never_negative() -> None:
    b = IterationBudget(5)
    b.try_consume(2)
    assert b.refund(10) == 2  # only what was consumed comes back
    assert b.consumed == 0 and b.remaining == 5


def test_refund_zero_is_noop() -> None:
    b = IterationBudget(5)
    b.try_consume(1)
    assert b.refund(0) == 0
    assert b.consumed == 1


def test_refund_negative_raises() -> None:
    b = IterationBudget(5)
    with pytest.raises(ValueError):
        b.refund(-1)


def test_refunded_units_are_reusable() -> None:
    b = IterationBudget(1)
    assert b.try_consume(1) is True
    assert b.try_consume(1) is False  # pool empty
    b.refund(1)
    assert b.try_consume(1) is True  # refunded unit is available again

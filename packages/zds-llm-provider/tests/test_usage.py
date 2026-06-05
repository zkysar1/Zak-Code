"""Tests for token/cost accounting value objects."""

from __future__ import annotations

from zds_llm_provider.usage import Usage, UsageTracker


def test_usage_add() -> None:
    a = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.01)
    b = Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30, cost_usd=0.02)
    c = a + b
    assert c.prompt_tokens == 30
    assert c.completion_tokens == 15
    assert c.total_tokens == 45
    assert c.cost_usd == 0.03


def test_usage_defaults() -> None:
    u = Usage()
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0
    assert u.total_tokens == 0
    assert u.cost_usd == 0.0


def test_usage_tracker_accumulates() -> None:
    tracker = UsageTracker()
    assert tracker.total == Usage()

    u1 = Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8, cost_usd=0.005)
    result1 = tracker.add(u1)
    assert result1.prompt_tokens == 5
    assert tracker.total.prompt_tokens == 5

    u2 = Usage(prompt_tokens=10, completion_tokens=7, total_tokens=17, cost_usd=0.01)
    result2 = tracker.add(u2)
    assert result2.prompt_tokens == 15
    assert result2.completion_tokens == 10
    assert tracker.total == result2

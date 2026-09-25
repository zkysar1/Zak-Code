"""Tests for token/cost accounting value objects."""

from __future__ import annotations

from zakcode.usage import Usage, UsageTracker


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


def test_reasoning_tokens_default_zero() -> None:
    assert Usage().reasoning_tokens == 0


def test_reasoning_tokens_sum() -> None:
    a = Usage(completion_tokens=40, reasoning_tokens=12)
    b = Usage(completion_tokens=30, reasoning_tokens=0)
    c = a + b
    assert c.reasoning_tokens == 12
    # Still a subset of the completion total it is drawn from.
    assert c.reasoning_tokens <= c.completion_tokens


def test_reasoning_tokens_load_forward_compatibly() -> None:
    # A message persisted before the field existed must still load, reading 0.
    legacy = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost_usd": 0.1}
    assert Usage.model_validate(legacy).reasoning_tokens == 0


def test_reasoning_tokens_survive_a_round_trip() -> None:
    # The session store and the event stream both serialize Usage whole, so the field only
    # reaches a trace row if it round-trips through the model dump.
    u = Usage(completion_tokens=40, reasoning_tokens=12)
    assert Usage.model_validate(u.model_dump()).reasoning_tokens == 12


def test_envelope_survives_addition_only_when_shared() -> None:
    a = Usage(prompt_tokens=1, envelope="env-aaaaaaaaaaaa")
    assert (a + Usage(prompt_tokens=2, envelope="env-aaaaaaaaaaaa")).envelope == "env-aaaaaaaaaaaa"
    assert (a + Usage(prompt_tokens=2, envelope="env-bbbbbbbbbbbb")).envelope == ""

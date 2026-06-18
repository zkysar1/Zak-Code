"""Tests for the Groq fallback pricing table (providers/pricing.py)."""

from __future__ import annotations

import pytest

from zakcode.providers.pricing import estimate_cost_usd


def test_known_model_priced_from_tokens() -> None:
    # gpt-oss-20b: 0.075 in / 0.30 out per 1M; 1M of each = 0.075 + 0.30.
    cost = estimate_cost_usd("groq/openai/gpt-oss-20b", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.075 + 0.30)


def test_stem_matches_through_provider_prefix() -> None:
    bare = estimate_cost_usd("gpt-oss-120b", 1_000_000, 0)
    prefixed = estimate_cost_usd("groq/openai/gpt-oss-120b", 1_000_000, 0)
    assert bare == prefixed == pytest.approx(0.15)


def test_120b_does_not_mismatch_20b_rate() -> None:
    # The 120b model ($0.15/M) must never be priced at the 20b rate ($0.075/M).
    assert estimate_cost_usd("openai/gpt-oss-120b", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost_usd("openai/gpt-oss-20b", 1_000_000, 0) == pytest.approx(0.075)


def test_cached_input_billed_at_discount() -> None:
    # All prompt tokens served from cache -> billed at 50% of the input rate.
    full = estimate_cost_usd("gpt-oss-20b", 1_000_000, 0, cache_read_tokens=0)
    cached = estimate_cost_usd("gpt-oss-20b", 1_000_000, 0, cache_read_tokens=1_000_000)
    assert cached == pytest.approx(full * 0.5)


def test_unknown_or_empty_model_returns_zero() -> None:
    assert estimate_cost_usd("openai/gpt-4o", 1000, 1000) == 0.0
    assert estimate_cost_usd("", 1000, 1000) == 0.0


def test_negative_counts_clamped_to_zero() -> None:
    assert estimate_cost_usd("gpt-oss-20b", -5, -5) == 0.0


def test_all_groq_default_models_are_priced() -> None:
    # Guard: every Groq default in the routing table must resolve to a rate here,
    # so adding a model without a price can never silently bill $0.
    from zakcode.providers.routing import DEFAULT_CATEGORY_MODELS

    for spec in DEFAULT_CATEGORY_MODELS.values():
        if spec.source == "groq":
            assert estimate_cost_usd(spec.model, 1000, 0) > 0.0, spec.model

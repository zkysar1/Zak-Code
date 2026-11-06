"""A cache write the provider bills and the reply does not name (ADR-0370).

On a write-billed tier (gpt-6-luna) the provider charges every prompt token that is not a cache
read at the cache-WRITE rate, and the usage carries no creation count. The provider's own billing
lines for whole UTC days 2026-10-31 to 2026-11-05 read: cache writes $11.5294 on 92.24M tokens at
$0.125 per million, input $0.0342 on 0.34M tokens, cached input $9.6310 on 963.10M tokens, output
$0.9550. The unread prompt was priced at the plain input rate, so ``cost_usd`` rebuilt $19.84 of
that $22.15 and the vessel's budget meter, which trusts ``cost_usd``, ran about a tenth low.

These tests pin the reading that replaced it: for a declared model, where the reply names no
creation count, the prompt that was not read from the cache is counted as written and priced at the
write rate. Hermetic: no network, no credentials (the shipped price snapshot prices the model).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.providers.litellm_provider import LiteLLMProvider

#: Declared as write-billed, and priced with a write rate in the shipped snapshot.
MODEL = "openai/gpt-6-luna"

PROMPT, READ, COMPLETION = 1_000, 100, 50
UNREAD = PROMPT - READ


def _usage(**reported: Any) -> SimpleNamespace:
    """An OpenAI-shaped usage: a cached count nested in the details, no creation count."""
    return SimpleNamespace(
        prompt_tokens=PROMPT,
        completion_tokens=COMPLETION,
        total_tokens=PROMPT + COMPLETION,
        prompt_tokens_details=SimpleNamespace(cached_tokens=READ),
        **reported,
    )


def _extract(usage: Any, *, model: str = MODEL, hidden: dict[str, Any] | None = None) -> Any:
    message = SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=usage,
        model=model,
        _hidden_params={} if hidden is None else hidden,
    )
    return LiteLLMProvider._extract_usage(response)


def _rates(model: str) -> dict[str, Any]:
    return dict(lp.litellm.get_model_info(model))


def _price(model: str, *, read: int, written: int) -> float:
    """The call priced from the map: ``read`` at the read rate, ``written`` at the write rate, the
    rest of the prompt at the plain input rate, plus the completion."""
    rates = _rates(model)
    return (
        read * rates["cache_read_input_token_cost"]
        + written * rates["cache_creation_input_token_cost"]
        + (PROMPT - read - written) * rates["input_cost_per_token"]
        + COMPLETION * rates["output_cost_per_token"]
    )


def test_the_unread_prompt_is_counted_as_written_and_priced_at_the_write_rate() -> None:
    assert _rates(MODEL).get("cache_creation_input_token_cost") is not None, "premise: write rate"
    usage = _extract(_usage())
    assert (usage.cache_read_tokens, usage.cache_creation_tokens) == (READ, UNREAD)
    assert usage.cost_usd == pytest.approx(_price(MODEL, read=READ, written=UNREAD))
    # The write rate is dearer than the input rate, so the figure sits above the plain-rate one.
    assert usage.cost_usd > _price(MODEL, read=READ, written=0)


def test_a_cost_litellm_priced_at_the_plain_rate_is_raised_to_the_written_price() -> None:
    plain = _price(MODEL, read=READ, written=0)
    usage = _extract(_usage(), hidden={"response_cost": plain})
    assert usage.cost_usd > plain
    assert usage.cost_usd == pytest.approx(_price(MODEL, read=READ, written=UNREAD))


def test_a_higher_cost_from_the_backend_is_kept() -> None:
    assert _extract(_usage(), hidden={"response_cost": 5.0}).cost_usd == 5.0


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-6-luna", "gpt-6-luna", "OpenAI/GPT-6-Luna", "gpt-6-luna-2026-10-06"],
)
def test_the_declared_model_is_matched_under_any_prefix_or_snapshot(model: str) -> None:
    assert _extract(_usage(), model=model).cache_creation_tokens == UNREAD


@pytest.mark.parametrize("model", ["gpt-6-luna-mini", "gpt-6-lunar", "gpt-6", "notgpt-6-luna", ""])
def test_a_longer_or_different_id_is_not_the_declared_model(model: str) -> None:
    assert _extract(_usage(), model=model).cache_creation_tokens == 0


def test_a_reply_that_names_a_creation_count_is_taken_as_it_says() -> None:
    usage = _extract(_usage(cache_creation_input_tokens=300))
    assert (usage.cache_read_tokens, usage.cache_creation_tokens) == (READ, 300)
    assert usage.cost_usd == pytest.approx(_price(MODEL, read=READ, written=300))


def test_a_prompt_read_whole_from_the_cache_derives_nothing() -> None:
    whole = SimpleNamespace(
        prompt_tokens=PROMPT,
        completion_tokens=COMPLETION,
        total_tokens=PROMPT + COMPLETION,
        prompt_tokens_details=SimpleNamespace(cached_tokens=PROMPT),
    )
    usage = _extract(whole)
    assert (usage.cache_read_tokens, usage.cache_creation_tokens) == (PROMPT, 0)
    assert usage.cost_usd == pytest.approx(_price(MODEL, read=PROMPT, written=0))


def test_another_model_keeps_the_plain_rate_even_where_its_map_has_a_write_rate() -> None:
    """Anthropic names its creation counts and bills the rest at the plain input rate, so a write
    rate in the map is not a reason to count the unread prompt as written."""
    other = "claude-sonnet-4-5-20250929"
    assert _rates(other).get("cache_creation_input_token_cost") is not None, "premise: write rate"
    usage = _extract(_usage(), model=other)
    assert usage.cache_creation_tokens == 0
    assert usage.cost_usd == pytest.approx(_price(other, read=READ, written=0))


def test_the_declared_set_is_exactly_the_measured_model() -> None:
    """A model joins on its own provider line-item read (ADR-0370), never by the map's shape."""
    assert frozenset({"gpt-6-luna"}) == LiteLLMProvider._UNCACHED_BILLED_AS_WRITE

"""g-373-80: a PERMANENT quota refusal must never classify as a transient 429.

Measured on a live dev vessel 2026-09-14: an ``insufficient_quota`` refusal was
classified ``RateLimited`` and rode the loop's ~15-minute backoff horizon, so the
turn billed fifteen minutes of compute to produce zero tokens, and the provider's
real message surfaced only when the budget ran out.

The distinction cannot live in the exception TYPE: litellm maps HTTP 429 on the
status code alone (eight 429 branches, none reads the body's ``code``), so the
permanent and transient conditions arrive as the same class. It therefore lives in
the message TEXT, checked once at the value rather than inside any single branch —
guard-2521: ``_map_error`` has TWO returns that emit a bare ``RateLimited`` (the
429 arm and the transient-5xx arm), and a fix scoped to the measured route would
have left the other live and silent. Every route below is tested, and every
must-classify case is paired with a healthy-subject control that must NOT.
"""

from __future__ import annotations

from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from tests.provider_error_policy import QUOTA_EXHAUSTION_MARKERS as POLICY_MARKERS
from tests.provider_error_policy import environmental_skip_reason
from zakcode.messages import Message
from zakcode.providers.base import (
    QUOTA_EXHAUSTION_MARKERS,
    ProviderError,
    QuotaExhausted,
    RateLimited,
    RequestFailed,
    quota_exhaustion_marker,
)
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.secrets import redact_secrets

#: The verbatim shape OpenAI returns when the account is out of credit, as it
#: reaches us after litellm's mapper interpolates the provider message.
QUOTA_BODY = (
    "litellm.RateLimitError: RateLimitError: OpenAIException - You exceeded your "
    "current quota, please check your plan and billing details. "
    '{"error": {"message": "You have no credits remaining.", '
    '"type": "insufficient_quota", "code": "insufficient_quota"}}'
)

#: A genuine throttle — same exception class, same status code, different text.
THROTTLE_BODY = (
    "litellm.RateLimitError: RateLimitError: OpenAIException - Rate limit reached "
    "for gpt-4o in organization org-abc on tokens per min (TPM): Limit 30000."
)


class _FakeRateError(Exception):
    def __init__(self, message: str, retry_after: Any = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ServiceUnavailableError(Exception):
    """Named exactly, because ``_map_error``'s transient arm matches by MRO NAME."""


@pytest.fixture
def provider() -> LiteLLMProvider:
    return LiteLLMProvider(model="gpt-4o", temperature=0.0)


@pytest.fixture(autouse=True)
def _wire_exception_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lp, "_LiteLLMRateLimitError", _FakeRateError)


def _raise(exc: Exception) -> Any:
    async def _acompletion(**_kw: Any) -> Any:
        raise exc

    return _acompletion


# ── Route A: the 429 arm — the route actually measured on the vessel ──────────
async def test_quota_worded_429_is_permanent(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lp.litellm, "acompletion", _raise(_FakeRateError(QUOTA_BODY, 60.0)))
    with pytest.raises(QuotaExhausted) as ei:
        await provider.acomplete([Message.user("hi")])
    # The load-bearing assertion: NOT retryable, so neither `except RateLimited`
    # retry site in the loop can catch it and spend the backoff budget on it.
    assert not isinstance(ei.value, RateLimited)
    assert not hasattr(ei.value, "retry_after")


async def test_ordinary_429_is_still_a_transient_rate_limit(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the SAME class and status, differing only in message text."""
    monkeypatch.setattr(lp.litellm, "acompletion", _raise(_FakeRateError(THROTTLE_BODY, 60.0)))
    with pytest.raises(RateLimited) as ei:
        await provider.acomplete([Message.user("hi")])
    assert not isinstance(ei.value, QuotaExhausted)
    assert ei.value.retry_after == 60.0


# ── Route B: the transient-5xx arm — the route a branch-scoped fix would miss ──
def test_quota_text_on_the_5xx_route_is_also_permanent() -> None:
    mapped = LiteLLMProvider._map_error(ServiceUnavailableError(QUOTA_BODY))
    assert isinstance(mapped, QuotaExhausted)


def test_plain_5xx_stays_retryable() -> None:
    """Control: the 5xx arm must keep surfacing RateLimited for real infra blips."""
    mapped = LiteLLMProvider._map_error(ServiceUnavailableError("502 upstream_unavailable"))
    assert isinstance(mapped, RateLimited)
    assert not isinstance(mapped, QuotaExhausted)


# ── Route C: the unmapped default arm ─────────────────────────────────────────
def test_quota_text_on_an_unmapped_exception_is_permanent() -> None:
    assert isinstance(LiteLLMProvider._map_error(ValueError("out of credits")), QuotaExhausted)


def test_unmapped_without_a_marker_stays_request_failed() -> None:
    """Control: the catch-all must not start swallowing ordinary failures."""
    assert isinstance(LiteLLMProvider._map_error(ValueError("something odd")), RequestFailed)


# ── The class contract the loop's behaviour rests on ──────────────────────────
def test_quota_exhausted_escapes_an_except_rate_limited_block() -> None:
    """loop.py has TWO `except RateLimited` retry sites; neither may catch this."""
    caught = None
    try:
        raise QuotaExhausted("insufficient_quota")
    except RateLimited:
        caught = "retry"
    except ProviderError:
        caught = "terminal"
    assert caught == "terminal"


def test_the_provider_text_reaches_the_turn_error_string() -> None:
    """loop.py's `except ProviderError` terminal records `str(exc)` as turn_error,
    so the operator sees the real cause on the FIRST refusal (outcome 3)."""
    mapped = LiteLLMProvider._map_error(_FakeRateError(QUOTA_BODY))
    assert "You exceeded your current quota" in str(mapped)


@pytest.mark.parametrize("marker", QUOTA_EXHAUSTION_MARKERS)
def test_every_marker_classifies_permanent(marker: str) -> None:
    assert isinstance(LiteLLMProvider._map_error(Exception(f"429: {marker}")), QuotaExhausted)


def test_marker_match_is_case_insensitive() -> None:
    assert quota_exhaustion_marker("ERROR: INSUFFICIENT_QUOTA") == "insufficient_quota"


@pytest.mark.parametrize("marker", QUOTA_EXHAUSTION_MARKERS)
def test_redaction_preserves_every_marker(marker: str) -> None:
    """``_map_error`` classifies the REDACTED text, so a future widening of the
    credential patterns must not eat a marker — it would silently disable the
    classification with no error anywhere."""
    body = f'{{"error": {{"code": "{marker}", "message": "no credits"}}}}'
    assert quota_exhaustion_marker(redact_secrets(body)[0]) == marker


# ── The test-side policy consumes the product's verdict ───────────────────────
def test_policy_fails_loudly_on_a_quota_exhaustion() -> None:
    assert environmental_skip_reason(QuotaExhausted(QUOTA_BODY)) is None


def test_policy_still_skips_an_ordinary_rate_limit() -> None:
    """Control: the live suites must keep skipping a genuine transient throttle."""
    assert environmental_skip_reason(RateLimited(THROTTLE_BODY)) is not None


def test_policy_markers_are_the_products_markers() -> None:
    """One home for the vendor phrasing: a second copy would fail open when stale."""
    assert POLICY_MARKERS is QUOTA_EXHAUSTION_MARKERS

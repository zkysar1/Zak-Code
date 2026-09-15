"""Both branches of the live-suite skip/fail discriminator, executed (g-373-81).

The diagnosis that produced this policy was read out of source by three agents;
none of them ever RAN either branch. These tests are that execution, and they
need no live provider: every condition is constructed directly from the taxonomy.

Each "must FAIL" case is paired with a healthy-subject control that must still
SKIP, so a policy that simply failed everything could not pass this file
(guard-3366).
"""

from __future__ import annotations

import pytest

from tests.provider_error_policy import (
    QUOTA_EXHAUSTION_MARKERS,
    environmental_skip_reason,
    quota_exhaustion_marker,
)
from zakcode.providers.base import (
    AuthError,
    ContextWindowExceeded,
    ModelOutputRejected,
    ProviderError,
    RateLimited,
    RequestFailed,
    TimedOut,
)


class _UnclassifiedProviderError(ProviderError):
    """A subclass the policy has never been taught about."""


# ── The FAIL branch: environmental_skip_reason returns None ──────────────────


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(ContextWindowExceeded("request too long"), id="product-defect-context"),
        pytest.param(ModelOutputRejected("tool_use_failed"), id="product-defect-tool-call"),
        pytest.param(RateLimited("insufficient_quota: you exceeded"), id="quota-exhaustion"),
        pytest.param(_UnclassifiedProviderError("brand new"), id="unclassified-subclass"),
        pytest.param(ValueError("not a provider error"), id="not-a-provider-error"),
    ],
)
def test_these_must_fail_the_live_suite(exc: BaseException) -> None:
    assert environmental_skip_reason(exc) is None


# ── The SKIP branch: a reason is returned ────────────────────────────────────


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(AuthError("no api key"), id="no-credential"),
        pytest.param(TimedOut("timed out", retry_after=1.0), id="client-timeout"),
        pytest.param(RateLimited("rate limit reached", retry_after=2.0), id="transient-429"),
        pytest.param(RequestFailed("connection reset"), id="transport"),
    ],
)
def test_these_must_still_skip(exc: BaseException) -> None:
    reason = environmental_skip_reason(exc)
    assert reason is not None
    assert type(exc).__name__ in reason


# ── The discriminator that the exception TYPE cannot carry ───────────────────


def test_quota_text_flips_an_otherwise_identical_rate_limit() -> None:
    """Same class, same construction -- only the message differs (g-373-80)."""
    transient = RateLimited("Rate limit reached for gpt-4o-mini", retry_after=20.0)
    exhausted = RateLimited(
        "RateLimitError: OpenAIException - You exceeded your current quota, "
        "please check your plan and billing details.",
        retry_after=20.0,
    )
    assert type(transient) is type(exhausted)
    assert environmental_skip_reason(transient) is not None, "control: throttle still skips"
    assert environmental_skip_reason(exhausted) is None, "exhaustion must fail the suite"


@pytest.mark.parametrize("marker", QUOTA_EXHAUSTION_MARKERS)
def test_every_declared_marker_is_detected_case_insensitively(marker: str) -> None:
    assert quota_exhaustion_marker(RateLimited(f"Error: {marker.upper()} here")) == marker


def test_a_timeout_is_not_mistaken_for_quota_exhaustion() -> None:
    """TimedOut subclasses RateLimited; precedence must keep it environmental."""
    assert environmental_skip_reason(TimedOut("request timed out")) is not None


def test_no_marker_in_an_ordinary_message() -> None:
    assert quota_exhaustion_marker(RateLimited("slow down")) is None

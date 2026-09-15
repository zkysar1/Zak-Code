"""Which live-provider failures are the ENVIRONMENT's fault, and which are ours.

The live suites (``test_live_provider_smoke``, ``test_live_decomposition``) opt in
behind ``LIVE_TESTS=1`` and then call a real provider. Both used to funnel every
exception through ``isinstance(exc, ProviderError) -> pytest.skip``. That predicate
is the whole base class, so it converted the product's OWN defects into a green
skip — including the one condition the suite exists to catch.

Measured 2026-09-14/15 (g-373-81, and g-373-80 for the classification leg):

* The ``LIVE_TESTS=1`` module-level ``skipif`` already disposes of "not opted in".
  A run that reaches this policy has declared it intends to talk to a real
  provider, so a provider-side refusal is not an environmental excuse by default.
* ``ProviderError``'s subtree carries two of the product's own defects --
  ``ContextWindowExceeded`` (we built an over-long request) and
  ``ModelOutputRejected`` (the model emitted a malformed tool call). Skipping
  those is skipping the bug.
* On 2026-09-14 the fleet key ran out of credit and residents could not think,
  while these suites would have reported SKIP.

So the default here is FAIL, and only the conditions named below are skipped. A
``ProviderError`` subclass added later therefore fails loudly until someone
classifies it, rather than silently joining the quiet set (guard-1718: a
fail-open default poisons every assertion of its own value; guard-373: catch the
specific class for the documented failure mode, never the base).
"""

from __future__ import annotations

from zakcode.providers.base import (
    QUOTA_EXHAUSTION_MARKERS,
    AuthError,
    ContextWindowExceeded,
    ModelOutputRejected,
    ProviderError,
    QuotaExhausted,
    RateLimited,
    RequestFailed,
    TimedOut,
    quota_exhaustion_marker,
)

#: The marker list and its discriminator now live in the PRODUCT
#: (:mod:`zakcode.providers.base`), because as of g-373-80 the product classifies a
#: permanent refusal itself — ``_map_error`` raises :class:`QuotaExhausted` instead
#: of a retryable ``RateLimited``. They are re-exported here so a test still imports
#: one name from one module, while the vendor-phrasing list has exactly ONE home:
#: two copies would drift the moment either is extended, and the copy that fell
#: behind would fail open (a real outage read as an ordinary throttle).
__all__ = [
    "QUOTA_EXHAUSTION_MARKERS",
    "environmental_skip_reason",
    "quota_exhaustion_marker",
]


def environmental_skip_reason(exc: BaseException) -> str | None:
    """Why ``exc`` is an environment condition, or None when the suite must FAIL.

    None means "not environmental" for every caller: a non-provider exception, a
    product defect, a permanent quota refusal, or an unrecognised
    ``ProviderError`` subclass.
    """
    if not isinstance(exc, ProviderError):
        # Not part of the provider taxonomy at all -- never ours to excuse.
        return None

    # Product defects first: these are never environmental, whatever the text says,
    # and they are precisely what a live suite exists to surface.
    if isinstance(exc, (ContextWindowExceeded, ModelOutputRejected)):
        return None

    # TimedOut subclasses RateLimited for its retry semantics, but it is a
    # CLIENT-side condition (an uncached local backend genuinely needing longer
    # than ZAKCODE_REQUEST_TIMEOUT). Checked before the quota probe below because
    # its own docstring insists it must not be reported as a rate limit.
    if isinstance(exc, TimedOut):
        return f"client-side timeout ({type(exc).__name__})"

    # A permanent quota/credit refusal is a REAL failure wearing a 429's clothes.
    # Two checks, and both earn their place. The TYPE check is the product's own
    # verdict (g-373-80): ``_map_error`` now raises QuotaExhausted for any message
    # carrying a marker, whatever status code it arrived under. The TEXT check
    # behind it is the net for a refusal that reaches a test WITHOUT that verdict —
    # a provider litellm never mapped to 429 at all (landing in RequestFailed), or
    # an exception constructed directly by a fixture. Dropping either one would make
    # the quieter route skip, which is the exact failure this module exists to stop.
    if isinstance(exc, QuotaExhausted):
        return None
    if quota_exhaustion_marker(exc) is not None:
        return None

    if isinstance(exc, AuthError):
        return f"no usable credential ({type(exc).__name__})"
    if isinstance(exc, RateLimited):
        return f"transient rate limit ({type(exc).__name__})"
    if isinstance(exc, RequestFailed):
        return f"provider transport failure ({type(exc).__name__})"

    # An unclassified ProviderError subclass. FAIL rather than skip -- see the
    # module docstring.
    return None

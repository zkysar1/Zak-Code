"""Where a call actually goes, and whether that costs money.

Two questions live here, because they are the same question asked twice:

1. **Does a configured generic ``api_base`` apply to this model?**
   :func:`model_uses_generic_endpoint` — the allowlist that keeps a self-hosted
   base off a named cloud call (the 2026-06-17 ``groq/... -> localhost`` bug).
2. **Will a call on this model spend money at a metered API?**
   :func:`is_local_model` / :func:`classify_destination` — the predicate behind
   ``Settings.local_only``.

They share one fact — *which endpoint receives the request* — so they share one
implementation. Keeping them apart is how the two drift into disagreeing about
the same model, and a cost guarantee that disagrees with the router is not a
guarantee.

This module imports **no** vendor SDK (no litellm), so ``config`` can import it
without pulling the vendor adapter into every settings load, and the clean-room
contract stays green. ``litellm_provider`` re-exports the predicate rather than
keeping its own copy.
"""

from __future__ import annotations

from collections.abc import Sequence

#: litellm providers that speak the generic OpenAI wire protocol and therefore accept an
#: injected ``api_base``. A configured generic base (a self-hosted llama-server / vLLM /
#: gateway) must NEVER be forwarded to a provider that carries its OWN base URL — doing so
#: reroutes a cloud call to the local endpoint, which is the 2026-06-17 failover bug where
#: ``groq/openai/gpt-oss-20b`` got the llama-server base, failed, and fell back to
#: ``openai/gpt-4o-mini``.
#:
#: This is an ALLOWLIST, deliberately, so a newly-added cloud prefix can never silently
#: inherit a local base by omission. SINGLE SOURCE OF TRUTH — ``litellm_provider`` imports
#: this name rather than defining its own.
GENERIC_OPENAI_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "openai_like", "hosted_vllm", "text-completion-openai"}
)

#: litellm prefixes served by a local Ollama daemon (its own base, ``OLLAMA_API_BASE``).
OLLAMA_PROVIDERS: frozenset[str] = frozenset({"ollama", "ollama_chat"})

#: Sentinels that occupy ``default_model`` but name no backend: ``"auto"`` defers to the
#: availability resolver and ``"zakpick"`` defers to per-category routing. Splitting a
#: sentinel on "/" yields the sentinel itself, which matches no real provider prefix — the
#: cause of the bug fixed in ``ZakCode._build_provider`` (under ``zakpick`` the "same
#: backend?" test compared every routed model against the literal string ``"zakpick"``, so
#: it was false for all of them and the configured ``api_base`` was dropped on every call).
MODEL_SENTINELS: frozenset[str] = frozenset({"auto", "zakpick"})


class LocalOnlyViolation(RuntimeError):
    """A call was about to reach a metered API while ``local_only`` is set.

    Deliberately **not** a :class:`~zakcode.providers.base.ProviderError`. Provider errors
    are recoverable and ride the loop's failover machinery, which responds by trying a
    DIFFERENT model — the one reaction that could turn a single refusal into the very cloud
    spend the switch exists to prevent. This is a configuration fault: it is fatal, it names
    the offending model, and nothing retries it.
    """


def provider_prefix(model: str) -> str:
    """The litellm provider prefix of ``model``, lowercased; ``""`` for a bare model name."""
    if not model or "/" not in model:
        return ""
    return model.split("/", 1)[0].strip().lower()


def is_sentinel(model: str | None) -> bool:
    """Whether ``model`` is a routing sentinel (``auto`` / ``zakpick``) rather than a model."""
    return (model or "").strip().lower() in MODEL_SENTINELS


def model_uses_generic_endpoint(model: str) -> bool:
    """Whether a configured generic ``api_base`` should be forwarded for ``model``.

    True for the OpenAI-compatible generic path — a bare model name, or an ``openai`` /
    ``openai_like`` / ``hosted_vllm`` / ``text-completion-openai`` prefix (the self-hosted
    llama-server case the base is configured for). False for any NAMED provider (``groq/``,
    ``anthropic/``, ``ollama_chat/``, …) that litellm routes via its own base URL.
    """
    prefix = provider_prefix(model)
    if not prefix:
        return True
    return prefix in GENERIC_OPENAI_PROVIDERS


def is_ollama_model(model: str) -> bool:
    """Whether ``model`` is served by a local Ollama daemon."""
    return provider_prefix(model) in OLLAMA_PROVIDERS


def _normalize_base(base: str) -> str:
    """Compare api_base values without tripping on a trailing slash or case."""
    return base.strip().rstrip("/").lower()


def api_base_is_trusted(api_base: str | None, local_api_bases: Sequence[str] | None) -> bool:
    """Whether ``api_base`` is one the operator has declared genuinely local.

    An EMPTY (or unset) allowlist trusts any base — that is the historical behavior and
    it stays the default, so no working configuration starts refusing (guard-1562).

    Setting the allowlist closes a real hole. ``local_only`` classifies by MODEL PREFIX,
    so any ``openai/*`` model with an ``api_base`` counted as local — including a base
    pointing at an LLM **gateway** that fans out to metered providers. A gateway speaks
    the generic OpenAI protocol, so it is indistinguishable from a self-hosted pod at
    this layer; only the operator knows which endpoints are theirs. Measured 2026-08-21:
    a litellm gateway fronting deepinfra/groq/openai passed the guard untouched.
    """
    if not local_api_bases:
        return True
    if not api_base:
        return False
    target = _normalize_base(api_base)
    return any(_normalize_base(b) == target for b in local_api_bases if b and b.strip())


def classify_destination(
    model: str,
    api_base: str | None,
    local_api_bases: Sequence[str] | None = None,
) -> tuple[bool, str]:
    """``(is_local, reason)`` — where a call on ``model`` lands, and why.

    The reason string is the whole point: a refusal that says only "not local" leaves the
    operator guessing which of model / prefix / ``api_base`` was wrong. It is written to be
    pasted into an error message verbatim.

    "Local" means **no metered third-party API is billed**, not "on this machine":

    * Ollama (``ollama``/``ollama_chat``) — a local daemon, always free.
    * A generic-OpenAI model WITH an ``api_base`` — the base redirects the call to a
      self-hosted server, so no cloud is reached. This is the self-hosted-pod case.
    * Everything else is metered: a named cloud prefix (``groq/``, ``anthropic/``) ignores
      ``api_base`` entirely by the allowlist above, and a generic model WITHOUT a base goes
      to the vendor's own default host (``openai/gpt-4o`` -> api.openai.com).

    Sentinels are NOT classified here — they name no destination. Callers resolve them to a
    concrete model first; :func:`is_local_model` refuses them loudly rather than guessing.
    """
    if is_sentinel(model):
        raise ValueError(
            f"cannot classify the routing sentinel {model!r} as local or metered — "
            "resolve it to a concrete model first"
        )
    if is_ollama_model(model):
        return True, f"{model} is served by the local Ollama daemon"
    if model_uses_generic_endpoint(model):
        if api_base:
            if not api_base_is_trusted(api_base, local_api_bases):
                return False, (
                    f"{model} is routed to {api_base}, which is NOT listed in "
                    f"ZAKCODE_LOCAL_API_BASES. A generic-OpenAI base may be a self-hosted "
                    f"pod OR a gateway that forwards to metered providers, and the two are "
                    f"indistinguishable here — add the base to the allowlist if it is yours"
                )
            return True, f"{model} is routed to the self-hosted endpoint at {api_base}"
        return False, (
            f"{model} uses the generic OpenAI path but no api_base is configured, so it "
            f"would be billed at the vendor's own host — set ZAKCODE_API_BASE to your "
            f"self-hosted endpoint"
        )
    prefix = provider_prefix(model)
    return False, (
        f"{model} targets the metered '{prefix}' API (a named cloud provider carries its "
        f"own base URL, so ZAKCODE_API_BASE does not redirect it)"
    )


def is_local_model(
    model: str,
    api_base: str | None,
    local_api_bases: Sequence[str] | None = None,
) -> bool:
    """Whether a call on ``model`` avoids every metered API. See :func:`classify_destination`."""
    return classify_destination(model, api_base, local_api_bases)[0]


__all__ = [
    "GENERIC_OPENAI_PROVIDERS",
    "OLLAMA_PROVIDERS",
    "MODEL_SENTINELS",
    "LocalOnlyViolation",
    "api_base_is_trusted",
    "provider_prefix",
    "is_sentinel",
    "model_uses_generic_endpoint",
    "is_ollama_model",
    "classify_destination",
    "is_local_model",
]

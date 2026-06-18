"""Fallback token pricing for provider responses litellm cannot cost.

litellm computes ``response_cost`` from its built-in model price map. Groq's
open-source lineup (``gpt-oss-20b``/``gpt-oss-120b``, ``qwen3-32b``,
``llama-3.1-8b-instant``), served through an OpenAI-compatible endpoint, is NOT
in that map, so litellm returns a 0 cost for those models — ``/cost`` and the
budget ceiling would under-report to $0. ``litellm_provider._extract_usage``
calls :func:`estimate_cost_usd` as a fallback ONLY when litellm reports no cost,
so a real ``response_cost`` (Anthropic, OpenAI proper, …) always wins.

This module is the single source of truth for the Groq rates (``routing.py``'s
lineup comment points here). It imports nothing from the package, so it can be
consumed by any layer without an import cycle.
"""

from __future__ import annotations

#: (input, output) USD per 1,000,000 tokens. Groq published lineup, 2026-06
#: (principal-verified). Keyed by a model-id STEM matched as a substring of the
#: response model string, so it survives provider-prefix variation
#: (``groq/openai/gpt-oss-120b``, ``openai/gpt-oss-120b``, ``gpt-oss-120b`` all
#: match ``gpt-oss-120b``). Order matters: the longer ``gpt-oss-120b`` stem is
#: listed before ``gpt-oss-20b`` so a 120b model can never mis-match the 20b rate.
GROQ_RATES_PER_M: dict[str, tuple[float, float]] = {
    "llama-3.1-8b-instant": (0.05, 0.08),
    "gpt-oss-120b": (0.15, 0.60),
    "gpt-oss-20b": (0.075, 0.30),
    "qwen3-32b": (0.29, 0.59),
}

#: Groq prompt-cache discount: cached input tokens bill at ~50% of the input rate.
#: Applied to the ``cache_read_tokens`` subset of ``prompt_tokens`` (relevant once
#: Groq prompt caching is enabled — Lever C). With no caching, ``cache_read`` is 0
#: and this is inert.
GROQ_CACHED_INPUT_FACTOR: float = 0.5


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Best-effort USD cost for a Groq open-model call litellm did not price.

    Returns ``0.0`` for any model not in :data:`GROQ_RATES_PER_M` (it never
    guesses). ``cache_read_tokens`` (a subset of ``prompt_tokens``) is billed at
    the cached-input discount; the remaining prompt tokens at the full input rate.
    """
    if not model:
        return 0.0
    rate: tuple[float, float] | None = None
    for stem, r in GROQ_RATES_PER_M.items():
        if stem in model:
            rate = r
            break
    if rate is None:
        return 0.0
    in_rate, out_rate = rate
    prompt = max(0, int(prompt_tokens))
    completion = max(0, int(completion_tokens))
    cached = max(0, min(int(cache_read_tokens), prompt))
    full_in = prompt - cached
    cost = (
        full_in * in_rate + cached * in_rate * GROQ_CACHED_INPUT_FACTOR + completion * out_rate
    ) / 1_000_000.0
    return cost


__all__ = ["GROQ_RATES_PER_M", "GROQ_CACHED_INPUT_FACTOR", "estimate_cost_usd"]

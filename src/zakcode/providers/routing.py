"""zakpick — task-category model routing (the general form of per-role routing).

When ``default_model`` is the sentinel ``"zakpick"`` the engine stops using one model for
everything and instead **picks a model per task category**. The categories ARE the interface:
the user parks a concrete model on each one — a small/cheap model on the easy/bounded
categories (classification, summaries, quick coding), a capable model on the hard ones
(deep coding, planning). Each entry is a ``(model, source)`` pair, so the same model name can
live at different providers (``qwen3-32b`` at Groq vs locally is the user's explicit choice).

Crucially, zakpick does **not** own any local-vs-cloud tradeoff. It never curates tiers, masks
sources, or degrades between providers at runtime. It routes each prompt to exactly the model
the user assigned to that category — and the user owns the consequences (a slow local model on
a weak GPU is slow; a rate-limited cloud model fails like any other provider error, handled by
the existing ``fallback_model`` seam). The one automatic decision is the cheap, deterministic
quick-vs-deep coder split (:func:`classify_main_turn`), which only ever chooses between the two
*coder models the user already configured*.

Out of the box (no config) every category has a sensible default drawn from Groq's published
lineup (Groq serves only open-source models, so the defaults also tell a user which open-source
model to download to run that category locally). Defaults all use ``source="groq"``; override
any category — model and/or source — via ``Settings.zakpick_models``.

This module imports **no** vendor SDK / litellm, so the clean-room contract test stays green.
Model strings here are data (like the candidate lists in ``resolve._EXTERNAL_SOURCES``), not a
hardcoded vendor *sort*.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

#: The task categories zakpick routes on. These ARE the strings passed as the routing ``task``.
#: ``classify`` is reserved for a future cheap difficulty-classifier side-call and has no live
#: call site in v1 (so it is never advertised to the user) — see docs/DECISIONS.md ADR-0009.
ZakpickCategory = Literal["quick_code", "deep_code", "summarize", "plan", "delegate", "classify"]

#: Recognized override keys for ``Settings.zakpick_models`` (the validator rejects others).
ZAKPICK_CATEGORIES: frozenset[str] = frozenset(
    {"quick_code", "deep_code", "summarize", "plan", "delegate", "classify"}
)


class ZakpickModel(BaseModel):
    """A ``(model, source)`` assignment for one category.

    ``model`` is the provider's own model id (e.g. ``"openai/gpt-oss-120b"`` as Groq names it,
    or ``"qwen3:32b"`` as Ollama tags it). ``source`` is the provider/runtime: ``"groq"`` (the
    default), ``"local"`` (Ollama), or any litellm provider prefix (``"openai"`` / ``"anthropic"``
    / …). The two are deliberately separate because a model name alone is ambiguous — the same
    weights run at Groq and locally — so the user states both. ``litellm_string`` joins them
    into the ``provider/model`` form the provider layer consumes.
    """

    model: str
    source: str = "groq"

    @property
    def litellm_string(self) -> str:
        """The ``provider/model`` litellm id. ``local``/``ollama`` map to the ``ollama_chat``
        backend; every other source is used as the litellm provider prefix verbatim."""
        src = self.source.strip().lower()
        if src in ("local", "ollama", "ollama_chat"):
            return f"ollama_chat/{self.model}"
        return f"{self.source}/{self.model}"


def _g(model: str) -> ZakpickModel:
    """A Groq-hosted default (source defaults to groq)."""
    return ZakpickModel(model=model)


#: Out-of-the-box defaults, drawn from Groq's published lineup and graduated by cost/capability
#: (prices $/1M in·out as of 2026-06): llama-3.1-8b-instant 0.05·0.08, gpt-oss-20b 0.075·0.30,
#: qwen3-32b 0.29·0.59, gpt-oss-120b 0.15·0.60. ``llama-3.3-70b-versatile`` is deliberately
#: avoided — the registry flags it tools_unreliable. Each doubles as a "download this to run the
#: category locally" hint (Groq serves only open-source models).
DEFAULT_CATEGORY_MODELS: dict[str, ZakpickModel] = {
    "classify": _g("llama-3.1-8b-instant"),  # cheapest/fastest — JSON gates
    "summarize": _g("openai/gpt-oss-20b"),  # cheap, fast, decent prose; no tools
    "quick_code": _g("openai/gpt-oss-20b"),  # cheap + tools-reliable family — easy turns
    "plan": _g("qwen/qwen3-32b"),  # strong reasoning for decomposition (read-only)
    "deep_code": _g("openai/gpt-oss-120b"),  # tools-RELIABLE, strongest — hard turns
    "delegate": _g("openai/gpt-oss-120b"),  # tools-reliable, capable — general execution
}

#: Categories with a LIVE routable call site in v1, so the info panel only advertises routes the
#: engine actually takes (a receipt must never claim a route it doesn't make). ``classify`` is a
#: documented seam with no in-process caller yet, so it is omitted here.
ROUTED_CATEGORIES: tuple[str, ...] = ("deep_code", "quick_code", "plan", "summarize", "delegate")

#: Plain-English labels for the internal category names (no jargon leaks to the user).
CATEGORY_LABEL: dict[str, str] = {
    "deep_code": "hard coding",
    "quick_code": "easy coding",
    "summarize": "summaries",
    "plan": "planning",
    "delegate": "delegated work",
    "classify": "classification",
}


def model_spec_for_category(category: str, settings: object) -> ZakpickModel:
    """The effective ``(model, source)`` for ``category``: a user override else the built-in
    default. An unknown category falls back to the ``deep_code`` default (the safe, capable one).
    """
    overrides = getattr(settings, "zakpick_models", None) or {}
    spec = overrides.get(category)
    if spec is not None:
        return spec
    return DEFAULT_CATEGORY_MODELS.get(category, DEFAULT_CATEGORY_MODELS["deep_code"])


def model_for_category(category: str, settings: object) -> str:
    """The litellm model string the provider layer should build for ``category``."""
    return model_spec_for_category(category, settings).litellm_string


def classify_main_turn(
    *, last_user_len: int, context_frac: float, signal_latched: bool
) -> Literal["quick_code", "deep_code"]:
    """Classify the main turn's difficulty — deterministic, offline, and biased to **fail UP**.

    Picks which of the user's two configured coder models drives this turn:

    * ``signal_latched`` (a stuck / doom-loop / verify-gate signal fired this turn, latched
      one-way) ALWAYS returns ``deep_code`` — the only promotion trigger, so a turn that starts
      cheap moves to the user's capable coder the moment it struggles. Iteration count is NOT an
      input, so steady-state tool loops stay on the cheap model until a real struggle signal.
    * Otherwise a short request over a small context (``< 600`` chars and ``< 25%`` of the
      window) is ``quick_code``; anything longer/larger is ``deep_code``.
    * Any ambiguity resolves to ``deep_code``. Never raises.
    """
    if signal_latched:
        return "deep_code"
    if last_user_len < 600 and context_frac < 0.25:
        return "quick_code"
    return "deep_code"


def describe_zakpick(settings: object) -> str:
    """One-line, friendly summary of the live per-category routing for the info panel / ``/model``.

    Shows only categories with a real call site (:data:`ROUTED_CATEGORIES`) and uses
    ``model (source)`` form, never a raw litellm slug. Never raises.
    """
    parts: list[str] = []
    for category in ROUTED_CATEGORIES:
        try:
            spec = model_spec_for_category(category, settings)
            parts.append(f"{CATEGORY_LABEL[category]} → {spec.model} ({spec.source})")
        except Exception:  # noqa: BLE001 — the panel is diagnostic; never crash it
            parts.append(f"{CATEGORY_LABEL[category]} → (unset)")
    return "zakpick — " + "; ".join(parts)


__all__ = [
    "ZakpickCategory",
    "ZAKPICK_CATEGORIES",
    "ZakpickModel",
    "DEFAULT_CATEGORY_MODELS",
    "ROUTED_CATEGORIES",
    "CATEGORY_LABEL",
    "model_spec_for_category",
    "model_for_category",
    "classify_main_turn",
    "describe_zakpick",
]

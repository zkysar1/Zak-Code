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
#: ``classify`` powers the cheap difficulty side-call that routes the main turn quick-vs-deep
#: (:func:`should_consult_classifier` → :func:`classify_main_turn`'s ``difficulty_hint``). It is
#: an INTERNAL meta-route, not user-facing main work, so it stays out of
#: :data:`ROUTED_CATEGORIES` (the info panel only advertises where real work runs). See ADR-0009.
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


def _o(model: str) -> ZakpickModel:
    """A first-party OpenAI default (litellm reads ``OPENAI_API_KEY`` directly)."""
    return ZakpickModel(model=model, source="openai")


#: Out-of-the-box defaults, graduated by cost/capability. The cheap, read-only, and
#: easy-turn categories stay on Groq's fast open models; the TOOL-HEAVY capable tier
#: (deep_code, delegate) runs on a first-party model whose NATIVE function-calling is
#: reliable. The live Groq $/1M in·out rates are the single source of truth in
#: ``providers/pricing.py`` (``GROQ_RATES_PER_M``); openai-source models are priced by
#: litellm directly.
#:
#: Why deep_code/delegate moved OFF Groq's open models (live-verified 2026-06-18):
#:   * ``openai/gpt-oss-120b`` emits malformed NATIVE tool calls that Groq's strict parser
#:     rejects with ``tool_use_failed`` — near-deterministically on a multi-tool schema, so
#:     every hard turn died with provider_error (it is a REASONING model, so the text-tool
#:     fallback returns empty content — broken in both modes).
#:   * ``groq/llama-3.3-70b-versatile`` is also tools_unreliable natively (pseudo-XML
#:     rejected); the text protocol works for FOCUSED tasks but stalls on complex multi-file
#:     prompts, and Groq does not cache it (costly across agentic iterations).
#:   * ``openai/gpt-4o-mini`` completed the full deep-task benchmark (incl. a multi-file
#:     task) with plain native tool calling and ~85% prompt-cache hit — most reliable AND
#:     cheapest per completed task. Requires ``OPENAI_API_KEY``; a Groq-only fork should
#:     override deep_code/delegate to ``llama-3.3-70b-versatile`` with
#:     ``ZAKCODE_TOOL_CALLING_MODE=text`` (focused tasks) instead.
#:
#: RE-TESTED 2026-07-29 UNDER LEAN RULES (g-016-84), because the caching argument above was
#: measured against an ~8.2k-token rule block re-sent every turn. With ``lean_rules`` on that
#: block is ~0.9k, so the uncached per-turn penalty that made Groq expensive should have
#: largely vanished. It did not change the answer:
#:   * 3 bench tasks, both arms with lean rules active (ZBENCH_RULES_ROOT + ZAKCODE_LEAN_RULES).
#:   * ``openai/gpt-4o-mini``: 1/3 verified, $0.0328 total — real attempts on all three
#:     (the two failures were a tie-break and an even-length-median bug, not missing work).
#:   * ``groq/qwen/qwen3.6-27b`` (native): 0/3 verified, $0.0330 total. On 2 of 3 it wrote NO
#:     FILE AT ALL and stopped after ONE iteration ("wordfreq.py not found", "lru.py missing")
#:     — the same tool-unreliability signature as the other Groq open models above.
#: CONCLUSION: the binding constraint was never per-turn token cost, so shrinking the prompt
#: cannot fix it. Cost came out within 0.6% of each other; reliability did not move. Keep
#: gpt-4o-mini. CAVEAT — this tested qwen3.6-27b NATIVE only; the ``llama-3.3-70b-versatile``
#: + ``ZAKCODE_TOOL_CALLING_MODE=text`` fork path above remains UNTESTED under lean rules, and
#: n=3 is small (gpt-4o-mini's own 1/3 did not reproduce the "completed the full benchmark"
#: claim above, so treat both numbers as directional).
DEFAULT_CATEGORY_MODELS: dict[str, ZakpickModel] = {
    "classify": _g("llama-3.1-8b-instant"),  # cheapest/fastest — JSON gates
    "summarize": _g("openai/gpt-oss-20b"),  # cheap, fast prose; NO tools, so the flag is moot
    # Repointed 2026-07-29 (g-016-83) off qwen3-32b, which Groq decommissioned
    # ~2026-07-19 and which is confirmed ABSENT from the live /v1/models catalog.
    # qwen3.6-27b is its catalog successor and the same tool-capable tier
    # (supports_tools, no tools_unreliable), so the auto-resolver keeps it for
    # tool sessions. test_routed_models_are_not_decommissioned pins this.
    "quick_code": _g("qwen/qwen3.6-27b"),  # tools-RELIABLE Groq model (gpt-oss-20b's tools flake)
    "plan": _g("qwen/qwen3.6-27b"),  # strong reasoning for decomposition (read-only)
    "deep_code": _o("gpt-4o-mini"),  # tools-RELIABLE native + cached — hard turns
    "delegate": _o("gpt-4o-mini"),  # tools-reliable native — general execution
}

#: Categories with a LIVE routable call site in v1, so the info panel only advertises routes the
#: engine actually takes (a receipt must never claim a route it doesn't make). ``classify`` IS
#: now called (the difficulty side-call), but it is an internal meta-route that picks BETWEEN
#: quick_code/deep_code rather than running user-facing work, so it stays omitted here.
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


# ── quick-vs-deep classifier thresholds (EXPECT TO TUNE THESE) ───────────────────────
# These two numbers are the entire heuristic that decides whether a main turn runs on the
# user's cheap (quick_code) or capable (deep_code) coder. They are deliberately conservative
# *first* values, chosen to bias toward deep_code (fail UP) so a misjudged turn errs on the side
# of quality, not a broken-but-cheap result. They WILL need tuning as we gather real usage —
# treat them as the one knob to turn, and prefer adjusting these constants over adding new signals.
#
# How to know they need tuning (the signals to watch, once usage data exists):
#   * Raise (loosen) them if lots of genuinely-easy turns are landing on deep_code — i.e. cheap
#     turns we're overpaying for. Symptom: deep_code share is high but those turns rarely trip a
#     struggle signal (so they didn't actually need the strong model).
#   * Lower (tighten) them if quick_code turns frequently escalate via the soft latch (stuck /
#     doom-loop / verify-gate) — i.e. we routed too cheap and paid for it in churn before the
#     latch flipped to deep_code. Symptom: high latch rate / verify-gate failures on quick turns.
# Whatever you change, keep the fail-UP bias: when uncertain, deep_code is the safe default.
#
#: A request shorter than this many characters is a candidate for the cheap coder. ~600 chars is
#: roughly a few sentences — a typo fix, a one-liner, a small tweak — vs. a multi-paragraph spec.
_QUICK_MAX_USER_CHARS = 600
#: ...and only when the conversation is still using less than this fraction of the model's context
#: window. Past ~25% there is usually enough accumulated state (files read, prior edits) that the
#: turn is no longer "quick" regardless of how short the latest message is.
_QUICK_MAX_CONTEXT_FRAC = 0.25


def classify_main_turn(
    *,
    last_user_len: int,
    context_frac: float,
    signal_latched: bool,
    difficulty_hint: Literal["quick_code", "deep_code"] | None = None,
) -> Literal["quick_code", "deep_code"]:
    """Classify the main turn's difficulty — deterministic, offline, and biased to **fail UP**.

    Picks which of the user's two configured coder models drives this turn:

    * ``signal_latched`` (a stuck / doom-loop / verify-gate signal fired this turn, latched
      one-way) ALWAYS returns ``deep_code`` — the only promotion trigger, so a turn that starts
      cheap moves to the user's capable coder the moment it struggles. Iteration count is NOT an
      input, so steady-state tool loops stay on the cheap model until a real struggle signal.
    * ``difficulty_hint`` — a SCOPE judgment from the cheap ``classify``-model side-call
      (:func:`should_consult_classifier`), supplied when available. It OVERRIDES the length
      heuristic below, because message length is a poor proxy for difficulty (a terse "build a
      PDF reader and maker" is a deep task no character count reveals). A ``"deep_code"`` hint
      routes straight to the capable coder; a ``"quick_code"`` hint still yields to the
      context-size guard (lots of accumulated state is no longer a quick turn). Once a hint
      exists, length is NEVER used to pick ``quick_code``.
    * Otherwise (no hint — the classifier was unavailable, or a bare/legacy loop) a short
      request over a small context (under :data:`_QUICK_MAX_USER_CHARS` chars AND under
      :data:`_QUICK_MAX_CONTEXT_FRAC` of the window) is ``quick_code``; anything longer/larger
      is ``deep_code``. Those two thresholds are the legacy tuning knob.
    * Any ambiguity resolves to ``deep_code``. Never raises.
    """
    if signal_latched:
        return "deep_code"
    if difficulty_hint is not None:
        if difficulty_hint == "deep_code":
            return "deep_code"
        # A "quick" scope verdict still defers to the context-size guard.
        return "quick_code" if context_frac < _QUICK_MAX_CONTEXT_FRAC else "deep_code"
    if last_user_len < _QUICK_MAX_USER_CHARS and context_frac < _QUICK_MAX_CONTEXT_FRAC:
        return "quick_code"
    return "deep_code"


# ── difficulty side-call (the activated ``classify`` category) ───────────────────────
# The quick-vs-deep split above used to read message LENGTH, which mis-routes a terse but huge
# request ("build a pdf reader and maker") onto the cheap coder. Instead, for exactly the
# ambiguous case (short request, small context — where the length rule would say quick), the
# engine spends ONE cheap ``classify``-model call to judge the request's SCOPE and feeds the
# verdict back as :func:`classify_main_turn`'s ``difficulty_hint``. The call is made by the
# Agent (it needs a provider); these helpers are the pure policy it uses, kept here next to the
# thresholds. Failure of that call FAILS UP to ``deep_code`` — never a length-based guess.

#: JSON schema for the difficulty side-call: a single enum, so even a small/fast model can
#: emit it reliably (it is a plain JSON gate, no tools — the path Groq's open models handle).
DIFFICULTY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"difficulty": {"type": "string", "enum": ["quick", "deep"]}},
    "required": ["difficulty"],
    "additionalProperties": False,
}


def should_consult_classifier(user_text: str, context_frac: float) -> bool:
    """Whether the difficulty side-call is worth a model call for this turn.

    Only the AMBIGUOUS case — a short request over a small context, i.e. exactly where the
    length heuristic would (often wrongly) pick ``quick_code`` — is worth the call. A long
    request or an already-large context is decided ``deep_code`` with NO call (the safe
    direction; length only ever fast-paths UP to deep, never down to quick).
    """
    return len(user_text) < _QUICK_MAX_USER_CHARS and context_frac < _QUICK_MAX_CONTEXT_FRAC


def difficulty_system_prompt() -> str:
    """System prompt for the one-shot difficulty classifier (cheap model, JSON out)."""
    return (
        "You are a fast router for a coding agent. Classify the user's request by the SCOPE of "
        "work it implies, NOT by how long the message is.\n"
        '- "quick": a small, bounded change a capable coder finishes in a few steps — a typo or '
        "one-line fix, a tiny tweak, a single short function, or answering a question about the "
        "code.\n"
        '- "deep": substantial or open-ended work — building a feature, adding a module/tool, '
        "multi-file changes, design or refactoring, or anything ambiguous or ambitious.\n"
        'A request can be SHORT yet "deep" — e.g. "build a pdf reader and maker" or "add auth" '
        'are deep. When unsure, answer "deep".\n'
        'Reply with ONLY a JSON object: {"difficulty": "quick"} or {"difficulty": "deep"}.'
    )


def parse_difficulty(data: object) -> Literal["quick_code", "deep_code"]:
    """Map a validated classifier result to a routing category; anything unexpected FAILS UP.

    Only an explicit ``{"difficulty": "quick"}`` yields ``quick_code``; every other value —
    ``"deep"``, a missing/odd field, or a non-dict — resolves to ``deep_code`` (the capable,
    reliable coder), so a garbled verdict can never strand a real task on the cheap model.
    """
    if isinstance(data, dict) and data.get("difficulty") == "quick":
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
    "DIFFICULTY_SCHEMA",
    "should_consult_classifier",
    "difficulty_system_prompt",
    "parse_difficulty",
    "describe_zakpick",
]

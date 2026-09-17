"""How a backend spells "thinking off" — rendered at the one request chokepoint.

Zak Code carries ONE internal spelling for a reasoning model's thinking switch,
:func:`zakcode.providers.routing.thinking_extra_body` (``{"chat_template_kwargs":
{"enable_thinking": <bool>}}``). It is llama.cpp's / vLLM's request-body form, measured
to work on the self-hosted pod (2026-08-17: completion_tokens 36 → 4, answer unchanged),
and it was carried on the assumption that "a server that does not understand the key
ignores it". That assumption is FALSE for the strict-schema clouds: Vertex AI validates
the whole JSON payload and refuses it (measured 2026-09-17 on a served Mind,
``vertex_ai_beta``: ``400 INVALID_ARGUMENT — Invalid JSON payload received. Unknown name
"chat_template_kwargs": Cannot find field.``), and OpenAI / Anthropic reject unknown body
fields the same way. The reasoning-overflow retry (ADR-0056) therefore turned a
recoverable "the model thought and delivered nothing" into a fatal ``provider_error``.

This module is the fix at the SPELLING level (ADR-0181): the provider renders the one
internal form into whatever the destination understands, at request-build time.

* An OpenAI-compatible server reached through a configured ``api_base`` (llama.cpp,
  vLLM, the zds pod) keeps the body key verbatim — the measured, working case.
* A Gemini model (``vertex_ai`` / ``vertex_ai_beta`` / ``gemini``) gets litellm's
  first-class ``reasoning_effort="minimal"`` for "off": litellm maps it per model to the
  tightest thinking budget the model accepts (128 for 2.5-pro, which cannot switch
  thinking off at all; 1 for 2.5-flash; a ``thinkingLevel`` for Gemini 3) — read from the
  installed litellm source, not assumed — and ``drop_params`` discards it for a Gemini
  model litellm does not flag as reasoning-capable. ``"disable"`` was rejected on
  purpose: it maps to a budget of 0, which 2.5-pro refuses.
* Every other destination (hosted OpenAI, Anthropic, Ollama, …) gets NO thinking
  override: the key is dropped and the retry runs with its rail alone, exactly the
  degraded path ADR-0056 already documents for a server without the key. Nothing there
  is measured, so nothing there is sent — the miss costs a retry that may overflow
  again; a wrong native param costs a 400 and the turn.

"on" is never rendered for a cloud model: thinking is the model's own default there.

The reasoning DEPTH (ADR-0182) is rendered at the same chokepoint by the same rule. It is a
level — litellm's ``reasoning_effort`` (``none`` … ``xhigh``), configured fleet-wide
(``Settings.reasoning_effort``) or per zakpick category — and it is sent ONLY where litellm
flags the model reasoning-capable, because every per-backend mapping litellm has for the
kwarg (a ``thinkingLevel`` on Gemini 3+, a ``thinkingBudget`` on Gemini 2.5, OpenAI's own
field on the gpt-5 family, a budget on Claude, ``think`` on Ollama) sits behind that same
predicate. Against any other model it is inert — never a 400. A self-hosted
OpenAI-compatible server is never given it: litellm's generic-OpenAI path does not list the
kwarg, so ``drop_params`` discards it before the request, and the servers that take a level
take it in their own body form (llama.cpp serving gpt-oss reads it from
``chat_template_kwargs``), which is what ``extra_body`` is for and what the switch rule above
already keeps verbatim there. "Off" wins over a depth: a request whose switch says off (a
category's ``thinking: false``, the ADR-0056 retry) gets the off rendering and no level.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from zakcode.providers.endpoints import model_uses_generic_endpoint, provider_prefix

#: The internal spelling's body key (llama.cpp / vLLM ``chat_template_kwargs``).
THINKING_SWITCH_KEY = "chat_template_kwargs"

#: litellm prefixes whose Gemini models take ``reasoning_effort`` (mapped by litellm to
#: ``thinkingConfig``). A Claude model served through Vertex takes Anthropic's mapping
#: instead, so the model name must ALSO name Gemini — see :func:`_is_gemini`.
GEMINI_PROVIDERS: frozenset[str] = frozenset({"vertex_ai", "vertex_ai_beta", "gemini"})

#: The ``reasoning_effort`` value that renders "thinking off" for Gemini: the tightest
#: budget every Gemini model accepts (litellm maps it per model).
GEMINI_THINKING_OFF_EFFORT = "minimal"

#: Names a Gemini 400 may use for the rendered switch on the wire — litellm rewrites
#: ``reasoning_effort`` into ``thinkingConfig`` before Vertex sees it, so a rejection
#: names the WIRE field, never the kwarg. Any of these in a rejection means the rendered
#: switch is what was refused (see :func:`rendered_thinking_wire_names`).
_THINKING_WIRE_NAMES: tuple[str, ...] = (
    "reasoning_effort",
    "thinkingConfig",
    "thinking_config",
    "thinkingBudget",
    "thinking_budget",
    "thinkingLevel",
    "thinking_level",
    "includeThoughts",
    "include_thoughts",
)


def _is_gemini(model: str) -> bool:
    return provider_prefix(model) in GEMINI_PROVIDERS and "gemini" in model.lower()


def thinking_switch_enabled(body: dict[str, Any]) -> bool | None:
    """The switch's value in ``body`` (``True``/``False``), or None when it carries none."""
    fragment = body.get(THINKING_SWITCH_KEY)
    if not isinstance(fragment, dict):
        return None
    value = fragment.get("enable_thinking")
    return value if isinstance(value, bool) else None


def render_thinking_switch(
    model: str, api_base: str | None, body: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render the internal thinking switch in ``body`` for ``model``'s destination.

    Returns ``(body, kwargs)``: the request body to send (the switch kept, or removed)
    and any first-class litellm kwargs that express it instead. A body without the
    switch is returned unchanged with no kwargs — the default request shape is
    byte-identical to before.
    """
    enabled = thinking_switch_enabled(body)
    if THINKING_SWITCH_KEY not in body:
        return body, {}
    if api_base is not None and model_uses_generic_endpoint(model):
        # A self-hosted OpenAI-compatible server: the measured, working spelling.
        return body, {}
    rest = {k: v for k, v in body.items() if k != THINKING_SWITCH_KEY}
    if enabled is False and _is_gemini(model):
        return rest, {"reasoning_effort": GEMINI_THINKING_OFF_EFFORT}
    # "on" is the cloud model's own default; any other destination has no measured
    # per-call switch — send nothing rather than a guess.
    return rest, {}


#: litellm's first-class kwarg for a reasoning DEPTH (a level, not the on/off switch).
REASONING_EFFORT_KWARG = "reasoning_effort"


def reasoning_effort_reaches(
    model: str, api_base: str | None, *, supports_reasoning: Callable[[str], bool]
) -> bool:
    """Whether a configured reasoning level is SENT for ``model`` (ADR-0182).

    Only where litellm has a per-backend mapping for the kwarg AND flags the model
    reasoning-capable — ``supports_reasoning`` is litellm's own predicate, injected so this
    module stays free of the SDK. A self-hosted OpenAI-compatible server reached through
    ``api_base`` never gets it: litellm's generic-OpenAI path does not carry the kwarg
    (``drop_params`` discards it before the request), and the servers that take a level
    take it in their own body form, which is ``extra_body``'s job.
    """
    if api_base is not None and model_uses_generic_endpoint(model):
        return False
    return supports_reasoning(model)


def render_reasoning_effort(
    model: str,
    api_base: str | None,
    level: str | None,
    body: dict[str, Any],
    *,
    supports_reasoning: Callable[[str], bool],
) -> dict[str, Any]:
    """The litellm kwargs expressing a configured reasoning ``level`` for ``model`` — or
    ``{}``: no level configured, the destination does not take one, or ``body`` (the request
    body BEFORE the switch is rendered) says thinking is OFF for this request — a category's
    ``thinking: false``, the loop's reasoning-overflow retry — and "off" is not a depth.
    """
    if level is None or thinking_switch_enabled(body) is False:
        return {}
    if not reasoning_effort_reaches(model, api_base, supports_reasoning=supports_reasoning):
        return {}
    return {REASONING_EFFORT_KWARG: level}


def rendered_thinking_wire_names(kwargs: dict[str, Any]) -> tuple[str, ...]:
    """The wire-level names a provider might use to refuse the rendered switch — or the
    rendered depth, which rides the same kwarg — in ``kwargs``; empty when neither was
    rendered."""
    return _THINKING_WIRE_NAMES if REASONING_EFFORT_KWARG in kwargs else ()

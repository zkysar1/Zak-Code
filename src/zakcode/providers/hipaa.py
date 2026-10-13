"""HIPAA mode: which models a covered home may call, and through which endpoint.

``Settings.hipaa_mode`` narrows every model call to ``Settings.hipaa_allowed_models``. The one
predicate lives here so the startup sweep (``Agent._assert_hipaa_models``) and the request
builder (``LiteLLMProvider._build_kwargs``) cannot disagree about what a model name means, the
same reason ``endpoints`` holds both of ``local_only``'s predicates.

The same mode also keeps a covered call on Chat Completions, the one OpenAI endpoint the owner
has confirmed: ``check_chat_completions_only`` refuses a request that would be stored, run in the
background, search the web, use a vendor-hosted tool, or be moved onto the Responses API.

This module imports no vendor SDK, so ``config`` and the Agent can import it freely. The route
check takes the vendor probe as an argument (``route_mode``): the litellm provider, the one
module allowed to import litellm, supplies it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

#: The litellm prefix OpenAI's own API is addressed by. ``openai/gpt-4o`` and ``gpt-4o`` name
#: the same hosted model, so the allowlist reads them as one. No other prefix is stripped: a
#: named provider, or a generic-OpenAI base such as ``openai_like/``, is a different
#: destination and has to be listed by its full name.
_OPENAI_PREFIX = "openai/"


class HipaaModelViolation(RuntimeError):
    """A call was about to use a model that is not on the HIPAA allowlist.

    Deliberately **not** a :class:`~zakcode.providers.base.ProviderError`, for the reason
    :class:`~zakcode.providers.endpoints.LocalOnlyViolation` is not. Provider errors are
    recoverable and ride the loop's failover, which answers by trying a DIFFERENT model, the
    one reaction that would turn a single refusal into patient data reaching a model nobody
    confirmed. This is a configuration fault: it is fatal, it names the offending model, and
    nothing retries it.
    """


class HipaaEndpointViolation(RuntimeError):
    """A call was about to leave Chat Completions, the one endpoint a HIPAA home may use.

    Not a :class:`~zakcode.providers.base.ProviderError`, for the reason
    :class:`HipaaModelViolation` is not: failover answers a ProviderError by trying another
    route, and another route is the very thing this refusal exists to stop. It is a
    configuration fault: it is fatal, it names the field or the route, and nothing retries it.
    """


def normalize_model(model: str | None) -> str:
    """The comparable form of a model name: trimmed, lowercased, a leading ``openai/`` removed."""
    name = (model or "").strip().lower()
    return name[len(_OPENAI_PREFIX) :] if name.startswith(_OPENAI_PREFIX) else name


def check_model(model: str | None, allowed: Sequence[str] | None) -> tuple[bool, str]:
    """Whether ``model`` may be called under HIPAA mode, with an operator-actionable reason.

    Fail CLOSED: a blank model, or an empty allowlist, permits nothing. Matching is exact on
    the normalized name, so a dated snapshot (``gpt-4o-2024-08-06``) is a different model from
    ``gpt-4o`` and has to be listed on its own.
    """
    name = normalize_model(model)
    if not name:
        return False, "there is no model name to check"
    listed = sorted({entry for entry in (normalize_model(e) for e in (allowed or ())) if entry})
    if not listed:
        return False, (
            f"{model!r} cannot be used: hipaa_allowed_models is empty, and an empty "
            "allowlist permits no model"
        )
    if name in listed:
        return True, f"{model!r} is on the allowlist"
    return False, f"{model!r} is not on the allowlist (hipaa_allowed_models: {', '.join(listed)})"


#: Chat Completions defines two kinds of tool. A request that declares any other ``type``
#: (``web_search*``, ``file_search``, ``code_interpreter``, ``mcp``, ``computer_use``, ...)
#: asks the vendor to run something on its side, and none of that is on the covered endpoint.
_CHAT_TOOL_TYPES = frozenset({"function", "custom"})

#: Request fields refused only when truthy. ``store=false`` is what HIPAA mode wants, so it is
#: not an offence; ``store=true`` asks the vendor to keep the exchange.
_REFUSED_WHEN_TRUTHY = {
    "store": "it asks the vendor to retain the exchange",
    "background": "it runs the request detached on the vendor's side",
}

#: Request fields that address another endpoint's stored state, or reach the live web, refused
#: whenever they are present. ``reasoning_summary`` is a Responses-only field: litellm answers
#: one by moving the call onto /v1/responses.
_REFUSED_WHEN_PRESENT = {
    "web_search_options": "it asks the vendor to search the live web",
    "previous_response_id": "it chains onto a stored Responses object",
    "conversation": "it addresses a stored Conversations object",
    "reasoning_summary": "it is a Responses-only field",
    "reasoningSummary": "it is a Responses-only field",
}


def _request_layers(call_kwargs: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The two places a request field can sit: the top level and the ``extra_body`` passthrough.

    Both reach the wire (litellm forwards ``extra_body`` into the JSON body), and which one
    wins when they disagree is litellm's to say. So each layer is judged on its own and an
    offence in either one refuses.
    """
    body = call_kwargs.get("extra_body")
    layers: list[Mapping[str, Any]] = [call_kwargs]
    if isinstance(body, Mapping):
        layers.append(body)
    return layers


def _tool_type(tool: Any) -> Any:
    return tool.get("type") if isinstance(tool, Mapping) else getattr(tool, "type", None)


def check_request_shape(call_kwargs: Mapping[str, Any]) -> tuple[bool, str]:
    """Whether the request body keeps nothing and runs nothing on the vendor's side.

    Judges the fields named above and the tool kinds, not the whole body, in the top level and
    in the ``extra_body`` passthrough alike.
    """
    for layer in _request_layers(call_kwargs):
        for name, why in _REFUSED_WHEN_TRUTHY.items():
            if layer.get(name):
                return False, f"{name} is set, and {why}"
        for name, why in _REFUSED_WHEN_PRESENT.items():
            if layer.get(name) is not None:
                return False, f"{name} is set, and {why}"
        tools = layer.get("tools") or ()
        hosted = sorted(
            {str(_tool_type(t)) for t in tools if _tool_type(t) not in _CHAT_TOOL_TYPES}
        )
        if hosted:
            return False, (
                f"the request declares vendor-hosted tool(s) {', '.join(hosted)}, and Chat "
                "Completions defines only function and custom tools"
            )
    return True, "the request body is a plain Chat Completions body"


#: The vendor probe ``check_route`` is given: ``(model, call_kwargs, api_base) -> mode``, the
#: endpoint family the installed client library would use for that call (``"responses"``,
#: ``"chat"``, ...), or ``None`` when it cannot say.
RouteProbe = Callable[[str, Mapping[str, Any], str | None], str | None]


def check_route(
    model: str | None,
    call_kwargs: Mapping[str, Any],
    *,
    route_mode: RouteProbe,
    api_base: str | None = None,
) -> tuple[bool, str]:
    """Whether litellm would send this call to Chat Completions and not to the Responses API.

    The client library is asked (``route_mode``), never re-derived: its answer moves with the
    installed release, the tools, the reasoning depth and the base, so a copy of its rules is
    right only on the day it was written.

    Fail CLOSED: a blank model, or a route the probe cannot name, is refused.
    """
    name = (model or "").strip()
    if not name:
        return False, "there is no model name to route"
    if "responses" in name.lower().split("/")[:-1]:
        return False, f"{model!r} names the Responses route outright (a responses/ prefix)"
    mode = route_mode(name, call_kwargs, api_base)
    if mode is None:
        return False, (
            f"litellm could not say which endpoint {model!r} would use, and a route that "
            "cannot be shown to be Chat Completions is refused"
        )
    if mode == "responses":
        return False, (
            f"litellm would send this {model!r} call to /v1/responses (the model is "
            "Responses-only, or the request carries tools with a reasoning depth that "
            "routes it there)"
        )
    return True, f"litellm would send this {model!r} call to Chat Completions"


def check_chat_completions_only(
    model: str | None,
    call_kwargs: Mapping[str, Any],
    *,
    route_mode: RouteProbe,
    api_base: str | None = None,
) -> tuple[bool, str]:
    """Whether a call about to be sent stays on Chat Completions, vendor-side state aside.

    "Vendor-side state" is anything the vendor would keep or run for the call (see
    :func:`check_request_shape`). ``call_kwargs`` is the request exactly as it will be handed
    to litellm, so the check sees every default the provider added (the reasoning depth that
    decides the route included).
    """
    ok, reason = check_request_shape(call_kwargs)
    if not ok:
        return False, reason
    return check_route(model, call_kwargs, route_mode=route_mode, api_base=api_base)


__all__ = [
    "HipaaEndpointViolation",
    "HipaaModelViolation",
    "check_chat_completions_only",
    "check_model",
    "check_request_shape",
    "check_route",
    "normalize_model",
]

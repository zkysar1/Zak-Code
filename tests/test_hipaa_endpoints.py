"""The ``hipaa_mode`` endpoint gate: a covered call stays on Chat Completions.

``tests/test_hipaa_models.py`` pins WHO a HIPAA home may call. This pins WHICH door the call
goes through: nothing kept or run on the vendor's side, and no move onto the Responses API.
Like the model gate it must never fail open, so these tests pin the refusal as well as the
pass, at the three places the guarantee lives:

* ``check_request_shape`` - the fields and tool kinds that ask the vendor to keep or run something.
* ``check_route`` - litellm's own answer about the endpoint, asked rather than re-derived.
* ``LiteLLMProvider._build_kwargs`` - the request that will actually be sent.

The route depends on the installed litellm (the lock pins 1.86.2, a dev venv may hold newer),
so a test that needs a particular answer either picks a model that is Responses-only in both,
or computes the expected answer from litellm itself.
"""

from __future__ import annotations

import contextlib
import inspect
import socket
from types import SimpleNamespace
from typing import Any

import pytest

from zakcode.config import Settings
from zakcode.messages import Message
from zakcode.providers import hipaa
from zakcode.providers import litellm_provider as provider_mod
from zakcode.providers.base import ProviderError
from zakcode.providers.hipaa import (
    HipaaEndpointViolation,
    HipaaModelViolation,
    check_request_shape,
)
from zakcode.providers.litellm_provider import LiteLLMProvider, _litellm_route

MESSAGES = [{"role": "user", "content": "hi"}]
FUNCTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "f",
        "description": "d",
        "parameters": {"type": "object", "properties": {}},
    },
}

#: Models litellm's registry marks ``mode: responses`` in the pinned release and in newer ones.
RESPONSES_ONLY = ["openai/gpt-5-pro", "openai/gpt-5-codex", "openai/gpt-5.4-pro"]
#: Models that serve function tools on Chat Completions in the pinned release and in newer ones.
CHAT_MODELS = ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-5-mini"]


def check_route(
    model: str | None, call_kwargs: dict[str, Any], *, api_base: str | None = None
) -> tuple[bool, str]:
    """The route check as the provider runs it: with litellm's own bridge check as the probe."""
    return hipaa.check_route(model, call_kwargs, route_mode=_litellm_route, api_base=api_base)


def check_chat_completions_only(
    model: str | None, call_kwargs: dict[str, Any], *, api_base: str | None = None
) -> tuple[bool, str]:
    """The composite check as the provider runs it, with the same probe."""
    return hipaa.check_chat_completions_only(
        model, call_kwargs, route_mode=_litellm_route, api_base=api_base
    )


# ── the request shape ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "call_kwargs",
    [
        {},
        {"tools": [FUNCTION_TOOL]},
        {"tools": [{"type": "custom", "custom": {"name": "grammar"}}]},
        # store=false is what HIPAA mode wants; it is the provider's own default.
        {"store": False},
        {"extra_body": {"store": False}},
        {"background": False},
        {"extra_body": {"unrelated": 1}},
        {"extra_body": None},
    ],
)
def test_a_plain_chat_body_passes(call_kwargs: dict[str, Any]) -> None:
    ok, reason = check_request_shape(call_kwargs)
    assert ok, reason


@pytest.mark.parametrize(
    ("call_kwargs", "fragment"),
    [
        ({"store": True}, "store"),
        ({"extra_body": {"store": True}}, "store"),
        ({"background": True}, "background"),
        ({"extra_body": {"background": True}}, "background"),
        # Present at all is an offence, even empty: the field itself is the request.
        ({"web_search_options": {}}, "web_search_options"),
        (
            {"extra_body": {"web_search_options": {"search_context_size": "low"}}},
            "web_search_options",
        ),
        ({"previous_response_id": "resp_1"}, "previous_response_id"),
        ({"extra_body": {"conversation": "conv_1"}}, "conversation"),
        ({"reasoning_summary": "auto"}, "reasoning_summary"),
        ({"extra_body": {"reasoningSummary": "auto"}}, "reasoningSummary"),
        ({"tools": [{"type": "web_search_preview"}]}, "web_search_preview"),
        ({"tools": [{"type": "file_search"}]}, "file_search"),
        ({"tools": [{"type": "code_interpreter"}]}, "code_interpreter"),
        ({"tools": [{"type": "mcp"}]}, "mcp"),
        ({"tools": [FUNCTION_TOOL, {"type": "computer_use_preview"}]}, "computer_use_preview"),
        ({"extra_body": {"tools": [{"type": "web_search"}]}}, "web_search"),
        ({"tools": [SimpleNamespace(type="image_generation")]}, "image_generation"),
        # The two layers are judged separately: a false at the top does not hide a true below.
        ({"store": False, "extra_body": {"store": True}}, "store"),
    ],
)
def test_a_request_that_keeps_or_runs_something_is_refused(
    call_kwargs: dict[str, Any], fragment: str
) -> None:
    ok, reason = check_request_shape(call_kwargs)
    assert not ok
    assert fragment in reason


def test_a_hosted_tool_is_named_among_function_tools() -> None:
    """The headline case: a real function-tool list with one hosted tool slipped in."""
    ok, reason = check_request_shape({"tools": [FUNCTION_TOOL, FUNCTION_TOOL, {"type": "mcp"}]})
    assert not ok
    assert "mcp" in reason


# ── the route ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model", ["openai/responses/gpt-5-pro", "responses/gpt-4o", "OpenAI/Responses/x"]
)
def test_a_responses_prefix_is_refused_outright(model: str) -> None:
    ok, reason = check_route(model, {})
    assert not ok
    assert "outright" in reason


@pytest.mark.parametrize("model", RESPONSES_ONLY)
def test_a_responses_only_model_is_refused_even_with_no_tools(model: str) -> None:
    ok, reason = check_route(model, {})
    assert not ok
    assert "/v1/responses" in reason


@pytest.mark.parametrize("model", CHAT_MODELS)
@pytest.mark.parametrize("tools", [None, [FUNCTION_TOOL]])
def test_a_chat_model_keeps_the_chat_route(model: str, tools: list[dict[str, Any]] | None) -> None:
    ok, reason = check_route(model, {"tools": tools})
    assert ok, reason


@pytest.mark.parametrize("model", [None, "", "   "])
def test_a_blank_model_has_no_route(model: str | None) -> None:
    ok, reason = check_route(model, {})
    assert not ok
    assert "no model name" in reason


@pytest.mark.parametrize(("mode", "allowed"), [("chat", True), ("responses", False), (None, False)])
def test_the_policy_follows_the_probe_it_is_given(mode: str | None, allowed: bool) -> None:
    """The policy module is vendor-free: any probe will do, and one that cannot say refuses."""
    asked: list[tuple[str, str | None]] = []

    def probe(model: str, call_kwargs: object, api_base: str | None) -> str | None:
        asked.append((model, api_base))
        return mode

    ok, _ = hipaa.check_route(" openai/gpt-4o ", {}, route_mode=probe, api_base="https://b/v1")
    assert ok is allowed
    assert asked == [("openai/gpt-4o", "https://b/v1")]


@pytest.mark.parametrize("model", ["responses/gpt-4o", "openai/responses/gpt-4o", "", None])
def test_a_model_that_needs_no_probe_never_reaches_it(model: str | None) -> None:
    def probe(*args: object) -> str | None:
        raise AssertionError("the probe must not be asked")

    ok, _ = hipaa.check_route(model, {}, route_mode=probe)
    assert not ok


def test_a_route_litellm_cannot_name_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail CLOSED. If litellm cannot be asked, the call is not shown to be Chat Completions."""
    import litellm.main

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("the installed litellm changed its bridge check")

    monkeypatch.setattr(litellm.main, "responses_api_bridge_check", boom)
    ok, reason = check_route("openai/gpt-4o", {})
    assert not ok
    assert "could not say" in reason


def test_the_verdict_is_litellms_own_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm.main

    answers = iter([({"mode": "responses"}, "m"), ({"mode": "chat"}, "m"), ({}, "m")])

    def bridge(model: str, custom_llm_provider: str) -> tuple[dict[str, str], str]:
        return next(answers)

    monkeypatch.setattr(litellm.main, "responses_api_bridge_check", bridge)
    assert check_route("openai/gpt-4o", {})[0] is False
    assert check_route("openai/gpt-4o", {})[0] is True
    # No mode at all is not Responses: litellm only bridges on an explicit "responses".
    assert check_route("openai/gpt-4o", {})[0] is True


def test_only_the_arguments_the_installed_signature_accepts_are_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lock pins a litellm whose bridge check has no ``api_base``; a newer one adds it.

    Offering an argument the installed function lacks is a TypeError, which would refuse every
    call on the pinned release. The stub has no ``api_base``, so a filter that stopped working
    would turn this into a refusal.
    """
    import litellm.main

    seen: dict[str, Any] = {}

    def bridge(
        model: str,
        custom_llm_provider: str,
        web_search_options: object = None,
        tools: object = None,
        reasoning_effort: object = None,
    ) -> tuple[dict[str, str], str]:
        seen.update(model=model, provider=custom_llm_provider, tools=tools, effort=reasoning_effort)
        return {"mode": "chat"}, model

    monkeypatch.setattr(litellm.main, "responses_api_bridge_check", bridge)
    ok, reason = check_route(
        "openai/gpt-4o",
        {"tools": [FUNCTION_TOOL], "reasoning_effort": "low"},
        api_base="https://api.openai.com/v1",
    )
    assert ok, reason
    assert seen == {
        "model": "gpt-4o",
        "provider": "openai",
        "tools": [FUNCTION_TOOL],
        "effort": "low",
    }


def test_api_base_is_offered_when_the_installed_signature_accepts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.main

    seen: dict[str, Any] = {}

    def bridge(
        model: str, custom_llm_provider: str, api_base: str | None = None
    ) -> tuple[dict[str, str], str]:
        seen["api_base"] = api_base
        return {"mode": "chat"}, model

    monkeypatch.setattr(litellm.main, "responses_api_bridge_check", bridge)
    assert check_route("openai/gpt-4o", {}, api_base="https://example.test/v1")[0]
    assert seen["api_base"] == "https://example.test/v1"


def test_the_shape_is_judged_before_the_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal on a field must not depend on litellm being askable."""
    import litellm.main

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("the route must not be asked once the shape has failed")

    monkeypatch.setattr(litellm.main, "responses_api_bridge_check", boom)
    ok, reason = check_chat_completions_only("openai/gpt-4o", {"store": True})
    assert not ok
    assert "store" in reason


# ── the application gate (the actual guarantee) ───────────────────


def _provider(
    model: str = "openai/gpt-4o", *, mode: bool = True, **settings: Any
) -> LiteLLMProvider:
    listed = [model.split("/", 1)[-1]]
    return LiteLLMProvider(
        Settings(
            default_model=model,
            hipaa_mode=mode,
            hipaa_allowed_models=listed if mode else [],
            _env_file=None,
            **settings,
        )
    )


def test_a_listed_chat_model_with_function_tools_builds_the_request() -> None:
    kwargs = _provider()._build_kwargs(MESSAGES, [FUNCTION_TOOL])
    assert kwargs["model"] == "openai/gpt-4o"
    assert kwargs["tools"] == [FUNCTION_TOOL]


def test_the_providers_own_store_default_does_not_trip_the_gate() -> None:
    """The provider writes ``store: false`` into the body for OpenAI's API itself; the gate
    refuses ``store: true`` only, or HIPAA mode would refuse every call it was built for."""
    kwargs = _provider()._build_kwargs(MESSAGES, None)
    assert (kwargs.get("extra_body") or {}).get("store") in (False, None)


@pytest.mark.parametrize("model", RESPONSES_ONLY)
def test_a_listed_responses_only_model_is_still_refused(model: str) -> None:
    """Listing a model says the owner may send it patient data. It does not make the Responses
    API an endpoint the owner confirmed."""
    with pytest.raises(HipaaEndpointViolation, match="/v1/responses"):
        _provider(model)._build_kwargs(MESSAGES, [FUNCTION_TOOL])


def test_store_true_from_the_settings_body_is_refused() -> None:
    with pytest.raises(HipaaEndpointViolation, match="store"):
        _provider(extra_body={"store": True})._build_kwargs(MESSAGES, None)


def test_store_true_from_a_per_call_body_is_refused() -> None:
    with pytest.raises(HipaaEndpointViolation, match="store"):
        _provider()._build_kwargs(MESSAGES, None, extra_body={"store": True})


def test_a_hosted_tool_is_refused_at_the_request_builder() -> None:
    with pytest.raises(HipaaEndpointViolation, match="web_search_preview"):
        _provider()._build_kwargs(MESSAGES, [FUNCTION_TOOL, {"type": "web_search_preview"}])


def test_a_per_call_web_search_option_is_refused() -> None:
    with pytest.raises(HipaaEndpointViolation, match="web_search_options"):
        _provider()._build_kwargs(MESSAGES, None, web_search_options={"search_context_size": "low"})


@pytest.mark.parametrize(
    "model", [*CHAT_MODELS, "openai/gpt-5.4", "openai/gpt-5.6-luna", "openai/gpt-6-luna"]
)
def test_the_gate_judges_the_request_the_provider_actually_built(model: str) -> None:
    """Wiring: the verdict equals litellm's own answer for the very kwargs the provider built.

    The provider injects a reasoning depth on a tool call for the gpt-5.4+ tiers, and under some
    litellm releases that depth is what moves the call onto /v1/responses. A gate that ran
    before that injection, or asked about different arguments, would pass a call that litellm
    then routes to the Responses API. So the expectation is computed from litellm on the
    request as built with the mode OFF, and the HIPAA-on provider must agree with it on
    whichever litellm is installed.
    """
    import litellm
    from litellm.main import responses_api_bridge_check

    built = _provider(model, mode=False)._build_kwargs(MESSAGES, [FUNCTION_TOOL])
    bare, provider, _key, _base = litellm.get_llm_provider(model=model)
    accepted = inspect.signature(responses_api_bridge_check).parameters
    offered = {"tools": built.get("tools"), "reasoning_effort": built.get("reasoning_effort")}
    info, _ = responses_api_bridge_check(
        model=bare,
        custom_llm_provider=provider,
        **{k: v for k, v in offered.items() if k in accepted},
    )
    bridged = info.get("mode") == "responses"

    gated = _provider(model)
    if bridged:
        with pytest.raises(HipaaEndpointViolation, match="/v1/responses"):
            gated._build_kwargs(MESSAGES, [FUNCTION_TOOL])
    else:
        assert gated._build_kwargs(MESSAGES, [FUNCTION_TOOL])["model"] == model


#: Both sides of the line on the pinned release and on a newer one: chat models, Responses-only
#: models, and the gpt-5.4 / gpt-5.6 tiers whose side depends on the installed litellm.
WIRE_MODELS = [
    *CHAT_MODELS,
    "openai/gpt-5.4",
    "openai/gpt-5.6-luna",
    "openai/gpt-6-luna",
    *RESPONSES_ONLY,
]


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
@pytest.mark.parametrize("model", WIRE_MODELS)
def test_the_gate_agrees_with_the_url_litellm_actually_calls(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wire: the verdict equals the endpoint the request reaches, at OpenAI's default base.

    The test above compares the gate with litellm's decision function; this one compares it with
    the request litellm sends. ``completion`` runs on the kwargs the provider built, every httpx
    request is answered in-process (so the URL is the real ``api.openai.com`` one and nothing
    leaves the machine), and the gate must refuse exactly the calls that did not reach
    ``/chat/completions``. The base is left unset on purpose: litellm 1.98 decides the unset-depth
    case differently on a custom base, so a request redirected to a local server is a different
    request.
    """
    import httpx
    import litellm

    reached: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        reached.append(request.url.path)
        return httpx.Response(400, json={"error": {"message": "wire test"}}, request=request)

    async def answer_async(self: object, request: httpx.Request) -> httpx.Response:
        return answer(request)

    real_connect = socket.socket.connect

    def loopback_only(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else address
        if str(host) not in ("127.0.0.1", "::1", "localhost") and not str(host).startswith("/"):
            raise OSError(f"the wire test does not leave the machine: {address!r}")
        real_connect(self, address)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req: answer(req))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", answer_async)
    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "api_base", None, raising=False)

    built = dict(_provider(model, mode=False)._build_kwargs(MESSAGES, [FUNCTION_TOOL]))
    built["api_key"] = "sk-wire-test"
    with contextlib.suppress(Exception):  # the stand-in answers 400 on purpose
        litellm.completion(**built)
    model_calls = [p for p in reached if p.endswith(("/chat/completions", "/responses"))]
    assert len(model_calls) == 1, reached

    gated = _provider(model)
    if model_calls[0].endswith("/chat/completions"):
        assert gated._build_kwargs(MESSAGES, [FUNCTION_TOOL])["model"] == model
    else:
        with pytest.raises(HipaaEndpointViolation, match="/v1/responses"):
            gated._build_kwargs(MESSAGES, [FUNCTION_TOOL])


def test_a_route_the_gate_cannot_prove_is_refused_by_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.main

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("no such function")

    monkeypatch.setattr(litellm.main, "responses_api_bridge_check", boom)
    with pytest.raises(HipaaEndpointViolation, match="could not say"):
        _provider()._build_kwargs(MESSAGES, None)


def test_the_provider_asks_about_the_base_it_will_actually_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wiring: the configured ``api_base`` reaches the route check, where litellm uses it to
    decide whether the endpoint is the real OpenAI one (releases that accept it)."""
    import litellm.main

    seen: dict[str, Any] = {}

    def bridge(
        model: str, custom_llm_provider: str, api_base: str | None = None
    ) -> tuple[dict[str, str], str]:
        seen["api_base"] = api_base
        return {"mode": "chat"}, model

    monkeypatch.setattr(litellm.main, "responses_api_bridge_check", bridge)
    _provider(api_base="https://example.test/v1")._build_kwargs(MESSAGES, None)
    assert seen["api_base"] == "https://example.test/v1"


def test_the_refusal_names_the_remedy() -> None:
    with pytest.raises(HipaaEndpointViolation) as exc:
        _provider(extra_body={"store": True})._build_kwargs(MESSAGES, None)
    message = str(exc.value)
    assert "Chat Completions" in message
    assert "ZAKCODE_HIPAA_MODE" in message


def test_hipaa_mode_off_changes_nothing() -> None:
    """The gate may newly refuse only a home that opted in."""
    off = _provider("openai/gpt-5-pro", mode=False, extra_body={"store": True})
    kwargs = off._build_kwargs(MESSAGES, [FUNCTION_TOOL, {"type": "web_search_preview"}])
    assert kwargs["model"] == "openai/gpt-5-pro"
    assert (kwargs.get("extra_body") or {}).get("store") is True


def test_the_model_gate_still_speaks_first() -> None:
    """An unlisted model is a HipaaModelViolation, whatever else is wrong with the request."""
    provider = LiteLLMProvider(
        Settings(
            default_model="openai/gpt-5-pro",
            hipaa_mode=True,
            hipaa_allowed_models=["gpt-4o"],
            _env_file=None,
        )
    )
    with pytest.raises(HipaaModelViolation):
        provider._build_kwargs(MESSAGES, [FUNCTION_TOOL])


def test_the_violation_is_not_a_provider_error() -> None:
    """Failover answers a ProviderError by trying another route, which is what this refusal
    exists to stop, so the violation stays outside that hierarchy."""
    assert not issubclass(HipaaEndpointViolation, ProviderError)
    assert issubclass(HipaaEndpointViolation, RuntimeError)


async def test_acomplete_refuses_before_any_request_leaves_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[object] = []

    async def must_not_be_reached(*args: object, **kwargs: object) -> object:
        called.append(kwargs)
        raise AssertionError("a refused call reached litellm")

    monkeypatch.setattr(provider_mod.litellm, "acompletion", must_not_be_reached)
    with pytest.raises(HipaaEndpointViolation, match="store"):
        await _provider(extra_body={"store": True}).acomplete(
            [Message.user("hi")], tools=[FUNCTION_TOOL]
        )
    assert called == []

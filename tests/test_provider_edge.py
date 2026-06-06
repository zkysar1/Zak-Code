"""Edge-case tests for the litellm-backed provider — hardening pass.

Every test here is hermetic: ``litellm.acompletion``, ``litellm.token_counter``,
``litellm.supports_function_calling`` and the litellm exception classes are
monkeypatched with fakes/scripted values. Nothing touches the network or a live
model. The focus is the error taxonomy, tool-call argument parsing, response
normalization edges, token-count fallback, and the registry's safe default.
"""

from __future__ import annotations

from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.messages import Message
from zakcode.providers.base import (
    AuthError,
    Capabilities,
    ContextWindowExceeded,
    LLMResult,
    ProviderError,
    RateLimited,
    RequestFailed,
)
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.registry import get_capabilities
from zakcode.usage import Usage


# ──────────────────────────────────────────────────────────────────────────────
# Fakes mirroring the litellm/openai response shape (attribute access).
# ──────────────────────────────────────────────────────────────────────────────
class _FakeFunction:
    def __init__(self, name: Any, arguments: Any) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: Any, name: Any, arguments: Any) -> None:
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content: Any, tool_calls: Any = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message: Any, finish_reason: str | None = "stop") -> None:
        self.message = message
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, choices: Any, usage: Any = None, hidden: Any = None) -> None:
        self.choices = choices
        self.usage = usage
        if hidden is not None:
            self._hidden_params = hidden


class _FakeUsage:
    def __init__(self, prompt: Any, completion: Any, total: Any) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


def _make_response(
    content: Any = "hi",
    tool_calls: Any = None,
    usage: Any = None,
    hidden: Any = None,
    finish_reason: str | None = "stop",
) -> _FakeResponse:
    return _FakeResponse(
        [_FakeChoice(_FakeMessage(content, tool_calls), finish_reason)], usage, hidden
    )


@pytest.fixture
def provider() -> LiteLLMProvider:
    return LiteLLMProvider(model="gpt-4o", temperature=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Error taxonomy
# ──────────────────────────────────────────────────────────────────────────────
class _FakeAuthError(Exception):
    pass


class _FakePermissionError(Exception):
    pass


class _FakeContextError(Exception):
    pass


class _FakeRateError(Exception):
    def __init__(self, message: str, retry_after: Any = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@pytest.fixture(autouse=True)
def _wire_exception_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the module's resolved exception classes at our fakes."""
    monkeypatch.setattr(lp, "_LiteLLMAuthError", _FakeAuthError)
    monkeypatch.setattr(lp, "_LiteLLMPermissionError", _FakePermissionError)
    monkeypatch.setattr(lp, "_LiteLLMContextWindowError", _FakeContextError)
    monkeypatch.setattr(lp, "_LiteLLMRateLimitError", _FakeRateError)


def _raise(exc: Exception) -> Any:
    async def _acompletion(**_kw: Any) -> Any:
        raise exc

    return _acompletion


async def test_auth_error_maps(provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lp.litellm, "acompletion", _raise(_FakeAuthError("bad key")))
    with pytest.raises(AuthError):
        await provider.acomplete([Message.user("hi")])


async def test_permission_error_maps_to_auth(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lp.litellm, "acompletion", _raise(_FakePermissionError("denied")))
    with pytest.raises(AuthError):
        await provider.acomplete([Message.user("hi")])


async def test_context_window_maps(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lp.litellm, "acompletion", _raise(_FakeContextError("too long")))
    with pytest.raises(ContextWindowExceeded):
        await provider.acomplete([Message.user("hi")])


async def test_rate_limit_carries_retry_after(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lp.litellm, "acompletion", _raise(_FakeRateError("slow down", 12.5)))
    with pytest.raises(RateLimited) as ei:
        await provider.acomplete([Message.user("hi")])
    assert ei.value.retry_after == 12.5


async def test_rate_limit_without_retry_after(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lp.litellm, "acompletion", _raise(_FakeRateError("slow down")))
    with pytest.raises(RateLimited) as ei:
        await provider.acomplete([Message.user("hi")])
    assert ei.value.retry_after is None


async def test_unknown_exception_maps_to_request_failed(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TotallyUnknownError(Exception):
        pass

    monkeypatch.setattr(lp.litellm, "acompletion", _raise(TotallyUnknownError("???")))
    with pytest.raises(RequestFailed):
        await provider.acomplete([Message.user("hi")])


async def test_raw_vendor_exception_never_leaks(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any escaping error must be a ProviderError, not the raw vendor type."""

    class VendorBoom(Exception):
        pass

    monkeypatch.setattr(lp.litellm, "acompletion", _raise(VendorBoom("kaboom")))
    with pytest.raises(ProviderError) as ei:
        await provider.acomplete([Message.user("hi")])
    assert not isinstance(ei.value, VendorBoom)


def test_map_error_redacts_credentials_from_message() -> None:
    # audit3 #8: a vendor exception that embeds a credential-shaped token must be scrubbed
    # before the ProviderError (which is displayed/logged) carries it.
    err = Exception("auth failed for api_key=sk-supersecretvalue1234567890 at gateway")
    mapped = LiteLLMProvider._map_error(err)
    assert "sk-supersecretvalue1234567890" not in str(mapped)


def test_map_error_falls_back_to_name_when_classes_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If litellm exposed no classes, name-based matching still maps correctly."""
    monkeypatch.setattr(lp, "_LiteLLMAuthError", None)
    monkeypatch.setattr(lp, "_LiteLLMPermissionError", None)
    monkeypatch.setattr(lp, "_LiteLLMContextWindowError", None)
    monkeypatch.setattr(lp, "_LiteLLMRateLimitError", None)

    class AuthenticationError(Exception):
        pass

    class ContextWindowExceededError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    assert isinstance(LiteLLMProvider._map_error(AuthenticationError("x")), AuthError)
    assert isinstance(
        LiteLLMProvider._map_error(ContextWindowExceededError("x")), ContextWindowExceeded
    )
    assert isinstance(LiteLLMProvider._map_error(RateLimitError("x")), RateLimited)


def test_map_error_retry_after_from_response_headers() -> None:
    class _Headers:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def get(self, key: str) -> Any:
            return self._data.get(key)

    class _Resp:
        def __init__(self) -> None:
            self.headers = _Headers({"retry-after": "30"})

    err = _FakeRateError("slow")
    err.retry_after = None
    err.response = _Resp()  # type: ignore[attr-defined]
    mapped = LiteLLMProvider._map_error(err)
    assert isinstance(mapped, RateLimited)
    assert mapped.retry_after == 30.0


def test_map_error_ignores_bool_retry_after() -> None:
    err = _FakeRateError("slow", retry_after=True)
    mapped = LiteLLMProvider._map_error(err)
    assert isinstance(mapped, RateLimited)
    assert mapped.retry_after is None


def test_resolve_exc_type_skips_non_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-class / non-exception attribute is rejected during resolution."""
    monkeypatch.setattr(lp.litellm, "NotAnException", "i am a string", raising=False)
    assert lp._resolve_exc_type("NotAnException") is None

    class RealError(Exception):
        pass

    monkeypatch.setattr(lp.litellm, "RealError", RealError, raising=False)
    assert lp._resolve_exc_type("Missing", "RealError") is RealError


# ──────────────────────────────────────────────────────────────────────────────
# Tool-call argument parsing
# ──────────────────────────────────────────────────────────────────────────────
def test_parse_malformed_json_arguments_falls_back_to_raw() -> None:
    calls = LiteLLMProvider._parse_tool_calls([_FakeToolCall("c1", "do", "{not valid json")])
    assert len(calls) == 1
    assert calls[0].arguments == {"_raw": "{not valid json"}


def test_parse_non_object_json_arguments_falls_back_to_raw() -> None:
    # Valid JSON, but a list/scalar — not a dict; must be wrapped in _raw.
    calls = LiteLLMProvider._parse_tool_calls([_FakeToolCall("c1", "do", "[1, 2, 3]")])
    assert calls[0].arguments == {"_raw": "[1, 2, 3]"}


def test_parse_none_arguments_yields_empty_dict() -> None:
    calls = LiteLLMProvider._parse_tool_calls([_FakeToolCall("c1", "do", None)])
    assert calls[0].arguments == {}


def test_parse_dict_arguments_passthrough() -> None:
    calls = LiteLLMProvider._parse_tool_calls([_FakeToolCall("c1", "do", {"a": 1})])
    assert calls[0].arguments == {"a": 1}


def test_parse_valid_json_string_arguments() -> None:
    calls = LiteLLMProvider._parse_tool_calls([_FakeToolCall("c1", "do", '{"a": 1}')])
    assert calls[0].arguments == {"a": 1}


def test_parse_non_string_non_dict_arguments_wrapped() -> None:
    calls = LiteLLMProvider._parse_tool_calls([_FakeToolCall("c1", "do", 12345)])
    assert calls[0].arguments == {"_raw": "12345"}


def test_parse_multiple_tool_calls_preserve_order() -> None:
    calls = LiteLLMProvider._parse_tool_calls(
        [
            _FakeToolCall("a", "first", '{"x": 1}'),
            _FakeToolCall("b", "second", "{bad"),
            _FakeToolCall("c", "third", None),
        ]
    )
    assert [c.id for c in calls] == ["a", "b", "c"]
    assert [c.name for c in calls] == ["first", "second", "third"]
    assert calls[0].arguments == {"x": 1}
    assert calls[1].arguments == {"_raw": "{bad"}
    assert calls[2].arguments == {}


def test_parse_missing_id_and_name_become_empty_strings() -> None:
    calls = LiteLLMProvider._parse_tool_calls([_FakeToolCall(None, None, "{}")])
    assert calls[0].id == ""
    assert calls[0].name == ""


def test_parse_empty_or_none_tool_calls() -> None:
    assert LiteLLMProvider._parse_tool_calls(None) == []
    assert LiteLLMProvider._parse_tool_calls([]) == []


# ──────────────────────────────────────────────────────────────────────────────
# Response normalization edges
# ──────────────────────────────────────────────────────────────────────────────
async def test_normalize_missing_usage_zeroes(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _acompletion(**_kw: Any) -> Any:
        return _make_response(content="hello", usage=None)

    monkeypatch.setattr(lp.litellm, "acompletion", _acompletion)
    result = await provider.acomplete([Message.user("hi")])
    assert result.usage == Usage()
    assert result.usage.total_tokens == 0
    assert result.usage.cost_usd == 0.0


async def test_normalize_missing_hidden_params_zero_cost(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    usage = _FakeUsage(10, 5, 15)

    async def _acompletion(**_kw: Any) -> Any:
        return _make_response(content="hello", usage=usage, hidden=None)

    monkeypatch.setattr(lp.litellm, "acompletion", _acompletion)
    result = await provider.acomplete([Message.user("hi")])
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15
    assert result.usage.cost_usd == 0.0


async def test_normalize_cost_from_hidden_params(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    usage = _FakeUsage(10, 5, 15)

    async def _acompletion(**_kw: Any) -> Any:
        return _make_response(usage=usage, hidden={"response_cost": 0.0042})

    monkeypatch.setattr(lp.litellm, "acompletion", _acompletion)
    result = await provider.acomplete([Message.user("hi")])
    assert result.usage.cost_usd == pytest.approx(0.0042)


async def test_normalize_content_none_becomes_empty_text(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _acompletion(**_kw: Any) -> Any:
        return _make_response(content=None)

    monkeypatch.setattr(lp.litellm, "acompletion", _acompletion)
    result = await provider.acomplete([Message.user("hi")])
    assert result.text == ""


async def test_normalize_no_choices(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _acompletion(**_kw: Any) -> Any:
        return _FakeResponse(choices=[], usage=None)

    monkeypatch.setattr(lp.litellm, "acompletion", _acompletion)
    result = await provider.acomplete([Message.user("hi")])
    assert isinstance(result, LLMResult)
    assert result.text == ""
    assert result.tool_calls == []
    assert result.finish_reason is None


async def test_normalize_choices_not_a_list(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _acompletion(**_kw: Any) -> Any:
        return _FakeResponse(choices=None, usage=None)

    monkeypatch.setattr(lp.litellm, "acompletion", _acompletion)
    result = await provider.acomplete([Message.user("hi")])
    assert result.text == ""
    assert result.tool_calls == []


def test_extract_usage_derives_total_when_missing() -> None:
    usage = LiteLLMProvider._extract_usage(_FakeResponse([], _FakeUsage(7, 3, 0)))
    assert usage.total_tokens == 10


def test_extract_usage_coerces_string_counts() -> None:
    usage = LiteLLMProvider._extract_usage(_FakeResponse([], _FakeUsage("7", "3", "10")))
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (7, 3, 10)


def test_extract_usage_rejects_bool_and_junk() -> None:
    usage = LiteLLMProvider._extract_usage(_FakeResponse([], _FakeUsage(True, "junk", None)))
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_extract_cost_string_value() -> None:
    usage = LiteLLMProvider._extract_usage(
        _FakeResponse([], _FakeUsage(1, 1, 2), hidden={"response_cost": "0.5"})
    )
    assert usage.cost_usd == pytest.approx(0.5)


def test_extract_cost_bool_rejected() -> None:
    usage = LiteLLMProvider._extract_usage(
        _FakeResponse([], _FakeUsage(1, 1, 2), hidden={"response_cost": True})
    )
    assert usage.cost_usd == 0.0


async def test_normalize_preserves_tool_calls_through_acomplete(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_calls = [
        _FakeToolCall("a", "first", '{"x": 1}'),
        _FakeToolCall("b", "second", "{malformed"),
    ]

    async def _acompletion(**_kw: Any) -> Any:
        return _make_response(content=None, tool_calls=tool_calls, finish_reason="tool_calls")

    monkeypatch.setattr(lp.litellm, "acompletion", _acompletion)
    result = await provider.acomplete([Message.user("hi")])
    assert result.has_tool_calls
    assert [c.id for c in result.tool_calls] == ["a", "b"]
    assert result.tool_calls[1].arguments == {"_raw": "{malformed"}
    assert result.finish_reason == "tool_calls"


# ──────────────────────────────────────────────────────────────────────────────
# count_tokens fallback
# ──────────────────────────────────────────────────────────────────────────────
def test_count_tokens_uses_litellm(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lp.litellm, "token_counter", lambda **_kw: 123)
    assert provider.count_tokens([Message.user("hi")]) == 123


def test_count_tokens_falls_back_when_counter_raises(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(**_kw: Any) -> int:
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr(lp.litellm, "token_counter", _boom)
    count = provider.count_tokens([Message.user("hello world")])
    assert count > 0  # rough estimate, never a crash


def test_count_tokens_falls_back_on_non_numeric(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lp.litellm, "token_counter", lambda **_kw: "not a number")
    assert provider.count_tokens([Message.user("hello")]) > 0


def test_count_tokens_falls_back_on_negative(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lp.litellm, "token_counter", lambda **_kw: -5)
    assert provider.count_tokens([Message.user("hello")]) > 0


def test_count_tokens_falls_back_on_bool(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    # bool is an int subclass; must not be accepted as a real count.
    monkeypatch.setattr(lp.litellm, "token_counter", lambda **_kw: True)
    assert provider.count_tokens([Message.user("hello world")]) > 0


# ──────────────────────────────────────────────────────────────────────────────
# capabilities() probe
# ──────────────────────────────────────────────────────────────────────────────
def test_capabilities_probe_failure_is_swallowed(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(**_kw: Any) -> bool:
        raise RuntimeError("offline")

    monkeypatch.setattr(lp.litellm, "supports_function_calling", _boom)
    caps = provider.capabilities()
    assert isinstance(caps, Capabilities)
    assert caps.supports_tools is True  # gpt-4o static entry, probe failure ignored


def test_capabilities_probe_disables_tools(
    provider: LiteLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lp.litellm, "supports_function_calling", lambda **_kw: False)
    caps = provider.capabilities()
    assert caps.supports_tools is False


# ──────────────────────────────────────────────────────────────────────────────
# registry.get_capabilities — safe defaults, never raises
# ──────────────────────────────────────────────────────────────────────────────
def test_registry_unknown_model_safe_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the dynamic litellm lookup to contribute nothing.
    monkeypatch.setattr("zakcode.providers.registry._from_litellm", lambda _model: None)
    caps = get_capabilities("totally-made-up-model-xyz")
    assert caps.context_window == 8192
    assert caps.supports_tools is True


def test_registry_known_model() -> None:
    caps = get_capabilities("gpt-4o")
    assert caps.context_window == 128_000
    assert caps.supports_vision is True


def test_registry_prefixed_known_model() -> None:
    caps = get_capabilities("openai/gpt-4o")
    assert caps.context_window == 128_000


def test_registry_empty_and_blank_model() -> None:
    assert get_capabilities("").context_window == 8192
    assert get_capabilities("   ").context_window == 8192


def test_registry_non_string_model_does_not_raise() -> None:
    caps = get_capabilities(None)  # type: ignore[arg-type]
    assert caps.context_window == 8192


def test_registry_lookup_exception_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_model: str) -> Any:
        raise RuntimeError("lookup exploded")

    monkeypatch.setattr("zakcode.providers.registry._lookup_static", _boom)
    caps = get_capabilities("anything")
    assert caps.context_window == 8192

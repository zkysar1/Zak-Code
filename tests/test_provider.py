"""Hermetic tests for the litellm-backed provider.

litellm is monkeypatched throughout; no network or live model is ever touched.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.messages import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from zakcode.providers import (
    AuthError,
    ContextWindowExceeded,
    LiteLLMProvider,
    RateLimited,
    RequestFailed,
    get_capabilities,
)


# ---------------------------------------------------------------------------
# Fake litellm response objects (attribute-style, like litellm returns)
# ---------------------------------------------------------------------------
class _Obj:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def model_dump(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            out[k] = v
        return out


def _make_response(
    *,
    content: str | None = "",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    prompt: int = 10,
    completion: int = 5,
    total: int = 15,
    cost: float | None = 0.0012,
) -> _Obj:
    message = _Obj(content=content, tool_calls=tool_calls)
    choice = _Obj(message=message, finish_reason=finish_reason)
    usage = _Obj(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
    hidden = {} if cost is None else {"response_cost": cost}
    return _Obj(choices=[choice], usage=usage, _hidden_params=hidden)


def _tool_call(call_id: str, name: str, arguments: str) -> _Obj:
    return _Obj(id=call_id, type="function", function=_Obj(name=name, arguments=arguments))


def _provider() -> LiteLLMProvider:
    return LiteLLMProvider(model="openai/gpt-4o", temperature=0.3)


# ---------------------------------------------------------------------------
# Message translation
# ---------------------------------------------------------------------------
def test_translate_basic_messages_with_system() -> None:
    wire = LiteLLMProvider._translate_messages([Message.user("hello")], system="be terse")
    assert wire[0] == {"role": "system", "content": "be terse"}
    assert wire[1] == {"role": "user", "content": "hello"}


def test_translate_assistant_tool_use() -> None:
    assistant = Message(
        role="assistant",
        blocks=[
            TextBlock(text="calling"),
            ToolUseBlock(id="call_1", name="read_file", input={"path": "a.txt"}),
        ],
    )
    wire = LiteLLMProvider._translate_messages([assistant], system=None)
    msg = wire[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "calling"
    assert msg["tool_calls"][0]["id"] == "call_1"
    assert msg["tool_calls"][0]["type"] == "function"
    assert msg["tool_calls"][0]["function"]["name"] == "read_file"
    # arguments must be a JSON string, not a dict
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "a.txt"}


def test_translate_assistant_tool_only_content_none() -> None:
    assistant = Message(
        role="assistant",
        blocks=[ToolUseBlock(id="c1", name="ls", input={})],
    )
    wire = LiteLLMProvider._translate_messages([assistant], system=None)
    assert wire[0]["content"] is None
    assert "tool_calls" in wire[0]


def test_translate_tool_results() -> None:
    tool_msg = Message.tool_results(
        [
            ToolResultBlock(tool_use_id="call_1", output="file contents"),
            ToolResultBlock(tool_use_id="call_2", output="more"),
        ]
    )
    wire = LiteLLMProvider._translate_messages([tool_msg], system=None)
    assert wire[0] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "file contents",
    }
    assert wire[1]["tool_call_id"] == "call_2"


def test_translate_system_role_message() -> None:
    wire = LiteLLMProvider._translate_messages(
        [Message.system("from message"), Message.user("hi")], system=None
    )
    assert wire[0] == {"role": "system", "content": "from message"}
    assert wire[1] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# LLMResult normalization
# ---------------------------------------------------------------------------
async def test_acomplete_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _Obj:
        captured.update(kwargs)
        return _make_response(content="hi there", tool_calls=None)

    monkeypatch.setattr(lp.litellm, "acompletion", fake_acompletion)

    result = await _provider().acomplete([Message.user("hello")], system="sys")

    assert result.text == "hi there"
    assert result.tool_calls == []
    assert result.has_tool_calls is False
    assert result.finish_reason == "stop"
    # usage + cost captured
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15
    assert result.usage.cost_usd == pytest.approx(0.0012)
    assert result.raw is not None
    # call shape
    assert captured["model"] == "openai/gpt-4o"
    assert captured["temperature"] == 0.3
    assert captured["stream"] is False
    assert captured["drop_params"] is True
    # no tools -> tool_choice not sent
    assert "tool_choice" not in captured
    assert captured["messages"][0] == {"role": "system", "content": "sys"}


async def test_acomplete_with_tools_sends_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _Obj:
        captured.update(kwargs)
        return _make_response(content="", tool_calls=None)

    monkeypatch.setattr(lp.litellm, "acompletion", fake_acompletion)

    tools = [{"type": "function", "function": {"name": "ls", "parameters": {}}}]
    await _provider().acomplete([Message.user("go")], tools=tools)

    assert captured["tools"] == tools
    assert captured["tool_choice"] == "auto"


async def test_acomplete_forwards_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _Obj:
        captured.update(kwargs)
        return _make_response(content="{}", tool_calls=None)

    monkeypatch.setattr(lp.litellm, "acompletion", fake_acompletion)
    rf = {"type": "json_object"}
    await _provider().acomplete([Message.user("go")], response_format=rf)
    assert captured["response_format"] == rf


async def test_acomplete_omits_response_format_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Backward-compat guard: a normal call must NOT send a response_format key (drop-in safe).
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _Obj:
        captured.update(kwargs)
        return _make_response(content="hi", tool_calls=None)

    monkeypatch.setattr(lp.litellm, "acompletion", fake_acompletion)
    await _provider().acomplete([Message.user("go")])
    assert "response_format" not in captured


def test_build_kwargs_response_format_present_and_absent() -> None:
    p = _provider()
    wire = [{"role": "user", "content": "x"}]
    assert "response_format" not in p._build_kwargs(wire, None)
    rf = {"type": "json_schema", "json_schema": {"name": "o", "schema": {}, "strict": True}}
    assert p._build_kwargs(wire, None, response_format=rf)["response_format"] == rf


def test_build_kwargs_sets_request_timeout() -> None:
    # A per-call timeout is ALWAYS on the wire so a hung model call can't block the loop forever;
    # the default is a generous safety-net ceiling, and it is overridable per provider.
    p = _provider()
    wire = [{"role": "user", "content": "x"}]
    assert p._build_kwargs(wire, None)["timeout"] == 600.0
    p2 = LiteLLMProvider(model="openai/gpt-4o", request_timeout=42.0)
    assert p2._build_kwargs(wire, None)["timeout"] == 42.0


def test_request_timeout_resolves_from_settings() -> None:
    # Settings.request_timeout (ZAKCODE_REQUEST_TIMEOUT) reaches the wire — the knob a
    # self-hosted pod needs, where prefill alone can outlast the hosted-API default.
    from zakcode.config import load_settings

    wire = [{"role": "user", "content": "x"}]
    s = load_settings(default_model="openai/gpt-4o", request_timeout=1200.0)
    p = LiteLLMProvider(s)
    assert p._build_kwargs(wire, None)["timeout"] == 1200.0
    # Explicit kwarg wins over settings (same precedence as every other field).
    p2 = LiteLLMProvider(s, request_timeout=42.0)
    assert p2._build_kwargs(wire, None)["timeout"] == 42.0
    # No settings, no kwarg: the 600s safety net holds.
    assert LiteLLMProvider(model="openai/gpt-4o")._build_kwargs(wire, None)["timeout"] == 600.0


async def test_acomplete_parses_tool_calls_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(**kwargs: Any) -> _Obj:
        return _make_response(
            content=None,
            tool_calls=[_tool_call("call_99", "read_file", '{"path": "x.py"}')],
            finish_reason="tool_calls",
        )

    monkeypatch.setattr(lp.litellm, "acompletion", fake_acompletion)

    result = await _provider().acomplete([Message.user("read x")])

    assert result.has_tool_calls is True
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "call_99"
    assert call.name == "read_file"
    assert call.arguments == {"path": "x.py"}
    assert result.finish_reason == "tool_calls"


async def test_acomplete_tool_call_bad_json_falls_back_to_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(**kwargs: Any) -> _Obj:
        return _make_response(
            content=None,
            tool_calls=[_tool_call("c1", "bad", "{not valid json")],
        )

    monkeypatch.setattr(lp.litellm, "acompletion", fake_acompletion)

    result = await _provider().acomplete([Message.user("x")])

    assert result.tool_calls[0].arguments == {"_raw": "{not valid json"}


async def test_acomplete_tool_call_non_object_json_falls_back_to_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Valid JSON but not an object (a list) -> _raw fallback.
    async def fake_acompletion(**kwargs: Any) -> _Obj:
        return _make_response(
            content=None,
            tool_calls=[_tool_call("c1", "f", "[1, 2, 3]")],
        )

    monkeypatch.setattr(lp.litellm, "acompletion", fake_acompletion)

    result = await _provider().acomplete([Message.user("x")])
    assert result.tool_calls[0].arguments == {"_raw": "[1, 2, 3]"}


async def test_acomplete_missing_cost_defaults_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(**kwargs: Any) -> _Obj:
        return _make_response(content="ok", cost=None)

    monkeypatch.setattr(lp.litellm, "acompletion", fake_acompletion)

    result = await _provider().acomplete([Message.user("x")])
    assert result.usage.cost_usd == 0.0


def test_extract_usage_recovers_cloud_cost_when_response_cost_missing() -> None:
    # PROV-12: a streaming chunk carries no response_cost; for a non-Groq cloud model the Groq
    # fallback is also 0, so cost must be recomputed from litellm's price map -- else /cost and the
    # max_cost_usd ceiling silently read $0 on the streaming path.
    chunk = _Obj(
        usage=_Obj(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        _hidden_params={"response_cost": None},
        model="gpt-4o-mini",
    )
    usage = lp.LiteLLMProvider._extract_usage(chunk)
    assert usage.cost_usd > 0.0


def test_map_error_retries_transient_infra_errors() -> None:
    # PROV-10: transient infra errors (timeout, dropped connection, 503/500) must map to a
    # retryable error so the loop's backoff retry handles them instead of killing the turn.
    for name in ("Timeout", "APIConnectionError", "ServiceUnavailableError", "InternalServerError"):
        exc = type(name, (Exception,), {})("boom")
        mapped = lp.LiteLLMProvider._map_error(exc)
        assert isinstance(mapped, lp.RateLimited), f"{name} -> {type(mapped).__name__}"


def test_map_error_names_timeouts_truthfully() -> None:
    # A client-timeout is retryable (subclasses RateLimited) but must be distinguishable:
    # "rate limited" sent the operator to the wrong knob during the zc-03 boot wedges —
    # the real remedy was ZAKCODE_REQUEST_TIMEOUT on an uncached slow backend.
    from zakcode.providers.base import TimedOut

    for name in ("Timeout", "APITimeoutError"):
        exc = type(name, (Exception,), {})("boom")
        mapped = lp.LiteLLMProvider._map_error(exc)
        assert isinstance(mapped, TimedOut), f"{name} -> {type(mapped).__name__}"
    for name in ("APIConnectionError", "ServiceUnavailableError", "InternalServerError"):
        exc = type(name, (Exception,), {})("boom")
        mapped = lp.LiteLLMProvider._map_error(exc)
        assert not isinstance(mapped, TimedOut), f"{name} must stay a plain RateLimited"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------
class _FakeAuthError(Exception):
    pass


class _FakeContextWindowError(Exception):
    pass


class _FakeRateLimitError(Exception):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_FakeAuthError("nope"), AuthError),
        (_FakeContextWindowError("too big"), ContextWindowExceeded),
        (RuntimeError("boom"), RequestFailed),
    ],
)
async def test_error_mapping(
    monkeypatch: pytest.MonkeyPatch, exc: Exception, expected: type
) -> None:
    # Point the resolved exception classes at our fakes so isinstance works.
    monkeypatch.setattr(lp, "_LiteLLMAuthError", _FakeAuthError)
    monkeypatch.setattr(lp, "_LiteLLMContextWindowError", _FakeContextWindowError)
    monkeypatch.setattr(lp, "_LiteLLMRateLimitError", _FakeRateLimitError)

    async def fake_acompletion(**kwargs: Any) -> Any:
        raise exc

    monkeypatch.setattr(lp.litellm, "acompletion", fake_acompletion)

    with pytest.raises(expected):
        await _provider().acomplete([Message.user("x")])


async def test_rate_limit_mapping_carries_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lp, "_LiteLLMRateLimitError", _FakeRateLimitError)

    async def fake_acompletion(**kwargs: Any) -> Any:
        raise _FakeRateLimitError("slow down", retry_after=2.5)

    monkeypatch.setattr(lp.litellm, "acompletion", fake_acompletion)

    with pytest.raises(RateLimited) as info:
        await _provider().acomplete([Message.user("x")])
    assert info.value.retry_after == 2.5


def test_error_mapping_by_class_name_fallback() -> None:
    # When the resolved classes are unavailable, fall back to name matching.
    class AuthenticationError(Exception):
        pass

    mapped = LiteLLMProvider._map_error(AuthenticationError("x"))
    assert isinstance(mapped, AuthError)


def test_error_mapping_missing_google_stack_names_the_extra() -> None:
    # Vertex AI without the optional google extra: litellm lets the raw ImportError
    # propagate ("No module named 'google'"), which names no remedy. The mapping must
    # surface the install command — this is the live 2026-08-19 failure shape (a fresh
    # zakcode install + a vertex_ai model on a GCE box).
    exc = ModuleNotFoundError("No module named 'google'", name="google")
    mapped = LiteLLMProvider._map_error(exc)
    assert isinstance(mapped, RequestFailed)
    assert "zakcode[google]" in str(mapped)
    # Submodules of the google namespace get the same hint...
    sub = ModuleNotFoundError("No module named 'google.auth'", name="google.auth")
    assert "zakcode[google]" in str(LiteLLMProvider._map_error(sub))
    # ...but an unrelated missing module must NOT get a misleading google hint.
    other = ModuleNotFoundError("No module named 'left_pad'", name="left_pad")
    mapped_other = LiteLLMProvider._map_error(other)
    assert isinstance(mapped_other, RequestFailed)
    assert "zakcode[google]" not in str(mapped_other)


def test_error_mapping_groq_tool_use_failed_is_retryable() -> None:
    # Groq rejects a malformed model-emitted tool call with code "tool_use_failed".
    # It reaches us in two shapes; BOTH must map to the retryable ModelOutputRejected
    # (a RateLimited subclass, retry_after=0) with the real cause in the message —
    # never a dead turn showing a parser crash.
    from zakcode.providers.base import ModelOutputRejected

    # Shape 1: litellm's own error mapping crashes parsing the string code.
    crash = ValueError("invalid literal for int() with base 10: 'tool_use_failed'")
    mapped = LiteLLMProvider._map_error(crash)
    assert isinstance(mapped, ModelOutputRejected)
    assert isinstance(mapped, RateLimited)  # rides the loop's retry machinery
    assert mapped.retry_after == 0.0  # nothing to wait for — retry immediately
    assert "malformed tool call" in str(mapped)
    assert "invalid literal" not in str(mapped)  # parser crash never shown

    # Shape 2: a genuine BadRequestError carrying the groq error body.
    class BadRequestError(Exception):
        pass

    body = BadRequestError('{"error":{"code":"tool_use_failed","failed_generation":"..."}}')
    assert isinstance(LiteLLMProvider._map_error(body), ModelOutputRejected)


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------
def test_count_tokens_uses_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_counter(*, model: str, messages: list[dict[str, Any]]) -> int:
        seen["model"] = model
        seen["messages"] = messages
        return 42

    monkeypatch.setattr(lp.litellm, "token_counter", fake_counter)

    n = _provider().count_tokens([Message.user("hi")], system="sys")
    assert n == 42
    assert seen["model"] == "openai/gpt-4o"
    assert seen["messages"][0]["role"] == "system"


def test_count_tokens_fallback_when_litellm_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(**kwargs: Any) -> int:
        raise RuntimeError("offline")

    monkeypatch.setattr(lp.litellm, "token_counter", boom)

    n = _provider().count_tokens([Message.user("hello world")])
    assert n > 0  # rough estimate, never raises


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------
def test_capabilities_known_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lp.litellm, "supports_function_calling", lambda model: True)
    caps = LiteLLMProvider(model="gpt-4o").capabilities()
    assert caps.context_window == 128_000
    assert caps.supports_tools is True


def test_capabilities_gate_disables_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lp.litellm, "supports_function_calling", lambda model: False)
    caps = LiteLLMProvider(model="gpt-4o").capabilities()
    assert caps.supports_tools is False


def test_capabilities_probe_failure_is_tolerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(model: str) -> bool:
        raise RuntimeError("no")

    monkeypatch.setattr(lp.litellm, "supports_function_calling", boom)
    caps = LiteLLMProvider(model="gpt-4o").capabilities()
    assert caps.supports_tools is True  # default retained


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_get_capabilities_static_openai() -> None:
    caps = get_capabilities("gpt-4o-mini")
    assert caps.context_window == 128_000
    assert caps.supports_vision is True


def test_get_capabilities_strips_provider_prefix() -> None:
    caps = get_capabilities("openai/gpt-4o")
    assert caps.context_window == 128_000


def test_get_capabilities_ollama() -> None:
    caps = get_capabilities("ollama_chat/llama3.1")
    assert caps.supports_tools is True
    assert caps.context_window == 128_000


def test_get_capabilities_ollama_bare_prefix_matches_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    # 'ollama/X' and 'ollama_chat/X' name the same backend model and must resolve to the
    # same window via the static table. Force litellm metadata off so the assertion proves
    # the registry's prefix normalization, not a metadata coincidence. (audit #11)
    import zakcode.providers.registry as reg

    monkeypatch.setattr(reg, "_from_litellm", lambda model: None)
    chat = get_capabilities("ollama_chat/qwen2.5").context_window
    assert chat == 32_768
    assert get_capabilities("ollama/qwen2.5").context_window == chat
    assert get_capabilities("ollama/qwen2.5:3b").context_window == chat  # tagged variant too


def test_get_capabilities_unknown_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zakcode.providers.registry as reg

    # Force the litellm metadata lookup to yield nothing so we hit the default.
    monkeypatch.setattr(reg, "_from_litellm", lambda model: None)
    caps = get_capabilities("totally-made-up-model-xyz")
    assert caps.context_window == 8192
    assert caps.supports_tools is True


# ---------------------------------------------------------------------------
# Ollama env wiring
# ---------------------------------------------------------------------------
def test_ollama_sets_env_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    LiteLLMProvider(model="ollama_chat/llama3.1", ollama_base_url="http://host:11434")
    import os

    assert os.environ["OLLAMA_API_BASE"] == "http://host:11434"


# ---------------------------------------------------------------------------
# Traffic smoothing (_pace) — ADR-0014


@pytest.mark.asyncio
async def test_pace_spaces_rapid_request_starts(monkeypatch) -> None:
    """A request starting inside the interval sleeps out the remainder. The previous
    start is planted directly (patching time.monotonic globally breaks the event
    loop, which reads the same clock)."""
    import time as _time

    provider = _provider()
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(lp.asyncio, "sleep", fake_sleep)
    provider._last_request_started = _time.monotonic()  # a request JUST started
    await provider._pace()
    assert len(slept) == 1
    assert 0.9 <= slept[0] <= lp._MIN_REQUEST_INTERVAL_S


@pytest.mark.asyncio
async def test_pace_never_delays_after_a_normal_length_call(monkeypatch) -> None:
    import time as _time

    provider = _provider()

    async def fail_sleep(delay: float) -> None:  # pragma: no cover — must never run
        raise AssertionError(f"paced a non-burst request by {delay}s")

    monkeypatch.setattr(lp.asyncio, "sleep", fail_sleep)
    provider._last_request_started = 0.0
    await provider._pace()  # first request of the instance: never paced
    provider._last_request_started = _time.monotonic() - 5.0  # a normal-length call ago
    await provider._pace()


# ---------------------------------------------------------------------------
# Traffic smoothing under CONCURRENCY (shared provider instance)
#
# Sub-agents reuse the parent provider (subagent.py) and independent subtasks run
# concurrently via asyncio.gather (task.py), so several _pace() calls are in flight
# on ONE instance. Unsynchronized, they all read the same _last_request_started,
# compute the same wait, and wake together — the burst fires simultaneously and the
# smoothing does nothing in exactly the scenario it exists for.
#
# The harness below runs the competing calls in ONE uninterrupted gather with the
# REAL event-loop clock and the REAL asyncio.sleep: no fake_sleep, no per-call await
# in the test body, nothing that could serialize the callers for free. The interval
# is monkeypatched small purely to keep the test fast — the production constant is
# untouched (no knobs), which test_pace_interval_constant_is_not_configurable pins.
#
# _UnlockedPacer reimplements the pre-fix algorithm verbatim and is driven through
# the SAME harness as the negative control. Without a variant that FAILS, the
# spacing assertion would only describe the assertion's own shape, not the fix.
# ---------------------------------------------------------------------------
_TEST_INTERVAL_S = 0.05


class _UnlockedPacer:
    """The pre-fix _pace(), verbatim: no mutual exclusion. Negative control."""

    def __init__(self) -> None:
        self._last_request_started = 0.0

    async def _pace(self) -> None:
        import asyncio as _asyncio
        import time as _time

        wait = self._last_request_started + _TEST_INTERVAL_S - _time.monotonic()
        if wait > 0:
            await _asyncio.sleep(wait)
        self._last_request_started = _time.monotonic()


async def _gather_start_times(pacer: Any, n: int) -> list[float]:
    """Fire n _pace() calls concurrently; return the instant each one released."""
    import asyncio as _asyncio
    import time as _time

    starts: list[float] = []

    async def one() -> None:
        await pacer._pace()
        starts.append(_time.monotonic())

    # Seed a just-started request so every caller below is inside the interval.
    pacer._last_request_started = _time.monotonic()
    await _asyncio.gather(*(one() for _ in range(n)))
    return sorted(starts)


def _min_gap(starts: list[float]) -> float:
    return min(b - a for a, b in zip(starts, starts[1:], strict=False))


@pytest.mark.asyncio
async def test_pace_spaces_concurrent_starts_on_a_shared_instance(monkeypatch) -> None:
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", _TEST_INTERVAL_S)

    starts = await _gather_start_times(_provider(), 4)
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]

    # Every consecutive pair is at least one interval apart (10% slack for loop jitter).
    assert _min_gap(starts) >= _TEST_INTERVAL_S * 0.9, f"concurrent starts collapsed: {gaps=}"


@pytest.mark.asyncio
async def test_unlocked_pacer_collapses_under_the_same_harness() -> None:
    """NEGATIVE CONTROL for the test above.

    Same harness, same concurrency, same clock — only the lock is missing. If this
    ever starts passing, the harness has acquired a serialization point of its own
    and the sibling test above has stopped proving anything.
    """
    starts = await _gather_start_times(_UnlockedPacer(), 4)

    assert _min_gap(starts) < _TEST_INTERVAL_S * 0.5, (
        "the unlocked pacer spaced its starts — the harness is serializing callers "
        "for free, so the locked test is no longer evidence for the lock"
    )


@pytest.mark.asyncio
async def test_pace_lock_does_not_serialize_the_requests_themselves(monkeypatch) -> None:
    """Only request STARTS queue; the in-flight requests still overlap."""
    import asyncio as _asyncio

    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", _TEST_INTERVAL_S)

    in_flight = 0
    peak = 0

    async def fake_acompletion(**kwargs: Any) -> _Obj:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await _asyncio.sleep(_TEST_INTERVAL_S * 4)  # a normal-length call
        in_flight -= 1
        return _make_response(content="ok", tool_calls=None)

    monkeypatch.setattr(lp.litellm, "acompletion", fake_acompletion)

    provider = _provider()
    await _asyncio.gather(*(provider.acomplete([Message.user("hi")]) for _ in range(3)))

    assert peak > 1, "the pace lock is being held across the request, not just the start"


def test_pace_interval_constant_is_not_configurable() -> None:
    """No knobs (the smoothing is a constant, never settings-driven)."""
    assert lp._MIN_REQUEST_INTERVAL_S == 1.0
    import inspect

    src = inspect.getsource(lp)
    assert "ZAKCODE_MIN_REQUEST_INTERVAL" not in src

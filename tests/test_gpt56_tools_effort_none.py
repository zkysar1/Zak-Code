"""OpenAI's gpt-5.6 tier takes function tools on the chat route ONLY with
``reasoning_effort="none"`` (ADR-0188).

WHY THIS FILE EXISTS
Every served Mind since the zakpick OpenAI mix landed (2026-09-01) ran on the
gpt-5-mini FALLBACK, not on the terra/luna it was pinned to: the first tool call of
each session 400'd — "Function tools with reasoning_effort are not supported for
gpt-5.6-terra in /v1/chat/completions. To use function tools, use /v1/responses or
set reasoning_effort to 'none'." — and the runtime failover moved the session to
the fallback for good (serve.log, 167 vessel sessions, 100% on the fallback; the
terra usage line never appeared once). A request that never mentions
reasoning_effort is refused too: the model's DEFAULT depth counts as "with".

Measured live against the fleet key, 2026-09-17: terra unset -> 400, terra low ->
400, terra none -> 200 with a tool call, luna none -> 200, gpt-5-mini none -> 400
("does not support 'none'"), gpt-5-mini unset -> 200. So the rule is the 5.6 tier
only, never the fallback tier, and it is sent only WITH tools.

Two halves, both hermetic. The predicate half pre-sets the flag from the model
name so the first tool call already has the accepted shape. The re-issue half
latches the flag from the provider's own remedy text, so a tier the predicate does
not know costs one re-issued call — never a failover to a model nobody chose.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.providers.base import RequestFailed
from zakcode.providers.litellm_provider import (
    LiteLLMProvider,
    _is_openai_gpt56_tools_effort_none_model,
)

MSGS = [{"role": "user", "content": "hi"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "report the answer",
            "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}},
        },
    }
]
REMEDY_400 = (
    "litellm.BadRequestError: OpenAIException - Function tools with reasoning_effort are "
    "not supported for gpt-5.7-nova in /v1/chat/completions. To use function tools, use "
    "/v1/responses or set reasoning_effort to 'none'."
)


# ── the predicate ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-5.6-terra", True),
        ("openai/gpt-5.6-luna", True),
        ("gpt-5.6-terra", True),
        ("openai/gpt-5.6-chat", False),  # not measured -> the re-issue half covers it
        ("openai/gpt-5-mini", False),  # the fallback tier refuses 'none' (measured)
        ("openai/gpt-5-nano", False),
        ("openai/gpt-5", False),
        ("openai/gpt-4o-mini", False),
        ("anthropic/claude-sonnet-5", False),
    ],
)
def test_predicate_is_the_gpt56_tier_only(model: str, expected: bool) -> None:
    assert _is_openai_gpt56_tools_effort_none_model(model) is expected


# ── request shape ────────────────────────────────────────────────────────────────


def test_tools_on_terra_send_effort_none() -> None:
    # THE DEFAULT, unchanged by ADR-0200: nothing configured, so the request would carry
    # no depth at all — which is the 400 — and 'none' is what it gets.
    p = LiteLLMProvider(model="openai/gpt-5.6-terra", context_window=400000)
    assert p.tools_require_effort_none is True
    assert p._build_kwargs(MSGS, TOOLS)["reasoning_effort"] == "none"


def test_no_tools_on_terra_leaves_the_depth_alone() -> None:
    p = LiteLLMProvider(model="openai/gpt-5.6-terra", context_window=400000)
    assert "reasoning_effort" not in p._build_kwargs(MSGS, None)


def test_a_configured_depth_rides_with_tools(effort: str = "low") -> None:
    """ADR-0200. What the request needs is an EXPLICIT depth, not the value 'none': any
    explicit level takes the call off the chat route onto /v1/responses, which accepts
    tools at any depth. Measured on that route over two pre-registered batches."""
    p = LiteLLMProvider(model="openai/gpt-5.6-luna", context_window=400000)
    assert p._build_kwargs(MSGS, TOOLS, reasoning_effort=effort)["reasoning_effort"] == effort


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
def test_every_configured_depth_rides(effort: str) -> None:
    p = LiteLLMProvider(model="openai/gpt-5.6-luna", reasoning_effort=effort, context_window=400000)
    assert p._build_kwargs(MSGS, TOOLS)["reasoning_effort"] == effort


def test_a_server_that_asked_for_none_keeps_it_over_a_configured_depth() -> None:
    """The one case where the override survives: the provider's own 4xx named 'none' as
    the remedy. That refusal is measured for THAT tier, and a depth would 400 every call.
    The predicate-known tier is ours, and there the depth is measured to work."""
    p = LiteLLMProvider(model="openai/gpt-5.6-luna", reasoning_effort="high", context_window=400000)
    assert p._build_kwargs(MSGS, TOOLS)["reasoning_effort"] == "high"
    p._effort_none_demanded_by_server = True
    assert p._build_kwargs(MSGS, TOOLS)["reasoning_effort"] == "none"


def test_the_fallback_tier_is_untouched() -> None:
    # gpt-5-mini refuses 'none' (measured), so the rule must never reach it.
    p = LiteLLMProvider(model="openai/gpt-5-mini", context_window=400000)
    assert p.tools_require_effort_none is False
    assert "reasoning_effort" not in p._build_kwargs(MSGS, TOOLS)


def test_a_local_gpt56_named_server_is_exempt() -> None:
    p = LiteLLMProvider(
        model="openai/gpt-5.6-terra",
        api_base="http://10.0.0.205:9090/v1",
        api_key="local",
        context_window=32000,
    )
    assert "reasoning_effort" not in p._build_kwargs(MSGS, TOOLS)


# ── the re-issue seam ────────────────────────────────────────────────────────────


class _FakeBadRequest(Exception):
    status_code = 400


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self) -> None:
        self.id = "call_1"
        self.function = _FakeFunction("answer", '{"value": 4}')


class _FakeMessage:
    content = None
    tool_calls = [_FakeToolCall()]


class _FakeChoice:
    message = _FakeMessage()
    finish_reason = "tool_calls"


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = _FakeUsage()
    model = "gpt-5.7-nova"


def _refuse_then_accept(seen: list[dict[str, Any]]) -> Any:
    async def _acompletion(**kw: Any) -> Any:
        seen.append(dict(kw))
        if len(seen) == 1:
            raise _FakeBadRequest(REMEDY_400)
        return _FakeResponse()

    return _acompletion


async def test_a_remedy_400_is_reissued_once_on_the_same_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(lp.litellm, "acompletion", _refuse_then_accept(seen))
    p = LiteLLMProvider(model="openai/gpt-5.7-nova", context_window=400000)
    assert p.tools_require_effort_none is False  # the predicate does not know this tier

    result = await p.acomplete([lp.Message.user("2+2?")], tools=TOOLS)

    assert len(seen) == 2, "exactly one re-issue"
    assert "reasoning_effort" not in seen[0]
    assert seen[1]["reasoning_effort"] == "none"
    assert seen[0]["model"] == seen[1]["model"] == "openai/gpt-5.7-nova", "never a failover"
    assert p.tools_require_effort_none is True, "latched for the rest of the session"
    assert p._effort_none_demanded_by_server is True, "the server itself named the remedy"
    assert [c.name for c in result.tool_calls] == ["answer"]


async def test_after_a_remedy_400_a_configured_depth_is_no_longer_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The safety net ADR-0200 keeps: once a provider has answered a tool call by asking
    for 'none', every later call of this session sends 'none', configured depth or not —
    otherwise the operator's depth would buy one 400 and one re-issue on every turn."""
    # On a tier the predicate KNOWS, so the configured depth reaches the wire on the
    # first call — which is the whole point of ADR-0200 and the precondition for this
    # net to be needed at all.
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(lp.litellm, "acompletion", _refuse_then_accept(seen))
    p = LiteLLMProvider(model="openai/gpt-5.6-luna", reasoning_effort="high", context_window=400000)

    await p.acomplete([lp.Message.user("2+2?")], tools=TOOLS)

    assert seen[0]["reasoning_effort"] == "high", "the configured depth went out first"
    assert seen[1]["reasoning_effort"] == "none", "the re-issue takes the named remedy"
    assert p._build_kwargs(MSGS, TOOLS)["reasoning_effort"] == "none", "and every later call"
    assert p.reasoning_effort == "high", "the operator's configuration is left intact"


async def test_a_second_refusal_is_reported_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def _always_refuse(**_kw: Any) -> Any:
        nonlocal calls
        calls += 1
        raise _FakeBadRequest(REMEDY_400)

    monkeypatch.setattr(lp.litellm, "acompletion", _always_refuse)
    p = LiteLLMProvider(model="openai/gpt-5.7-nova", context_window=400000)
    with pytest.raises(RequestFailed):
        await p.acomplete([lp.Message.user("2+2?")], tools=TOOLS)
    assert calls == 2, "one re-issue, then the taxonomy — never the same 400 twice"


async def test_the_remedy_text_without_tools_is_not_latched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No tools in the request means the pairing cannot be the cause; the 400 maps
    # through the taxonomy as before and the flag stays off.
    monkeypatch.setattr(lp.litellm, "acompletion", _refuse_then_accept([]))
    p = LiteLLMProvider(model="openai/gpt-5.7-nova", context_window=400000)
    with pytest.raises(RequestFailed):
        await p.acomplete([lp.Message.user("2+2?")])
    assert p.tools_require_effort_none is False


# ── the depth on the wire ────────────────────────────────────────────────────────
# Setting a kwarg is not sending it. litellm decides which API the call reaches and
# how the depth is rendered there, so the request it actually builds is asserted here
# against a canned response — no network, no key. If an upgrade reroutes this call or
# stops rendering the depth, this fails instead of silently running at the old depth.

_RESPONSES_BODY: dict[str, Any] = {
    "id": "resp_test",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-5.6-luna",
    "output": [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "answer",
            "arguments": '{"value": 4}',
            "status": "completed",
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "error": None,
    "incomplete_details": None,
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 3},
    },
}


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> httpx.Response:
        seen.append({"path": request.url.path, "body": json.loads(request.content.decode("utf-8"))})
        return httpx.Response(200, json=_RESPONSES_BODY, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    return seen


@pytest.mark.parametrize("effort", ["none", "low"])
async def test_the_depth_reaches_the_responses_api_with_the_tool_call(
    wire: list[dict[str, Any]], effort: str
) -> None:
    """Both the default and a configured depth take the SAME route — which is why the
    value is free to change. litellm bridges a gpt-5.4+ chat call carrying tools and any
    non-None effort onto /v1/responses, and 'none' counts (ADR-0200)."""
    configured = None if effort == "none" else effort
    p = LiteLLMProvider(model="openai/gpt-5.6-luna", reasoning_effort=configured)
    result = await p.acomplete([lp.Message.user("2+2?")], tools=TOOLS)

    assert [c.name for c in result.tool_calls] == ["answer"]
    assert len(wire) == 1
    assert wire[0]["path"] == "/v1/responses", (
        "litellm no longer bridges this call: the 400 ADR-0188 was written against is a "
        "CHAT-route refusal, so a call that stays on chat carries the model's default "
        "depth and is refused"
    )
    assert wire[0]["body"]["reasoning"] == {"effort": effort}

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

from typing import Any

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
    p = LiteLLMProvider(model="openai/gpt-5.6-terra", context_window=400000)
    assert p.tools_require_effort_none is True
    assert p._build_kwargs(MSGS, TOOLS)["reasoning_effort"] == "none"


def test_no_tools_on_terra_leaves_the_depth_alone() -> None:
    p = LiteLLMProvider(model="openai/gpt-5.6-terra", context_window=400000)
    assert "reasoning_effort" not in p._build_kwargs(MSGS, None)


def test_a_configured_depth_yields_to_none_when_tools_ride() -> None:
    # With tools in the request a depth cannot be honoured on this route at all — the
    # alternative is a 400 on every call and a whole session on the fallback model.
    p = LiteLLMProvider(model="openai/gpt-5.6-luna", context_window=400000)
    assert p._build_kwargs(MSGS, TOOLS, reasoning_effort="high")["reasoning_effort"] == "none"


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
    assert [c.name for c in result.tool_calls] == ["answer"]


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

"""A new OpenAI model family gets the request OpenAI wants, not the one litellm's name table
remembers (ADR-0279).

WHY THIS FILE EXISTS
Measured 2026-10-05 and 06 against ``openai/gpt-6-luna`` through this provider, litellm 1.86.2
(the ``uv.lock`` pin). litellm's param mapper picks its OpenAI handling by MODEL NAME: the
families it has shipped support for (gpt-5*, the o-series) get the reasoning-model mapping, and
any other OpenAI name is a plain chat model. gpt-6-luna is the second kind, although litellm's
own model map flags it ``supports_reasoning`` and ``supports_none_reasoning_effort`` exactly as
it flags gpt-5.6-luna. Three things followed, each a live 400 or a silent drop:

* ``max_tokens`` went out as it was, and OpenAI answered 400 "Unsupported parameter:
  'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead." Every
  call of the provider's defaults failed. The provider now says OpenAI's spelling itself, for
  the destination, so the next family needs no change.
* ``reasoning_effort`` was DROPPED (``drop_params``), so the ``none`` that a tool call needs on
  this tier never reached the wire, and the model refused the tool call with the 400 that asks
  for it. ``allowed_openai_params`` is litellm's own opt-in to forward a named parameter as it
  is; where the mapper already handles the name it changes nothing.
* A temperature other than the default is a 400 on gpt-6-luna as on gpt-5, and tool calls on its
  chat route need ``reasoning_effort="none"`` as on the 5.6 tier. Both rules were written as
  gpt-5 name prefixes. Their predicates are in ``test_gpt5_temperature.py`` and
  ``test_gpt56_tools_effort_none.py``; the requests they produce are pinned here.

Everything litellm sends is asserted at the WIRE, offline, with ``httpx`` patched and a canned
body, because the failures above were all decided inside litellm after this provider's kwargs
were already built. The other destinations are pinned the same way: a request that did not
change is a claim, and a test is how it stays one.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

import httpx
import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.messages import Message
from zakcode.providers.litellm_provider import LiteLLMProvider

MSGS = [{"role": "user", "content": "hi"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]
POD = "http://10.0.0.205:9090/v1"
GROQ_BASE = "https://api.groq.com/openai/v1"
WINDOW = 1_050_000  # stated, so no test reads the window from litellm's model map
_ROOT = Path(__file__).resolve().parents[1]


# ---- the kwargs the provider builds ----
@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-6-luna",
        "gpt-6-luna",
        "openai/gpt-5.6-luna",
        "gpt-5.6-terra",
        "openai/gpt-5-mini",
        "openai/gpt-4o-mini",
        "openai/o4-mini",
    ],
)
def test_a_call_to_openais_own_api_says_max_completion_tokens(model: str) -> None:
    kwargs = LiteLLMProvider(model=model, context_window=WINDOW)._build_kwargs(MSGS, None)
    assert kwargs["max_completion_tokens"] == lp._MAX_COMPLETION_TOKENS
    assert "max_tokens" not in kwargs


def test_a_per_call_max_tokens_is_renamed_too() -> None:
    """The compaction summary asks for its own room per call, through ``kw``."""
    provider = LiteLLMProvider(model="openai/gpt-6-luna", context_window=WINDOW)
    kwargs = provider._build_kwargs(MSGS, None, max_tokens=1234)
    assert kwargs["max_completion_tokens"] == 1234
    assert "max_tokens" not in kwargs


def test_a_callers_own_max_completion_tokens_wins() -> None:
    """The workaround the first measurement used: ``max_tokens=None`` and the new name, per
    call. It must keep meaning what it says, and the default ``max_tokens`` must not ride
    beside a cap the caller chose."""
    provider = LiteLLMProvider(model="openai/gpt-6-luna", context_window=WINDOW)
    explicit = provider._build_kwargs(MSGS, None, max_tokens=None, max_completion_tokens=64)
    assert explicit["max_completion_tokens"] == 64
    assert "max_tokens" not in explicit
    beside_default = provider._build_kwargs(MSGS, None, max_completion_tokens=64)
    assert beside_default["max_completion_tokens"] == 64
    assert "max_tokens" not in beside_default


@pytest.mark.parametrize(
    ("model", "extra"),
    [
        ("openai/zds-qwen3.6-35b", {"api_base": POD}),
        ("openai/gpt-6-luna", {"api_base": POD}),  # the same name behind a configured server
        ("openai/gpt-5.6-luna", {"api_base": POD}),
        ("openai/llama-3.3-70b-versatile", {"api_base": GROQ_BASE}),
        ("groq/qwen/qwen3-32b", {}),
        ("anthropic/claude-sonnet-4-5", {}),
        ("ollama_chat/llama3", {}),
        ("vertex_ai/gemini-2.5-pro", {}),
        ("azure/gpt-5", {}),
    ],
)
def test_every_other_destination_keeps_the_request_it_had(
    model: str, extra: dict[str, Any]
) -> None:
    """``openai/`` with an ``api_base`` is the pod, a llama-server or Groq's OpenAI-compatible
    endpoint: somebody else's API, which takes ``max_tokens``. What each of them does with the
    new spelling was never measured, so none of them is sent it."""
    kwargs = LiteLLMProvider(model=model, context_window=131072, **extra)._build_kwargs(MSGS, TOOLS)
    assert kwargs["max_tokens"] == lp._MAX_COMPLETION_TOKENS
    assert "max_completion_tokens" not in kwargs
    assert "allowed_openai_params" not in kwargs


def test_a_chosen_depth_is_offered_to_litellm_to_forward() -> None:
    provider = LiteLLMProvider(model="openai/gpt-6-luna", context_window=WINDOW)
    kwargs = provider._build_kwargs(MSGS, None, reasoning_effort="low")
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["allowed_openai_params"] == ["reasoning_effort"]


def test_an_existing_allowed_list_is_extended_not_replaced() -> None:
    provider = LiteLLMProvider(model="openai/gpt-6-luna", context_window=WINDOW)
    kwargs = provider._build_kwargs(
        MSGS, None, reasoning_effort="low", allowed_openai_params=["verbosity"]
    )
    assert kwargs["allowed_openai_params"] == ["verbosity", "reasoning_effort"]


def test_no_depth_means_nothing_is_offered() -> None:
    provider = LiteLLMProvider(model="openai/gpt-6-luna", context_window=WINDOW)
    assert "allowed_openai_params" not in provider._build_kwargs(MSGS, None)


def test_a_tool_call_on_gpt_6_luna_carries_effort_none_and_the_offer() -> None:
    provider = LiteLLMProvider(model="openai/gpt-6-luna", context_window=WINDOW)
    assert provider.tools_require_effort_none is True
    kwargs = provider._build_kwargs(MSGS, TOOLS)
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["allowed_openai_params"] == ["reasoning_effort"]


# ---- the request litellm actually sends: pinned offline against canned responses ----
_CHAT_BODY: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-6-luna",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ready"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}
_RESPONSES_BODY: dict[str, Any] = {
    "id": "resp_test",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-5.6-luna",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ready", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "error": None,
    "incomplete_details": None,
    "usage": {
        "input_tokens": 5,
        "output_tokens": 1,
        "total_tokens": 6,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every request litellm sends, answered from a canned body. No network, no real key."""
    seen: list[dict[str, Any]] = []

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append({"host": request.url.host, "path": request.url.path, "body": body})
        canned = _RESPONSES_BODY if request.url.path.endswith("/responses") else _CHAT_BODY
        return httpx.Response(200, json=canned, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    return seen


async def test_gpt_6_luna_chat_request_says_max_completion_tokens(
    wire: list[dict[str, Any]],
) -> None:
    provider = LiteLLMProvider(model="openai/gpt-6-luna", context_window=WINDOW)
    result = await provider.acomplete([Message.user("say ready")], system="S")
    assert result.text == "ready"
    assert [(w["host"], w["path"]) for w in wire] == [("api.openai.com", "/v1/chat/completions")]
    body = wire[0]["body"]
    assert body["max_completion_tokens"] == lp._MAX_COMPLETION_TOKENS
    assert "max_tokens" not in body, (
        "OpenAI refuses max_tokens on this model with a 400: the default kwargs of every call "
        "would fail, as they did before ADR-0279"
    )


async def test_gpt_6_luna_tool_call_stays_on_chat_and_carries_effort_none(
    wire: list[dict[str, Any]],
) -> None:
    provider = LiteLLMProvider(model="openai/gpt-6-luna", context_window=WINDOW)
    await provider.acomplete([Message.user("where am I")], system="S", tools=TOOLS)
    assert len(wire) == 1
    sent = wire[0]
    assert (sent["host"], sent["path"]) == ("api.openai.com", "/v1/chat/completions")
    assert sent["body"].get("reasoning_effort") == "none", (
        "reasoning_effort did not reach the wire: litellm dropped it for a model name its param "
        "mapper does not list, and OpenAI refuses a tool call that carries no depth"
    )
    assert sent["body"]["max_completion_tokens"] == lp._MAX_COMPLETION_TOKENS
    assert sent["body"].get("store") is False


async def test_gpt_6_luna_configured_depth_reaches_the_wire(wire: list[dict[str, Any]]) -> None:
    provider = LiteLLMProvider(model="openai/gpt-6-luna", context_window=WINDOW)
    await provider.acomplete([Message.user("say ready")], system="S", reasoning_effort="low")
    assert wire[0]["body"].get("reasoning_effort") == "low"


async def test_gpt_6_luna_temperature_never_reaches_the_wire(wire: list[dict[str, Any]]) -> None:
    provider = LiteLLMProvider(model="openai/gpt-6-luna", temperature=0.7, context_window=WINDOW)
    await provider.acomplete([Message.user("say ready")], system="S")
    assert "temperature" not in wire[0]["body"]


async def test_gpt_5_6_luna_requests_are_the_requests_they_were(
    wire: list[dict[str, Any]],
) -> None:
    """The family litellm DOES know. Its chat request carries the same cap under the same name,
    and a tool call still goes to the Responses API with its own effort and ``store: false``."""
    provider = LiteLLMProvider(model="openai/gpt-5.6-luna", context_window=400000)
    await provider.acomplete([Message.user("say ready")], system="S")
    chat = wire[-1]
    assert chat["path"] == "/v1/chat/completions"
    assert chat["body"]["max_completion_tokens"] == lp._MAX_COMPLETION_TOKENS
    assert "max_tokens" not in chat["body"]
    assert "reasoning_effort" not in chat["body"]

    await provider.acomplete([Message.user("where am I")], system="S", tools=TOOLS)
    bridged = wire[-1]
    assert bridged["path"] == "/v1/responses", (
        "litellm no longer sends a gpt-5.6 tool call to the Responses API: the effort bench's "
        "'product route' and the retention default both rest on this routing"
    )
    assert bridged["body"]["reasoning"] == {"effort": "none"}
    assert bridged["body"].get("store") is False
    assert "reasoning_effort" not in bridged["body"], (
        "the Responses request carries its depth as reasoning.effort: a top-level reasoning_effort "
        "beside it means the opt-in leaked through the bridge"
    )


@pytest.mark.parametrize("model", ["openai/zds-qwen3.6-35b", "openai/gpt-6-luna"])
async def test_the_pod_keeps_max_tokens_whatever_the_model_is_called(
    wire: list[dict[str, Any]], model: str
) -> None:
    provider = LiteLLMProvider(model=model, api_base=POD, context_window=131072)
    await provider.acomplete([Message.user("say ready")], system="S")
    assert [(w["host"], w["path"]) for w in wire] == [("10.0.0.205", "/v1/chat/completions")]
    body = wire[0]["body"]
    assert body["max_tokens"] == lp._MAX_COMPLETION_TOKENS
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body


async def test_a_groq_compatible_base_keeps_max_tokens(wire: list[dict[str, Any]]) -> None:
    provider = LiteLLMProvider(
        model="openai/llama-3.3-70b-versatile",
        api_base=GROQ_BASE,
        api_key="offline-test-key",
        context_window=131072,
    )
    await provider.acomplete([Message.user("say ready")], system="S")
    assert [(w["host"], w["path"]) for w in wire] == [
        ("api.groq.com", "/openai/v1/chat/completions")
    ]
    body = wire[0]["body"]
    assert body["max_tokens"] == lp._MAX_COMPLETION_TOKENS
    assert "max_completion_tokens" not in body


def test_the_declared_litellm_floor_knows_the_opt_in() -> None:
    """``allowed_openai_params`` first ships in litellm 1.66: the 1.65.0 wheel has no trace of it
    and 1.66.0 through 1.70.0 do (read from the wheels, 2026-10-06). An older litellm does not
    consume the kwarg, so it rides into the request body, and OpenAI answers 400 "Unrecognized
    request argument supplied: allowed_openai_params" (live, same day) on every call that carries a
    reasoning depth. So the floor this package declares must not admit one."""
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    specs = [d for d in pyproject["project"]["dependencies"] if re.match(r"litellm\b", d)]
    assert len(specs) == 1, specs
    floor = re.search(r">=\s*(\d+)\.(\d+)", specs[0])
    assert floor is not None, specs[0]
    assert (int(floor.group(1)), int(floor.group(2))) >= (1, 66), specs[0]

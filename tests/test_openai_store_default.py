"""OpenAI's own API is asked NOT to store what a served conversation sends it (ADR-0199).

WHY THIS FILE EXISTS
Measured 2026-09-19. litellm moves a chat call for a gpt-5.4+ model onto OpenAI's Responses API
whenever function tools ride with a reasoning effort, and since the forced-none rule
(`tests/test_gpt56_tools_effort_none.py`) that is every tool call of the 5.6 tier. The Responses
API STORES a response unless the request says `store: false`: a request with no field came back
`store: true` and was retrievable by its id; with `false` the same id was a 404. Chat completions
stores only on request. So a change that was about reasoning effort switched provider-side
retention on, and nobody had decided that.

Two wire-level facts pin the implementation, and the last two tests here pin THEM, offline:

* `store` must ride `extra_body`. A top-level `store=False` handed to litellm never reaches the
  Responses request (measured: no `store` key in the body), although it sits in litellm's own
  supported-parameter lists. `extra_body` reaches it, beside the cache routing key.
* Which API a call reaches is litellm's decision, not ours. If an upgrade changes that routing
  or stops forwarding `extra_body`, the retention default would change without any error, so
  the request litellm actually builds is asserted here against a canned response.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

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


# ---- the request the provider builds ----
@pytest.mark.parametrize("model", ["openai/gpt-5.6-luna", "openai/gpt-5-mini", "gpt-5.6-terra"])
def test_a_call_to_openais_own_api_asks_not_to_be_stored(model: str) -> None:
    kwargs = LiteLLMProvider(model=model)._build_kwargs(MSGS, TOOLS)
    assert kwargs["extra_body"] == {"store": False}
    # NEVER top-level: litellm's bridge drops it there, silently.
    assert "store" not in kwargs


def test_it_rides_the_chat_route_too_so_a_routing_change_cannot_switch_it_back_on() -> None:
    kwargs = LiteLLMProvider(model="openai/gpt-5.6-luna")._build_kwargs(MSGS, None)
    assert kwargs["extra_body"] == {"store": False}


def test_the_cache_routing_key_keeps_riding_beside_it() -> None:
    provider = LiteLLMProvider(model="openai/gpt-5.6-luna")
    kwargs = provider._build_kwargs(MSGS, TOOLS, prompt_cache_key="zakcode/sess-1")
    assert kwargs["extra_body"] == {"store": False, "prompt_cache_key": "zakcode/sess-1"}


def test_a_configured_server_behind_the_same_prefix_gets_nothing() -> None:
    """``openai/`` with an ``api_base`` is the pod or a llama-server: somebody else's API. What
    a backend does with a field it does not know is per backend and was never measured there."""
    provider = LiteLLMProvider(model="openai/zds-qwen3.6-35b", api_base=POD, context_window=131072)
    assert "store" not in (provider._build_kwargs(MSGS, TOOLS).get("extra_body") or {})
    same_name = LiteLLMProvider(model="openai/gpt-5.6-luna", api_base=POD, context_window=131072)
    assert "store" not in (same_name._build_kwargs(MSGS, TOOLS).get("extra_body") or {})


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4-5",
        "groq/qwen/qwen3-32b",
        "vertex_ai/gemini-2.5-pro",
        "azure/gpt-5",
    ],
)
def test_every_other_destination_gets_nothing(model: str) -> None:
    """Named clouds refuse an unknown body argument outright. ``azure/`` takes the same litellm
    bridge but was never measured, so nothing is assumed about it."""
    kwargs = LiteLLMProvider(model=model, context_window=131072)._build_kwargs(MSGS, TOOLS)
    assert "store" not in (kwargs.get("extra_body") or {})
    assert "store" not in kwargs


def test_an_operator_who_configures_store_keeps_their_word() -> None:
    provider = LiteLLMProvider(model="openai/gpt-5.6-luna", extra_body={"store": True})
    assert provider._build_kwargs(MSGS, TOOLS)["extra_body"] == {"store": True}
    default = LiteLLMProvider(model="openai/gpt-5.6-luna")
    per_call = default._build_kwargs(MSGS, TOOLS, extra_body={"store": True})
    assert per_call["extra_body"] == {"store": True}
    assert default.extra_body == {}  # the instance's configuration is never mutated


def test_a_server_that_refuses_it_by_name_loses_it_for_the_session() -> None:
    """ADR-0181's repair covers it because it is a key of the body this call sent."""
    provider = LiteLLMProvider(model="openai/gpt-5.6-luna")
    provider.rejected_request_fields.append("store")
    assert "extra_body" not in provider._build_kwargs(MSGS, TOOLS)


# ---- the request litellm actually sends: pinned offline against canned responses ----
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
            "name": "shell",
            "arguments": '{"command": "pwd"}',
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
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}
_CHAT_BODY: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-5.6-luna",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ready"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
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


async def test_a_tool_call_reaches_the_responses_api_asking_not_to_be_stored(
    wire: list[dict[str, Any]],
) -> None:
    provider = LiteLLMProvider(model="openai/gpt-5.6-luna")
    result = await provider.acomplete([Message.user("where am I")], system="S", tools=TOOLS)
    assert [(call.name, call.arguments) for call in result.tool_calls] == [
        ("shell", {"command": "pwd"})
    ]
    assert len(wire) == 1
    sent = wire[0]
    assert (sent["host"], sent["path"]) == ("api.openai.com", "/v1/responses"), (
        "litellm no longer sends a tool call of this tier to the Responses API: the retention "
        "default and the effort bench's 'product route' both rest on this routing"
    )
    assert sent["body"].get("store") is False, (
        "store: false did not reach the Responses request, so OpenAI stores the response: "
        "litellm stopped forwarding extra_body on its chat-to-responses bridge"
    )
    assert sent["body"]["reasoning"] == {"effort": "none"}


async def test_a_call_without_tools_reaches_chat_completions_with_the_same_field(
    wire: list[dict[str, Any]],
) -> None:
    provider = LiteLLMProvider(model="openai/gpt-5.6-luna")
    result = await provider.acomplete([Message.user("say ready")], system="S")
    assert result.text == "ready"
    assert [(w["host"], w["path"]) for w in wire] == [("api.openai.com", "/v1/chat/completions")]
    assert wire[0]["body"].get("store") is False

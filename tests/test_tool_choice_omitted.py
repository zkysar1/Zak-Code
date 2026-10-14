"""Tools ride without ``tool_choice`` (ADR-0295).

Reported 2026-10-14 against an OpenAI-compatible gateway, a LiteLLM proxy in front of Bedrock: a
request carrying ``tools`` was answered, and the same request with ``tool_choice: "auto"`` added
was a 400 ``UnsupportedParamsError`` naming ``{'tool_choice': 'auto'}``. Native tool calls through
that route were unusable because of a field that said nothing: "auto" is what a backend does when
it gets tools and no choice. The provider no longer sends the field, so a gateway that refuses it
by name is never asked.

These tests pin that. Hermetic: no network, no credentials (``conftest`` points ``ZAKCODE_HOME``
at a temporary folder, so no ``~/.zakcode/.env`` is read). The refusal is scripted from the
report; it was not captured from the gateway.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import zakcode.providers.litellm_provider as lp
from tests.test_provider_stream import _chunk
from zakcode.agent.compact import Compactor
from zakcode.agent.loop import AgentLoop
from zakcode.config import PermissionTier, load_settings
from zakcode.messages import Message
from zakcode.providers.base import RequestFailed
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.session.store import Session
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

GATEWAY = "https://gateway.invalid/v1"
WINDOW = 200_000
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

#: What the gateway answers a request that carries the field, in the shape litellm gives the
#: error (the report names the class and the field; the rest is litellm's own message).
REFUSAL = (
    "litellm.UnsupportedParamsError: openai does not support parameters: {'tool_choice': 'auto'}, "
    "for model=Claude4.5. To drop these, set `litellm.drop_params=True` or for proxy:\n\n"
    "`litellm_settings:\n drop_params: true`\n."
)


@pytest.fixture(autouse=True)
def _no_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider spaces request starts a second apart; that sleep is not under test."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


def _provider(model: str = "openai/Claude4.5") -> LiteLLMProvider:
    return LiteLLMProvider(model=model, api_base=GATEWAY, api_key="sk-test", context_window=WINDOW)


# ── the request the provider builds ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("openai/Claude4.5", id="gateway-alias"),
        pytest.param("openai/gpt-4o", id="openai"),
        pytest.param("anthropic/claude-sonnet-4-6", id="anthropic"),
        pytest.param("gemini/gemini-2.5-pro", id="gemini"),
        pytest.param("ollama_chat/llama3.1", id="ollama"),
    ],
)
def test_tools_ride_alone_on_every_route(model: str) -> None:
    """The tools still go out; only the field that said nothing is gone."""
    kwargs = _provider(model)._build_kwargs(MSGS, TOOLS)
    assert kwargs["tools"] == TOOLS
    assert "tool_choice" not in kwargs


def test_a_call_without_tools_sends_neither() -> None:
    kwargs = _provider()._build_kwargs(MSGS, None)
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


def test_a_caller_that_wants_a_choice_still_gets_it_through() -> None:
    """The omission is a default, not a ban: a per-call ``tool_choice`` is forwarded as given."""
    kwargs = _provider()._build_kwargs(MSGS, TOOLS, tool_choice="required")
    assert kwargs["tool_choice"] == "required"


# ── a gateway that refuses the field by name ─────────────────────────────────────────────────


def _answer() -> SimpleNamespace:
    message = SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=usage,
        model="Claude4.5",
        _hidden_params={},
    )


class RefusingGateway:
    """``litellm.acompletion`` pointed at the gateway of the report: a request carrying
    ``tool_choice`` is a 400, a request carrying tools alone is answered. ``requests`` records
    the kwargs of every call."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if "tool_choice" in kwargs:
            raise lp.litellm.BadRequestError(
                message=REFUSAL, model="Claude4.5", llm_provider="openai"
            )
        if not kwargs.get("stream"):
            return _answer()

        async def _stream() -> AsyncIterator[Any]:
            yield _chunk(content="ok")
            yield _chunk(content="", finish_reason="stop")

        return _stream()


async def test_the_scripted_gateway_does_refuse_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: the fake is the gateway of the report, so the answers below mean
    something. A request that asks for the field is the 400 the report names."""
    gateway = RefusingGateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    with pytest.raises(RequestFailed, match="tool_choice"):
        await _provider().acomplete(
            [Message.user("hi")], system="S", tools=TOOLS, tool_choice="auto"
        )


async def test_a_buffered_tool_call_is_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = RefusingGateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    result = await _provider().acomplete([Message.user("hi")], system="S", tools=TOOLS)

    assert result.text == "ok"
    [request] = gateway.requests
    assert request["tools"] == TOOLS
    assert "tool_choice" not in request


async def test_a_streamed_tool_call_is_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = RefusingGateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    events = [e async for e in _provider().astream([Message.user("hi")], system="S", tools=TOOLS)]

    assert events, "the stream produced nothing"
    [request] = gateway.requests
    assert request["stream"] is True
    assert request["tools"] == TOOLS
    assert "tool_choice" not in request


# ── what litellm actually puts on the wire ───────────────────────────────────────────────────

_CHAT_BODY: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "Claude4.5",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ready"}, "finish_reason": "stop"}
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
        return httpx.Response(200, json=_CHAT_BODY, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    return seen


async def test_the_request_litellm_sends_carries_tools_and_no_tool_choice(
    wire: list[dict[str, Any]],
) -> None:
    """Building the kwargs is not sending them: litellm decides what reaches the gateway, and
    it adds no ``tool_choice`` of its own."""
    result = await _provider().acomplete([Message.user("hi")], system="S", tools=TOOLS)

    assert result.text == "ready"
    [sent] = wire
    assert (sent["host"], sent["path"]) == ("gateway.invalid", "/v1/chat/completions")
    assert [t["function"]["name"] for t in sent["body"]["tools"]] == ["shell"]
    assert "tool_choice" not in sent["body"]


async def test_the_wire_does_show_a_tool_choice_when_one_is_asked_for(
    wire: list[dict[str, Any]],
) -> None:
    """Positive control: the fixture sees the field when it is there, so its absence above is
    a reading and not blindness."""
    await _provider().acomplete([Message.user("hi")], system="S", tools=TOOLS, tool_choice="auto")

    assert wire[0]["body"].get("tool_choice") == "auto"


# ── the agent loop, whose registry is what supplies the tools ────────────────────────────────


class _Probe(Tool):
    spec = ToolSpec(
        name="probe",
        description="a tool the model may call",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("done")


def _loop(tmp_path: Path) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(_Probe())
    session = Session(cwd=str(tmp_path), model="openai/Claude4.5")
    return AgentLoop(
        _provider(),
        registry,
        session,
        settings=load_settings(workspace_root=tmp_path),
        compactor=Compactor(),
    )


def _tool_names(request: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in request.get("tools") or []]


async def test_a_turn_with_tools_completes_against_the_refusing_gateway(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway = RefusingGateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    result = await _loop(tmp_path).arun_turn("continue")

    assert result.stop_reason == "completed", result.error
    assert "probe" in _tool_names(gateway.requests[0]), "premise: the loop sent its tools"
    assert all("tool_choice" not in request for request in gateway.requests)


async def test_a_streamed_turn_with_tools_completes_against_the_refusing_gateway(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway = RefusingGateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    events = [event async for event in _loop(tmp_path).astream_turn("continue")]

    assert events[-1].stop_reason == "completed", events[-1].error
    assert "probe" in _tool_names(gateway.requests[0]), "premise: the loop sent its tools"
    assert all("tool_choice" not in request for request in gateway.requests)

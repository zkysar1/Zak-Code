"""Tests for the server wire layer (event round-trips + request/response models)."""

from __future__ import annotations

import pytest

from zakcode.agent.loop import TurnResult
from zakcode.config import PermissionTier
from zakcode.events import (
    AgentDone,
    AgentStatus,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.permissions import PermissionRequest
from zakcode.server.wire import (
    ChatRequest,
    ChatResponse,
    SessionInfo,
    ToolInfo,
    WSActionRequired,
    client_message_from_dict,
    event_from_dict,
    event_to_dict,
)
from zakcode.session.store import Session
from zakcode.tools.base import ConcurrencyClass, ToolSpec
from zakcode.usage import Usage

# ── AgentEvent round-trips (every member) ─────────────────────────────────────


def test_every_event_round_trips_through_dict() -> None:
    events = [
        AgentTextDelta(text="hi"),
        AgentToolCall(id="c1", name="bash", arguments={"command": "ls"}),
        AgentToolResult(tool_use_id="c1", output="ok", is_error=False),
        AgentStatus(message="thinking"),
        AgentUsage(usage=Usage(total_tokens=5)),
        AgentDone(stop_reason="completed", iterations=2, usage=Usage(total_tokens=9)),
    ]
    for original in events:
        data = event_to_dict(original)
        assert isinstance(data, dict)
        assert data["event"] == original.event  # discriminator preserved
        restored = event_from_dict(data)
        assert restored == original


def test_event_to_dict_is_json_safe() -> None:
    import json

    data = event_to_dict(AgentToolCall(id="c1", name="x", arguments={"n": 1}))
    # Must serialize with the stdlib json encoder (no pydantic/enum leftovers).
    assert json.loads(json.dumps(data)) == data


def test_event_from_dict_rejects_unknown_event() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        event_from_dict({"event": "nope"})


# ── ChatRequest / ChatResponse ────────────────────────────────────────────────


def test_chat_request_defaults() -> None:
    req = ChatRequest(message="hello")
    assert req.message == "hello"
    assert req.session_id is None
    assert req.model is None


def test_chat_response_from_turn() -> None:
    result = TurnResult(
        assistant_messages=[
            Message(
                role="assistant",
                blocks=[
                    TextBlock(text="done"),
                    ToolUseBlock(id="c1", name="write_file", input={"path": "a"}),
                ],
            )
        ],
        tool_results=[ToolResultBlock(tool_use_id="c1", output="wrote a", is_error=False)],
        iterations=2,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.02),
        stop_reason="completed",
    )
    resp = ChatResponse.from_turn("sess-1", result)
    assert resp.session_id == "sess-1"
    assert resp.text == "done"
    assert resp.iterations == 2
    assert resp.stop_reason == "completed"
    assert resp.usage.total_tokens == 15
    assert resp.cost_usd == pytest.approx(0.02)
    assert resp.tool_results[0].output == "wrote a"


def test_chat_response_json_round_trip() -> None:
    resp = ChatResponse(session_id="s", text="hi", usage=Usage(total_tokens=3))
    restored = ChatResponse.model_validate_json(resp.model_dump_json())
    assert restored == resp


# ── SessionInfo / ToolInfo ────────────────────────────────────────────────────


def test_session_info_from_session() -> None:
    session = Session(cwd="/work", model="openai/gpt-4o")
    session.add_message(Message.user("hello"))
    session.add_message(Message.assistant_text("hi"))
    session.add_usage(Usage(total_tokens=7))
    info = SessionInfo.from_session(session)
    assert info.id == session.id
    assert info.model == "openai/gpt-4o"
    assert info.cwd == "/work"
    assert info.message_count == 2
    assert info.usage.total_tokens == 7


def test_tool_info_from_spec() -> None:
    spec = ToolSpec(
        name="bash",
        description="run a shell command",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        required_permission=PermissionTier.DANGER_FULL_ACCESS,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )
    info = ToolInfo.from_spec(spec)
    assert info.name == "bash"
    assert info.required_permission == "DANGER_FULL_ACCESS"
    assert info.concurrency == "never_parallel"
    assert info.parameters["properties"]["command"]["type"] == "string"


# ── WebSocket control frames ──────────────────────────────────────────────────


def test_ws_client_message_discriminates() -> None:
    assert client_message_from_dict({"type": "input", "message": "go"}).message == "go"
    assert client_message_from_dict({"type": "interrupt"}).type == "interrupt"
    appr = client_message_from_dict({"type": "approval", "outcome": "allow_once"})
    assert appr.outcome == "allow_once"


def test_ws_client_message_rejects_unknown_type() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        client_message_from_dict({"type": "mystery"})


def test_ws_action_required_from_request() -> None:
    req = PermissionRequest(
        tool_name="bash",
        tier=PermissionTier.DANGER_FULL_ACCESS,
        arguments={"command": "rm x"},
        reason="danger tier",
    )
    frame = WSActionRequired.from_request(req)
    assert frame.type == "action_required"
    assert frame.tool_name == "bash"
    assert frame.tier == "DANGER_FULL_ACCESS"
    assert frame.arguments == {"command": "rm x"}
    assert frame.reason == "danger tier"


def test_chat_response_carries_provider_error_detail() -> None:
    """ChatResponse.from_turn maps TurnResult.error so a REST consumer can tell a
    rate-limit from an auth failure without reading server logs (PR-2 review)."""
    failed = TurnResult(stop_reason="provider_error", error="RateLimited: 429", degraded=True)
    resp = ChatResponse.from_turn("s1", failed)
    assert resp.stop_reason == "provider_error"
    assert resp.error == "RateLimited: 429"
    clean = ChatResponse.from_turn("s1", TurnResult())
    assert clean.error == ""

"""Unit tests for SafeEventProjection — the whitelist filter for the public watch surface.

Pure (no server, no network): asserts the projection strips tool arguments/output, drops
usage/unknown events, correlates tool_use_id -> name for result badges, and runs every
surviving string field through the extended redaction (planted secrets must not survive).
"""

from __future__ import annotations

from zakcode.events import (
    AgentDone,
    AgentStatus,
    AgentTaskUpdate,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.server.safe_projection import SafeEventProjection
from zakcode.usage import Usage


def test_text_delta_is_kept_and_redacted() -> None:
    frame = SafeEventProjection().project(AgentTextDelta(text="Researching coral reefs."))
    assert frame == {"event": "text", "text": "Researching coral reefs."}


def test_status_is_kept() -> None:
    frame = SafeEventProjection().project(AgentStatus(message="Searching databases..."))
    assert frame == {"event": "status", "message": "Searching databases..."}


def test_tool_call_strips_arguments() -> None:
    frame = SafeEventProjection().project(
        AgentToolCall(id="c1", name="bash", arguments={"cmd": "cat /etc/passwd"})
    )
    assert frame == {"event": "tool_summary", "name": "bash", "status": "running"}
    assert "arguments" not in frame
    assert "passwd" not in str(frame)


def test_tool_result_strips_output_and_labels_from_prior_call() -> None:
    proj = SafeEventProjection()
    proj.project(AgentToolCall(id="c1", name="read_file", arguments={"path": "secrets.txt"}))
    frame = proj.project(
        AgentToolResult(tool_use_id="c1", output="API_KEY=sk-abcdef0123456789xyz", is_error=False)
    )
    assert frame == {"event": "tool_summary", "name": "read_file", "status": "completed"}
    assert "sk-abcdef" not in str(frame)
    assert "secrets.txt" not in str(frame)


def test_tool_result_failed_status() -> None:
    proj = SafeEventProjection()
    proj.project(AgentToolCall(id="c2", name="bash", arguments={}))
    frame = proj.project(AgentToolResult(tool_use_id="c2", output="boom", is_error=True))
    assert frame == {"event": "tool_summary", "name": "bash", "status": "failed"}


def test_tool_result_without_prior_call_labels_generic() -> None:
    frame = SafeEventProjection().project(
        AgentToolResult(tool_use_id="unknown", output="x", is_error=False)
    )
    assert frame == {"event": "tool_summary", "name": "tool", "status": "completed"}


def test_usage_is_dropped() -> None:
    usage_event = AgentUsage(usage=Usage(total_tokens=100, cost_usd=1.23))
    assert SafeEventProjection().project(usage_event) is None


def test_done_strips_internals() -> None:
    frame = SafeEventProjection().project(
        AgentDone(
            stop_reason="completed",
            iterations=5,
            usage=Usage(cost_usd=9.9),
            error="boom detail",
            degraded=True,
        )
    )
    assert frame == {"event": "done", "stop_reason": "completed"}
    assert "boom detail" not in str(frame)


def test_task_update_keeps_only_title_and_status() -> None:
    event = AgentTaskUpdate(
        plan="1. inspect /secret/path/plan.md",
        tasks=[
            {
                "id": "t1",
                "title": "Find sources",
                "status": "done",
                "kind": "task",
                "note": "/tmp/leak-path",
                "children": [{"title": "child-leak"}],
            }
        ],
        finished=1,
        total=3,
        complete=False,
    )
    frame = SafeEventProjection().project(event)
    assert frame["event"] == "task_update"
    assert frame["tasks"] == [{"title": "Find sources", "status": "done"}]
    assert frame["finished"] == 1
    assert frame["total"] == 3
    assert frame["complete"] is False
    assert "plan" not in frame  # raw model-authored checklist dropped
    assert "leak" not in str(frame)  # note + children dropped


def test_planted_secrets_in_text_are_redacted() -> None:
    proj = SafeEventProjection(
        secret_values=["hunter2supersecretvalue"],
        workspace_paths=["/home/ec2-user/ws"],
    )
    frame = proj.project(
        AgentTextDelta(
            text=(
                "key gsk_0123456789abcdef0123 value hunter2supersecretvalue "
                "at /home/ec2-user/ws/journal.md"
            )
        )
    )
    text = frame["text"]
    assert "gsk_0123456789abcdef0123" not in text
    assert "hunter2supersecretvalue" not in text
    assert "/home/ec2-user/ws" not in text
    assert "[path]" in text


def test_unknown_event_fails_closed() -> None:
    class _Weird:
        event = "weird"

    assert SafeEventProjection().project(_Weird()) is None  # type: ignore[arg-type]

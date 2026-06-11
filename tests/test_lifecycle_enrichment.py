"""Tests for SESSION_END and PRE_COMPACT lifecycle payload enrichment (PR-T7).

Verifies that ``data.session_summary`` is present with expected fields.
"""

from __future__ import annotations

from pathlib import Path

from zakcode.hooks import HookEvent, LifecyclePayload


async def test_session_end_enriched_data(tmp_path: Path) -> None:
    """SESSION_END payload carries session_summary with expected fields."""
    from zakcode.evals.harness import ScriptedProvider, make_agent, reply

    captured: list[LifecyclePayload] = []
    agent = make_agent(
        ScriptedProvider(script=[reply("ok")]),
        workspace_root=str(tmp_path),
        permission_mode="allow",
    )
    agent.hook_manager.register_lifecycle(HookEvent.SESSION_END, lambda p: captured.append(p))
    await agent.arun_turn("hello")
    await agent.aclose()

    assert len(captured) == 1
    payload = captured[0]
    assert payload.data.get("trigger") == "session_end"
    summary = payload.data.get("session_summary")
    assert summary is not None
    assert "session_id" in summary
    assert summary["session_id"] == agent.session.id
    assert "message_count" in summary
    assert isinstance(summary["message_count"], int)
    assert "total_usage" in summary
    assert isinstance(summary["total_usage"], dict)
    assert "created_at" in summary


async def test_pre_compact_enriched_data(tmp_path: Path) -> None:
    """PRE_COMPACT payload carries session_summary with session_id + message_count."""
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply
    from zakcode.messages import Message

    captured: list[LifecyclePayload] = []
    provider = ScriptedProvider(responder=lambda msgs, system, idx: reply("summary"))
    agent = Agent(
        provider=provider,
        enable_compaction=True,
        default_model="scripted/test",
        workspace_root=str(tmp_path),
    )
    agent.hook_manager.register_lifecycle(HookEvent.PRE_COMPACT, lambda p: captured.append(p))
    for i in range(6):
        agent.session.add_message(Message.user(f"m{i}"))
        agent.session.add_message(Message.assistant_text(f"r{i}"))
    await agent.loop.compact_now()

    assert len(captured) == 1
    payload = captured[0]
    # compact_now() uses trigger="manual"; _maybe_compact() uses "auto".
    assert payload.data.get("trigger") == "manual"
    summary = payload.data.get("session_summary")
    assert summary is not None
    assert "session_id" in summary
    assert summary["session_id"] == agent.session.id
    assert "message_count" in summary
    assert isinstance(summary["message_count"], int)


async def test_session_end_shell_hook_receives_enriched_data(tmp_path: Path) -> None:
    """Shell hook receives JSON with data.session_summary on stdin."""
    import json
    import sys

    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply
    from zakcode.hooks import HookManager, HookSpec

    marker = tmp_path / "session_end_data.json"
    body = (
        "import sys, json\n"
        "d = json.load(sys.stdin)\n"
        f"open(r'{marker}', 'w').write(json.dumps(d.get('data', {{}})))\n"
    )
    script = tmp_path / "hook.py"
    script.write_text(body, encoding="utf-8")
    spec = HookSpec(event=HookEvent.SESSION_END, command=[sys.executable, str(script)])
    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        hook_manager=HookManager([spec]),
        default_model="scripted/test",
        workspace_root=str(tmp_path),
    )
    await agent.arun_turn("hello")
    await agent.aclose()

    data = json.loads(marker.read_text())
    assert data.get("trigger") == "session_end"
    assert "session_summary" in data
    assert "session_id" in data["session_summary"]

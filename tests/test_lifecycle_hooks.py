"""Tests for session-lifecycle hooks (SessionStart / SessionEnd / PreCompact)."""

from __future__ import annotations

import sys
from pathlib import Path

from zakcode.hooks import HookEvent, HookManager, HookSpec, LifecyclePayload


def _payload(event: HookEvent = HookEvent.SESSION_START) -> LifecyclePayload:
    return LifecyclePayload(event=event, session_id="s1", cwd="/w")


# ── HookManager.fire (in-process + shell, observe-only) ───────────────────────


async def test_in_process_lifecycle_hook_runs() -> None:
    seen: list[str] = []
    mgr = HookManager()
    mgr.register_lifecycle(HookEvent.SESSION_START, lambda p: seen.append(p.session_id))
    await mgr.fire(_payload())
    assert seen == ["s1"]


async def test_async_lifecycle_hook_runs() -> None:
    seen: list[str] = []

    async def hook(p: LifecyclePayload) -> None:
        seen.append(p.event)

    mgr = HookManager()
    mgr.register_lifecycle(HookEvent.SESSION_END, hook)
    await mgr.fire(_payload(HookEvent.SESSION_END))
    assert seen == ["SessionEnd"]


async def test_lifecycle_hook_only_fires_for_its_event() -> None:
    seen: list[str] = []
    mgr = HookManager()
    mgr.register_lifecycle(HookEvent.PRE_COMPACT, lambda p: seen.append("compact"))
    await mgr.fire(_payload(HookEvent.SESSION_START))  # different event
    assert seen == []
    await mgr.fire(_payload(HookEvent.PRE_COMPACT))
    assert seen == ["compact"]


async def test_raising_lifecycle_hook_is_isolated() -> None:
    def boom(p: LifecyclePayload) -> None:
        raise RuntimeError("kaboom")

    mgr = HookManager()
    mgr.register_lifecycle(HookEvent.SESSION_START, boom)
    await mgr.fire(_payload())  # must not raise


def test_has_lifecycle_hooks_precheck() -> None:
    assert HookManager().has_lifecycle_hooks(HookEvent.SESSION_START) is False
    mgr = HookManager()
    mgr.register_lifecycle(HookEvent.SESSION_START, lambda p: None)
    assert mgr.has_lifecycle_hooks(HookEvent.SESSION_START) is True
    spec = HookSpec(event=HookEvent.PRE_COMPACT, command=["x"])
    assert HookManager([spec]).has_lifecycle_hooks(HookEvent.PRE_COMPACT) is True


async def test_shell_lifecycle_hook_receives_payload(tmp_path: Path) -> None:
    marker = tmp_path / "fired.txt"
    body = (
        "import sys, json\n"
        "d = json.load(sys.stdin)\n"
        f"open(r'{marker}', 'w').write(d['session_id'])\n"
    )
    script = tmp_path / "hook.py"
    script.write_text(body, encoding="utf-8")
    spec = HookSpec(event=HookEvent.SESSION_START, command=[sys.executable, str(script)])
    await HookManager([spec]).fire(_payload())
    assert marker.read_text() == "s1"


# ── fire points through the agent loop / facade ───────────────────────────────


async def test_session_start_fires_once_per_session(tmp_path: Path) -> None:
    from zakcode.evals.harness import ScriptedProvider, make_agent, reply

    starts: list[str] = []
    provider = ScriptedProvider(script=[reply("a"), reply("b")])
    agent = make_agent(provider, workspace_root=str(tmp_path), permission_mode="allow")
    agent.hook_manager.register_lifecycle(
        HookEvent.SESSION_START, lambda p: starts.append(p.session_id)
    )

    await agent.arun_turn("first")
    await agent.arun_turn("second")
    assert len(starts) == 1  # fired once across both turns
    assert starts[0] == agent.session.id


async def test_session_end_fires_on_aclose(tmp_path: Path) -> None:
    from zakcode.evals.harness import ScriptedProvider, make_agent, reply

    ended: list[str] = []
    agent = make_agent(
        ScriptedProvider(script=[reply("ok")]),
        workspace_root=str(tmp_path),
        permission_mode="allow",
    )
    agent.hook_manager.register_lifecycle(HookEvent.SESSION_END, lambda p: ended.append(p.event))
    await agent.aclose()
    await agent.aclose()  # idempotent: SESSION_END must not double-fire
    assert ended == ["SessionEnd"]


async def test_pre_compact_fires_on_manual_compaction(tmp_path: Path) -> None:
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply
    from zakcode.messages import Message

    fired: list[str] = []
    # A responder that returns a summary when asked to compact, else a reply.
    provider = ScriptedProvider(responder=lambda msgs, system, idx: reply("summary"))
    agent = Agent(
        provider=provider,
        enable_compaction=True,
        default_model="scripted/test",
        workspace_root=str(tmp_path),
    )
    agent.hook_manager.register_lifecycle(HookEvent.PRE_COMPACT, lambda p: fired.append(p.event))
    # Seed enough history that there is something to compact.
    for i in range(6):
        agent.session.add_message(Message.user(f"m{i}"))
        agent.session.add_message(Message.assistant_text(f"r{i}"))
    await agent.loop.compact_now()
    assert fired == ["PreCompact"]

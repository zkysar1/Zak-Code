"""Tests for session-lifecycle hooks (SessionStart / SessionEnd / PreCompact)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from zakcode._subprocess import find_bash
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


@pytest.mark.skipif(sys.platform == "win32", reason="a POSIX shell's `&` child")
async def test_a_session_start_hook_that_leaves_a_daemon_running_is_over_when_it_exits(
    tmp_path: Path,
) -> None:
    # A mind world's SessionStart hook starts the framework's daemon when none is running. The
    # daemon redirects all three of its streams, so the hook is over when its own process is.
    # Measured 2026-09-18 on the served path under uvloop: 95.0 s, the hook's whole timeout,
    # against 0.7 s on the stdlib loop (ADR-0197 pins the served process to the stdlib loop).
    bash = find_bash()
    assert bash is not None
    spec = HookSpec(
        event=HookEvent.SESSION_START,
        command=[bash, "-c", "sleep 8 </dev/null >/dev/null 2>&1 & echo started"],
        timeout=6,
    )
    start = time.monotonic()
    await HookManager([spec]).fire(
        LifecyclePayload(event=HookEvent.SESSION_START, session_id="s1", cwd=str(tmp_path))
    )
    assert time.monotonic() - start < 4  # did NOT wait out the timeout


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
        context_window=8192,
        workspace_root=str(tmp_path),
    )
    agent.hook_manager.register_lifecycle(HookEvent.PRE_COMPACT, lambda p: fired.append(p.event))
    # Seed enough history that there is something to compact.
    for i in range(6):
        agent.session.add_message(Message.user(f"m{i}"))
        agent.session.add_message(Message.assistant_text(f"r{i}"))
    await agent.loop.compact_now()
    assert fired == ["PreCompact"]

"""Tests for TURN_END hook event, payload, response models, and dispatch.

Shell hooks are exercised with tiny throwaway Python scripts invoked via
``sys.executable`` so the tests are hermetic and cross-platform (no real shell).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from zakcode.agent.stuck import StuckTracker
from zakcode.hooks import (
    HookEvent,
    HookManager,
    HookSpec,
    TurnEndPayload,
    TurnEndResult,
)


def _te_payload(**kw: object) -> TurnEndPayload:
    base: dict = {
        "session_id": "sess-42",
        "cwd": "/workspace",
        "stop_reason": "completed",
        "iterations": 5,
        "max_iterations": 50,
        "last_assistant_message": "Done.",
    }
    base.update(kw)
    return TurnEndPayload(**base)


def _script(tmp_path: Path, name: str, body: str) -> list[str]:
    """Write a Python hook script and return the argv array to invoke it."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return [sys.executable, str(path)]


# ── payload serialization ────────────────────────────────────────────────────


def test_turn_end_payload_serialization() -> None:
    """TurnEndPayload JSON has session_id and last_assistant_message at top level."""
    payload = _te_payload()
    doc = json.loads(payload.model_dump_json())
    assert doc["session_id"] == "sess-42"
    assert doc["last_assistant_message"] == "Done."
    assert doc["event"] == "TurnEnd"
    assert doc["stop_reason"] == "completed"
    assert doc["iterations"] == 5
    assert doc["max_iterations"] == 50
    assert doc["degraded"] is False


# ── shell hook: Claude Code compat (exit 0 + decision JSON) ─────────────────


async def test_run_turn_end_shell_block_claude_code_compat(tmp_path: Path) -> None:
    """Exit 0 + {"decision":"block","reason":"..."} on stdout = vetoed."""
    body = (
        "import json, sys\n"
        'print(json.dumps({"decision": "block", "reason": "keep going"}))\n'
        "sys.exit(0)\n"
    )
    cmd = _script(tmp_path, "block.py", body)
    mgr = HookManager([HookSpec(event=HookEvent.TURN_END, command=cmd)])
    result = await mgr.run_turn_end(_te_payload())
    assert result.vetoed is True
    assert result.continuation_prompt == "keep going"


async def test_run_turn_end_shell_allow_empty_stdout(tmp_path: Path) -> None:
    """Exit 0 + empty stdout = allow."""
    cmd = _script(tmp_path, "allow.py", "import sys; sys.exit(0)")
    mgr = HookManager([HookSpec(event=HookEvent.TURN_END, command=cmd)])
    result = await mgr.run_turn_end(_te_payload())
    assert result.vetoed is False


async def test_run_turn_end_shell_allow_nonzero_exit(tmp_path: Path) -> None:
    """Exit 1 = fail-open allow."""
    cmd = _script(tmp_path, "fail.py", "import sys; sys.exit(1)")
    mgr = HookManager([HookSpec(event=HookEvent.TURN_END, command=cmd)])
    result = await mgr.run_turn_end(_te_payload())
    assert result.vetoed is False


async def test_run_turn_end_shell_block_native_exit2(tmp_path: Path) -> None:
    """Exit 2 + stdout message = vetoed (native protocol)."""
    body = "import sys; print('continue please'); sys.exit(2)\n"
    cmd = _script(tmp_path, "native.py", body)
    mgr = HookManager([HookSpec(event=HookEvent.TURN_END, command=cmd)])
    result = await mgr.run_turn_end(_te_payload())
    assert result.vetoed is True
    assert result.continuation_prompt == "continue please"


async def test_run_turn_end_shell_timeout(tmp_path: Path) -> None:
    """Hook sleeps past timeout -> process killed, fail-open allow."""
    cmd = _script(tmp_path, "slow.py", "import time; time.sleep(30)")
    spec = HookSpec(event=HookEvent.TURN_END, command=cmd, timeout=0.5)
    mgr = HookManager([spec])
    result = await mgr.run_turn_end(_te_payload())
    assert result.vetoed is False


async def test_run_turn_end_shell_crash() -> None:
    """Hook raises OSError on spawn -> fail-open allow, warning logged."""
    spec = HookSpec(
        event=HookEvent.TURN_END,
        command=["this_executable_does_not_exist_zzz_turn_end"],
    )
    mgr = HookManager([spec])
    result = await mgr.run_turn_end(_te_payload())
    assert result.vetoed is False


async def test_run_turn_end_shell_env_scrubbed(tmp_path: Path) -> None:
    """drop_env=["ANTHROPIC_API_KEY"] -> key absent from child env."""
    body = (
        "import os, json, sys\n"
        "val = os.environ.get('ANTHROPIC_API_KEY', 'ABSENT')\n"
        f"open(r'{tmp_path / 'env_check.txt'}', 'w').write(val)\n"
        "sys.exit(0)\n"
    )
    cmd = _script(tmp_path, "env_check.py", body)
    spec = HookSpec(event=HookEvent.TURN_END, command=cmd)
    mgr = HookManager([spec])
    # Set the key in the environment so the test is meaningful.
    old = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "sk-secret-test-key"
    try:
        await mgr.run_turn_end(_te_payload(), drop_env=["ANTHROPIC_API_KEY"])
    finally:
        if old is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = old
    result_text = (tmp_path / "env_check.txt").read_text()
    assert result_text == "ABSENT"


# ── StuckTracker.reset() ────────────────────────────────────────────────────


def test_stuck_tracker_reset() -> None:
    """StuckTracker.reset() zeroes _streak, clears _prev_sig, clears _error_counts."""
    from collections import Counter

    tracker = StuckTracker()
    # Simulate some state accumulation.
    tracker._streak = 5
    tracker._prev_sig = (("bash", '{"command":"ls"}'),)
    tracker._error_counts = Counter({("bash", '{"command":"ls"}'): 3})

    tracker.reset()
    assert tracker._streak == 0
    assert tracker._prev_sig is None
    assert len(tracker._error_counts) == 0


# ── in-process hook veto ─────────────────────────────────────────────────────


async def test_run_turn_end_in_process_veto() -> None:
    """In-process hook returns TurnEndResult(vetoed=True) -> first veto wins."""

    def veto_hook(payload: TurnEndPayload) -> TurnEndResult:
        return TurnEndResult(vetoed=True, continuation_prompt="in-process veto")

    mgr = HookManager()
    mgr.register_turn_end(veto_hook)
    result = await mgr.run_turn_end(_te_payload())
    assert result.vetoed is True
    assert result.continuation_prompt == "in-process veto"


async def test_run_turn_end_in_process_allow() -> None:
    """In-process hook returning None -> allow."""

    def allow_hook(payload: TurnEndPayload) -> None:
        return None

    mgr = HookManager()
    mgr.register_turn_end(allow_hook)
    result = await mgr.run_turn_end(_te_payload())
    assert result.vetoed is False


async def test_run_turn_end_in_process_error_failopen() -> None:
    """In-process hook that raises -> fail-open allow."""

    def boom(payload: TurnEndPayload) -> TurnEndResult:
        raise RuntimeError("kaboom")

    mgr = HookManager()
    mgr.register_turn_end(boom)
    result = await mgr.run_turn_end(_te_payload())
    assert result.vetoed is False


async def test_run_turn_end_in_process_first_veto_wins() -> None:
    """First veto in in-process chain stops the chain."""
    calls: list[str] = []

    def first(payload: TurnEndPayload) -> TurnEndResult:
        calls.append("first")
        return TurnEndResult(vetoed=True, continuation_prompt="first wins")

    def second(payload: TurnEndPayload) -> TurnEndResult:
        calls.append("second")
        return TurnEndResult(vetoed=True, continuation_prompt="second")

    mgr = HookManager()
    mgr.register_turn_end(first)
    mgr.register_turn_end(second)
    result = await mgr.run_turn_end(_te_payload())
    assert calls == ["first"]
    assert result.continuation_prompt == "first wins"


async def test_run_turn_end_in_process_before_shell(tmp_path: Path) -> None:
    """In-process hooks run before shell hooks; a veto from in-process stops the chain."""
    calls: list[str] = []

    def ip_veto(payload: TurnEndPayload) -> TurnEndResult:
        calls.append("in-process")
        return TurnEndResult(vetoed=True, continuation_prompt="ip")

    marker = tmp_path / "shell_ran.txt"
    body = f"open(r'{marker}', 'w').write('ran')\nimport sys; sys.exit(0)\n"
    cmd = _script(tmp_path, "shell.py", body)
    mgr = HookManager([HookSpec(event=HookEvent.TURN_END, command=cmd)])
    mgr.register_turn_end(ip_veto)
    result = await mgr.run_turn_end(_te_payload())
    assert result.vetoed is True
    assert calls == ["in-process"]
    assert not marker.exists()  # shell hook never ran


# ── has_hooks pre-check ──────────────────────────────────────────────────────


def test_has_hooks_turn_end_shell() -> None:
    spec = HookSpec(event=HookEvent.TURN_END, command=["x"])
    mgr = HookManager([spec])
    assert mgr.has_hooks(HookEvent.TURN_END, "*") is True
    assert mgr.has_hooks(HookEvent.PRE_TOOL_USE, "bash") is False


def test_has_hooks_turn_end_in_process() -> None:
    mgr = HookManager()
    assert mgr.has_hooks(HookEvent.TURN_END, "*") is False
    mgr.register_turn_end(lambda p: None)
    assert mgr.has_hooks(HookEvent.TURN_END, "*") is True


# ── TE-R2: wire-fidelity fixture ────────────────────────────────────────────


async def test_wire_fidelity_stop_hook_contract(tmp_path: Path) -> None:
    """Contract pin: a script that reads stdin exactly like Mind's stop-hook.sh
    (json.load(sys.stdin).get('session_id'), .get('last_assistant_message'))
    and emits the decision-block JSON. The full round trip must work."""
    body = (
        "import sys, json\n"
        "payload = json.load(sys.stdin)\n"
        "sid = payload.get('session_id', '')\n"
        "lam = payload.get('last_assistant_message', '')\n"
        "# Mind's stop-hook reads these two fields and decides.\n"
        "# Here we always block, echoing the fields back as proof.\n"
        "print(json.dumps({\n"
        '    "decision": "block",\n'
        '    "reason": f"BLOCK session={sid} msg={lam}"\n'
        "}))\n"
        "sys.exit(0)\n"
    )
    cmd = _script(tmp_path, "stop_hook_compat.py", body)
    mgr = HookManager([HookSpec(event=HookEvent.TURN_END, command=cmd)])
    payload = TurnEndPayload(
        session_id="test-session-abc",
        last_assistant_message="I completed the task.",
        stop_reason="completed",
        iterations=10,
        max_iterations=50,
    )
    result = await mgr.run_turn_end(payload)
    assert result.vetoed is True
    assert "test-session-abc" in result.continuation_prompt
    assert "I completed the task." in result.continuation_prompt


# ── async in-process hook ────────────────────────────────────────────────────


async def test_run_turn_end_in_process_async_hook() -> None:
    """An async in-process TURN_END hook is awaited correctly."""

    async def async_veto(payload: TurnEndPayload) -> TurnEndResult:
        return TurnEndResult(vetoed=True, continuation_prompt="async veto")

    mgr = HookManager()
    mgr.register_turn_end(async_veto)
    result = await mgr.run_turn_end(_te_payload())
    assert result.vetoed is True
    assert result.continuation_prompt == "async veto"

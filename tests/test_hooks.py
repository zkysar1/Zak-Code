"""Tests for the lifecycle hook runtime (shell + in-process), focused on the
exit-code protocol, argument mutation, and the error-isolation invariant.

Shell hooks are exercised with tiny throwaway Python scripts invoked via
``sys.executable`` so the tests are hermetic and cross-platform (no real shell).
"""

from __future__ import annotations

import sys
from pathlib import Path

from zakcode.hooks import (
    HookDecision,
    HookEvent,
    HookManager,
    HookPayload,
    HookResult,
    HookSpec,
)


def _payload(event: HookEvent = HookEvent.PRE_TOOL_USE, **kw) -> HookPayload:
    base = {"event": event, "tool_name": "bash", "arguments": {"command": "ls"}}
    base.update(kw)
    return HookPayload(**base)


def _script(tmp_path: Path, name: str, body: str) -> list[str]:
    """Write a Python hook script and return the argv array to invoke it."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return [sys.executable, str(path)]


# ── in-process hooks ──────────────────────────────────────────────────────────


async def test_in_process_allow_by_default() -> None:
    mgr = HookManager(in_process={HookEvent.PRE_TOOL_USE: [lambda p: None]})
    result = await mgr.run(_payload())
    assert not result.blocked
    assert result.decision is HookDecision.ALLOW


async def test_in_process_block_vetoes() -> None:
    def veto(payload: HookPayload) -> HookResult:
        return HookResult(decision=HookDecision.BLOCK, messages=["nope"])

    mgr = HookManager(in_process={HookEvent.PRE_TOOL_USE: [veto]})
    result = await mgr.run(_payload())
    assert result.blocked
    assert "nope" in result.message


async def test_in_process_async_hook_supported() -> None:
    async def veto(payload: HookPayload) -> HookResult:
        return HookResult(decision=HookDecision.BLOCK, messages=["async no"])

    mgr = HookManager(in_process={HookEvent.PRE_TOOL_USE: [veto]})
    assert (await mgr.run(_payload())).blocked


async def test_in_process_mutates_arguments() -> None:
    def rewrite(payload: HookPayload) -> HookResult:
        return HookResult(mutated_arguments={"command": "ls -la"})

    mgr = HookManager(in_process={HookEvent.PRE_TOOL_USE: [rewrite]})
    result = await mgr.run(_payload())
    assert result.mutated_arguments == {"command": "ls -la"}


async def test_in_process_raising_hook_is_isolated() -> None:
    def boom(payload: HookPayload) -> HookResult:
        raise RuntimeError("kaboom")

    mgr = HookManager(in_process={HookEvent.PRE_TOOL_USE: [boom]})
    result = await mgr.run(_payload())
    # The turn is NOT broken: a raising hook downgrades to a warning.
    assert not result.blocked
    assert result.decision is HookDecision.WARN
    assert "kaboom" in result.message


async def test_first_block_wins_and_short_circuits() -> None:
    calls: list[str] = []

    def first(payload: HookPayload) -> HookResult:
        calls.append("first")
        return HookResult(decision=HookDecision.BLOCK, messages=["stop"])

    def second(payload: HookPayload) -> HookResult:
        calls.append("second")
        return HookResult()

    mgr = HookManager(in_process={HookEvent.PRE_TOOL_USE: [first, second]})
    assert (await mgr.run(_payload())).blocked
    assert calls == ["first"]  # second never ran


async def test_mutation_threads_through_chain() -> None:
    def widen(payload: HookPayload) -> HookResult:
        cmd = payload.arguments.get("command", "")
        return HookResult(mutated_arguments={"command": cmd + " -1"})

    mgr = HookManager(in_process={HookEvent.PRE_TOOL_USE: [widen, widen]})
    result = await mgr.run(_payload())
    # Second hook saw the first hook's rewrite.
    assert result.mutated_arguments == {"command": "ls -1 -1"}


async def test_post_tool_use_cannot_block() -> None:
    def veto(payload: HookPayload) -> HookResult:
        return HookResult(decision=HookDecision.BLOCK, messages=["too late"])

    mgr = HookManager(in_process={HookEvent.POST_TOOL_USE: [veto]})
    result = await mgr.run(_payload(event=HookEvent.POST_TOOL_USE))
    # The tool already ran; a post-hook block is downgraded to a warning.
    assert not result.blocked
    assert result.decision is HookDecision.WARN
    assert "too late" in result.message


# ── shell hooks: exit-code protocol ───────────────────────────────────────────


async def test_shell_hook_allow_exit_0(tmp_path: Path) -> None:
    cmd = _script(tmp_path, "ok.py", "import sys; sys.exit(0)")
    mgr = HookManager([HookSpec(event=HookEvent.PRE_TOOL_USE, command=cmd)])
    assert not (await mgr.run(_payload())).blocked


async def test_shell_hook_block_exit_2(tmp_path: Path) -> None:
    cmd = _script(tmp_path, "no.py", "import sys; print('denied by policy'); sys.exit(2)")
    mgr = HookManager([HookSpec(event=HookEvent.PRE_TOOL_USE, command=cmd)])
    result = await mgr.run(_payload())
    assert result.blocked
    assert "denied by policy" in result.message


async def test_shell_hook_other_exit_is_warn(tmp_path: Path) -> None:
    cmd = _script(tmp_path, "warn.py", "import sys; sys.exit(7)")
    mgr = HookManager([HookSpec(event=HookEvent.PRE_TOOL_USE, command=cmd)])
    result = await mgr.run(_payload())
    assert not result.blocked
    assert result.decision is HookDecision.WARN


async def test_shell_hook_reads_payload_on_stdin(tmp_path: Path) -> None:
    # Block only if the payload (read from stdin) names the bash tool.
    body = (
        "import sys, json\n"
        "data = json.load(sys.stdin)\n"
        "sys.exit(2 if data.get('tool_name') == 'bash' else 0)\n"
    )
    cmd = _script(tmp_path, "inspect.py", body)
    mgr = HookManager([HookSpec(event=HookEvent.PRE_TOOL_USE, command=cmd)])
    assert (await mgr.run(_payload(tool_name="bash"))).blocked
    assert not (await mgr.run(_payload(tool_name="read_file"))).blocked


async def test_shell_hook_mutates_arguments_via_stdout_json(tmp_path: Path) -> None:
    body = (
        "import sys, json\n"
        "print(json.dumps({'arguments': {'command': 'echo safe'}, 'message': 'rewrote'}))\n"
        "sys.exit(0)\n"
    )
    cmd = _script(tmp_path, "mutate.py", body)
    mgr = HookManager([HookSpec(event=HookEvent.PRE_TOOL_USE, command=cmd)])
    result = await mgr.run(_payload())
    assert result.mutated_arguments == {"command": "echo safe"}
    assert "rewrote" in result.message


async def test_shell_hook_nonexistent_command_is_isolated() -> None:
    spec = HookSpec(
        event=HookEvent.PRE_TOOL_USE,
        command=["this_executable_does_not_exist_zzz", "--x"],
    )
    mgr = HookManager([spec])
    result = await mgr.run(_payload())
    # A hook that can't even start must not break the turn.
    assert not result.blocked
    assert result.decision is HookDecision.WARN


async def test_shell_hook_timeout_is_isolated(tmp_path: Path) -> None:
    cmd = _script(tmp_path, "slow.py", "import time; time.sleep(30)")
    spec = HookSpec(event=HookEvent.PRE_TOOL_USE, command=cmd, timeout=0.5)
    mgr = HookManager([spec])
    result = await mgr.run(_payload())
    assert not result.blocked
    assert result.decision is HookDecision.WARN
    assert "timed out" in result.message


# ── matching & pre-checks ─────────────────────────────────────────────────────


async def test_matcher_filters_by_tool_name(tmp_path: Path) -> None:
    cmd = _script(tmp_path, "block_bash.py", "import sys; sys.exit(2)")
    spec = HookSpec(event=HookEvent.PRE_TOOL_USE, command=cmd, matcher="bash")
    mgr = HookManager([spec])
    assert (await mgr.run(_payload(tool_name="bash"))).blocked
    # read_file doesn't match "bash" → hook never runs → allowed.
    assert not (await mgr.run(_payload(tool_name="read_file"))).blocked


def test_has_hooks_precheck() -> None:
    spec = HookSpec(event=HookEvent.PRE_TOOL_USE, command=["x"], matcher="bash")
    mgr = HookManager([spec])
    assert mgr.has_hooks(HookEvent.PRE_TOOL_USE, "bash") is True
    assert mgr.has_hooks(HookEvent.PRE_TOOL_USE, "read_file") is False
    assert mgr.has_hooks(HookEvent.POST_TOOL_USE, "bash") is False


async def test_empty_manager_allows_everything() -> None:
    mgr = HookManager()
    result = await mgr.run(_payload())
    assert not result.blocked
    assert result.decision is HookDecision.ALLOW
    assert result.mutated_arguments is None


def test_register_adds_in_process_hook() -> None:
    mgr = HookManager()
    mgr.register(HookEvent.PRE_TOOL_USE, lambda p: None)
    assert mgr.has_hooks(HookEvent.PRE_TOOL_USE, "bash") is True

"""Tests for the project-verifier gate (R1): ``zakcode.agent.verify`` + loop wiring."""

from __future__ import annotations

from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.agent.verify import (
    VerificationGate,
    _commands_match,
    _exit_status_is_the_commands,
)
from zakcode.config import PermissionTier, Settings
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.usage import Usage

# ── pure gate unit tests ──────────────────────────────────────────────────────


def _calls_results(name: str, args: dict, *, is_error: bool = False):
    call = ToolCall(id="x", name=name, arguments=args)
    block = ToolResultBlock(tool_use_id="x", output="out", is_error=is_error)
    return [call], [block]


def test_gate_is_inert_without_a_command() -> None:
    gate = VerificationGate(command=None)
    assert not gate.enabled
    gate.observe(*_calls_results("write_file", {"path": "a.py"}))
    assert not gate.needs_verification()  # never arms


def test_gate_arms_on_code_change_then_passes_on_successful_run() -> None:
    gate = VerificationGate(command="uv run poe check")
    assert gate.enabled and not gate.needs_verification()
    gate.observe(*_calls_results("write_file", {"path": "a.py"}))
    assert gate.needs_verification()  # code changed, not yet verified
    gate.observe(*_calls_results("bash", {"command": "uv run poe check"}))
    assert gate.passed and not gate.needs_verification()


def test_failing_run_keeps_gate_open_and_records_output() -> None:
    gate = VerificationGate(command="check")
    gate.observe(*_calls_results("edit_file", {"path": "a.py"}))
    fail = ToolResultBlock(tool_use_id="r", output="2 tests failed", is_error=True)
    gate.observe([ToolCall(id="r", name="bash", arguments={"command": "check"})], [fail])
    assert gate.needs_verification()
    assert "2 tests failed" in gate.nudge() and "check" in gate.nudge()


def test_command_matching_is_token_contiguous() -> None:
    assert _commands_match("uv run poe check", "uv run poe check")
    assert _commands_match("cd repo && uv run poe check", "uv run poe check")  # wrapped
    assert not _commands_match("uv run poe", "uv run poe check")  # too short
    assert not _commands_match("echo uv check", "uv run poe check")  # not contiguous


def test_attempt_cap_bounds_the_gate() -> None:
    gate = VerificationGate(command="check", attempt_cap=2)
    assert gate.can_attempt()
    gate.consume_attempt()
    assert gate.can_attempt()
    gate.consume_attempt()
    assert not gate.can_attempt()


# ── loop integration (hermetic: fake write_file/bash tools, no real subprocess) ──


class _FakeWrite(Tool):
    spec = ToolSpec(
        name="write_file",
        description="fake write",
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("written", data={"path": args.get("path", "notes.txt")})


class _FakeBash(Tool):
    """A 'bash' that returns a fixed pass/fail without touching a real shell."""

    def __init__(self, *, ok: bool) -> None:
        self._ok = ok
        self.runs = 0
        self.spec = ToolSpec(
            name="bash",
            description="fake bash",
            required_permission=PermissionTier.DANGER_FULL_ACCESS,
            concurrency=ConcurrencyClass.NEVER_PARALLEL,
        )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.runs += 1
        return ToolResult.ok("checks passed") if self._ok else ToolResult.error("checks failed")


class _Scripted(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[i]

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


def _registry(bash: _FakeBash) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_FakeWrite())
    reg.register(bash)
    return reg


def _write_then_done(n_done: int = 8) -> list[LLMResult]:
    write = LLMResult(
        text="",
        tool_calls=[ToolCall(id="w1", name="write_file", arguments={"path": "notes.txt"})],
        usage=Usage(total_tokens=1),
    )
    done = LLMResult(text="done", tool_calls=[], usage=Usage(total_tokens=1))
    return [write] + [done] * n_done


def _loop(provider: Provider, registry: ToolRegistry) -> AgentLoop:
    # permission_policy=None -> the harness runs the (fake) bash directly, exercising the
    # harness-run path without a real subprocess.
    settings = Settings(verify_command="check")
    return AgentLoop(
        provider, registry, Session(cwd="/tmp", model="t/m"), settings=settings, max_iterations=20
    )


@pytest.mark.asyncio
async def test_loop_completes_when_verifier_passes() -> None:
    bash = _FakeBash(ok=True)
    result = await _loop(_Scripted(_write_then_done()), _registry(bash)).arun_turn("change code")
    assert result.stop_reason == "completed"
    assert bash.runs == 1  # the harness ran the project check exactly once, it passed


@pytest.mark.asyncio
async def test_loop_ends_verification_failed_when_checks_never_pass() -> None:
    bash = _FakeBash(ok=False)
    result = await _loop(_Scripted(_write_then_done()), _registry(bash)).arun_turn("change code")
    assert result.stop_reason == "verification_failed"
    assert result.degraded is True
    assert bash.runs == 3  # attempt_cap (default 3) harness runs, all failing


@pytest.mark.asyncio
async def test_loop_is_inert_without_verify_command() -> None:
    bash = _FakeBash(ok=False)  # would fail IF it ran
    provider = _Scripted(_write_then_done())
    loop = AgentLoop(
        provider,
        _registry(bash),
        Session(cwd="/tmp", model="t/m"),
        settings=Settings(verify_command=None),
        max_iterations=20,
    )
    result = await loop.arun_turn("change code")
    assert result.stop_reason == "completed"
    assert bash.runs == 0  # gate never armed -> the verifier was never invoked


def test_a_piped_verify_run_does_not_pass_the_gate_on_the_pipes_exit_code() -> None:
    """ADR-0139, sibling half: `uv run poe check | tail -40` reports TAIL's status.

    A model pipes the verify command precisely because its output is long, so the shape is the
    common one, not an edge case (observed verbatim in a live run: `uv run poe check 2>&1 |
    tail -40`). Before this, a FAILING check exited 0 through the pipe, set passed=True, and
    additionally CLEARED last_output -- so the nudge that would have shown the model its own
    failure was never built.
    """
    assert _exit_status_is_the_commands("uv run poe check", "uv run poe check") is True
    assert _exit_status_is_the_commands("cd repo && uv run poe check", "uv run poe check") is True
    assert (
        _exit_status_is_the_commands("uv run poe check 2>&1 | tail -40", "uv run poe check")
        is False
    )
    # A pipeline ENDING in the verify command keeps its own status.
    assert _exit_status_is_the_commands("echo go | uv run poe check", "uv run poe check") is True

    def gate(command: str, is_error: bool) -> VerificationGate:
        g = VerificationGate(command="uv run poe check")
        g.observe(
            [ToolCall(id="w", name="write_file", arguments={})],
            [ToolResultBlock(tool_use_id="w", output="ok", is_error=False)],
        )
        g.observe(
            [ToolCall(id="r", name="bash", arguments={"command": command})],
            [ToolResultBlock(tool_use_id="r", output="2 failed, 10 passed", is_error=is_error)],
        )
        return g

    # The defect: shell exit is tail's (0), so is_error is False though the checks FAILED.
    piped = gate("uv run poe check 2>&1 | tail -40", False)
    assert piped.passed is False
    assert piped.needs_verification() is True
    assert piped.last_output  # kept, so nudge() can show the model what failed
    assert "2 failed" in piped.nudge()

    # Unpiped behaviour is untouched in both directions.
    assert gate("uv run poe check", False).passed is True
    assert gate("uv run poe check", False).last_output == ""
    assert gate("uv run poe check", True).passed is False

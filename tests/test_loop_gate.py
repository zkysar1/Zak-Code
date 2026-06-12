"""Integration tests for the permission + hook gate wired into the agent loop.

These assert the security-critical behavior: a tool call is authorized on a code
path the model cannot reach, the *same* gate runs on both the buffered
(``arun_turn``) and streaming (``astream_turn``) paths, and a denial/veto becomes
a recoverable error result rather than aborting the turn.

Everything is hermetic — a scripted :class:`Provider` (no network) drives a chosen
tool call, and a recording fake tool tells us whether execution actually happened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.config import PermissionTier
from zakcode.hooks import HookDecision, HookEvent, HookManager, HookPayload, HookResult
from zakcode.messages import ToolResultBlock
from zakcode.permissions import (
    PermissionMode,
    PermissionOutcome,
    PermissionPolicy,
    PermissionRequest,
)
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


class _RecordingTool(Tool):
    """A tool that records every (args) it was actually executed with."""

    def __init__(self, name: str, tier: PermissionTier) -> None:
        self.spec = ToolSpec(
            name=name,
            description=f"{name} (recording test tool)",
            required_permission=tier,
            concurrency=ConcurrencyClass.NEVER_PARALLEL,
        )
        self.calls: list[dict] = []

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult.ok(f"ran {self.name} with {args}")


class _OneToolThenDone(Provider):
    """Requests one tool call on the first turn, then completes.

    Only ``acomplete`` is implemented; the base ``astream`` default wraps it, so
    the identical script drives both the buffered and streaming loop paths.
    """

    def __init__(self, tool_name: str, arguments: dict) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                text="",
                tool_calls=[ToolCall(id="c1", name=self.tool_name, arguments=self.arguments)],
                usage=Usage(total_tokens=1),
            )
        return LLMResult(text="done", tool_calls=[], usage=Usage(total_tokens=1))

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


class _ScriptedPrompter:
    def __init__(self, outcome: PermissionOutcome) -> None:
        self.outcome = outcome
        self.requests: list[PermissionRequest] = []

    async def confirm(self, request: PermissionRequest) -> PermissionOutcome:
        self.requests.append(request)
        return self.outcome


def _loop(
    tmp_path: Path,
    provider: Provider,
    registry: ToolRegistry,
    *,
    policy: PermissionPolicy | None,
    hooks: HookManager | None = None,
) -> AgentLoop:
    session = Session(cwd=str(tmp_path), model="scripted/test")
    return AgentLoop(
        provider,
        registry,
        session,
        workspace_root=tmp_path,
        permission_policy=policy,
        hook_manager=hooks,
        max_iterations=5,
    )


async def _collect(loop: AgentLoop, text: str, *, stream: bool) -> list[ToolResultBlock]:
    """Run one turn on either path; return the tool-result blocks produced."""
    if not stream:
        result = await loop.arun_turn(text)
        return result.tool_results
    blocks: list[ToolResultBlock] = []
    async for event in loop.astream_turn(text):
        if event.event == "tool_result":
            blocks.append(
                ToolResultBlock(
                    tool_use_id=event.tool_use_id,
                    output=event.output,
                    is_error=event.is_error,
                )
            )
    return blocks


# ── permission gate, on BOTH paths ────────────────────────────────────────────


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_deny_mode_blocks_write_both_paths(tmp_path: Path, stream: bool) -> None:
    tool = _RecordingTool("write_thing", PermissionTier.WORKSPACE_WRITE)
    registry = ToolRegistry()
    registry.register(tool)
    provider = _OneToolThenDone("write_thing", {"path": "x", "content": "y"})
    loop = _loop(tmp_path, provider, registry, policy=PermissionPolicy(PermissionMode.DENY))

    blocks = await _collect(loop, "do it", stream=stream)

    assert len(blocks) == 1
    assert blocks[0].is_error
    assert "permission denied" in blocks[0].output.lower()
    # The crucial property: the tool NEVER executed, regardless of what the model asked.
    assert tool.calls == []


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
async def test_readonly_tool_auto_allows_both_paths(tmp_path: Path, stream: bool) -> None:
    tool = _RecordingTool("peek", PermissionTier.READ_ONLY)
    registry = ToolRegistry()
    registry.register(tool)
    provider = _OneToolThenDone("peek", {"path": "x"})
    # ask mode, no prompter: read-only still runs (below the ceiling, never prompts).
    loop = _loop(tmp_path, provider, registry, policy=PermissionPolicy(PermissionMode.ASK))

    blocks = await _collect(loop, "peek", stream=stream)

    assert len(blocks) == 1
    assert not blocks[0].is_error
    assert tool.calls == [{"path": "x"}]


async def test_dangerous_command_blocked_independent_of_model(tmp_path: Path) -> None:
    # Even if the tool's tier would allow it, a catastrophic command is gated.
    tool = _RecordingTool("bash", PermissionTier.DANGER_FULL_ACCESS)
    registry = ToolRegistry()
    registry.register(tool)
    provider = _OneToolThenDone("bash", {"command": "rm -rf /"})
    # 'allow' mode would auto-run a normal command, but the dangerous pattern forces
    # a prompt; with no prompter that fails closed → denied.
    loop = _loop(tmp_path, provider, registry, policy=PermissionPolicy(PermissionMode.ALLOW))

    blocks = await _collect(loop, "destroy", stream=False)

    assert blocks[0].is_error
    assert tool.calls == []


async def test_unknown_tool_is_fail_closed(tmp_path: Path) -> None:
    # Model requests a tool not in the registry: no spec → strongest tier → denied
    # (ask mode, no prompter). The tool obviously can't run.
    registry = ToolRegistry()  # empty
    provider = _OneToolThenDone("mystery", {"command": "whatever"})
    loop = _loop(tmp_path, provider, registry, policy=PermissionPolicy(PermissionMode.ASK))

    blocks = await _collect(loop, "go", stream=False)

    assert blocks[0].is_error
    assert "permission denied" in blocks[0].output.lower()


async def test_session_approval_persists_across_calls(tmp_path: Path) -> None:
    tool = _RecordingTool("bash", PermissionTier.DANGER_FULL_ACCESS)
    registry = ToolRegistry()
    registry.register(tool)
    prompter = _ScriptedPrompter(PermissionOutcome.ALLOW_SESSION)
    policy = PermissionPolicy(PermissionMode.ASK, prompter=prompter)
    session = Session(cwd=str(tmp_path), model="scripted/test")
    workspace = tmp_path

    # Two separate turns that each request the same command.
    for _ in range(2):
        provider = _OneToolThenDone("bash", {"command": "ls"})
        loop = AgentLoop(
            provider,
            registry,
            session,
            workspace_root=workspace,
            permission_policy=policy,  # shared policy → session memory carries over
            max_iterations=5,
        )
        await loop.arun_turn("run ls")

    assert len(tool.calls) == 2  # both runs executed
    assert len(prompter.requests) == 1  # but the operator was asked only once


# ── hook gate ──────────────────────────────────────────────────────────────────


async def test_pre_hook_veto_blocks_tool(tmp_path: Path) -> None:
    tool = _RecordingTool("write_thing", PermissionTier.WORKSPACE_WRITE)
    registry = ToolRegistry()
    registry.register(tool)
    provider = _OneToolThenDone("write_thing", {"path": "x", "content": "y"})

    def veto(payload: HookPayload) -> HookResult:
        return HookResult(decision=HookDecision.BLOCK, messages=["policy says no"])

    hooks = HookManager(in_process={HookEvent.PRE_TOOL_USE: [veto]})
    # allow mode so permission isn't what stops it — the hook is.
    loop = _loop(
        tmp_path, provider, registry, policy=PermissionPolicy(PermissionMode.ALLOW), hooks=hooks
    )

    blocks = await _collect(loop, "go", stream=False)

    assert blocks[0].is_error
    assert "policy says no" in blocks[0].output
    assert tool.calls == []


async def test_pre_hook_mutates_arguments(tmp_path: Path) -> None:
    tool = _RecordingTool("write_thing", PermissionTier.WORKSPACE_WRITE)
    registry = ToolRegistry()
    registry.register(tool)
    provider = _OneToolThenDone("write_thing", {"path": "x", "content": "original"})

    def rewrite(payload: HookPayload) -> HookResult:
        new = dict(payload.arguments)
        new["content"] = "rewritten"
        return HookResult(mutated_arguments=new)

    hooks = HookManager(in_process={HookEvent.PRE_TOOL_USE: [rewrite]})
    loop = _loop(
        tmp_path, provider, registry, policy=PermissionPolicy(PermissionMode.ALLOW), hooks=hooks
    )

    await loop.arun_turn("go")

    # The tool ran with the hook's rewritten arguments, not the model's originals.
    assert tool.calls == [{"path": "x", "content": "rewritten"}]


async def test_pre_hook_cannot_rewrite_into_a_dangerous_command(tmp_path: Path) -> None:
    # audit3 #5: the gate authorized the ORIGINAL (benign) command; a PreToolUse hook then
    # rewrites it into a catastrophic one. The never-waivable blocklist must re-check the
    # POST-mutation args, so the rewritten command is blocked and never executes.
    tool = _RecordingTool("bash", PermissionTier.DANGER_FULL_ACCESS)
    registry = ToolRegistry()
    registry.register(tool)
    provider = _OneToolThenDone("bash", {"command": "echo hi"})  # benign as authorized

    def rewrite(payload: HookPayload) -> HookResult:
        return HookResult(mutated_arguments={"command": "rm -rf /"})

    hooks = HookManager(in_process={HookEvent.PRE_TOOL_USE: [rewrite]})
    loop = _loop(
        tmp_path, provider, registry, policy=PermissionPolicy(PermissionMode.ALLOW), hooks=hooks
    )

    blocks = await _collect(loop, "go", stream=False)
    assert blocks[0].is_error
    assert "never waived" in blocks[0].output
    assert tool.calls == []  # the rewritten dangerous command never ran


async def test_pre_hook_cannot_rewrite_into_an_undeclared_install(tmp_path: Path) -> None:
    # The dependency gate is a never-waivable floor too: the gate authorized a benign
    # 'echo hi'; a PreToolUse hook then rewrites it into an undeclared 'pip install evil'.
    # The post-rewrite re-check must re-apply the gate, so the install never executes.
    tool = _RecordingTool("bash", PermissionTier.DANGER_FULL_ACCESS)
    registry = ToolRegistry()
    registry.register(tool)
    provider = _OneToolThenDone("bash", {"command": "echo hi"})  # benign as authorized

    def rewrite(payload: HookPayload) -> HookResult:
        return HookResult(mutated_arguments={"command": "pip install evil"})

    hooks = HookManager(in_process={HookEvent.PRE_TOOL_USE: [rewrite]})
    policy = PermissionPolicy(PermissionMode.ALLOW, declared_packages=lambda: {"pypi:requests"})
    loop = _loop(tmp_path, provider, registry, policy=policy, hooks=hooks)

    blocks = await _collect(loop, "go", stream=False)
    assert blocks[0].is_error
    assert "never waived" in blocks[0].output
    assert "evil" in blocks[0].output
    assert tool.calls == []  # the rewritten undeclared install never ran


async def test_post_hook_message_appended_to_result(tmp_path: Path) -> None:
    tool = _RecordingTool("peek", PermissionTier.READ_ONLY)
    registry = ToolRegistry()
    registry.register(tool)
    provider = _OneToolThenDone("peek", {"path": "x"})

    def note(payload: HookPayload) -> HookResult:
        return HookResult(decision=HookDecision.WARN, messages=["audited"])

    hooks = HookManager(in_process={HookEvent.POST_TOOL_USE: [note]})
    loop = _loop(
        tmp_path, provider, registry, policy=PermissionPolicy(PermissionMode.ASK), hooks=hooks
    )

    blocks = await _collect(loop, "go", stream=False)

    assert tool.calls == [{"path": "x"}]  # post-hook never blocks
    assert not blocks[0].is_error
    assert "audited" in blocks[0].output


async def test_turn_continues_after_denial(tmp_path: Path) -> None:
    # A denied tool yields an error result and the turn proceeds to completion
    # (the model gets a second acomplete and finishes) — denial is recoverable.
    tool = _RecordingTool("write_thing", PermissionTier.WORKSPACE_WRITE)
    registry = ToolRegistry()
    registry.register(tool)
    provider = _OneToolThenDone("write_thing", {"path": "x", "content": "y"})
    loop = _loop(tmp_path, provider, registry, policy=PermissionPolicy(PermissionMode.DENY))

    result = await loop.arun_turn("go")

    assert result.stop_reason == "completed"
    assert provider.calls == 2  # ran again after the denial → recovered
    assert result.tool_results[0].is_error


async def test_no_policy_means_ungated(tmp_path: Path) -> None:
    # A bare loop with no injected policy is a pure mechanism (library/tests):
    # tools run without authorization. The Agent facade always injects a policy;
    # this just documents/locks the explicit no-policy contract.
    tool = _RecordingTool("bash", PermissionTier.DANGER_FULL_ACCESS)
    registry = ToolRegistry()
    registry.register(tool)
    provider = _OneToolThenDone("bash", {"command": "ls"})
    loop = _loop(tmp_path, provider, registry, policy=None)

    await loop.arun_turn("go")

    assert tool.calls == [{"command": "ls"}]

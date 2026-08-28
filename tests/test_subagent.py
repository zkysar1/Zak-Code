"""Tests for the sub-agent factory (M4-3): isolation, filtering, shared budget, summary.

These exercise the structural guarantees delegation relies on, using a scripted
provider so no model or network is touched:

* a child runs on a FRESH session (no parent-context bleed),
* a child sees only its FILTERED tools,
* children draw from the SAME shared budget (and respect its child cap),
* the parent gets back a condensed SUMMARY (final text), not the raw transcript.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from zakcode.agent.budget import ChildLimitExceeded, IterationBudget
from zakcode.agent.loop import TurnResult
from zakcode.agent.subagent import (
    GENERAL_PURPOSE,
    SubAgentDefinition,
    SubAgentManager,
    SubAgentResult,
    SubAgentRunner,
)
from zakcode.config import Settings
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolCall,
)
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.usage import Usage


class _RecordingTool(Tool):
    """A read-only tool that records every (registry-scoped) invocation."""

    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(name=name, description=f"the {name} tool")
        self.invocations: list[dict] = []

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.invocations.append(args)
        return ToolResult.ok(f"{self.spec.name} ran")


class _OneShotProvider(Provider):
    """Emits one assistant text then completes — a single, deterministic turn.

    With ``call_tool`` set, the first completion instead requests that tool once
    (then the second completion finishes), so tool-filtering can be observed.
    """

    def __init__(self, text: str, *, call_tool: str | None = None) -> None:
        self.text = text
        self.call_tool = call_tool
        self._calls = 0

    async def acomplete(
        self, messages: list, *, tools: list | None = None, system: str | None = None, **kw
    ) -> LLMResult:
        self._calls += 1
        if self.call_tool and self._calls == 1:
            return LLMResult(
                tool_calls=[ToolCall(id="t1", name=self.call_tool, arguments={"x": 1})],
                usage=Usage(total_tokens=1),
            )
        return LLMResult(text=self.text, usage=Usage(total_tokens=1))

    async def astream(
        self, messages: list, *, tools: list | None = None, system: str | None = None, **kw
    ) -> AsyncIterator[ProviderStreamEvent]:
        result = await self.acomplete(messages, tools=tools, system=system)
        if result.text:
            yield StreamTextDelta(text=result.text)
        for i, call in enumerate(result.tool_calls):
            yield StreamToolCallDelta(
                index=i, id=call.id, name=call.name, arguments_delta=json.dumps(call.arguments)
            )
        yield StreamDone()

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


def _runner(
    tmp_path: Path,
    provider: Provider,
    registry: ToolRegistry,
    budget: IterationBudget,
) -> SubAgentRunner:
    return SubAgentRunner(
        provider=provider,
        registry=registry,
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        budget=budget,
        workspace_root=tmp_path,
    )


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


async def test_child_returns_summary_not_transcript(tmp_path: Path) -> None:
    runner = _runner(
        tmp_path, _OneShotProvider("the answer is 42"), _registry(), IterationBudget(10)
    )
    result = await runner.run(GENERAL_PURPOSE, "what is the answer?")
    assert isinstance(result, SubAgentResult)
    assert result.summary == "the answer is 42"
    assert result.stop_reason == "completed"
    assert result.name == "general-purpose"


async def test_child_runs_on_fresh_session(tmp_path: Path) -> None:
    # The child's loop builds its own Session; the parent passes no history in, so
    # the only messages are the child's own prompt + reply. We assert via the tool
    # context that the workspace is the child's and that two runs don't share state.
    runner = _runner(tmp_path, _OneShotProvider("done"), _registry(), IterationBudget(10))
    r1 = await runner.run(GENERAL_PURPOSE, "first")
    r2 = await runner.run(GENERAL_PURPOSE, "second")
    # Each child completed independently in exactly one iteration (no accumulated
    # history that would change iteration counts).
    assert r1.iterations == 1
    assert r2.iterations == 1


async def test_child_tool_filtering_blocks_disallowed_tool(tmp_path: Path) -> None:
    allowed = _RecordingTool("read_file")
    forbidden = _RecordingTool("write_file")
    registry = _registry(allowed, forbidden)
    # The child is restricted to read_file but the model tries to call write_file.
    provider = _OneShotProvider("ok", call_tool="write_file")
    definition = SubAgentDefinition(name="reader", allowed_tools=["read_file"])
    runner = _runner(tmp_path, provider, registry, IterationBudget(10))

    result = await runner.run(definition, "go")
    # write_file was never executed (not in the child's filtered registry)…
    assert forbidden.invocations == []
    # …and the child still completes (the disallowed call returns an error result
    # the model recovers from), reporting the final summary.
    assert result.summary == "ok"


def test_child_registry_is_subset(tmp_path: Path) -> None:
    a, b, c = _RecordingTool("read_file"), _RecordingTool("write_file"), _RecordingTool("bash")
    registry = _registry(a, b, c)
    runner = _runner(tmp_path, _OneShotProvider("x"), registry, IterationBudget(10))
    sub = runner.child_registry(SubAgentDefinition(name="r", allowed_tools=["read_file", "bash"]))
    assert set(sub.names()) == {"read_file", "bash"}
    # No filter → the parent's full registry is reused unchanged.
    full = runner.child_registry(GENERAL_PURPOSE)
    assert full is registry


async def test_children_share_the_budget(tmp_path: Path) -> None:
    budget = IterationBudget(10)
    runner = _runner(tmp_path, _OneShotProvider("done"), _registry(), budget)
    await runner.run(GENERAL_PURPOSE, "one")
    consumed_after_first = budget.consumed
    assert consumed_after_first >= 1
    await runner.run(GENERAL_PURPOSE, "two")
    # The second child drew from the SAME pool (consumption strictly increased).
    assert budget.consumed > consumed_after_first


async def test_child_cap_enforced(tmp_path: Path) -> None:
    budget = IterationBudget(100, max_children=1)
    runner = _runner(tmp_path, _OneShotProvider("done"), _registry(), budget)
    await runner.run(GENERAL_PURPOSE, "first child ok")
    with pytest.raises(ChildLimitExceeded):
        await runner.run(GENERAL_PURPOSE, "second child over cap")
    assert budget.children_spawned == 1


def test_prompt_builder_carries_system_suffix(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _OneShotProvider("x"), _registry(), IterationBudget(10))
    definition = SubAgentDefinition(name="planner", system_suffix="Produce a plan; do not edit.")
    builder = runner.prompt_builder_for(definition)
    # The definition's suffix is carried, with the shared structured-handoff instruction appended.
    assert builder.extra_instructions is not None
    assert "Produce a plan; do not edit." in builder.extra_instructions
    assert "Handoff:" in builder.extra_instructions


# ── summary fallback (M4-review MAJOR-2) ─────────────────────────────────────────


def test_summarize_prefers_assistant_text() -> None:
    result = TurnResult(
        assistant_messages=[Message.assistant_text("the final answer")],
        stop_reason="completed",
    )
    assert SubAgentRunner._summarize(result) == "the final answer"


def test_summarize_falls_back_to_last_tool_result() -> None:
    # A child that stopped mid-tool-loop (no final assistant text) still hands back
    # something useful: its last tool output.
    result = TurnResult(
        assistant_messages=[],
        tool_results=[ToolResultBlock(tool_use_id="t1", output="found 3 matches")],
        stop_reason="max_iterations",
        iterations=3,
    )
    assert SubAgentRunner._summarize(result) == "found 3 matches"


def test_summarize_falls_back_to_status_line() -> None:
    result = TurnResult(assistant_messages=[], stop_reason="max_iterations", iterations=4)
    summary = SubAgentRunner._summarize(result)
    assert "no final text" in summary
    assert "max_iterations" in summary


# ── SubAgentManager (the concrete spawner, M4-4c-2) ──────────────────────────────


def _manager(tmp_path: Path, provider: Provider, defs: list, default: str) -> SubAgentManager:
    runner = _runner(tmp_path, provider, _registry(), IterationBudget(10))
    return SubAgentManager(runner, defs, default=default)


def test_manager_reports_types_and_explicit_default(tmp_path: Path) -> None:
    mgr = _manager(
        tmp_path,
        _OneShotProvider("hi"),
        [GENERAL_PURPOSE, SubAgentDefinition(name="plan")],
        default="general-purpose",
    )
    assert mgr.available_types() == ["general-purpose", "plan"]
    assert mgr.default_type() == "general-purpose"


async def test_manager_spawn_runs_the_named_definition(tmp_path: Path) -> None:
    mgr = _manager(tmp_path, _OneShotProvider("answer"), [GENERAL_PURPOSE], "general-purpose")
    res = await mgr.spawn(type_name="general-purpose", prompt="q")
    assert isinstance(res, SubAgentResult)
    assert res.summary == "answer"


async def test_manager_unknown_type_raises(tmp_path: Path) -> None:
    mgr = _manager(tmp_path, _OneShotProvider("x"), [GENERAL_PURPOSE], "general-purpose")
    with pytest.raises(KeyError):
        await mgr.spawn(type_name="nope", prompt="q")


def test_manager_rejects_empty_definitions(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _OneShotProvider("x"), _registry(), IterationBudget(10))
    with pytest.raises(ValueError):
        SubAgentManager(runner, [], default="x")


def test_manager_rejects_default_not_in_definitions(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _OneShotProvider("x"), _registry(), IterationBudget(10))
    with pytest.raises(ValueError):
        SubAgentManager(runner, [GENERAL_PURPOSE], default="missing")


def test_manager_satisfies_spawner_protocol(tmp_path: Path) -> None:
    from zakcode.tools.base import SubAgentSpawner

    mgr = _manager(tmp_path, _OneShotProvider("x"), [GENERAL_PURPOSE], "general-purpose")
    assert isinstance(mgr, SubAgentSpawner)


async def test_child_loop_constructed_without_obsolete_flags(tmp_path: Path, monkeypatch) -> None:
    # Bet 1: write-grounding and the verify-before-finish gate are ALWAYS ON in AgentLoop
    # (not configurable — one way of doing things), so the runner must NOT forward the
    # removed verify_writes/recipe_* kwargs. A delegated child inherits grounding + the gate
    # by construction; this guards against a stray kwarg reappearing and reintroducing a flag.
    import zakcode.agent.subagent as sub

    captured: dict = {}

    class _FakeLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.update(kwargs)

        async def aclose(self) -> None:  # the real AgentLoop has this; the runner calls it
            pass

        async def arun_turn(self, prompt: str) -> TurnResult:
            return TurnResult(stop_reason="completed")

    monkeypatch.setattr(sub, "AgentLoop", _FakeLoop)
    runner = SubAgentRunner(
        provider=_OneShotProvider("x"),
        registry=_registry(_RecordingTool("read_file")),
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        budget=IterationBudget(10),
        workspace_root=tmp_path,
    )
    await runner.run(GENERAL_PURPOSE, "do it")
    for obsolete in (
        "verify_writes",
        "recipe_mode",
        "recipe_harness_run",
        "recipe_acceptance_compare",
        "recipe_attempt_cap",
        "single_tool_per_turn",
    ):
        assert obsolete not in captured, obsolete


async def test_child_gets_isolated_permission_view(tmp_path: Path, monkeypatch) -> None:
    # audit2 #10: the child must receive a child_view() of the parent policy (isolated
    # session grants), not the shared instance, so a child ALLOW_SESSION can't bleed up.
    import zakcode.agent.subagent as sub
    from zakcode.permissions import PermissionMode, PermissionPolicy

    parent_policy = PermissionPolicy(PermissionMode.ASK)
    captured: dict = {}

    class _FakeLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.update(kwargs)

        async def aclose(self) -> None:  # the real AgentLoop has this; the runner calls it
            pass

        async def arun_turn(self, prompt: str) -> TurnResult:
            return TurnResult(stop_reason="completed")

    monkeypatch.setattr(sub, "AgentLoop", _FakeLoop)
    runner = SubAgentRunner(
        provider=_OneShotProvider("x"),
        registry=_registry(_RecordingTool("read_file")),
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        budget=IterationBudget(10),
        permission_policy=parent_policy,
        workspace_root=tmp_path,
    )
    await runner.run(GENERAL_PURPOSE, "do it")
    child_policy = captured["permission_policy"]
    assert child_policy is not parent_policy  # not the shared instance
    assert child_policy.mode == parent_policy.mode
    child_policy._session_allow.add("bash")
    assert "bash" not in parent_policy._session_allow  # grant does not bleed up


async def test_child_inherits_extra_workspace_roots(tmp_path: Path, monkeypatch) -> None:
    # audit4 #4: a delegated child must get the SAME multi-root sandbox as the parent, so a
    # --skill-dir-granted path isn't wrongly rejected inside a sub-agent.
    import zakcode.agent.subagent as sub

    captured: dict = {}

    class _FakeLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.update(kwargs)

        async def aclose(self) -> None:  # the real AgentLoop has this; the runner calls it
            pass

        async def arun_turn(self, prompt: str) -> TurnResult:
            return TurnResult(stop_reason="completed")

    monkeypatch.setattr(sub, "AgentLoop", _FakeLoop)
    extra = [tmp_path / "mind", tmp_path / "world"]
    runner = SubAgentRunner(
        provider=_OneShotProvider("x"),
        registry=_registry(_RecordingTool("read_file")),
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        budget=IterationBudget(10),
        extra_workspace_roots=extra,
        workspace_root=tmp_path,
    )
    await runner.run(GENERAL_PURPOSE, "do it")
    assert captured["extra_workspace_roots"] == extra


async def test_subagent_does_not_refire_session_start(tmp_path: Path) -> None:
    # A sub-agent is a sub-task within the parent's already-started session: it must NOT re-run the
    # workspace's SessionStart hooks. On a Mind those are a heavy boot, so re-firing them per
    # sub-agent -- and, worse, concurrently across a parallel delegation -- makes the boots contend
    # and can make parallel delegation SLOWER than sequential. Regression guard for that.
    from zakcode.hooks import HookEvent, HookManager

    fired: list[str] = []
    mgr = HookManager()
    mgr.register_lifecycle(HookEvent.SESSION_START, lambda p: fired.append(p.session_id))
    runner = SubAgentRunner(
        provider=_OneShotProvider("done"),
        registry=_registry(_RecordingTool("read_file")),
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        budget=IterationBudget(10),
        workspace_root=tmp_path,
        hook_manager=mgr,
    )
    result = await runner.run(GENERAL_PURPOSE, "do it")
    assert result.stop_reason == "completed"  # the sub-agent actually ran a turn ...
    assert fired == []  # ... but did NOT fire SESSION_START (the parent's session already started)


async def test_subagent_gets_its_own_empty_hooks_not_the_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CC-faithful: a sub-agent runs ONLY its own (frontmatter) hooks -- it does NOT inherit the
    # parent's project hooks. The runner builds the child loop with hook_manager=None (-> a fresh
    # empty HookManager), NOT the parent's, so the parent's per-tool gates (e.g. a Mind's
    # PreToolUse[Write] hooks) + SessionStart boot do not fire per sub-agent.
    import zakcode.agent.subagent as sub
    from zakcode.hooks import HookManager

    captured: dict = {}

    class _FakeLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.update(kwargs)

        async def aclose(self) -> None:
            pass

        async def arun_turn(self, prompt: str) -> TurnResult:
            return TurnResult(stop_reason="completed")

    monkeypatch.setattr(sub, "AgentLoop", _FakeLoop)
    parent_hooks = HookManager()
    runner = SubAgentRunner(
        provider=_OneShotProvider("x"),
        registry=_registry(_RecordingTool("read_file")),
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        budget=IterationBudget(10),
        workspace_root=tmp_path,
        hook_manager=parent_hooks,
    )
    await runner.run(GENERAL_PURPOSE, "do it")
    assert captured["hook_manager"] is None  # the child gets its OWN empty hook set ...
    assert captured["hook_manager"] is not parent_hooks  # ... NOT the parent's

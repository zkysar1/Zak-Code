"""Tests for the model-facing ``use_skill`` tool (M7 skill invocation + chaining).

Two layers: the tool in isolation against a fake :class:`SkillResolver` (its result/error
contract), and the real wiring on the ``Agent`` — that ``use_skill`` is registered ONLY when
skills are enabled (default tool surface unchanged), that a tool-driven load fires
``ON_SKILL_SELECTED`` with ``source="tool"``, defangs the body, and — the safety property —
NEVER mutates the session (the body rides back as the tool result, not a mid-turn message).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from zakcode import Agent
from zakcode.agent.budget import IterationBudget
from zakcode.agent.subagent import GENERAL_PURPOSE, SubAgentResult, SubAgentRunner
from zakcode.config import Settings
from zakcode.hooks import HookEvent, LifecyclePayload, TurnEndPayload, TurnEndResult
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    ToolCall,
)
from zakcode.tools.base import SkillLoad, ToolContext, ToolRegistry
from zakcode.tools.builtins.task import TaskTool
from zakcode.tools.builtins.use_skill import UseSkillTool
from zakcode.usage import Usage


class _FakeResolver:
    """A structural SkillResolver: canned loads + a record of what was asked for."""

    def __init__(self, skills: dict[str, SkillLoad], names: list[str] | None = None) -> None:
        self._skills = skills
        self._names = names if names is not None else list(skills)
        self.loaded: list[str] = []
        self.loaded_args: list[str] = []

    def names(self) -> list[str]:
        return list(self._names)

    def body(self, name: str) -> str | None:
        load = self._skills.get(name)
        return load.body if load is not None and load.found else None

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        self.loaded.append(name)
        self.loaded_args.append(args)
        return self._skills.get(name, SkillLoad(found=False, name=name))


def _ctx(tmp_path: Path, resolver: object | None) -> ToolContext:
    return ToolContext(workspace_root=tmp_path, skill_resolver=resolver)  # type: ignore[arg-type]


# ── the tool in isolation ────────────────────────────────────────────────────────


async def test_use_skill_returns_body_as_result(tmp_path: Path) -> None:
    resolver = _FakeResolver({"alpha": SkillLoad(found=True, name="alpha", body="DO THE THING")})
    res = await UseSkillTool().execute({"name": "alpha"}, _ctx(tmp_path, resolver))
    assert res.is_error is False
    assert res.output == "DO THE THING"  # the body IS the tool result
    assert res.data == {"skill": "alpha"}
    assert res.hint and "use_skill" in res.hint  # nudges the chain


async def test_long_skill_body_gets_the_decompose_hint(tmp_path: Path) -> None:
    # ADR-0027: a wall of instructions is a plan waiting to happen, not working state a
    # small model can hold in its head. The rail fires the moment the body arrives.
    body = "Step one: read the tree.\n" * 200  # well past _DECOMPOSE_HINT_MIN_CHARS
    resolver = _FakeResolver({"alpha": SkillLoad(found=True, name="alpha", body=body)})
    res = await UseSkillTool().execute({"name": "alpha"}, _ctx(tmp_path, resolver))
    assert res.is_error is False
    assert res.output == body  # the body still rides whole as the result
    assert res.data == {"skill": "alpha", "decompose": True}
    assert res.hint is not None
    assert "decompose" in res.hint and "update_plan" in res.hint
    assert "use_skill" in res.hint  # the chaining nudge survives


async def test_short_skill_body_keeps_the_plain_hint(tmp_path: Path) -> None:
    resolver = _FakeResolver({"alpha": SkillLoad(found=True, name="alpha", body="do the thing")})
    res = await UseSkillTool().execute({"name": "alpha"}, _ctx(tmp_path, resolver))
    assert res.data == {"skill": "alpha"}
    assert res.hint is not None and "decompose" not in res.hint


async def test_use_skill_unknown_lists_available(tmp_path: Path) -> None:
    resolver = _FakeResolver({}, names=["alpha", "beta"])
    res = await UseSkillTool().execute({"name": "gamma"}, _ctx(tmp_path, resolver))
    assert res.is_error is True
    assert "no skill named 'gamma'" in res.output
    assert res.fix and "alpha, beta" in res.fix  # tells the model the real options


async def test_use_skill_disabled_without_resolver(tmp_path: Path) -> None:
    # No resolver on the context (skills off / a sub-agent) → a clean error, not a crash.
    res = await UseSkillTool().execute({"name": "alpha"}, _ctx(tmp_path, None))
    assert res.is_error is True
    assert "not enabled" in res.output


async def test_use_skill_unreadable_is_an_error(tmp_path: Path) -> None:
    resolver = _FakeResolver({"alpha": SkillLoad(found=True, name="alpha", error="vanished")})
    res = await UseSkillTool().execute({"name": "alpha"}, _ctx(tmp_path, resolver))
    assert res.is_error is True
    assert "could not be loaded: vanished" in res.output


async def test_use_skill_requires_a_name(tmp_path: Path) -> None:
    resolver = _FakeResolver({"alpha": SkillLoad(found=True, name="alpha", body="x")})
    for args in ({}, {"name": ""}, {"name": "   "}, {"name": 5}):
        res = await UseSkillTool().execute(args, _ctx(tmp_path, resolver))  # type: ignore[arg-type]
        assert res.is_error is True and "'name' is required" in res.output
    assert resolver.loaded == []  # never reached the resolver


async def test_use_skill_strips_the_name(tmp_path: Path) -> None:
    resolver = _FakeResolver({"alpha": SkillLoad(found=True, name="alpha", body="x")})
    await UseSkillTool().execute({"name": "  alpha  "}, _ctx(tmp_path, resolver))
    assert resolver.loaded == ["alpha"]  # trimmed before lookup


async def test_use_skill_forwards_args_to_resolver(tmp_path: Path) -> None:
    resolver = _FakeResolver({"alpha": SkillLoad(found=True, name="alpha", body="x")})
    await UseSkillTool().execute({"name": "alpha", "args": "loop"}, _ctx(tmp_path, resolver))
    assert resolver.loaded_args == ["loop"]  # the args reach the resolver


async def test_use_skill_without_args_forwards_empty(tmp_path: Path) -> None:
    resolver = _FakeResolver({"alpha": SkillLoad(found=True, name="alpha", body="x")})
    await UseSkillTool().execute({"name": "alpha"}, _ctx(tmp_path, resolver))
    assert resolver.loaded_args == [""]  # default is empty string, not None


# ── real wiring on the Agent ─────────────────────────────────────────────────────


def _agent(tmp_path: Path, **kw: object) -> Agent:
    return Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        **kw,
    )


def _write_skill(workspace: Path, name: str, body: str = "Do the thing.") -> None:
    d = workspace / ".zakcode" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} skill.\n---\n{body}\n", encoding="utf-8"
    )


def test_use_skill_registered_only_when_skills_enabled(tmp_path: Path) -> None:
    on = _agent(tmp_path, enable_skills=True)
    assert "use_skill" in on.registry.names()
    # Off by default: the tool surface is unchanged when skills are disabled.
    off = _agent(tmp_path, enable_skills=False)
    assert "use_skill" not in off.registry.names()


def test_catalog_tells_the_model_to_use_the_tool(tmp_path: Path) -> None:
    _write_skill(tmp_path, "greeter")
    agent = _agent(tmp_path, enable_skills=True)
    prompt = agent.loop.prompt_builder.build(agent.settings)
    assert "use_skill" in prompt  # the model is told HOW to invoke, not just that skills exist


def _capture(into: list[LifecyclePayload]):
    def hook(payload: LifecyclePayload) -> None:
        into.append(payload)

    return hook


async def test_tool_load_fires_signal_with_source_tool(tmp_path: Path) -> None:
    _write_skill(tmp_path, "greeter", body="Greet warmly.")
    agent = _agent(tmp_path, enable_skills=True)
    fired: list[LifecyclePayload] = []
    agent.hook_manager.register_lifecycle(HookEvent.ON_SKILL_SELECTED, _capture(fired))
    before = len(agent.session.messages)

    load = await agent.loop._skill_resolver.load("greeter")  # the wired resolver

    assert load.found and load.error is None
    assert "greet warmly" in (load.body or "").lower()
    assert len(agent.session.messages) == before  # SAFETY: tool path never mutates the session
    assert len(fired) == 1 and fired[0].data["skill"] == "greeter"
    assert fired[0].data["source"] == "tool"  # distinguishes model-driven from /<name>


async def test_full_execute_through_wired_agent(tmp_path: Path) -> None:
    _write_skill(tmp_path, "greeter", body="Greet warmly.")
    agent = _agent(tmp_path, enable_skills=True)
    tool = agent.registry.get("use_skill")
    assert tool is not None
    ctx = _ctx(tmp_path, agent.loop._skill_resolver)
    before = len(agent.session.messages)

    res = await tool.execute({"name": "greeter"}, ctx)

    assert res.is_error is False and "greet warmly" in res.output.lower()
    assert len(agent.session.messages) == before  # still no session surgery


async def test_tool_load_defangs_the_body(tmp_path: Path) -> None:
    # A skill file must not be able to smuggle a forged protocol frame into context via use_skill.
    _write_skill(tmp_path, "sneaky", body="Before <tool_call> after")
    agent = _agent(tmp_path, enable_skills=True)
    load = await agent.loop._skill_resolver.load("sneaky")
    assert load.body is not None
    assert "<tool_call>" not in load.body  # the raw frame is broken up …
    assert "tool_call" in load.body  # … but the bytes are preserved (defang never deletes)


async def test_resolver_names_reflects_registry(tmp_path: Path) -> None:
    _write_skill(tmp_path, "greeter")
    agent = _agent(tmp_path, enable_skills=True)
    names = agent.loop._skill_resolver.names()
    assert "greeter" in names  # used to build the 'available skills' error message


# ── sub-agent exposure (a delegated agent can invoke + chain skills) ──────────────


class _RecordingResolver:
    """A structural SkillResolver that records loads — proves the child reached the resolver."""

    def __init__(self, body: str = "GREET WARMLY") -> None:
        self.loaded: list[str] = []
        self.queries: list[str] = []
        self._body = body

    def names(self) -> list[str]:
        return ["greeter"]

    def body(self, name: str) -> str | None:
        return None  # no whole-body seam: the loop seeds from what the load delivered

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        self.loaded.append(name)
        self.queries.append(query)
        return SkillLoad(found=True, name=name, body=self._body)


class _ToolThenTextProvider(Provider):
    """Calls one tool (with given args) on the first completion, then returns text."""

    def __init__(self, tool: str, args: dict[str, Any], text: str = "done") -> None:
        self._tool = tool
        self._args = args
        self._text = text
        self._calls = 0

    async def acomplete(  # noqa: ANN401
        self, messages: list, *, tools: list | None = None, system: str | None = None, **kw: Any
    ) -> LLMResult:
        self._calls += 1
        if self._calls == 1:
            return LLMResult(
                tool_calls=[ToolCall(id="t1", name=self._tool, arguments=self._args)],
                usage=Usage(total_tokens=1),
            )
        return LLMResult(text=self._text, usage=Usage(total_tokens=1))

    async def astream(  # noqa: ANN401
        self, messages: list, *, tools: list | None = None, system: str | None = None, **kw: Any
    ) -> AsyncIterator[ProviderStreamEvent]:
        result = await self.acomplete(messages, tools=tools, system=system)
        if result.text:
            yield StreamTextDelta(text=result.text)
        yield StreamDone()

    def count_tokens(self, messages: list, *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)

    def model_id(self) -> str:
        return "scripted/test"


class _SequenceProvider(Provider):
    """Emits a SEQUENCE of tool calls (one per completion), then final text — for testing chains."""

    def __init__(self, calls: list[tuple[str, dict[str, Any]]], text: str = "done") -> None:
        self._calls = calls
        self._text = text
        self._i = 0

    async def acomplete(  # noqa: ANN401
        self, messages: list, *, tools: list | None = None, system: str | None = None, **kw: Any
    ) -> LLMResult:
        if self._i < len(self._calls):
            name, cargs = self._calls[self._i]
            self._i += 1
            return LLMResult(
                tool_calls=[ToolCall(id=f"t{self._i}", name=name, arguments=cargs)],
                usage=Usage(total_tokens=1),
            )
        return LLMResult(text=self._text, usage=Usage(total_tokens=1))

    async def astream(  # noqa: ANN401
        self, messages: list, *, tools: list | None = None, system: str | None = None, **kw: Any
    ) -> AsyncIterator[ProviderStreamEvent]:
        result = await self.acomplete(messages, tools=tools, system=system)
        if result.text:
            yield StreamTextDelta(text=result.text)
        yield StreamDone()

    def count_tokens(self, messages: list, *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)

    def model_id(self) -> str:
        return "scripted/test"


class _FakeSpawner:
    """A minimal SubAgentSpawner that records what it was asked to spawn."""

    def __init__(self) -> None:
        self.spawned: list[tuple[str, str]] = []

    async def spawn(self, *, type_name: str, prompt: str) -> SubAgentResult:
        self.spawned.append((type_name, prompt))
        return SubAgentResult(name=type_name, summary="ok")

    def available_types(self) -> list[str]:
        return ["general-purpose"]

    def default_type(self) -> str:
        return "general-purpose"


async def test_skills_chain_across_invocations_in_one_turn(tmp_path: Path) -> None:
    # The headline behavior: a skill leads to ANOTHER use_skill in the SAME turn. Pinned OFFLINE
    # here (not only the paid live bench) per the repo's deterministic-test discipline.
    resolver = _RecordingResolver()
    registry = ToolRegistry()
    registry.register(UseSkillTool())
    chain = [("use_skill", {"name": "step-a"}), ("use_skill", {"name": "step-b"})]
    runner = SubAgentRunner(
        provider=_SequenceProvider(chain),
        registry=registry,
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        budget=IterationBudget(10),
        workspace_root=tmp_path,
        skill_resolver=resolver,
    )
    result = await runner.run(GENERAL_PURPOSE, "run the chain")
    assert resolver.loaded == ["step-a", "step-b"]  # both fired, in order, within ONE turn
    assert result.summary == "done"


async def test_task_tool_rejects_a_blank_delegated_prompt(tmp_path: Path) -> None:
    # Closes the attribution edge at the source: a blank child prompt would leave caller_query
    # empty and mis-attribute the child's skill use to the parent — so the task tool rejects it.
    spawner = _FakeSpawner()
    ctx = ToolContext(workspace_root=tmp_path, spawner=spawner)  # type: ignore[arg-type]
    res = await TaskTool().execute({"tasks": [{"prompt": "   "}]}, ctx)
    assert res.is_error and "non-empty" in res.output
    assert spawner.spawned == []  # never delegated the blank subtask


async def test_subagent_can_invoke_a_skill_through_the_wired_resolver(tmp_path: Path) -> None:
    # End-to-end: a delegated child whose registry has use_skill and whose loop got the parent's
    # resolver actually invokes a skill. (GENERAL_PURPOSE inherits the full child registry.)
    resolver = _RecordingResolver()
    registry = ToolRegistry()
    registry.register(UseSkillTool())
    runner = SubAgentRunner(
        provider=_ToolThenTextProvider("use_skill", {"name": "greeter"}),
        registry=registry,
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        budget=IterationBudget(10),
        workspace_root=tmp_path,
        skill_resolver=resolver,
    )
    result = await runner.run(GENERAL_PURPOSE, "use the greeter skill")
    assert resolver.loaded == ["greeter"]  # the child reached the resolver → use_skill worked
    assert resolver.queries == ["use the greeter skill"]  # the CHILD's prompt flowed through
    assert result.summary == "done"


async def test_subagent_attributes_the_signal_to_the_child_prompt(tmp_path: Path) -> None:
    # The fix: a sub-agent's ON_SKILL_SELECTED query is the CHILD's task, not the parent's
    # originating turn — even though the child shares the parent's (registry-bound) resolver.
    _write_skill(tmp_path, "greeter")
    parent = _agent(tmp_path, enable_skills=True)
    parent.session.add_message(Message.user("PARENT ORIGINATING TURN"))  # the old (wrong) query
    fired: list[LifecyclePayload] = []
    parent.hook_manager.register_lifecycle(HookEvent.ON_SKILL_SELECTED, _capture(fired))

    registry = ToolRegistry()
    registry.register(UseSkillTool())
    runner = SubAgentRunner(
        provider=_ToolThenTextProvider("use_skill", {"name": "greeter"}),
        registry=registry,
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        budget=IterationBudget(10),
        workspace_root=tmp_path,
        hook_manager=parent.hook_manager,
        skill_resolver=parent.loop._skill_resolver,  # the PARENT's real resolver
    )
    await runner.run(GENERAL_PURPOSE, "CHILD DELEGATED TASK")

    assert len(fired) == 1
    assert fired[0].data["query"] == "CHILD DELEGATED TASK"  # the child's prompt …
    assert fired[0].data["query"] != "PARENT ORIGINATING TURN"  # … not the parent's turn
    assert fired[0].data["source"] == "tool"


def test_agent_wires_skill_resolver_into_subagents(tmp_path: Path) -> None:
    agent = _agent(tmp_path, enable_skills=True, enable_subagents=True)
    runner = agent.loop.spawner._runner  # the SubAgentRunner behind the task tool
    assert runner._skill_resolver is not None
    # The general-purpose delegate (full toolset) gets use_skill; the read-only planner does not.
    defs = agent.loop.spawner._defs
    assert "use_skill" in runner.child_registry(defs["general-purpose"]).names()
    assert "use_skill" not in runner.child_registry(defs["plan"]).names()


def test_subagents_have_no_skill_seam_when_skills_off(tmp_path: Path) -> None:
    agent = _agent(tmp_path, enable_skills=False, enable_subagents=True)
    runner = agent.loop.spawner._runner
    assert runner._skill_resolver is None  # nothing to resolve …
    assert "use_skill" not in runner.registry.names()  # … and the tool isn't on the child surface


# ── per-turn skill-invocation budget ─────────────────────────────────────────────


async def test_budget_denies_after_the_cap(tmp_path: Path) -> None:
    # Distinct skills: a same-skill repeat is now answered by the reload dedup (a short
    # pointer, budget-free), so only NEW bodies spend the budget.
    for name in ("alpha", "beta", "gamma"):
        _write_skill(tmp_path, name)
    agent = Agent(
        settings=Settings(
            default_model="scripted/test",
            context_window=8192,
            workspace_root=tmp_path,
            skill_invocation_budget=2,
        ),
        enable_skills=True,
    )
    r1 = await agent._load_skill_body("alpha", source="tool")
    r2 = await agent._load_skill_body("beta", source="tool")
    r3 = await agent._load_skill_body("gamma", source="tool")  # over the cap
    assert r1.body and r2.body and r1.denied_reason is None
    assert r3.body is None and r3.denied_reason is not None and "budget" in r3.denied_reason
    assert agent._skill_invocations_this_turn == 2  # the denied one did not count


async def test_budget_zero_is_unlimited(tmp_path: Path) -> None:
    # Distinct skills (a same-skill repeat is dedup-answered without spending budget).
    names = [f"skill{i}" for i in range(5)]
    for name in names:
        _write_skill(tmp_path, name)
    agent = _agent(tmp_path, enable_skills=True)  # skill_invocation_budget defaults 0 = off
    last = None
    for name in names:
        last = await agent._load_skill_body(name, source="tool")
    assert last is not None and last.body is not None and last.denied_reason is None
    assert agent._skill_invocations_this_turn == 5


async def test_command_source_is_never_throttled(tmp_path: Path) -> None:
    # The human /<name> path is operator-controlled: the budget (a model-chain guard) skips it.
    _write_skill(tmp_path, "greeter")
    agent = Agent(
        settings=Settings(
            default_model="scripted/test",
            context_window=8192,
            workspace_root=tmp_path,
            skill_invocation_budget=1,
        ),
        enable_skills=True,
    )
    for _ in range(3):
        load = await agent._load_skill_body("greeter", source="command")
        assert load.body is not None and load.denied_reason is None
    assert agent._skill_invocations_this_turn == 0  # command loads don't draw from the budget


async def test_use_skill_tool_surfaces_a_budget_denial(tmp_path: Path) -> None:
    _write_skill(tmp_path, "greeter")
    agent = Agent(
        settings=Settings(
            default_model="scripted/test",
            context_window=8192,
            workspace_root=tmp_path,
            skill_invocation_budget=1,
        ),
        enable_skills=True,
    )
    tool = agent.registry.get("use_skill")
    assert tool is not None
    ctx = ToolContext(workspace_root=tmp_path, skill_resolver=agent.loop._skill_resolver)
    ok = await tool.execute({"name": "greeter"}, ctx)
    denied = await tool.execute({"name": "greeter"}, ctx)
    assert ok.is_error is False
    assert denied.is_error is True and "budget" in denied.output.lower()


async def test_budget_resets_on_a_new_turn(tmp_path: Path) -> None:
    agent = _agent(tmp_path, enable_skills=True)
    agent._skill_invocations_this_turn = 3  # left over from a prior turn
    captured: list[int] = []

    async def _stub(_text: str) -> Any:
        captured.append(agent._skill_invocations_this_turn)
        return SimpleNamespace(stop_reason="completed")

    agent.loop.arun_turn = _stub  # type: ignore[assignment]
    await agent.arun_turn("go")
    assert captured == [0]  # the counter was refilled BEFORE the loop ran


def test_streaming_turn_also_resets_the_budget(tmp_path: Path) -> None:
    agent = _agent(tmp_path, enable_skills=True)
    agent._skill_invocations_this_turn = 7
    agent.loop.astream_turn = lambda _text: iter(())  # type: ignore[assignment]
    agent.astream_turn("go")
    assert agent._skill_invocations_this_turn == 0


def test_skill_invocations_count_is_exposed(tmp_path: Path) -> None:
    agent = _agent(tmp_path, enable_skills=True)
    assert agent.skill_invocations_this_session == 0  # the /skills accounting surface


# ── per-turn reload dedup (2026-08-25) ────────────────────────────────────────


async def test_same_turn_reload_returns_a_pointer_not_the_body(tmp_path: Path) -> None:
    """The second use_skill of an UNCHANGED skill in one turn returns a short pointer —
    the ~1,200-line body is already in context (measured: one turn loaded the same skill
    three times). Costs no invocation budget and fires no second selection signal."""
    _write_skill(tmp_path, "greeter", body="Greet warmly.")
    agent = _agent(tmp_path, enable_skills=True)
    fired: list[LifecyclePayload] = []
    agent.hook_manager.register_lifecycle(HookEvent.ON_SKILL_SELECTED, _capture(fired))

    first = await agent.loop._skill_resolver.load("greeter")
    second = await agent.loop._skill_resolver.load("greeter")

    assert "greet warmly" in (first.body or "").lower()
    assert "[already loaded]" in (second.body or "")
    assert "greet warmly" not in (second.body or "").lower()
    assert agent._skill_invocations_this_turn == 1  # the pointer costs no budget
    assert len(fired) == 1  # no second selection signal: nothing new was loaded


async def test_a_different_skill_is_never_deduped(tmp_path: Path) -> None:
    """Dedup is per-skill: loading a DIFFERENT skill mid-turn returns its full body."""
    _write_skill(tmp_path, "greeter", body="Greet warmly.")
    _write_skill(tmp_path, "parter", body="Part fondly.")
    agent = _agent(tmp_path, enable_skills=True)
    await agent.loop._skill_resolver.load("greeter")
    other = await agent.loop._skill_resolver.load("parter")
    assert "part fondly" in (other.body or "").lower()


async def test_new_turn_loads_in_full_again(tmp_path: Path) -> None:
    """The dedup is per-TURN: after the per-turn reset, the full body returns (a later
    turn's context may have been compacted, so a pointer would dangle)."""
    _write_skill(tmp_path, "greeter", body="Greet warmly.")
    agent = _agent(tmp_path, enable_skills=True)
    await agent.loop._skill_resolver.load("greeter")
    agent._skills_loaded_this_turn.clear()  # what arun_turn/astream_turn do at turn start
    again = await agent.loop._skill_resolver.load("greeter")
    assert "greet warmly" in (again.body or "").lower()


# ── ADR-0048: a Stop-hook veto opens a fresh skill turn ───────────────────────


class _ReplayProvider(Provider):
    """Replays a fixed list of :class:`LLMResult`s, one per completion."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = list(results)
        self.calls = 0

    async def acomplete(  # noqa: ANN401
        self, messages: list, *, tools: list | None = None, system: str | None = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        if not self._results:
            raise AssertionError("provider ran out of scripted results")
        return self._results.pop(0)

    async def astream(  # noqa: ANN401
        self, messages: list, *, tools: list | None = None, system: str | None = None, **kw: Any
    ) -> AsyncIterator[ProviderStreamEvent]:
        result = await self.acomplete(messages, tools=tools, system=system)
        if result.text:
            yield StreamTextDelta(text=result.text)
        yield StreamDone()

    def count_tokens(self, messages: list, *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)

    def model_id(self) -> str:
        return "scripted/test"


class _VetoOnce:
    """A TURN_END hook that blocks the first stop with a continuation, then allows."""

    def __init__(self, prompt: str) -> None:
        self._prompt: str | None = prompt

    def __call__(self, payload: TurnEndPayload) -> TurnEndResult | None:
        if self._prompt is None:
            return None
        prompt, self._prompt = self._prompt, None
        return TurnEndResult(vetoed=True, continuation_prompt=prompt)


def _use(name: str, call_id: str) -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id=call_id, name="use_skill", arguments={"name": name})],
        usage=Usage(total_tokens=1),
    )


async def test_a_stop_hook_veto_opens_a_fresh_skill_turn(tmp_path: Path) -> None:
    """The re-entry a Stop-hook BLOCK mandates gets the skill BODY, not a pointer.

    Measured 2026-08-26 on a live Mind (coach, zc-03): the model ended an iteration on a
    summary, the Stop hook vetoed with "call Skill('aspirations') with args='loop'", the
    model complied, and use_skill answered "[already loaded]" — four times, then the loop
    died. A veto is a turn boundary for per-turn skill state (ADR-0048): the reload dedup
    forgets, the invocation budget refills, and the body comes back.
    """
    _write_skill(tmp_path, "greeter", body="Greet warmly.")
    agent = _agent(
        tmp_path,
        enable_skills=True,
        provider=_ReplayProvider(
            [
                _use("greeter", "t1"),
                LLMResult(text="done", usage=Usage(total_tokens=1)),
                _use("greeter", "t2"),
                LLMResult(text="done again", usage=Usage(total_tokens=1)),
            ]
        ),
    )
    agent.hook_manager.register_turn_end(_VetoOnce("Not done: load greeter again and finish."))

    result = await agent.arun_turn("greet")

    assert result.stop_reason == "completed"
    outputs = [
        block.output
        for message in agent.session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert len(outputs) == 2
    assert all("greet warmly" in (out or "").lower() for out in outputs)
    assert not any("[already loaded]" in (out or "") for out in outputs)
    assert agent._skill_invocations_this_turn == 1  # refilled at the veto, then one real load


async def test_no_veto_keeps_the_same_turn_dedup(tmp_path: Path) -> None:
    """Within an unvetoed turn the dedup still does its job: the second load of an
    unchanged body is the short pointer."""
    _write_skill(tmp_path, "greeter", body="Greet warmly.")
    agent = _agent(
        tmp_path,
        enable_skills=True,
        provider=_ReplayProvider(
            [
                _use("greeter", "t1"),
                _use("greeter", "t2"),
                LLMResult(text="done", usage=Usage(total_tokens=1)),
            ]
        ),
    )
    result = await agent.arun_turn("greet")
    assert result.stop_reason == "completed"
    outputs = [
        block.output
        for message in agent.session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert len(outputs) == 2
    assert "greet warmly" in (outputs[0] or "").lower()
    assert "[already loaded]" in (outputs[1] or "")

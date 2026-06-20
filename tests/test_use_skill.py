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
from zakcode.agent.subagent import GENERAL_PURPOSE, SubAgentRunner
from zakcode.config import Settings
from zakcode.hooks import HookEvent, LifecyclePayload
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
from zakcode.tools.builtins.use_skill import UseSkillTool
from zakcode.usage import Usage


class _FakeResolver:
    """A structural SkillResolver: canned loads + a record of what was asked for."""

    def __init__(self, skills: dict[str, SkillLoad], names: list[str] | None = None) -> None:
        self._skills = skills
        self._names = names if names is not None else list(skills)
        self.loaded: list[str] = []

    def names(self) -> list[str]:
        return list(self._names)

    async def load(self, name: str) -> SkillLoad:
        self.loaded.append(name)
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


# ── real wiring on the Agent ─────────────────────────────────────────────────────


def _agent(tmp_path: Path, **kw: object) -> Agent:
    return Agent(settings=Settings(default_model="scripted/test", workspace_root=tmp_path), **kw)


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
        self._body = body

    def names(self) -> list[str]:
        return ["greeter"]

    async def load(self, name: str) -> SkillLoad:
        self.loaded.append(name)
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
        return Capabilities()

    def model_id(self) -> str:
        return "scripted/test"


async def test_subagent_can_invoke_a_skill_through_the_wired_resolver(tmp_path: Path) -> None:
    # End-to-end: a delegated child whose registry has use_skill and whose loop got the parent's
    # resolver actually invokes a skill. (GENERAL_PURPOSE inherits the full child registry.)
    resolver = _RecordingResolver()
    registry = ToolRegistry()
    registry.register(UseSkillTool())
    runner = SubAgentRunner(
        provider=_ToolThenTextProvider("use_skill", {"name": "greeter"}),
        registry=registry,
        settings=Settings(default_model="scripted/test", workspace_root=tmp_path),
        budget=IterationBudget(10),
        workspace_root=tmp_path,
        skill_resolver=resolver,
    )
    result = await runner.run(GENERAL_PURPOSE, "use the greeter skill")
    assert resolver.loaded == ["greeter"]  # the child reached the resolver → use_skill worked
    assert result.summary == "done"


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
    _write_skill(tmp_path, "greeter")
    agent = Agent(
        settings=Settings(
            default_model="scripted/test", workspace_root=tmp_path, skill_invocation_budget=2
        ),
        enable_skills=True,
    )
    r1 = await agent._load_skill_body("greeter", source="tool")
    r2 = await agent._load_skill_body("greeter", source="tool")
    r3 = await agent._load_skill_body("greeter", source="tool")  # over the cap
    assert r1.body and r2.body and r1.denied_reason is None
    assert r3.body is None and r3.denied_reason is not None and "budget" in r3.denied_reason
    assert agent._skill_invocations_this_turn == 2  # the denied one did not count


async def test_budget_zero_is_unlimited(tmp_path: Path) -> None:
    _write_skill(tmp_path, "greeter")
    agent = _agent(tmp_path, enable_skills=True)  # skill_invocation_budget defaults 0 = off
    last = None
    for _ in range(5):
        last = await agent._load_skill_body("greeter", source="tool")
    assert last is not None and last.body is not None and last.denied_reason is None
    assert agent._skill_invocations_this_turn == 5


async def test_command_source_is_never_throttled(tmp_path: Path) -> None:
    # The human /<name> path is operator-controlled: the budget (a model-chain guard) skips it.
    _write_skill(tmp_path, "greeter")
    agent = Agent(
        settings=Settings(
            default_model="scripted/test", workspace_root=tmp_path, skill_invocation_budget=1
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
            default_model="scripted/test", workspace_root=tmp_path, skill_invocation_budget=1
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

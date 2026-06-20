"""Tests for the model-facing ``use_skill`` tool (M7 skill invocation + chaining).

Two layers: the tool in isolation against a fake :class:`SkillResolver` (its result/error
contract), and the real wiring on the ``Agent`` — that ``use_skill`` is registered ONLY when
skills are enabled (default tool surface unchanged), that a tool-driven load fires
``ON_SKILL_SELECTED`` with ``source="tool"``, defangs the body, and — the safety property —
NEVER mutates the session (the body rides back as the tool result, not a mid-turn message).
"""

from __future__ import annotations

from pathlib import Path

from zakcode import Agent
from zakcode.config import Settings
from zakcode.hooks import HookEvent, LifecyclePayload
from zakcode.tools.base import SkillLoad, ToolContext
from zakcode.tools.builtins.use_skill import UseSkillTool


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

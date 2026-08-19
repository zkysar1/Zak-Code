"""Tests for skills wiring into the Agent facade + CLI (M7-2)."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from zakcode import Agent
from zakcode.cli import _render_skills, _skill_command_turn
from zakcode.config import Settings
from zakcode.hooks import HookEvent, LifecyclePayload
from zakcode.messages import Message

_SKILL = """\
---
name: greeter
description: Greet the user warmly.
---
# Greeter
Always greet the user by name before doing anything else.
"""


def _write_skill(workspace: Path, name: str, text: str = _SKILL) -> None:
    d = workspace / ".zakcode" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")


def _agent(tmp_path: Path, **kw: object) -> Agent:
    return Agent(settings=Settings(default_model="scripted/test", workspace_root=tmp_path), **kw)


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    return Console(file=buf, width=100), buf


def test_no_skills_by_default(tmp_path: Path) -> None:
    _write_skill(tmp_path, "g")
    agent = _agent(tmp_path)  # enable_skills defaults False
    assert agent.skill_registry is None


def test_enable_skills_discovers_catalog(tmp_path: Path) -> None:
    _write_skill(tmp_path, "g")
    agent = _agent(tmp_path, enable_skills=True)
    assert agent.skill_registry is not None
    assert "greeter" in agent.skill_registry.names()
    # L0 catalog is in the (cacheable) system prompt; the body is NOT loaded yet.
    assert "greeter" in agent.loop.prompt_builder.build(agent.settings)
    assert agent.skill_registry.get("greeter").body_loaded is False


def test_render_skills_lists_them(tmp_path: Path) -> None:
    _write_skill(tmp_path, "g")
    agent = _agent(tmp_path, enable_skills=True)
    console, buf = _console()
    _render_skills(console, agent)
    out = buf.getvalue()
    assert "greeter" in out
    assert "Greet the user warmly" in out


def test_render_skills_none(tmp_path: Path) -> None:
    # Empty state: with skills DISABLED there is no registry to show. (With skills enabled the
    # bundled `research` playbook is always discovered, so the empty state is unreachable there.)
    agent = _agent(tmp_path, enable_skills=False)
    console, buf = _console()
    _render_skills(console, agent)
    assert "no skills discovered" in buf.getvalue()


def test_render_skills_includes_bundled_research(tmp_path: Path) -> None:
    # The shipped bundled skill shows up even in a fresh workspace with nothing authored.
    agent = _agent(tmp_path, enable_skills=True)
    console, buf = _console()
    _render_skills(console, agent)
    assert "research" in buf.getvalue()


def test_slash_skill_composes_the_turn(tmp_path: Path) -> None:
    # Claude Code slash semantics: /<skill> RUNS now. The helper composes the body as THIS
    # turn's input for the REPL's shared streaming path — it must NOT inject into the session
    # itself (delivery is the turn runner's job; injecting here would double the body).
    _write_skill(tmp_path, "g")
    agent = _agent(tmp_path, enable_skills=True)
    before = len(agent.session.messages)
    console, buf = _console()
    outcome = _skill_command_turn(console, agent, "greeter")
    assert outcome.handled is True
    assert outcome.turn_text is not None
    assert outcome.turn_text.startswith("[skill: greeter]")
    assert "greet the user by name" in outcome.turn_text.lower()
    # The body was loaded (lazily) but the session is untouched until the turn runs.
    assert agent.skill_registry.get("greeter").body_loaded is True
    assert len(agent.session.messages) == before
    assert "running skill" in buf.getvalue()


def test_slash_skill_body_unreadable_is_handled(tmp_path: Path) -> None:
    # A skill discovered at startup whose SKILL.md vanishes before invocation must not
    # crash the REPL: the helper reports the error and stays handled (it WAS a skill).
    _write_skill(tmp_path, "g")
    agent = _agent(tmp_path, enable_skills=True)
    (tmp_path / ".zakcode" / "skills" / "g" / "SKILL.md").unlink()
    before = len(agent.session.messages)
    console, buf = _console()
    outcome = _skill_command_turn(console, agent, "greeter")
    assert outcome.handled is True  # still a skill name; do not fall through to plugins
    assert outcome.turn_text is None  # nothing to run
    assert len(agent.session.messages) == before  # nothing injected
    assert "could not load skill" in buf.getvalue()


def test_slash_unknown_skill_falls_through(tmp_path: Path) -> None:
    agent = _agent(tmp_path, enable_skills=True)
    console, _ = _console()
    assert _skill_command_turn(console, agent, "nope").handled is False


# ── skill-selection signal (ON_SKILL_SELECTED): the seam a learning mind records from ──


def _capture(into: list[LifecyclePayload]):
    def hook(payload: LifecyclePayload) -> None:
        into.append(payload)

    return hook


async def test_invoke_skill_fires_selection_signal(tmp_path: Path) -> None:
    _write_skill(tmp_path, "g")
    agent = _agent(tmp_path, enable_skills=True)
    fired: list[LifecyclePayload] = []
    agent.hook_manager.register_lifecycle(HookEvent.ON_SKILL_SELECTED, _capture(fired))

    result = await agent.invoke_skill("greeter")

    assert result.invoked is True and result.error is None
    assert len(fired) == 1
    payload = fired[0]
    assert payload.event is HookEvent.ON_SKILL_SELECTED
    assert payload.data["skill"] == "greeter"
    assert payload.data["source"] == "command"
    assert payload.session_id == agent.session.id
    # The body was injected too (the signal accompanies the actual use).
    assert "greet the user by name" in agent.session.messages[-1].text.lower()


async def test_skill_signal_carries_triggering_query(tmp_path: Path) -> None:
    _write_skill(tmp_path, "g")
    agent = _agent(tmp_path, enable_skills=True)
    agent.session.add_message(Message.user("help me greet bob"))  # the prior turn
    fired: list[LifecyclePayload] = []
    agent.hook_manager.register_lifecycle(HookEvent.ON_SKILL_SELECTED, _capture(fired))

    await agent.invoke_skill("greeter")

    assert fired and fired[0].data["query"] == "help me greet bob"


async def test_load_failure_does_not_fire_signal(tmp_path: Path) -> None:
    _write_skill(tmp_path, "g")
    agent = _agent(tmp_path, enable_skills=True)
    (tmp_path / ".zakcode" / "skills" / "g" / "SKILL.md").unlink()
    fired: list[LifecyclePayload] = []
    agent.hook_manager.register_lifecycle(HookEvent.ON_SKILL_SELECTED, _capture(fired))

    result = await agent.invoke_skill("greeter")

    assert result.invoked is True and result.error is not None
    assert fired == []  # a skill that did not actually load is not a "selection"


async def test_unknown_skill_does_not_fire_signal(tmp_path: Path) -> None:
    agent = _agent(tmp_path, enable_skills=True)
    fired: list[LifecyclePayload] = []
    agent.hook_manager.register_lifecycle(HookEvent.ON_SKILL_SELECTED, _capture(fired))

    result = await agent.invoke_skill("nope")

    assert result.invoked is False and fired == []


def test_slash_skill_no_registry_is_safe(tmp_path: Path) -> None:
    agent = _agent(tmp_path)  # skills disabled → no registry
    console, _ = _console()
    assert _skill_command_turn(console, agent, "anything").handled is False


async def test_composed_skill_turn_runs_like_any_turn(tmp_path: Path) -> None:
    # End-to-end on the offline scripted provider: the composed text IS a normal turn — the
    # session gains the [skill: …] user message plus the model's response, exactly the shape
    # invoke_skill + a follow-up message used to need two steps for.
    _write_skill(tmp_path, "g")
    agent = _agent(tmp_path, enable_skills=True)
    outcome = await agent.compose_skill_turn("greeter")
    assert outcome.turn_text is not None
    await agent.arun_turn(outcome.turn_text)
    user_texts = [m.text for m in agent.session.messages if m.role == "user"]
    assert any(t.startswith("[skill: greeter]") for t in user_texts)
    assert agent.session.messages[-1].role == "assistant"


# ── extra_skill_dirs (--skill-dir) ──────────────────────────────────────────────


_EXT_SKILL = """\
---
name: ext-greeter
description: Greet from an external directory.
---
# External Greeter
Always greet with an external greeting.
"""


def _write_ext_skill(root: Path, name: str, text: str = _EXT_SKILL) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")


def test_agent_extra_skill_dirs(tmp_path: Path) -> None:
    ext = tmp_path / "external-skills"
    _write_ext_skill(ext, "ext")
    agent = _agent(tmp_path, enable_skills=True, extra_skill_dirs=[str(ext)])
    assert agent.skill_registry is not None
    assert "ext-greeter" in agent.skill_registry.names()
    # L0 catalog includes the external skill
    prompt = agent.loop.prompt_builder.build(agent.settings)
    assert "ext-greeter" in prompt


def test_agent_extra_skill_dir_body_is_lazy(tmp_path: Path) -> None:
    ext = tmp_path / "external-skills"
    _write_ext_skill(ext, "ext")
    agent = _agent(tmp_path, enable_skills=True, extra_skill_dirs=[str(ext)])
    skill = agent.skill_registry.get("ext-greeter")
    assert skill is not None
    assert skill.body_loaded is False
    body = skill.body()
    assert "external greeting" in body
    assert skill.body_loaded is True


def test_agent_extra_skill_dir_shadows_project(tmp_path: Path) -> None:
    # Write a project skill and an external skill with the same name.
    _write_skill(tmp_path, "g")  # name="greeter" in project .zakcode/skills
    ext = tmp_path / "external-skills"
    _write_ext_skill(ext, "g2", "---\nname: greeter\ndescription: External wins.\n---\nbody\n")
    agent = _agent(tmp_path, enable_skills=True, extra_skill_dirs=[str(ext)])
    skill = agent.skill_registry.get("greeter")
    assert skill is not None
    assert skill.description == "External wins."


def test_invoke_external_skill(tmp_path: Path) -> None:
    ext = tmp_path / "external-skills"
    _write_ext_skill(ext, "ext")
    agent = _agent(tmp_path, enable_skills=True, extra_skill_dirs=[str(ext)])
    console, buf = _console()
    outcome = _skill_command_turn(console, agent, "ext-greeter")
    assert outcome.handled is True
    assert agent.skill_registry.get("ext-greeter").body_loaded is True
    assert outcome.turn_text is not None
    assert "external greeting" in outcome.turn_text.lower()
    assert "running skill" in buf.getvalue()

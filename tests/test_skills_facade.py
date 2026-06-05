"""Tests for skills wiring into the Agent facade + CLI (M7-2)."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from zakcode import Agent
from zakcode.cli import _invoke_skill, _render_skills
from zakcode.config import Settings

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
    agent = _agent(tmp_path, enable_skills=True)  # enabled, nothing on disk
    console, buf = _console()
    _render_skills(console, agent)
    assert "no skills discovered" in buf.getvalue()


def test_invoke_skill_injects_body(tmp_path: Path) -> None:
    _write_skill(tmp_path, "g")
    agent = _agent(tmp_path, enable_skills=True)
    before = len(agent.session.messages)
    console, buf = _console()
    handled = _invoke_skill(console, agent, "greeter")
    assert handled is True
    # The body was loaded (lazily) and injected as a user message.
    assert agent.skill_registry.get("greeter").body_loaded is True
    assert len(agent.session.messages) == before + 1
    last = agent.session.messages[-1]
    assert last.role == "user"
    assert "greet the user by name" in last.text.lower()
    assert "loaded skill" in buf.getvalue()


def test_invoke_skill_body_unreadable_is_handled(tmp_path: Path) -> None:
    # A skill discovered at startup whose SKILL.md vanishes before invocation must not
    # crash the REPL: _invoke_skill reports the error and returns True (it WAS a skill).
    _write_skill(tmp_path, "g")
    agent = _agent(tmp_path, enable_skills=True)
    (tmp_path / ".zakcode" / "skills" / "g" / "SKILL.md").unlink()
    before = len(agent.session.messages)
    console, buf = _console()
    handled = _invoke_skill(console, agent, "greeter")
    assert handled is True  # still a skill name; do not fall through to plugin dispatch
    assert len(agent.session.messages) == before  # nothing injected
    assert "could not load skill" in buf.getvalue()


def test_invoke_unknown_skill_returns_false(tmp_path: Path) -> None:
    agent = _agent(tmp_path, enable_skills=True)
    console, _ = _console()
    assert _invoke_skill(console, agent, "nope") is False


def test_invoke_skill_no_registry_is_safe(tmp_path: Path) -> None:
    agent = _agent(tmp_path)  # skills disabled → no registry
    console, _ = _console()
    assert _invoke_skill(console, agent, "anything") is False


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
    handled = _invoke_skill(console, agent, "ext-greeter")
    assert handled is True
    assert agent.skill_registry.get("ext-greeter").body_loaded is True
    last = agent.session.messages[-1]
    assert last.role == "user"
    assert "external greeting" in last.text.lower()
    assert "loaded skill" in buf.getvalue()

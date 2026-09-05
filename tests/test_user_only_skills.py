"""User-only skills are invisible to the model's seams (ADR-0109).

Field transcript 2026-09-05: "ok, clear that plan, and lets start from scratch" was classified
as implying the Mind's ``/start`` — a control command whose own description says "USER-ONLY —
Claude must NEVER invoke /start" — a ``run /start`` step was seeded, the plan gate refused the
model's text finish, and the model ran ``use_skill(start)``. Claude Code's
``disable-model-invocation: true`` is the machine-readable form of that prose; these tests pin
every seam that honors it: the parsed flag, the model-facing catalog and prompt, the
``use_skill`` refusal (the human command path untouched), the classify side-call, and the
loop's plan seeders on both routes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import zakcode
from zakcode.agent.loop import AgentLoop
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.providers.routing import DifficultyVerdict
from zakcode.session.store import Session
from zakcode.skills import Skill, SkillRegistry, parse_frontmatter
from zakcode.tools.base import SkillLoad, ToolRegistry

START_MD = """\
---
name: start
description: Creates or resumes an agent. USER-ONLY — Claude must NEVER invoke /start.
disable-model-invocation: true
triggers:
  - "/start"
---
# /start
Bring the agent up.
"""

FORGE_MD = """\
---
name: forge-skill
description: Forge a new skill from a description
---
# /forge-skill
Forge it.
"""


def _registry(tmp_path: Path) -> SkillRegistry:
    reg = SkillRegistry()
    for dirname, text in (("start", START_MD), ("forge-skill", FORGE_MD)):
        d = tmp_path / dirname
        d.mkdir()
        (d / "SKILL.md").write_text(text, encoding="utf-8")
        fm, _ = parse_frontmatter(text)
        reg.add(Skill(fm, d / "SKILL.md"))
    return reg


# ── the flag and the catalogs ────────────────────────────────────────────────────────────


def test_the_flag_parses_and_defaults_to_invocable(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    start, forge = reg.get("start"), reg.get("forge-skill")
    assert start is not None and forge is not None
    assert start.frontmatter.extras["disable_model_invocation"] == "true"  # hyphen normalized
    assert start.model_invocable is False
    assert forge.model_invocable is True


def test_model_catalog_omits_user_only_skills_but_the_operator_catalog_keeps_them(
    tmp_path: Path,
) -> None:
    reg = _registry(tmp_path)
    assert [n for n, _ in reg.catalog()] == ["start", "forge-skill"]  # /skills: they CAN type it
    assert [n for n, _ in reg.model_catalog()] == ["forge-skill"]
    assert reg.user_only_names() == ["start"]


def test_the_prompt_names_user_only_commands_without_a_use_skill_call(tmp_path: Path) -> None:
    rendered = _registry(tmp_path).render_catalog()
    assert 'use_skill(name="forge-skill")' in rendered
    assert 'use_skill(name="start")' not in rendered
    assert "User-only commands (/start)" in rendered
    assert "never call, plan, or seed one" in rendered


# ── the tool path refuses; the human command path does not ──────────────────────────────


@pytest.mark.asyncio
async def test_use_skill_refuses_a_user_only_skill_and_the_command_path_runs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = zakcode.Agent(workspace_root=tmp_path)
    monkeypatch.setattr(agent, "skill_registry", _registry(tmp_path))

    tool = await agent._load_skill_body("start", source="tool")
    assert tool.found and tool.body is None
    assert tool.denied_reason is not None and "user-only" in tool.denied_reason
    assert "/start" in tool.denied_reason

    command = await agent._load_skill_body("start", source="command")
    assert command.found and command.denied_reason is None
    assert command.body is not None and "Bring the agent up." in command.body

    other = await agent._load_skill_body("forge-skill", source="tool")
    assert other.found and other.denied_reason is None and other.body is not None


# ── the classify side-call never names a user-only skill ─────────────────────────────────


class _Stub(Provider):
    def __init__(self, json_text: str) -> None:
        self.json_text = json_text
        self.systems: list[str] = []

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: Any = None,
        response_format: Any = None,
        **kw: Any,
    ) -> LLMResult:
        self.systems.append(system or "")
        return LLMResult(text=self.json_text)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


@pytest.mark.asyncio
async def test_side_call_catalog_excludes_user_only_and_drops_the_guess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    monkeypatch.setattr(agent, "skill_registry", _registry(tmp_path))
    stub = _Stub('{"difficulty": "quick", "skill": "start"}')
    monkeypatch.setattr(agent, "_resolve_task_provider", lambda c: (stub, "classify/m"))
    verdict = await agent._classify_difficulty("run /start alpha", 0.0)
    assert verdict == DifficultyVerdict("quick_code", None)  # category kept, skill dropped
    assert "forge-skill" in stub.systems[-1] and "- start" not in stub.systems[-1]


# ── the loop never seeds a step for a user-only skill ────────────────────────────────────


class _Text(Provider):
    def __init__(self, text: str) -> None:
        self.text = text

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        return LLMResult(text=self.text)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=200_000)


class _Resolver:
    def names(self) -> list[str]:
        return ["start", "boot", "forge-skill"]

    def user_only_names(self) -> list[str]:
        return ["start"]

    def body(self, name: str) -> str | None:
        return None

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        raise AssertionError("no skill is loaded in these tests")


def _loop(tmp_path: Path, provider: Provider, verdict: DifficultyVerdict) -> AgentLoop:
    async def classifier(user_text: str, context_frac: float) -> DifficultyVerdict:
        return verdict

    return AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=6,
        main_provider_for=lambda category: provider,
        difficulty_classifier=classifier,
        skill_resolver=_Resolver(),
    )


def test_compound_seeder_skips_the_user_only_skill(tmp_path: Path) -> None:
    loop = _loop(tmp_path, _Text("Sure."), DifficultyVerdict("quick_code", None))
    asyncio.run(loop.arun_turn("do /start alpha and /boot"))
    titles = [t.title for t in loop.session.task_network.tasks]
    assert titles == ["run /boot"]  # /start never becomes a step the model cannot execute


def test_an_implied_user_only_skill_seeds_nothing_and_arms_nothing(tmp_path: Path) -> None:
    """Belt and braces: even if a classifier named it, the loop refuses the seed."""
    loop = _loop(tmp_path, _Text("Plan cleared."), DifficultyVerdict("quick_code", "start"))
    result = asyncio.run(loop.arun_turn("ok, clear that plan, and lets start from scratch"))
    assert result.stop_reason == "completed"
    assert loop.session.task_network.tasks == []
    assert not any(m.role == "user" and "/start" in m.text for m in loop.session.messages)


def test_an_implied_skill_step_is_marked_as_a_cancellable_guess(tmp_path: Path) -> None:
    loop = _loop(tmp_path, _Text("Working on it."), DifficultyVerdict("quick_code", "forge-skill"))
    asyncio.run(loop.arun_turn("finish forging this skill"))
    step = next(t for t in loop.session.task_network.tasks if t.title == "run /forge-skill")
    assert "harness guess" in (step.note or "")
    assert "cancelled if the request did not ask for it" in (step.note or "")

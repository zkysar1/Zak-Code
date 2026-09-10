"""User-only skills are invisible to the model's seams (ADR-0109).

Field transcript 2026-09-05: "ok, clear that plan, and lets start from scratch" was classified
as implying the Mind's ``/start`` — a control command whose own description says "USER-ONLY —
Claude must NEVER invoke /start" — a ``run /start`` step was seeded, the plan gate refused the
model's text finish, and the model ran ``use_skill(start)``. Claude Code's
``disable-model-invocation: true`` is the machine-readable form of that prose; these tests pin
every seam that honors it: the parsed flag, the model-facing catalog and prompt, the
``use_skill`` refusal (the human command path untouched), the classify side-call, and the
loop's plan seeders on both routes.

ADR-0127 closes the gap the first cut left: the side-call could not NAME a user-only command,
so a request for one was matched to the nearest skill the model MAY run and that got seeded
(field 2026-09-10: "Start yourself as coach in assistant mode" → ``/prime``, thirty iterations
inside the wrong skill, then a report that it had started). The commands are now listed to the
classifier under their own heading, the verdict keeps the name, and the loop hands it to the
operator with a rail — no step, no backstop.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

import zakcode
from zakcode.agent.loop import AgentLoop
from zakcode.events import AgentStatus
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
)
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


#: The Mind's real /start description (2026-09-10), so the anchor floor sees what the field
#: sees: "Start yourself as coach in assistant mode" shares ``assi`` and ``mode`` with it.
MIND_START_MD = START_MD.replace(
    "Creates or resumes an agent.",
    "Creates or resumes an agent in reader (read-only), assistant (user-directed), or "
    "autonomous mode (perpetual loop). USER-ONLY: the user types /start {agent-name} "
    "[--mode {mode}].",
)


def _registry(tmp_path: Path, start_md: str = START_MD) -> SkillRegistry:
    reg = SkillRegistry()
    for dirname, text in (("start", start_md), ("forge-skill", FORGE_MD)):
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


# ── the classify side-call names a user-only command APART, and the floor still applies ──


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
async def test_side_call_lists_operator_only_commands_apart_and_keeps_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0127: the command is offered under its own heading — never among the agent's
    skills — and a verdict that names it is KEPT, for the loop to hand to the operator."""
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    monkeypatch.setattr(agent, "skill_registry", _registry(tmp_path))
    stub = _Stub('{"difficulty": "quick", "skill": "start"}')
    monkeypatch.setattr(agent, "_resolve_task_provider", lambda c: (stub, "classify/m"))
    verdict = await agent._classify_difficulty("run /start alpha", 0.0)
    assert verdict == DifficultyVerdict("quick_code", "start")
    agent_block, heading, operator_block = stub.systems[-1].partition("ONLY the operator can run")
    assert heading and "- forge-skill" in agent_block and "- start" not in agent_block
    assert "- start: Creates or resumes an agent" in operator_block
    assert "names THAT command, never a neighbouring skill" in operator_block


@pytest.mark.asyncio
async def test_side_call_still_drops_the_everyday_word_for_a_user_only_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ADR-0109 incident string: one everyday word is not a request for /start."""
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    monkeypatch.setattr(agent, "skill_registry", _registry(tmp_path, MIND_START_MD))
    stub = _Stub('{"difficulty": "quick", "skill": "start"}')
    monkeypatch.setattr(agent, "_resolve_task_provider", lambda c: (stub, "classify/m"))
    verdict = await agent._classify_difficulty(
        "ok, clear that plan, and lets start from scratch", 0.0
    )
    assert verdict == DifficultyVerdict("quick_code", None)


@pytest.mark.asyncio
async def test_side_call_keeps_a_user_only_command_the_request_describes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The field request of 2026-09-10 describes /start in its own words (assistant, mode)."""
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    monkeypatch.setattr(agent, "skill_registry", _registry(tmp_path, MIND_START_MD))
    stub = _Stub('{"difficulty": "deep", "skill": "start"}')
    monkeypatch.setattr(agent, "_resolve_task_provider", lambda c: (stub, "classify/m"))
    verdict = await agent._classify_difficulty(
        "Start yourself as coach in assistant mode. Once you are started, save a note to "
        "your working memory.",
        0.0,
    )
    assert verdict == DifficultyVerdict("deep_code", "start")


# ── the loop never seeds a step for a user-only skill ────────────────────────────────────


class _Text(Provider):
    def __init__(self, text: str) -> None:
        self.text = text

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        return LLMResult(text=self.text)

    async def astream(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> AsyncIterator[ProviderStreamEvent]:
        yield StreamTextDelta(text=self.text)
        yield StreamDone(finish_reason="stop")

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


#: A hand-off in the shape the rail asks for — no future-tense "I will …", which the intent
#: gate (ADR-0053) would read as announced-but-unperformed work.
HANDED_BACK = (
    "That command is yours to type: /start coach --mode assistant. The note can be saved "
    "once the agent is up."
)
FIELD_REQUEST = "Start yourself as coach in assistant mode, then save a note to working memory."


def test_an_implied_user_only_skill_is_handed_to_the_operator(tmp_path: Path) -> None:
    """ADR-0127: no step, no backstop — and ONE rail that says whose command it is."""
    loop = _loop(tmp_path, _Text(HANDED_BACK), DifficultyVerdict("deep_code", "start"))
    result = asyncio.run(loop.arun_turn(FIELD_REQUEST))
    assert result.stop_reason == "completed"
    assert loop.session.task_network.tasks == []  # never a step the model cannot execute
    rails = [
        m.text for m in loop.session.messages if m.role == "user" and m.text.startswith("[harness]")
    ]
    # Exactly one harness message: the hand-off. A coverage-backstop nudge would be a second.
    assert len(rails) == 1
    assert "/start" in rails[0] and "only the operator can run" in rails[0]
    assert "await_user" in rails[0] and "not stand in for it with another skill" in rails[0]
    notes = [
        e for e in loop._trace.of_kind("intervention") if e.data.get("kind") == "user_only_skill"
    ]
    assert len(notes) == 1 and "/start" in notes[0].detail


def test_the_reviewer_is_told_what_was_handed_to_the_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0128: the fresh-eyes critic reads the bare request, so it flagged "failed to start
    itself" against a correct hand-off and sent the model hunting for a start script. The
    criteria now carry the rule for the commands handed back this turn — and only this turn."""
    import zakcode.agent.loop as loop_module
    from zakcode.quality.judge import BinaryVerdict
    from zakcode.usage import Usage

    seen: list[str] = []

    async def judge(provider: Any, *, criteria: str, artifact: str, **kw: Any) -> Any:
        seen.append(criteria)
        return BinaryVerdict(approved=True), Usage()

    monkeypatch.setattr(loop_module, "binary_judge", judge)
    loop = _loop(tmp_path, _Text(HANDED_BACK), DifficultyVerdict("deep_code", "start"))
    asyncio.run(loop.arun_turn(FIELD_REQUEST))
    assert loop._turn_handed_off == ["start"]
    asyncio.run(loop._completion_critic(FIELD_REQUEST, HANDED_BACK))
    assert seen[-1].startswith(FIELD_REQUEST)
    assert "/start may only be run by the human operator" in seen[-1]
    assert "Do not flag that part as unmet" in seen[-1]
    # The next turn hands nothing off: the clause is gone with it.
    loop.difficulty_classifier = None  # no verdict this turn
    asyncio.run(loop.arun_turn("what did you save?"))
    assert loop._turn_handed_off == []
    asyncio.run(loop._completion_critic("what did you save?", "the note"))
    assert seen[-1] == "what did you save?"


def test_streaming_hands_a_user_only_skill_to_the_operator_too(tmp_path: Path) -> None:
    loop = _loop(tmp_path, _Text(HANDED_BACK), DifficultyVerdict("deep_code", "start"))

    async def _collect() -> list[Any]:
        return [ev async for ev in loop.astream_turn(FIELD_REQUEST)]

    events = asyncio.run(_collect())
    statuses = [ev.message for ev in events if isinstance(ev, AgentStatus)]
    assert "request implies /start — operator-only, handing it back to them" in statuses
    assert loop.session.task_network.tasks == []
    assert sum(m.role == "user" and "/start" in m.text for m in loop.session.messages) == 1


def test_an_implied_skill_step_is_marked_as_a_cancellable_guess(tmp_path: Path) -> None:
    loop = _loop(tmp_path, _Text("Working on it."), DifficultyVerdict("quick_code", "forge-skill"))
    asyncio.run(loop.arun_turn("finish forging this skill"))
    step = next(t for t in loop.session.task_network.tasks if t.title == "run /forge-skill")
    assert "harness guess" in (step.note or "")
    assert "cancelled if the request did not ask for it" in (step.note or "")

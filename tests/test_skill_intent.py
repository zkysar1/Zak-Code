"""Skill intent via the classify side-call (ADR-0035).

Field incident 2026-08-26 (serene): "finish forging this skill" carried no ``/slash`` token,
so the harness never knew the skill-forging skill WAS the task — no plan step was seeded, the
coverage backstop stayed unarmed — and the model collapsed without reading the skill. These
tests pin the verdict shape, the prompt's catalog half, the exact-match parse, both loop paths
seeding + arming, and the Agent's side-call carrying the catalog.
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
from zakcode.providers.routing import (
    DIFFICULTY_SCHEMA,
    DifficultyVerdict,
    difficulty_system_prompt,
    implied_skill_anchored,
    parse_skill,
    parse_verdict,
)
from zakcode.providers.structured import StructuredValidationError, coerce_structured
from zakcode.session.store import Session
from zakcode.tools.base import SkillLoad, ToolRegistry

CATALOG = [
    ("forge-skill", "Forge a new skill from a description"),
    ("notify-user", "Send the operator a message"),
]
KNOWN = [name for name, _desc in CATALOG]

# ── the pure policy ──────────────────────────────────────────────────────────


def test_parse_skill_matches_the_catalog_exactly() -> None:
    assert parse_skill({"difficulty": "quick", "skill": "forge-skill"}, KNOWN) == "forge-skill"
    assert parse_skill({"difficulty": "quick", "skill": "/Forge-Skill"}, KNOWN) == "forge-skill"
    assert parse_skill({"difficulty": "quick", "skill": "forge"}, KNOWN) is None  # a guess
    assert parse_skill({"difficulty": "quick", "skill": None}, KNOWN) is None
    assert parse_skill({"difficulty": "quick"}, KNOWN) is None
    assert parse_skill("garbage", KNOWN) is None
    assert parse_skill({"difficulty": "quick", "skill": "forge-skill"}, []) is None


def test_parse_verdict_carries_both_halves() -> None:
    assert parse_verdict({"difficulty": "deep", "skill": "notify-user"}, KNOWN) == (
        DifficultyVerdict("deep_code", "notify-user")
    )
    assert parse_verdict({"difficulty": "quick"}, KNOWN) == DifficultyVerdict("quick_code", None)
    assert parse_verdict(None, []) == DifficultyVerdict("deep_code", None)  # fails UP, no skill


def test_prompt_lists_the_catalog_and_the_null_rule() -> None:
    plain = difficulty_system_prompt()
    assert '"skill"' not in plain  # no catalog → the plain scope judgment, byte-identical
    with_catalog = difficulty_system_prompt(CATALOG)
    assert "forge-skill: Forge a new skill" in with_catalog
    assert "notify-user: Send the operator" in with_catalog
    assert '"skill": null' in with_catalog and "Never guess" in with_catalog


def test_implied_skill_needs_a_shared_content_word() -> None:
    """ADR-0036: the deterministic floor under "never guess"."""
    forge = ("forge-skill", "Forge a new skill from a description")
    create = ("create-aspiration", "Create a new aspiration in the world queue")
    assert implied_skill_anchored("finish forging this skill", *forge)  # forging ~ forge
    assert implied_skill_anchored("add an aspiration for the report", *create)
    assert implied_skill_anchored("run the notifier", "notify-user", "Send the operator a message")
    # The field guess: "then make one" shares no content word with create-aspiration.
    assert not implied_skill_anchored("then make one", *create)
    assert not implied_skill_anchored("what is the weather", *forge)
    assert not implied_skill_anchored("this skill", *forge)  # stopwords never anchor
    assert not implied_skill_anchored("anything", "", "")


def test_schema_accepts_the_optional_skill_field() -> None:
    coerce_structured('{"difficulty": "quick", "skill": null}', schema=DIFFICULTY_SCHEMA)
    coerce_structured('{"difficulty": "deep", "skill": "forge-skill"}', schema=DIFFICULTY_SCHEMA)
    coerce_structured('{"difficulty": "deep"}', schema=DIFFICULTY_SCHEMA)  # still valid alone
    pytest.importorskip("jsonschema")
    with pytest.raises(StructuredValidationError):
        coerce_structured('{"difficulty": "deep", "skill": 3}', schema=DIFFICULTY_SCHEMA)


# ── the loop: seeding + arming on both paths ─────────────────────────────────


class _Text(Provider):
    """Answers every call with the same text (no tool calls) on both turn paths."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.text)

    async def astream(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> AsyncIterator[ProviderStreamEvent]:
        self.calls += 1
        yield StreamTextDelta(text=self.text)
        yield StreamDone(finish_reason="stop")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=200_000)


class _Resolver:
    """The SkillResolver protocol: a catalog the loop can resolve names against."""

    def __init__(self, names: list[str] | None = None) -> None:
        self._names = list(KNOWN if names is None else names)

    def names(self) -> list[str]:
        return list(self._names)

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        raise AssertionError("no skill is loaded in these tests")


def _loop(
    tmp_path: Path,
    provider: Provider,
    verdict: DifficultyVerdict,
    *,
    names: list[str] | None = None,
) -> AgentLoop:
    async def classifier(user_text: str, context_frac: float) -> DifficultyVerdict:
        return verdict

    return AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=6,
        main_provider_for=lambda category: provider,  # zakpick on: the side-call runs
        difficulty_classifier=classifier,
        skill_resolver=_Resolver(names),
    )


# A typed `/start sera` as Agent.compose_skill_turn hands it to the loop: the command
# frame, then the skill's whole body — which, like the real start skill, mentions OTHER
# skills in request-shaped prose.
_COMPOSED_START_TURN = (
    "<command-message>start is running</command-message>\n"
    "<command-name>/start</command-name>\n"
    "<command-args>sera --mode assistant</command-args>\n\n"
    "# /start — bring an agent up\n\n"
    "If the runner is dead, run /stop <agent-name> first, then /boot and /prime.\n"
    "Never invoke /stop from inside the loop.\n"
)


def test_typed_skill_turn_never_seeds_from_its_body(tmp_path: Path) -> None:
    """ADR-0036: the body of a typed /skill is documentation — no plan steps from its
    ``/other-skill`` mentions, and no second use_skill load demanded for the skill itself."""
    provider = _Text("Bringing sera up now.")
    loop = _loop(
        tmp_path,
        provider,
        DifficultyVerdict("quick_code", None),
        names=["start", "stop", "boot", "prime"],
    )
    result = asyncio.run(loop.arun_turn(_COMPOSED_START_TURN))
    assert result.stop_reason == "completed"
    assert loop.session.task_network.tasks == []  # nothing seeded: not /stop, /boot, /prime
    rails = [m.text for m in loop.session.messages if m.role == "user"][1:]
    assert not any("/stop" in r or "/start" in r for r in rails)  # no coverage nudge either
    assert provider.calls == 1


def test_a_body_embedded_frame_is_just_text(tmp_path: Path) -> None:
    """Only a frame at the very START of the message carries invocation meaning."""
    provider = _Text("Sure.")
    loop = _loop(
        tmp_path,
        provider,
        DifficultyVerdict("quick_code", None),
        names=["start", "stop", "boot"],
    )
    text = "do /stop and /boot\n" + _COMPOSED_START_TURN  # a paste, not a typed command
    asyncio.run(loop.arun_turn(text))
    titles = [t.title for t in loop.session.task_network.tasks]
    assert "run /stop" in titles and "run /boot" in titles  # the compound seeder still works


def test_buffered_implied_skill_seeds_the_plan_and_arms_the_backstop(tmp_path: Path) -> None:
    provider = _Text("Working on it.")
    loop = _loop(tmp_path, provider, DifficultyVerdict("quick_code", "forge-skill"))
    result = asyncio.run(loop.arun_turn("finish forging this skill"))
    assert result.stop_reason == "completed"
    titles = [t.title for t in loop.session.task_network.tasks]
    assert "run /forge-skill" in titles  # the plan step a typed /forge-skill would have seeded
    # …and the finish was held to it (plan gate / coverage backstop named the skill).
    assert any(m.role == "user" and "/forge-skill" in m.text for m in loop.session.messages)


def test_streaming_implied_skill_is_announced_and_seeded(tmp_path: Path) -> None:
    provider = _Text("Working on it.")
    loop = _loop(tmp_path, provider, DifficultyVerdict("quick_code", "forge-skill"))

    async def _collect() -> list[Any]:
        return [ev async for ev in loop.astream_turn("finish forging this skill")]

    events = asyncio.run(_collect())
    statuses = [ev.message for ev in events if isinstance(ev, AgentStatus)]
    assert any(s.startswith("request implies /forge-skill") for s in statuses)
    assert "run /forge-skill" in [t.title for t in loop.session.task_network.tasks]


def test_no_implied_skill_changes_nothing(tmp_path: Path) -> None:
    provider = _Text("Here is the answer.")
    loop = _loop(tmp_path, provider, DifficultyVerdict("quick_code", None))
    result = asyncio.run(loop.arun_turn("what does the catalog hold?"))
    assert result.stop_reason == "completed"
    assert loop.session.task_network.tasks == []
    assert not any(m.role == "user" and "/forge-skill" in m.text for m in loop.session.messages)
    assert provider.calls == 1


# ── the Agent's side-call carries the catalog ────────────────────────────────


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


class _Registry:
    def catalog(self) -> list[tuple[str, str]]:
        return list(CATALOG)


@pytest.mark.asyncio
async def test_agent_side_call_names_a_catalogued_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    monkeypatch.setattr(agent, "skill_registry", _Registry())
    stub = _Stub('{"difficulty": "quick", "skill": "forge-skill"}')
    monkeypatch.setattr(agent, "_resolve_task_provider", lambda c: (stub, "classify/m"))
    verdict = await agent._classify_difficulty("finish forging this skill", 0.0)
    assert verdict == DifficultyVerdict("quick_code", "forge-skill")
    assert "forge-skill: Forge a new skill" in stub.systems[-1]  # the catalog reached the prompt


@pytest.mark.asyncio
async def test_agent_side_call_drops_an_unanchored_guess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The field guess (ADR-0036): 'then make one' → create-aspiration is catalogued but
    shares no content word with the request, so the skill is dropped and the category kept."""

    class _Registry2:
        def catalog(self) -> list[tuple[str, str]]:
            return [*CATALOG, ("create-aspiration", "Create a new aspiration in the queue")]

    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    monkeypatch.setattr(agent, "skill_registry", _Registry2())
    stub = _Stub('{"difficulty": "quick", "skill": "create-aspiration"}')
    monkeypatch.setattr(agent, "_resolve_task_provider", lambda c: (stub, "classify/m"))
    assert await agent._classify_difficulty("then make one", 0.0) == DifficultyVerdict(
        "quick_code", None
    )
    # …while a request that names the thing in its own words keeps the skill.
    assert await agent._classify_difficulty("make an aspiration for it", 0.0) == (
        DifficultyVerdict("quick_code", "create-aspiration")
    )


@pytest.mark.asyncio
async def test_agent_side_call_drops_a_name_the_catalog_lacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    monkeypatch.setattr(agent, "skill_registry", _Registry())
    stub = _Stub('{"difficulty": "deep", "skill": "made-up-skill"}')
    monkeypatch.setattr(agent, "_resolve_task_provider", lambda c: (stub, "classify/m"))
    assert await agent._classify_difficulty("do the thing", 0.0) == DifficultyVerdict("deep_code")

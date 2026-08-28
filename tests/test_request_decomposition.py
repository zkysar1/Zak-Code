"""Compound-ask decomposition: plan seeding + the skill-coverage backstop.

Field incident (2026-08-26): a user message asked for TWO skills ("/fresh-eyes-code
review and a /encode-session"); a mid-turn interjection plus a session replay evicted
the first from conversation memory, and the turn ended "done" having run only the
second. These tests pin the two-layer fix: (1) a request naming >=2 registry-known
skills seeds one plan step per skill at turn entry (session state — survives what
conversation memory does not), enforced by the existing plan gate; (2) a one-shot
completion-time backstop nudges for any explicitly-requested skill that was neither
invoked via use_skill nor mentioned by the plan in any state.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop
from zakcode.events import AgentDone, AgentEvent, AgentStatus
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import Task
from zakcode.tools.base import SkillLoad, ToolRegistry
from zakcode.tools.builtins.use_skill import UseSkillTool


class _Resolver:
    """Structural SkillResolver: fixed names, canned bodies."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def names(self) -> list[str]:
        return list(self._names)

    def body(self, name: str) -> str | None:
        return None  # no whole-body seam: the loop seeds from what the load delivered

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        if name in self._names:
            return SkillLoad(found=True, name=name, body=f"instructions for {name}")
        return SkillLoad(found=False, name=name)


class _ScriptByCallProvider(Provider):
    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return self._factory(self.calls, messages)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _loop(provider: Provider, tmp_path: Path, names: list[str]) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(UseSkillTool())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=10,
        skill_resolver=_Resolver(names),
    )


SKILLS = ["fresh-eyes-code", "encode-session"]


# ── _skill_refs: registry-resolved slash tokens only ─────────────────────────────


def test_skill_refs_resolve_against_registry(tmp_path: Path) -> None:
    loop = _loop(_ScriptByCallProvider(lambda n, m: LLMResult(text="hi")), tmp_path, SKILLS)
    text = (
        "Perfect. Now do a /fresh-eyes-code review and a /encode-session — "
        "check /tmp and the a/b path too, and never /unknown-skill."
    )
    # Prose slashes (/tmp, a/b) and unknown names never match; known ones do, once, in order.
    assert loop._skill_refs(text) == ["fresh-eyes-code", "encode-session"]
    assert loop._skill_refs("do a /fresh-eyes-code and then /fresh-eyes-code again") == [
        "fresh-eyes-code"
    ]
    assert loop._skill_refs("no skills here") == []


def test_skill_refs_match_punctuation_adjacent_tokens(tmp_path: Path) -> None:
    # Fresh-eyes F1: "/a,/b" shorthand used to defeat BOTH layers at once (refs<2 so
    # nothing seeded, and the missing name was invisible to the coverage backstop).
    loop = _loop(_ScriptByCallProvider(lambda n, m: LLMResult(text="hi")), tmp_path, SKILLS)
    assert loop._skill_refs("do /fresh-eyes-code,/encode-session") == SKILLS
    assert loop._skill_refs("try /fresh-eyes-code!/encode-session; ok") == SKILLS


def test_skill_refs_ignore_documentation_mentions(tmp_path: Path) -> None:
    # ADR-0026: a pasted prompt whose PROSE discussed eight skills produced a coverage
    # nudge demanding all eight. Mention shapes never count as requests…
    loop = _loop(_ScriptByCallProvider(lambda n, m: LLMResult(text="hi")), tmp_path, SKILLS)
    assert loop._skill_refs("`/fresh-eyes-code` is buggy in assistant mode") == []
    assert loop._skill_refs("(/encode-session) runs the learning pass") == []
    assert loop._skill_refs("run [/encode-session] later maybe") == []
    assert loop._skill_refs('the doc says "/fresh-eyes-code" a lot') == []
    assert loop._skill_refs("> /fresh-eyes-code output was wrong") == []
    assert loop._skill_refs("```\nrun /encode-session\n```") == []
    # …while request shapes still do.
    assert loop._skill_refs("run /encode-session") == ["encode-session"]
    assert loop._skill_refs("/fresh-eyes-code please") == ["fresh-eyes-code"]


def test_plan_mention_requires_token_boundary(tmp_path: Path) -> None:
    # Fresh-eyes F2: with /test and /test-e2e both registered, a plan step naming the
    # longer must not read as "addressing" the shorter (substring false-suppression).
    provider = _ScriptByCallProvider(lambda n, m: LLMResult(text="hi"))
    loop = _loop(provider, tmp_path, ["test", "test-e2e"])
    loop.session.task_network.tasks.append(Task(title="run /test-e2e", kind="primitive"))
    loop.session.task_network.normalize()
    assert loop._plan_mentions_skill("test-e2e") is True
    assert loop._plan_mentions_skill("test") is False
    assert loop._seed_plan_from_request("do /test and /test-e2e") == ["test"]


def test_skill_refs_empty_when_skills_disabled(tmp_path: Path) -> None:
    registry = ToolRegistry()
    loop = AgentLoop(
        _ScriptByCallProvider(lambda n, m: LLMResult(text="hi")),
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=5,
        skill_resolver=None,
    )
    assert loop._skill_refs("/fresh-eyes-code and /encode-session") == []


# ── arrival-time plan seeding ────────────────────────────────────────────────────


def test_compound_request_seeds_one_step_per_skill(tmp_path: Path) -> None:
    provider = _ScriptByCallProvider(lambda n, m: LLMResult(text="on it"))
    loop = _loop(provider, tmp_path, SKILLS)
    asyncio.run(loop.arun_turn("do a /fresh-eyes-code review and a /encode-session"))
    titles = [t.title for t in loop.session.task_network.tasks]
    assert "run /fresh-eyes-code" in titles
    assert "run /encode-session" in titles


def test_single_skill_request_does_not_seed(tmp_path: Path) -> None:
    # One-part asks stay ceremony-free: the model handles them directly (the coverage
    # backstop still guards the finish — see below).
    use = ToolCall(id="u1", name="use_skill", arguments={"name": "encode-session"})
    provider = _ScriptByCallProvider(
        lambda n, m: LLMResult(tool_calls=[use]) if n == 1 else LLMResult(text="done")
    )
    loop = _loop(provider, tmp_path, SKILLS)
    asyncio.run(loop.arun_turn("run a /encode-session"))
    assert not any("run /" in t.title for t in loop.session.task_network.tasks)


def test_seeding_appends_and_never_duplicates(tmp_path: Path) -> None:
    provider = _ScriptByCallProvider(lambda n, m: LLMResult(text="ok"))
    loop = _loop(provider, tmp_path, SKILLS)
    # A model-authored plan already mentions one of the two skills.
    loop.session.task_network.tasks.append(
        Task(title="already planning the /encode-session pass", kind="primitive")
    )
    loop.session.task_network.normalize()
    seeded = loop._seed_plan_from_request("/fresh-eyes-code then /encode-session please")
    assert seeded == ["fresh-eyes-code"]  # only the unmentioned one
    titles = [t.title for t in loop.session.task_network.tasks]
    assert titles.count("run /fresh-eyes-code") == 1
    # Re-seeding the same request is a no-op (idempotent across turns/replays).
    assert loop._seed_plan_from_request("/fresh-eyes-code then /encode-session please") == []


# ── completion-time coverage backstop ────────────────────────────────────────────


def test_coverage_nudge_fires_once_and_names_the_missing_skill(tmp_path: Path) -> None:
    # Single-skill request (no seeding), model answers WITHOUT running it: one nudge
    # naming the skill, then — when the model still declines — the turn ends rather
    # than looping.
    provider = _ScriptByCallProvider(lambda n, m: LLMResult(text=f"answer {n}"))
    loop = _loop(provider, tmp_path, SKILLS)
    result = asyncio.run(loop.arun_turn("please run a /fresh-eyes-code on the diff"))
    assert result.stop_reason == "completed"
    assert provider.calls == 2  # nudged exactly once
    rails = [m.text for m in loop.session.messages if m.text and "also asked for" in m.text]
    assert len(rails) == 1
    assert "/fresh-eyes-code" in rails[0]


def test_coverage_satisfied_by_use_skill(tmp_path: Path) -> None:
    provider = _ScriptByCallProvider(
        lambda n, m: (
            LLMResult(
                tool_calls=[
                    ToolCall(id="u1", name="use_skill", arguments={"name": "fresh-eyes-code"})
                ]
            )
            if n == 1
            else LLMResult(text="review done")
        )
    )
    loop = _loop(provider, tmp_path, SKILLS)
    result = asyncio.run(loop.arun_turn("please run a /fresh-eyes-code on the diff"))
    assert result.stop_reason == "completed"
    assert not any(m.text and "also asked for" in m.text for m in loop.session.messages)


def test_failed_use_skill_load_does_not_count_as_coverage(tmp_path: Path) -> None:
    # An errored load (unknown name) is not an invocation — the nudge still fires.
    calls: list[ToolCall] = [ToolCall(id="u1", name="use_skill", arguments={"name": "nope"})]
    results = [ToolResultBlock(tool_use_id="u1", output="unknown skill", is_error=True)]
    seen: set[str] = set()
    AgentLoop._harvest_skill_invocations(calls, results, seen)
    assert seen == set()
    ok = [ToolResultBlock(tool_use_id="u1", output="body", is_error=False)]
    AgentLoop._harvest_skill_invocations(calls, ok, seen)
    assert seen == {"nope"}


def test_seeded_plan_owns_enforcement_not_the_backstop(tmp_path: Path) -> None:
    # Two-skill request seeds the plan; a model that just answers gets the PLAN gate's
    # nudges (open steps), while the coverage backstop stays silent — the plan mentions
    # both skills, so precedence holds and the two mechanisms never double-nag.
    provider = _ScriptByCallProvider(lambda n, m: LLMResult(text=f"answer {n}"))
    loop = _loop(provider, tmp_path, SKILLS)
    result = asyncio.run(loop.arun_turn("do a /fresh-eyes-code and a /encode-session"))
    assert result.stop_reason == "completed"
    transcript = [m.text for m in loop.session.messages if m.text]
    assert any("open step" in t for t in transcript)  # the plan gate spoke
    assert not any("also asked for" in t for t in transcript)  # the backstop did not
    assert result.degraded is True  # finished with seeded steps still open


# ── streaming twin ───────────────────────────────────────────────────────────────


def _drain(loop: AgentLoop, text: str) -> list[AgentEvent]:
    async def run() -> list[AgentEvent]:
        return [ev async for ev in loop.astream_turn(text)]

    return asyncio.run(run())


def test_streaming_seeds_and_announces(tmp_path: Path) -> None:
    provider = _ScriptByCallProvider(lambda n, m: LLMResult(text="ok"))
    loop = _loop(provider, tmp_path, SKILLS)
    events = _drain(loop, "do a /fresh-eyes-code and a /encode-session")
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("plan seeded from the request" in s for s in statuses)
    titles = [t.title for t in loop.session.task_network.tasks]
    assert "run /fresh-eyes-code" in titles and "run /encode-session" in titles


def test_streaming_coverage_nudge(tmp_path: Path) -> None:
    provider = _ScriptByCallProvider(lambda n, m: LLMResult(text=f"answer {n}"))
    loop = _loop(provider, tmp_path, SKILLS)
    events = _drain(loop, "run a /encode-session please")
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.stop_reason == "completed"
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("never ran" in s for s in statuses)
    rails = [m.text for m in loop.session.messages if m.text and "also asked for" in m.text]
    assert len(rails) == 1 and "/encode-session" in rails[0]

"""ADR-0192: a skill whose body fits the window is delivered WHOLE and seeds no plan; paging
and the seeded skeleton are the shape of a body that cannot fit.

Measured 2026-09-18 on a served Mind (Vinheim prod, gpt-5.6-terra, a 922k window): every
sectioned skill was paged and seeded regardless of the window, so one loop iteration cost
45 page deliveries and 101 ``update_plan`` calls out of 232 tool calls, the prompt grew to
360k tokens and the turn died ``veto_stall`` at $155 with no iteration completed. Three
smaller defects rode along: the fit check reserved the model's whole output cap (128,000
tokens) as answer room, so a recipe that pins the window to 131,072 could fit nothing; a
fold row echoed without its count (``75.1–75.6``) became a literal step; and a fenced
``# Phase 6 for non-recurring …`` comment line seeded a step titled with that sentence.
Hermetic: scripted providers, fake resolvers, a fake tokenizer of four chars a token.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from zakcode import tasks
from zakcode.agent.loop import _MAX_ANSWER_ROOM, _MIN_ANSWER_ROOM, AgentLoop, _answer_room
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import Task, TaskNetwork, skill_pages, skill_skeleton
from zakcode.tools.base import SkillLoad, ToolRegistry
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.tools.builtins.use_skill import UseSkillTool

FRAME = "<command-message>demo is running</command-message>\n<command-name>/demo</command-name>\n\n"


def _sectioned(chars_per_section: int, sections: int = 4) -> str:
    """A skill with ``sections`` numbered sections of about ``chars_per_section`` each."""
    filler = ("do the thing carefully " * (chars_per_section // 23 + 1))[:chars_per_section]
    body = "# /demo — a sectioned skill\n\nIntro the sections rely on.\n\n"
    for i in range(1, sections + 1):
        body += f"## Step {i}: Part {i}\n\n{filler}\n\n"
    return body + "## Return Protocol\n\nEnd with a tool call.\n"


class _Resolver:
    def __init__(self, bodies: dict[str, str]) -> None:
        self._bodies = bodies

    def names(self) -> list[str]:
        return list(self._bodies)

    def body(self, name: str) -> str | None:
        return self._bodies.get(name)

    async def load(self, name: str, *, caller_query: str = "", **_: Any) -> SkillLoad:
        body = self._bodies.get(name)
        if body is None:
            return SkillLoad(found=False, name=name)
        return SkillLoad(found=True, name=name, body=body, path=f"/skills/{name}/SKILL.md")


class _ScriptByCall(Provider):
    def __init__(self, script: Any, window: int) -> None:
        self._script = script
        self._window = window
        self.calls = 0
        self.seen: list[list[Message]] = []

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        self.seen.append(list(messages))
        return self._script(self.calls)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        chars = len(system or "")
        for message in messages:
            for block in message.blocks:
                chars += len(getattr(block, "text", "") or "")
        return chars // 4

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=self._window)


def _loop(provider: Provider, tmp_path: Path, bodies: dict[str, str]) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(UseSkillTool())
    registry.register(UpdatePlanTool())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=6,
        skill_resolver=_Resolver(bodies),
    )


def _use(name: str = "demo") -> LLMResult:
    return LLMResult(tool_calls=[ToolCall(id="t1", name="Skill", arguments={"name": name})])


def _skill_results(loop: AgentLoop) -> list[ToolResultBlock]:
    return [
        b
        for m in loop.session.messages
        for b in m.blocks
        if isinstance(b, ToolResultBlock) and b.tool_use_id == "t1"
    ]


def _user_texts(messages: list[Message]) -> list[str]:
    return [m.text or "" for m in messages if m.role == "user"]


# ── the delivery decision ─────────────────────────────────────


def test_a_sectioned_body_that_fits_is_delivered_whole_and_seeds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "PAGE_BUDGET_CHARS", 100)  # it pages: one section a page
    body = _sectioned(400)  # ~2 KB, four sections
    provider = _ScriptByCall(lambda n: _use() if n == 1 else LLMResult(text="done"), 32_768)
    loop = _loop(provider, tmp_path, {"demo": body})
    result = asyncio.run(loop.arun_turn("run /demo"))
    assert result.stop_reason == "completed"
    (res,) = _skill_results(loop)
    assert "## Step 4: Part 4" in res.output and "— page 1 of" not in res.output
    assert loop.session.task_network.tasks == []  # no skeleton
    second = _user_texts(provider.seen[1])
    assert not any("sections of /demo to your plan" in t for t in second)
    assert not any(e.data.get("kind") == "skill_skeleton" for e in loop._trace.events)
    assert loop._ensure_skill_pages("demo") is None


def test_a_body_that_cannot_fit_is_paged_and_seeded_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control for the test above: the same shape under a window it cannot fit."""
    monkeypatch.setattr(tasks, "PAGE_BUDGET_CHARS", 8_500)  # one ~8 KB section a page
    body = _sectioned(8_000)  # ~32 KB ≈ 8k tokens whole, ~2k a page
    provider = _ScriptByCall(lambda n: _use() if n == 1 else LLMResult(text="done"), 32_768)
    loop = _loop(provider, tmp_path, {"demo": body})  # a window is required at construction
    # Then size it from the measured system prompt: a page fits beside it with the answer
    # room (ADR-0066), the whole body does not — whatever the prompt grows to.
    system = provider.count_tokens([], system=loop._build_system())
    provider._window = system + _MIN_ANSWER_ROOM + 4_000
    asyncio.run(loop.arun_turn("run /demo"))
    (res,) = _skill_results(loop)
    assert "[/demo — page 1 of 4: Step 1: Part 1]" in res.output
    assert "## Step 4: Part 4" not in res.output
    assert [t.title for t in loop.session.task_network.tasks] == [
        f"Step {i}: Part {i}" for i in range(1, 5)
    ]
    assert loop._ensure_skill_pages("demo") is not None


def test_the_typed_door_delivers_a_fitting_body_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "PAGE_BUDGET_CHARS", 100)
    body = _sectioned(400)
    provider = _ScriptByCall(lambda n: LLMResult(text="done"), 32_768)
    loop = _loop(provider, tmp_path, {"demo": body})
    result = asyncio.run(loop.arun_turn(FRAME + body))
    assert result.stop_reason == "completed"
    first = _user_texts(provider.seen[0])
    assert not any("sections of /demo to your plan" in t for t in first)
    assert not any("Current plan" in t for t in first)
    assert loop.session.task_network.tasks == []


def test_the_decision_is_made_once_and_every_door_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "PAGE_BUDGET_CHARS", 100)
    body = _sectioned(400)
    provider = _ScriptByCall(lambda n: LLMResult(text="done"), 32_768)
    loop = _loop(provider, tmp_path, {"demo": body})
    assert skill_pages(body, skill="demo") is not None  # sectioned: paged before ADR-0192
    assert loop._skill_pages_for_delivery("demo", body) is None
    assert loop._skill_whole == {"demo": True}
    loop._skill_whole["demo"] = False  # the memo is the decision
    assert loop._skill_pages_for_delivery("demo", body) is not None


# ── the answer room ───────────────────────────────────────────


def test_the_answer_room_is_the_output_cap_bounded_above() -> None:
    assert _answer_room(Capabilities(max_output=128_000)) == _MAX_ANSWER_ROOM
    assert _answer_room(Capabilities(max_output=8_192)) == 8_192
    assert _answer_room(Capabilities(max_output=None)) == _MIN_ANSWER_ROOM


def test_a_pinned_window_below_the_output_cap_still_fits_a_skill(tmp_path: Path) -> None:
    """The 2026-09-18 recipe: window pinned to 131,072 under a 128,000 output cap."""

    class _Pinned(_ScriptByCall):
        def capabilities(self) -> Capabilities:
            return Capabilities(supports_tools=True, context_window=131_072, max_output=128_000)

    loop = _loop(_Pinned(lambda n: LLMResult(text="done"), 131_072), tmp_path, {})
    assert loop._verbatim_overflow("x " * 4_000, what="skill 'demo'") is None  # ~2k tokens
    assert loop._verbatim_overflow("x " * 300_000, what="skill 'demo'") is not None  # ~150k


# ── the fold round-trip (ADR-0184) ────────────────────────────


def _closed_compound() -> tuple[TaskNetwork, list[str]]:
    net = TaskNetwork()
    parent = Task(
        title="Phase 1 battery",
        status="done",
        children=[Task(title=f"sub {k}", status="done") for k in range(1, 7)],
    )
    net.replace_from_author([parent, Task(title="Phase 2 select", status="in_progress")])
    return net, [c.id for c in net.tasks[0].children]


def test_a_bare_range_echo_expands_to_the_closed_run_it_names() -> None:
    net, kids = _closed_compound()
    assert "(6 steps done)" in net.render(elide_done=True)
    echo = [
        Task(
            title="Phase 1 battery",
            status="done",
            children=[Task(title=f"{kids[0]}–{kids[-1]}", status="done")],
        ),
        Task(title="Phase 2 select", status="in_progress"),
    ]
    net.replace_from_author(echo)
    assert [c.title for c in net.tasks[0].children] == [f"sub {k}" for k in range(1, 7)]
    assert net.progress() == (6, 7)


def test_a_top_level_bare_range_echo_never_becomes_a_literal_step() -> None:
    net = TaskNetwork()
    net.replace_from_author(
        [Task(title=f"step {k}", status="done") for k in range(1, 7)]
        + [Task(title="next", status="in_progress")]
    )
    net.replace_from_author(
        [Task(title="1-6", status="done"), Task(title="next", status="in_progress")]
    )
    assert [t.title for t in net.tasks] == [f"step {k}" for k in range(1, 7)] + ["next"]


def test_a_bare_range_naming_no_closed_run_is_the_models_own_title() -> None:
    net = TaskNetwork()
    net.replace_from_author(
        [Task(title="step 1", status="done"), Task(title="step 2", status="in_progress")]
    )
    net.replace_from_author(
        [Task(title="step 1", status="done"), Task(title="1–2", status="in_progress")]
    )
    assert [t.title for t in net.tasks] == ["step 1", "1–2"]  # step 2 was open: not a fold


# ── the fenced section marker ─────────────────────────────────


def test_a_fenced_phase_comment_needs_a_separator_after_its_number() -> None:
    fence = (
        "# /w\n\n## The loop\n\n```\n"
        "# Phase -0.5a0: Orchestrator Entry Battery (ONE call replaces the\n"
        "# per-phase presence checks)\nbattery()\n\n"
        "# Phase 1 — SELECT (reuse the scorer)\nselect()\n\n"
        "# do_verify emits the imperative. Previously\n"
        "# Phase 6 for non-recurring deep closes rode on LLM memory alone and drifted,\n"
        "# observed miss g-115-2404.\nverify()\n\n"
        "#     Phase 3.9 note cites. The ONLY status that means done.\n"
        "# Step 2.95 — UNIT CLAIM\nclaim()\n\n"
        "# Phase -0.5e': Quiescence Wake-Verify (Change 2)\nwake()\n```\n"
    )
    assert [t.title for t in skill_skeleton(fence, skill="w")] == [
        "Phase -0.5a0: Orchestrator Entry Battery (ONE call replaces the",
        "Phase 1 — SELECT (reuse the scorer)",
        "Step 2.95 — UNIT CLAIM",
        "Phase -0.5e': Quiescence Wake-Verify (Change 2)",
    ]

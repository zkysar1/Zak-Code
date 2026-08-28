"""ADR-0062: a loaded skill's numbered sections become plan steps — the harness decomposes,
the model refines.

ADR-0027 asked the model to decompose a long skill body ("FIRST call update_plan …") and
left it a hint. Field 2026-08-28 (sera, gemini-2.5-flash): a say naming /encode-session
mid-sentence had the model load the skill via use_skill — 883 lines, hint attached — and
go straight to `git status`; no plan was ever written and nothing held it to the skill's
remaining sections. Now the harness seeds what the body's own headings spell out, at both
doors (the typed /<skill> turn and a use_skill load), and the existing plan gate holds the
finish. Hermetic: scripted providers, fake resolvers, tmp workspaces.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop, _composed_skill_body
from zakcode.events import AgentDone, AgentEvent, AgentStatus, AgentTaskUpdate
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import Task, TaskNetwork, skill_skeleton
from zakcode.tools.base import SkillLoad, ToolContext, ToolRegistry
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.tools.builtins.use_skill import UseSkillTool

ENCODE = """# /encode-session — Session Learning Consolidation

Some intro prose that mentions /fresh-eyes-code in passing.

## Sub-commands

```
/encode-session --quick
```

## Phase 0: Load Conventions

**Step 0** — run `load-conventions.sh` (this bold line is prose, not a heading).

## Phase 1: Establish Session Context

## Lane 1: Encoding (Knowledge Tree, Reasoning Bank)

### 1.0 Pre-Encoding Retrieval (G13 / R16)

### 1.1 Knowledge Tree

## Lane 2: Out-of-Cycle Work

```
## Phase 99: inside a fence — never a step
```

## Return Protocol

## Phase Final: Summary
"""

FRAME = (
    "<command-message>encode-session is running</command-message>\n"
    "<command-name>/encode-session</command-name>\n\n"
)


# ── the pure skeleton ──────────────────────────────────────────────────────────


def test_sections_become_steps_and_subsections_nest() -> None:
    steps = skill_skeleton(ENCODE, skill="encode-session")
    network = TaskNetwork(tasks=steps)
    network.normalize()
    assert [t.title for t in steps] == [
        "Phase 0: Load Conventions",
        "Phase 1: Establish Session Context",
        "Lane 1: Encoding (Knowledge Tree, Reasoning Bank)",
        "Lane 2: Out-of-Cycle Work",
        "Phase Final: Summary",
    ]
    lane = steps[2]
    assert lane.kind == "compound" and lane.id == "3"
    assert [c.title for c in lane.children] == [
        "1.0 Pre-Encoding Retrieval (G13 / R16)",
        "1.1 Knowledge Tree",
    ]
    assert [c.id for c in lane.children] == ["3.1", "3.2"]
    assert all(t.note.startswith("from /encode-session;") for t in network.leaves())
    assert network.progress() == (0, 6)  # six leaves owe work


def test_prose_and_fenced_headings_never_count() -> None:
    steps = skill_skeleton(ENCODE, skill="encode-session")
    titles = [t.title for t in steps]
    assert not any("Sub-commands" in t or "Return Protocol" in t for t in titles)
    assert not any("Phase 99" in t for t in titles)  # inside the fence
    assert skill_skeleton("Do the thing.\nThen the other thing.", skill="plain") == []
    assert skill_skeleton("## Steps\n\n## Step-by-step guide\n", skill="x") == []


def test_a_subsection_outside_a_section_stands_alone() -> None:
    body = "## Procedure\n\n### Step 1: look\n\n### Step 2: leap\n\n## Notes\n\n### 3. not nested\n"
    steps = skill_skeleton(body, skill="x")
    assert [t.title for t in steps] == ["Step 1: look", "Step 2: leap", "3. not nested"]
    assert all(t.kind == "primitive" for t in steps)


def test_titles_are_cleaned_and_trimmed() -> None:
    long_tail = " x" * 80
    body = f"## **Phase 4.2:** Post-Execution `Domain` Steps — lightweight{long_tail}\n"
    [step] = skill_skeleton(body, skill="x")
    assert step.title.startswith("Phase 4.2: Post-Execution Domain Steps — lightweight")
    assert 90 < len(step.title) <= 100 and step.title.endswith("…")


def test_caps_fold_the_rest_into_one_closing_step() -> None:
    body = "".join(f"## Step {i}\n" for i in range(1, 46))
    steps = skill_skeleton(body, skill="big")
    assert len(steps) == 40
    assert steps[-1].title == "Remaining sections of /big (6 more)"
    assert steps[-1].note.startswith("from /big;")
    body = "## Phase 1\n" + "".join(f"### 1.{i}\n" for i in range(1, 16))
    [section] = skill_skeleton(body, skill="wide")
    assert len(section.children) == 12
    assert section.note.endswith("(+3 more sub-sections not listed)")


def test_composed_skill_body_is_the_text_after_the_frame() -> None:
    assert _composed_skill_body(FRAME + ENCODE) == ENCODE
    assert _composed_skill_body("plain user text") == ""
    elided = FRAME + '<command-body elided="true" chars="9">gone</command-body>'
    assert _composed_skill_body(elided) == ""


# ── the loop seeds at both doors ─────────────────────────────────────────────────


class _Resolver:
    def __init__(self, bodies: dict[str, str]) -> None:
        self._bodies = bodies

    def names(self) -> list[str]:
        return list(self._bodies)

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        if name in self._bodies:
            return SkillLoad(found=True, name=name, body=self._bodies[name])
        return SkillLoad(found=False, name=name)


class _ScriptByCall(Provider):
    """``factory(call_number, messages)`` → LLMResult; records every message list seen."""

    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self.calls = 0
        self.seen: list[list[Message]] = []

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        self.seen.append(list(messages))
        return self._factory(self.calls, messages)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _loop(provider: Provider, tmp_path: Path, bodies: dict[str, str]) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(UseSkillTool())
    registry.register(UpdatePlanTool())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=10,
        skill_resolver=_Resolver(bodies),
    )


def _use(name: str, call_id: str = "t1") -> LLMResult:
    return LLMResult(tool_calls=[ToolCall(id=call_id, name="use_skill", arguments={"name": name})])


def _finish_plan(call_id: str = "p1") -> LLMResult:
    """The model's own full-replace plan: one step, already done (update_plan semantics)."""
    return LLMResult(
        tool_calls=[
            ToolCall(
                id=call_id,
                name="update_plan",
                arguments={"tasks": [{"title": "carried out", "status": "done"}]},
            )
        ]
    )


def _user_texts(messages: list[Message]) -> list[str]:
    return [m.text or "" for m in messages if m.role == "user"]


def test_typed_skill_turn_starts_from_its_sections(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        return _finish_plan() if n == 1 else LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path, {"encode-session": ENCODE})
    result = asyncio.run(loop.arun_turn(FRAME + ENCODE))
    assert result.stop_reason == "completed"
    first = _user_texts(provider.seen[0])
    # The rail and the seeded plan were both in front of the FIRST completion.
    assert any(
        "I added the 5 sections of /encode-session to your plan as steps (1–5)" in t for t in first
    )
    plan_seen = next(t for t in first if "Current plan (0/6 steps done):" in t)
    assert "[ ] 3.1 1.0 Pre-Encoding" in plan_seen
    assert any(e.data.get("kind") == "skill_skeleton" for e in result.trace.events)
    # update_plan is full-replace: the model's own plan won, and the turn finished clean.
    assert [t.title for t in loop.session.task_network.tasks] == ["carried out"]


def test_streaming_twin_announces_the_seeded_plan(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        return _finish_plan() if n == 1 else LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path, {"encode-session": ENCODE})

    async def run() -> list[AgentEvent]:
        return [ev async for ev in loop.astream_turn(FRAME + ENCODE)]

    events = asyncio.run(run())
    assert any(isinstance(ev, AgentDone) and ev.stop_reason == "completed" for ev in events)
    assert any(
        isinstance(ev, AgentStatus) and ev.message == "plan seeded from /encode-session: 5 steps"
        for ev in events
    )
    updates = [ev for ev in events if isinstance(ev, AgentTaskUpdate)]
    assert updates and "Lane 1: Encoding" in updates[0].plan and updates[0].total == 6


def test_use_skill_load_seeds_the_sections_after_its_result(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use("encode-session")
        if n == 2:
            return _finish_plan()
        return LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path, {"encode-session": ENCODE})
    result = asyncio.run(loop.arun_turn("Ok, I will deal with that later. Now run /encode-session"))
    assert result.stop_reason == "completed"
    second = provider.seen[1]
    # Order: the tool result, THEN the rail, then the plan re-injection last.
    kinds = [m.role for m in second]
    tool_at = max(i for i, m in enumerate(second) if m.role == "tool")
    assert kinds[tool_at + 1] == "user"
    assert "I added the 5 sections of /encode-session to your plan as steps (1–5)" in (
        second[tool_at + 1].text or ""
    )
    assert "Current plan (0/6 steps done):" in (second[-1].text or "")


def test_sections_nest_under_the_step_that_named_the_skill(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use("encode-session")
        if n == 2:
            return _finish_plan()
        return LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path, {"encode-session": ENCODE, "fresh-eyes-code": "Review."})
    # Two skills named → ADR-0017 seeds "run /fresh-eyes-code", "run /encode-session".
    result = asyncio.run(loop.arun_turn("do a /fresh-eyes-code and a /encode-session"))
    assert result.stop_reason == "completed"
    plan_seen = next(t for t in _user_texts(provider.seen[1]) if "Current plan" in t)
    assert "[ ] 2 run /encode-session" in plan_seen
    assert "[ ] 2.1 Phase 0: Load Conventions" in plan_seen  # nested under the naming step
    assert "[ ] 2.3.1 1.0 Pre-Encoding" in plan_seen
    rail = next(t for t in _user_texts(provider.seen[1]) if "I added the 5 sections" in t)
    assert "(2.1–2.5)" in rail


def test_a_skill_without_sections_is_left_to_the_model(tmp_path: Path) -> None:
    # No numbered sections → nothing to seed: ADR-0027's hint stands and the plan is the
    # model's own. No step, no rail — a short skill stays ceremony-free.
    def script(n: int, messages: list[Message]) -> LLMResult:
        return _use("plain") if n == 1 else LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path, {"plain": "Do the thing.\nThen the other thing."})
    result = asyncio.run(loop.arun_turn("run /plain"))
    assert result.stop_reason == "completed" and provider.calls == 2
    assert loop.session.task_network.tasks == []
    assert not any("I added" in t for t in _user_texts(provider.seen[1]))
    assert not any(e.data.get("kind") == "skill_skeleton" for e in result.trace.events)


def test_a_skeleton_is_seeded_once(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use("encode-session", "t1")
        if n == 2:
            return _use("encode-session", "t2")  # a second load in the same turn
        if n == 3:
            return _finish_plan()
        return LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path, {"encode-session": ENCODE})
    result = asyncio.run(loop.arun_turn("run /encode-session"))
    assert result.stop_reason == "completed"
    third = _user_texts(provider.seen[2])
    assert sum("I added the 5 sections" in t for t in third) == 1
    plan_seen = next(t for t in third if "Current plan" in t)
    assert plan_seen.count("Phase 0: Load Conventions") == 1
    # Across turns the plan's own marker is the guard: a re-load never seeds twice.
    loop.session.task_network.tasks = skill_skeleton(ENCODE, skill="encode-session")
    loop.session.task_network.normalize()
    assert loop._seed_skill_skeleton("encode-session", ENCODE, set()) == []
    assert len(loop.session.task_network.tasks) == 5


def test_a_step_the_model_already_broke_down_is_left_alone(tmp_path: Path) -> None:
    loop = _loop(_ScriptByCall(lambda n, m: LLMResult(text="hi")), tmp_path, {"x": ENCODE})
    network = loop.session.task_network
    network.tasks = [
        Task(
            title="run /encode-session",
            kind="compound",
            children=[Task(title="my own first step")],
        )
    ]
    network.normalize()
    assert loop._seed_skill_skeleton("encode-session", ENCODE, set()) == []
    assert [c.title for c in network.tasks[0].children] == ["my own first step"]


# ── the tool's hint matches what the loop did ─────────────────────────────────────


async def test_use_skill_names_the_seeded_sections_in_its_hint(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path, skill_resolver=_Resolver({"e": ENCODE}))  # type: ignore[arg-type]
    res = await UseSkillTool().execute({"name": "e"}, ctx)
    assert res.is_error is False
    assert res.data == {"skill": "e", "decompose": True, "sections": 5}
    assert res.hint and "5 numbered sections are now steps in your plan" in res.hint
    assert "use_skill" in res.hint  # the chaining nudge survives

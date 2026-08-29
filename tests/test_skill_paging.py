"""ADR-0067: a sectioned skill is paged through the plan — one section in context at a time.

ADR-0062 made a skill's sections the plan; ADR-0066 made a skill that cannot fit the window
refuse loudly. This closes the gap between them: the harness hands the model the skeleton
plus ONE section's body, and turns the page when ``update_plan`` moves the plan past it —
so context is bounded by the largest section, not the largest skill, and a 32k model can
run a 184 KB skill one lane at a time. Both doors (a ``use_skill`` load and a typed
``/<skill>`` turn) page the same way. Hermetic: scripted providers, fake resolvers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from zakcode import tasks
from zakcode.agent.loop import AgentLoop
from zakcode.events import AgentEvent, AgentStatus
from zakcode.messages import Message, ToolResultBlock
from zakcode.permissions import PermissionMode, PermissionPolicy
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.skills.fit import measure_skill_fit
from zakcode.tasks import SkillPage, skill_pages, skill_skeleton
from zakcode.tools.base import SkillLoad, ToolRegistry
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.tools.builtins.use_skill import UseSkillTool

DEMO = """# /demo — a paged skill

Intro prose the sections rely on.

## Rules

Always be brief.

## Step 1: First

Do the first thing.

### 1.1 A sub-step

Stays inside page 1.

## Step 2: Second

Do the second thing. Then check the second result before you go on to the next.

## Step 3: Third

Do the third thing. Then check the third result before you call the work done.

## Return Protocol

End with a tool call.
"""

FRAME = "<command-message>demo is running</command-message>\n<command-name>/demo</command-name>\n\n"
MARK = "from /demo; done when this section has been carried out"


@pytest.fixture(autouse=True)
def _small_page_budget(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bodies here are a few hundred bytes: under the production budget they pack into
    one page and are delivered whole (ADR-0088). A 100-char budget keeps each of DEMO's
    sections its own page, as the paging contract tests need; ``real_page_budget`` opts a
    test out (the budget-cutting and packing tests measure against the real one)."""
    if request.node.get_closest_marker("real_page_budget") is None:
        monkeypatch.setattr(tasks, "PAGE_BUDGET_CHARS", 100)


# ── the pure splitter ─────────────────────────────────────────────────────────────


def test_pages_are_the_top_level_step_sections() -> None:
    pages = skill_pages(DEMO, skill="demo")
    assert pages is not None and pages.count == 3
    assert [p.title for p in pages.pages] == ["Step 1: First", "Step 2: Second", "Step 3: Third"]
    assert [p.marker for p in pages.pages] == ["step 1", "step 2", "step 3"]
    # The preamble and every documentation section travel up front; a sub-step stays in its page.
    assert "Intro prose" in pages.front and "## Rules" in pages.front
    assert "## Return Protocol" in pages.front
    assert "### 1.1 A sub-step" in pages.pages[0].text
    assert "Do the second thing." not in pages.pages[0].text
    # Page k is skeleton step k.
    assert [t.title for t in skill_skeleton(DEMO, skill="demo")] == [p.title for p in pages.pages]


def test_delivery_carries_the_header_and_the_paging_contract() -> None:
    pages = skill_pages(DEMO, skill="demo")
    assert pages is not None
    first = pages.first()
    assert first.startswith("# /demo — a paged skill")
    assert "[/demo — page 1 of 3: Step 1: First]" in first
    assert "Do the first thing." in first and "Do the second thing." not in first
    assert "section 2 of 3 arrives in the reply to that call" in first
    assert pages.render(3).endswith("This is the last section.")


def test_fewer_than_two_sections_is_delivered_whole() -> None:
    assert skill_pages("# x\n\n## Step 1: Only\n\ntext\n", skill="x") is None
    assert skill_pages("# x\n\n## Syntax\n\n## Chaining\n", skill="x") is None
    assert skill_pages("Do the thing.\n\nThen the other thing.\n", skill="x") is None


def test_bold_lead_ins_and_fenced_phase_comments_page_too() -> None:
    # ADR-0084: the control skills' bold checklist and a loop skill's fenced pseudocode
    # markers are sections — /start (63 KB) and /worker-loop (84 KB) were delivered whole.
    filler = "x " * 30
    bold = skill_pages(
        f"# x\n\n**Step 1**: bold only.\n\n{filler}\n\n**Step 2**: still bold.\n\n{filler}\n",
        skill="x",
    )
    assert bold is not None and [p.title for p in bold.pages] == [
        "Step 1: bold only",
        "Step 2: still bold",
    ]
    assert bold.front == "# x"
    loop = (
        "# /w\n\nrules\n\n## The loop\n\n```\n# Phase -0.5 — LIGHT PRIME. Two tiers.\n"
        "prime()\n\n# Phase 1 — SELECT (reuse the scorer)\nselect()\n```\n\n"
        "## Return Protocol\n\nend\n"
    )
    pages = skill_pages(loop, skill="w")
    assert pages is not None and [p.title for p in pages.pages] == [
        "Phase -0.5 — LIGHT PRIME",
        "Phase 1 — SELECT (reuse the scorer)",
    ]
    assert [p.marker for p in pages.pages] == ["phase -0.5", "phase 1"]
    # A cut inside a fence is re-fenced on both sides: every page is markdown on its own.
    assert pages.pages[0].text == "```\n# Phase -0.5 — LIGHT PRIME. Two tiers.\nprime()\n\n```"
    assert pages.pages[1].text == "```\n# Phase 1 — SELECT (reuse the scorer)\nselect()\n```"
    assert "## The loop" in pages.front and "## Return Protocol" in pages.front
    assert [t.title for t in skill_skeleton(loop, skill="w")] == [p.title for p in pages.pages]


def _paragraphs(n: int) -> str:
    filler = ("lorem ipsum " * 60).strip()  # ~720 chars a paragraph
    return "\n\n".join(f"{filler} {i}" for i in range(n))


@pytest.mark.real_page_budget
def test_a_section_over_the_budget_is_cut_at_its_markers_then_headings_then_paragraphs() -> None:
    from zakcode.tasks import PAGE_BUDGET_CHARS

    body = (
        "# /big\n\n## Step 1: Cut at sub-steps\n\nintro\n\n"
        f"### Step 1.1: first\n\n{_paragraphs(4)}\n\n### Step 1.2: second\n\n{_paragraphs(30)}\n\n"
        f"## Step 2: Cut at headings\n\n### Notes A\n\n{_paragraphs(10)}\n\n"
        f"### Notes B\n\n{_paragraphs(10)}\n\n## Step 3: Small\n\ntext\n"
    )
    pages = skill_pages(body, skill="big")
    assert pages is not None
    assert [p.title for p in pages.pages] == [
        # The section's own intro (before its first sub-step) and Step 1.1 share a page —
        # consecutive small pieces pack to the budget (ADR-0088).
        "Step 1: Cut at sub-steps (+1 more)",
        "Step 1.2: second (1/2)",  # a sub-step over the budget: paragraphs, packed
        "Step 1.2: second (2/2)",
        "Notes A",  # no ordered-work marker inside Step 2: any heading cuts it
        "Notes B (+1 more)",  # Notes B and the small Step 3 fit one page together
    ]
    assert [p.marker for p in pages.pages] == ["step 1", "step 1.2", "step 1.2", "", ""]
    assert pages.pages[0].sections == (
        ("Step 1: Cut at sub-steps", "step 1"),
        ("Step 1.1: first", "step 1.1"),
    )
    assert pages.pages[4].matches("Step 3: Small") and pages.pages[4].matches("step 3 (tiny)")
    assert all(len(p.text) <= PAGE_BUDGET_CHARS for p in pages.pages)
    # A part keeps its heading with its first paragraph — never a page of just the title.
    assert pages.pages[1].text.startswith("### Step 1.2: second\n\nlorem ipsum")
    assert pages.pages[2].text.startswith("lorem ipsum")
    assert pages.front == "# /big"
    steps = skill_skeleton(body, skill="big")
    assert [t.title for t in steps] == [p.title for p in pages.pages]
    # A packed page's sections are its step's sub-steps.
    assert [c.title for c in steps[0].children] == ["Step 1: Cut at sub-steps", "Step 1.1: first"]
    assert [c.title for c in steps[4].children] == ["Notes B", "Step 3: Small"]


@pytest.mark.real_page_budget
def test_small_consecutive_sections_share_a_page() -> None:
    """ADR-0088: a page costs a model turn to deliver, so consecutive sections pack to the
    budget — measured on a Mind's 131 skills: 976 pages, 321 deliveries. Any packed
    section names the page; the sections are the step's sub-steps; a skill that packs
    into ONE page is delivered whole, its sections still the plan's steps."""
    body = (
        "# /p\n\n"
        + "\n\n".join(f"## Step {k}: {t}\n\n{_paragraphs(4)}" for k, t in enumerate("ABCDEF", 1))
        + "\n"
    )
    pages = skill_pages(body, skill="p")
    assert pages is not None and pages.count == 2
    assert [p.title for p in pages.pages] == ["Step 1: A (+3 more)", "Step 5: E (+1 more)"]
    assert pages.header(1) == "[/p — page 1 of 2: Step 1: A (+3 more)]"
    assert pages.pages[0].matches("Step 3: C") and pages.pages[0].matches("step 4 — rewritten")
    assert not pages.pages[0].matches("Step 5: E") and pages.pages[1].matches("Step 6: F")
    assert "## Step 4: D" in pages.pages[0].text and "## Step 5: E" not in pages.pages[0].text
    steps = skill_skeleton(body, skill="p")
    assert [t.title for t in steps] == [p.title for p in pages.pages]
    assert steps[0].kind == "compound"
    assert [c.title for c in steps[0].children] == [
        "Step 1: A",
        "Step 2: B",
        "Step 3: C",
        "Step 4: D",
    ]
    # DEMO's three short sections pack into one page: delivered whole, three plan steps.
    assert skill_pages(DEMO, skill="demo") is None
    assert [t.title for t in skill_skeleton(DEMO, skill="demo")] == [
        "Step 1: First",
        "Step 2: Second",
        "Step 3: Third",
    ]


def test_a_documentation_section_holding_steps_is_a_container() -> None:
    # /review-hypotheses: 21 `### Step` headings under `## Mode 1/2/3` — non-step `##`
    # headings that used to send the whole 51 KB up front.
    load, judge = "load " * 16, "judge " * 13
    body = (
        "# /modes\n\n## Syntax\n\n`/modes --resolve`\n\n## Mode 1: Resolve\n\n"
        "How resolving works.\n\n"
        f"### Step 1: Load\n\n{load}\n\n### Step 2: Judge\n\n{judge}\n\n## Return Protocol\n\nend\n"
    )
    pages = skill_pages(body, skill="modes")
    assert pages is not None and [p.title for p in pages.pages] == ["Step 1: Load", "Step 2: Judge"]
    assert pages.pages[0].text == f"### Step 1: Load\n\n{load.rstrip()}"
    # The container's own intro travels up front with the other documentation, in order.
    assert pages.front == (
        "# /modes\n\n## Syntax\n\n`/modes --resolve`\n\n## Mode 1: Resolve\n\nHow resolving works."
        "\n\n## Return Protocol\n\nend"
    )
    assert [t.title for t in skill_skeleton(body, skill="modes")] == [
        "Step 1: Load",
        "Step 2: Judge",
    ]


def test_fenced_headings_never_split_a_page() -> None:
    filler = "x " * 20
    fence = "```\n## Step 2: inside a fence\n```"
    body = f"## Step 1: A\n\n{filler}\n\n{fence}\n\n## Step 2: B\n\n{filler}\n"
    pages = skill_pages(body, skill="x")
    assert pages is not None and pages.count == 2
    assert "inside a fence" in pages.pages[0].text


def test_pages_fold_past_the_skeleton_cap_exactly_like_the_skeleton() -> None:
    body = "\n".join(f"## Step {i}: S{i}\n\n{'body ' * 12}\n" for i in range(1, 66))
    pages = skill_pages(body, skill="big")
    steps = skill_skeleton(body, skill="big")
    assert pages is not None and pages.count == len(steps) == 60
    assert pages.pages[-1].title == steps[-1].title == "Remaining sections of /big (6 more)"
    assert "## Step 65: S65" in pages.pages[-1].text


def test_a_page_finds_its_step_after_the_model_rewrote_the_title() -> None:
    page = SkillPage(index=2, title="Step 2: Second", marker="step 2", text="")
    assert page.matches("Step 2: Second")
    assert page.matches("step 2 + 3: second and third, merged")
    assert not page.matches("Step 20: something else")
    assert not page.matches("Step 2.1: a sub-step")
    assert not SkillPage(index=1, title="Only title", marker="", text="").matches("other")


# ── the loop turns the page ───────────────────────────────────────────────────────


class _Resolver:
    def __init__(self, bodies: dict[str, str]) -> None:
        self._bodies = bodies

    def names(self) -> list[str]:
        return list(self._bodies)

    def body(self, name: str) -> str | None:
        return self._bodies.get(name)

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        if name in self._bodies:
            return SkillLoad(found=True, name=name, body=self._bodies[name])
        return SkillLoad(found=False, name=name)


class _ScriptByCall(Provider):
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
        chars = len(system or "")
        for message in messages:
            for block in message.blocks:
                chars += len(getattr(block, "text", "") or "")
        return chars // 4

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=32_768)


def _loop(provider: Provider, tmp_path: Path, session: Session | None = None) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(UseSkillTool())
    registry.register(UpdatePlanTool())
    return AgentLoop(
        provider,
        registry,
        session or Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=10,
        skill_resolver=_Resolver({"demo": DEMO}),
    )


def _use(call_id: str = "t1") -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id=call_id, name="use_skill", arguments={"name": "demo"})]
    )


def _plan(tasks: list[dict[str, Any]], call_id: str = "p1") -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id=call_id, name="update_plan", arguments={"tasks": tasks})]
    )


def _seeded(*statuses: str) -> list[dict[str, Any]]:
    """The seeded skeleton with the given statuses (notes intact — the common shape)."""
    titles = ["Step 1: First", "Step 2: Second", "Step 3: Third"]
    return [{"title": t, "status": s, "note": MARK} for t, s in zip(titles, statuses, strict=True)]


def _user_texts(messages: list[Message]) -> list[str]:
    return [m.text or "" for m in messages if m.role == "user"]


def _notes(loop: AgentLoop, kind: str) -> list[dict[str, Any]]:
    return [e.data for e in loop._trace.events if e.data.get("kind") == kind]


def _page_after_plan(provider: _ScriptByCall, header: str) -> tuple[str, list[list[Message]]]:
    """The page text as the model first saw it (a user message directly after a tool-result
    message) and every message list the provider saw before that call."""
    for i, msgs in enumerate(provider.seen):
        for j, m in enumerate(msgs):
            if m.role == "user" and header in (m.text or ""):
                assert msgs[j - 1].role == "tool", "the page must follow the update_plan result"
                return m.text or "", provider.seen[:i]
    raise AssertionError(f"{header} never reached the model")


def test_use_skill_delivers_page_one_and_seeds_every_section(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        return _use() if n == 1 else LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    block = next(
        b for m in loop.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)
    )
    assert "[/demo — page 1 of 3: Step 1: First]" in block.output
    assert "Do the first thing." in block.output and "Do the second thing." not in block.output
    # The skeleton names EVERY section — seeded from the whole body, not from page 1.
    assert [t.title for t in loop.session.task_network.tasks] == [
        "Step 1: First",
        "Step 2: Second",
        "Step 3: Third",
    ]
    rail = next(t for t in _user_texts(provider.seen[1]) if "I added the 3 sections" in t)
    assert "You hold section 1's instructions now" in rail
    assert loop._skill_pages_delivered["demo"] == {1}


def test_marking_a_section_done_turns_the_page(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(_seeded("done", "in_progress", "pending"))
        return LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("run /demo"))
    assert result.stop_reason == "completed"
    # Page 2 arrived as a user message right after the update_plan result — the next call
    # the model made saw it there. (Side calls such as the plan-quality check share the
    # provider, so the transcript is searched, not indexed.)
    page, before = _page_after_plan(provider, "[/demo — page 2 of 3: Step 2: Second]")
    assert "Do the second thing." in page and "Do the third thing." not in page
    assert "section 3 of 3 arrives" in page
    # Page 2 was never in context before the plan reached it.
    assert not any("Do the second thing." in t for msgs in before for t in _user_texts(msgs))
    # The effectiveness signal: one page note, and the turn summary.
    (note,) = _notes(loop, "skill_page")
    assert (note["skill"], note["page"], note["of"], note["skipped"]) == ("demo", 2, 3, 0)
    assert note["tokens"] > 0
    (summary,) = _notes(loop, "skill_paging")
    assert (summary["pages"], summary["delivered"], summary["closed"]) == (3, 2, 1)
    assert summary["delivered_tokens"] > 0 and summary["body_tokens"] > 0


def test_the_plan_pulls_pages_in_order_and_never_twice(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(_seeded("done", "in_progress", "pending"))
        if n == 3:
            return _plan(_seeded("done", "in_progress", "pending"), call_id="p2")  # no change
        if n == 4:
            return _plan(_seeded("done", "done", "in_progress"), call_id="p3")
        return LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    pages = [n["page"] for n in _notes(loop, "skill_page")]
    assert pages == [2, 3]
    assert loop._skill_pages_delivered["demo"] == {1, 2, 3}
    assert loop.current_skill_page("demo") is not None  # section 3 is still open
    assert "[/demo — page 3 of 3" in (loop.current_skill_page("demo") or "")


def test_a_merged_step_finishes_both_pages_and_counts_the_skip(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            # The model rewrote the plan without notes and merged two sections.
            return _plan(
                [
                    {"title": "Step 1 + 2: First and second, together", "status": "done"},
                    {"title": "Step 3: Third", "status": "in_progress"},
                ]
            )
        return LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    (note,) = _notes(loop, "skill_page")
    assert (note["page"], note["skipped"]) == (3, 1)
    assert any("[/demo — page 3 of 3" in t for msgs in provider.seen for t in _user_texts(msgs))
    assert not any("Do the second thing." in t for msgs in provider.seen for t in _user_texts(msgs))


def test_every_section_closed_means_no_page(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            # Cancelled, not done: a section closed as done unseen is reopened (ADR-0086).
            return _plan(_seeded("done", "cancelled", "cancelled"))
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    assert _notes(loop, "skill_page") == []
    assert loop.current_skill_page("demo") is None


def test_the_typed_door_carries_page_one_and_seeds_the_rest(tmp_path: Path) -> None:
    pages = skill_pages(DEMO, skill="demo")
    assert pages is not None

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _plan(_seeded("done", "in_progress", "pending"))
        return LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn(FRAME + pages.first()))
    assert [t.title for t in loop.session.task_network.tasks][:3] == [
        "Step 1: First",
        "Step 2: Second",
        "Step 3: Third",
    ]
    assert loop._skill_pages_delivered["demo"] == {1, 2}
    assert any("[/demo — page 2 of 3" in t for t in _user_texts(provider.seen[1]))


def test_a_restart_reads_how_far_it_was_paged_from_the_transcript(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(_seeded("done", "in_progress", "pending"))
        return LLMResult(text="done")

    first = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(first.arun_turn("run /demo"))
    # A new loop over the same session (a restart) forgets nothing the transcript kept.
    second = _loop(_ScriptByCall(lambda n, m: LLMResult(text="x")), tmp_path, first.session)
    assert second._pages_in_transcript("demo") == {1, 2}
    assert second._ensure_skill_pages("demo") is not None
    assert second._skill_pages_delivered["demo"] == {1, 2}


def test_streaming_twin_announces_the_page(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(_seeded("done", "in_progress", "pending"))
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)

    async def run() -> list[AgentEvent]:
        return [ev async for ev in loop.astream_turn("run /demo")]

    events = asyncio.run(run())
    assert any(
        isinstance(ev, AgentStatus) and ev.message == "page 2/3 of /demo: Step 2: Second"
        for ev in events
    )


def test_a_page_that_cannot_fit_ends_the_turn_loudly(tmp_path: Path) -> None:
    huge = DEMO.replace("Do the second thing.", "Do the second thing. " + "x" * 200_000)

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(_seeded("done", "in_progress", "pending"))
        return LLMResult(text="done")

    registry = ToolRegistry()
    registry.register(UseSkillTool())
    registry.register(UpdatePlanTool())
    provider = _ScriptByCall(script)
    loop = AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=10,
        skill_resolver=_Resolver({"demo": huge}),
    )
    result = asyncio.run(loop.arun_turn("run /demo"))
    assert result.stop_reason == "skill_too_large"
    # Page 1 loaded fine; page 2 was never handed over, in any form, and the model was
    # not asked to continue without it.
    assert not any("Do the second thing." in t for msgs in provider.seen for t in _user_texts(msgs))
    assert not any(
        m.role == "assistant" and m.text == "done" for msgs in provider.seen for m in msgs
    )
    (note,) = [e for e in loop._trace.events if e.data.get("kind") == "skill_too_large"]
    assert "section 2 of skill 'demo'" in note.detail


def test_fit_report_names_the_largest_section() -> None:
    fits = measure_skill_fit(
        [("demo", "x" * 400)], count_tokens=lambda t: len(t) // 4, window=1000, paged={"demo"}
    )
    assert fits[0].paged is True
    assert "largest section" in fits[0].describe()


@pytest.mark.real_page_budget
def test_a_paged_load_without_a_whole_body_is_named_not_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolver that delivers page 1 but cannot hand the loop the whole body would leave a
    one-step plan and a skill that never turns a page — the loop says so in the trace."""
    # DEMO pages only under a small budget; the loop then reads page 1's text under the
    # real one, where it is a single section (front matter and all).
    with monkeypatch.context() as small:
        small.setattr(tasks, "PAGE_BUDGET_CHARS", 100)
        pages = skill_pages(DEMO, skill="demo")
    assert pages is not None

    class _PageOnlyResolver(_Resolver):
        def body(self, name: str) -> str | None:
            return None

        async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
            return SkillLoad(found=True, name=name, body=pages.first())

    def script(n: int, messages: list[Message]) -> LLMResult:
        return _use() if n == 1 else LLMResult(text="done")

    registry = ToolRegistry()
    registry.register(UseSkillTool())
    registry.register(UpdatePlanTool())
    loop = AgentLoop(
        _ScriptByCall(script),
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=10,
        skill_resolver=_PageOnlyResolver({}),
    )
    asyncio.run(loop.arun_turn("run /demo"))
    (note,) = _notes(loop, "skill_page_body_missing")
    assert note["skill"] == "demo"
    assert [t.title for t in loop.session.task_network.tasks] == ["Step 1: First"]


# ── the first field run's defects (2026-08-28, coach /boot) ──────────────────────


def _start_steps(*statuses: str) -> list[dict[str, Any]]:
    """Another skill's steps, seeded before ours — /start's, closed."""
    titles = ["Step 1: Check state", "Step 2: Set Mode", "Step 3: Activate"]
    note = "from /start; done when this section has been carried out"
    return [{"title": t, "status": s, "note": note} for t, s in zip(titles, statuses, strict=True)]


def test_another_skills_closed_steps_never_satisfy_our_pages(tmp_path: Path) -> None:
    """/boot's 'Step 1..3' pages were matched by /start's closed 'Step 1..3' steps and the
    boot jumped to page 14 — a page matches only steps seeded from its own skill or owned
    by none."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            # The model's rewrite: /start's steps kept, /demo's section 1 done, 2–3 dropped.
            return _plan(_start_steps("done", "done", "done") + _seeded("done", "done", "done")[:1])
        return LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    (note,) = _notes(loop, "skill_page")
    assert (note["page"], note["skipped"]) == (2, 0)
    page = next(t for msgs in provider.seen for t in _user_texts(msgs) if "page 2 of 3" in t)
    assert "Your plan dropped 2 of /demo's sections" in page
    assert "'Step 2: Second'" in page and "Do the second thing." in page


def test_dropped_sections_come_back_into_the_plan_in_order(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(
                [{"title": "Something else the model added", "status": "done"}]
                + _seeded("done", "done", "done")[:1]
            )
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    (restored,) = _notes(loop, "skill_sections_restored")
    assert (restored["restored"], restored["pages"]) == (2, [2, 3])
    titles = [t.title for t in loop.session.task_network.tasks]
    assert titles == [
        "Something else the model added",
        "Step 1: First",
        "Step 2: Second",
        "Step 3: Third",
    ]
    assert all(t.note.startswith("from /demo") for t in loop.session.task_network.tasks[1:])
    assert [t.status for t in loop.session.task_network.tasks[2:]] == ["pending", "pending"]
    assert loop._skill_pages_delivered["demo"] == {1, 2}


def test_a_section_the_plan_moved_past_is_not_restored(tmp_path: Path) -> None:
    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            # Section 2 dropped, section 3 under way: the model skipped 2 on purpose.
            return _plan(
                [
                    _seeded("done", "done", "in_progress")[0],
                    _seeded("done", "done", "in_progress")[2],
                ]
            )
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    assert _notes(loop, "skill_sections_restored") == []
    (note,) = _notes(loop, "skill_page")
    assert (note["page"], note["skipped"]) == (3, 1)


# ── the second field run's defects (2026-08-28, coach /aspirations-precheck) ────────


def _keep_first(rewrite: int = 0) -> list[dict[str, Any]]:
    """The model's collapse: section 1 kept (its page is held), sections 2–3 dropped. Each
    ``rewrite`` words the note differently, as the field run's eight collapses did — a
    byte-identical re-send is a different case (deduped before it runs)."""
    note = MARK if rewrite == 0 else f"{MARK} (rewrite {rewrite})"
    return [{"title": "Step 1: First", "status": "in_progress", "note": note}]


def test_a_restore_with_no_new_page_still_tells_the_model(tmp_path: Path) -> None:
    """Section 1 is current and already held, so no page turns — the restore used to be
    silent, and the model re-issued the identical collapse every iteration (ADR-0075)."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(_keep_first())
        return LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    (restored,) = _notes(loop, "skill_sections_restored")
    assert (restored["restored"], restored["pages"]) == (2, [2, 3])
    assert not any(n["page"] > 1 for n in _notes(loop, "skill_page"))  # nothing turned
    rail = next(
        t
        for msgs in provider.seen
        for t in _user_texts(msgs)
        if "Your plan dropped 2 of /demo's sections" in t
    )
    assert "Deleting a section from the plan does not close it" in rail
    assert "mark it done or cancelled" in rail
    assert "here is the one that is current now" not in rail  # no page rode along


def test_a_third_drop_is_the_models_decision(tmp_path: Path) -> None:
    """Restored and explained twice, dropped a third time: the sections stay out, the
    drop is noted, and the plan is the model's (ADR-0075)."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        # Three collapses, counted by the plan results the transcript holds — the
        # plan-quality judge shares the provider, so call numbers are not iterations.
        plans = sum(
            1
            for m in messages
            for b in m.blocks
            if isinstance(b, ToolResultBlock) and (b.output or "").startswith("Current plan")
        )
        if plans < 3:
            return _plan(_keep_first(rewrite=n), call_id=f"p{n}")
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    assert [n["pages"] for n in _notes(loop, "skill_sections_restored")] == [[2, 3], [2, 3]]
    (dropped,) = _notes(loop, "skill_sections_dropped")
    assert dropped["pages"] == [2, 3]
    assert [t.title for t in loop.session.task_network.tasks] == ["Step 1: First"]


def test_the_plan_can_come_back_to_a_page_it_jumped_over(tmp_path: Path) -> None:
    """Delivery is 'never the same page twice', not 'only forward': after a jump to page 3
    the model re-added section 2 and made it current, so page 2 is delivered."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(
                [
                    _seeded("done", "done", "in_progress")[0],
                    _seeded("done", "done", "in_progress")[2],
                ]
            )
        if n == 4:  # n == 3 is the plan-quality side call that follows every update_plan
            return _plan(_seeded("done", "pending", "pending"), call_id="p2")
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    pages = [(n["page"], n["skipped"]) for n in _notes(loop, "skill_page")]
    assert pages == [(3, 1), (2, 0)]
    assert loop._skill_pages_delivered["demo"] == {1, 2, 3}


# ── the third field run's defect (2026-08-29, coach worker /start) ────────────────────


def _plans_sent(messages: list[Message]) -> int:
    """How many plan results the transcript holds — the plan-quality judge shares the
    provider, so call numbers are not iterations (see ``test_a_third_drop_...``)."""
    return sum(
        1
        for m in messages
        for b in m.blocks
        if isinstance(b, ToolResultBlock) and (b.output or "").startswith("Current plan")
    )


def test_a_section_marked_done_unseen_is_reopened_and_its_page_arrives(tmp_path: Path) -> None:
    """Nine sections closed in one rewrite, the RUNNING branch among them, then "waiting for
    the next /start page" at an idle prompt for an hour: the pages never came because their
    steps were closed (ADR-0086). Done before its page was held → pending again, page
    delivered, and the model told why."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        plans = _plans_sent(messages)
        if plans == 0:
            return _plan(_seeded("done", "done", "done"), call_id=f"p{n}")
        if plans == 1:  # page 2 arrived: done for real this time, 3 still pending
            return _plan(_seeded("done", "done", "pending"), call_id=f"p{n}")
        if plans == 2:  # page 3 arrived
            return _plan(_seeded("done", "done", "done"), call_id=f"p{n}")
        return LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    (reopened,) = _notes(loop, "skill_sections_reopened")
    assert (reopened["reopened"], reopened["pages"]) == (2, [2, 3])
    assert [n["page"] for n in _notes(loop, "skill_page")] == [2, 3]
    page, _ = _page_after_plan(provider, "[/demo — page 2 of 3: Step 2: Second]")
    assert "You marked 2 of /demo's sections done before their instructions were delivered" in page
    assert "('Step 2: Second', 'Step 3: Third')" in page
    assert "here is the one that is current now" in page
    assert "Do the second thing." in page
    assert [t.status for t in loop.session.task_network.tasks] == ["done", "done", "done"]
    assert loop._skill_pages_delivered["demo"] == {1, 2, 3}


def test_a_section_closed_unseen_gets_its_own_page_not_the_one_left_behind(
    tmp_path: Path,
) -> None:
    """The model skipped section 2 and closed 3 without its page. Reopening 3 and re-sending
    page 2 — the earliest open one — was a doom loop in the field: the model wanted 3, closed
    it again, got page 2 again (coach-w, 2026-08-29). ADR-0089: page 3 arrives, the rail
    names the section left behind, and the second close stands."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        plans = _plans_sent(messages)
        if plans == 0:
            return _plan(_seeded("done", "pending", "done"), call_id=f"p{n}")
        if plans == 1:  # page 3 held now: the close stands
            tasks = _seeded("done", "pending", "done")
            tasks[2]["note"] = f"{MARK} (carried out)"
            return _plan(tasks, call_id=f"p{n}")
        return LLMResult(text="done")

    provider = _ScriptByCall(script)
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    assert [n["pages"] for n in _notes(loop, "skill_sections_reopened")] == [[3]]
    assert [(n["page"], n["skipped"]) for n in _notes(loop, "skill_page")] == [(3, 1)]
    page, _ = _page_after_plan(provider, "[/demo — page 3 of 3: Step 3: Third]")
    assert "the first of them is delivered below" in page
    assert "Sections before it are still open ('Step 2: Second')" in page
    assert "Do the third thing." in page
    assert [t.status for t in loop.session.task_network.tasks] == ["done", "pending", "done"]
    assert loop._skill_pages_delivered["demo"] == {1, 3}
    assert _plans_sent(loop.session.messages) == 2


def test_a_model_that_keeps_closing_unseen_sections_is_walked_through_them(
    tmp_path: Path,
) -> None:
    """Every section closed at once, three times over: each reply delivers the first section
    still unseen, so the walk is bounded by the section count — never a loop (ADR-0089)."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        plans = _plans_sent(messages)
        if plans < 3:
            tasks = _seeded("done", "done", "done")
            for task in tasks:
                task["note"] = f"{MARK} (rewrite {plans})"
            return _plan(tasks, call_id=f"p{n}")
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    assert [n["pages"] for n in _notes(loop, "skill_sections_reopened")] == [[2, 3], [3]]
    assert [n["page"] for n in _notes(loop, "skill_page")] == [2, 3]
    assert [t.status for t in loop.session.task_network.tasks] == ["done", "done", "done"]
    assert loop._skill_pages_delivered["demo"] == {1, 2, 3}


def test_the_page_delivered_is_the_one_the_plan_is_on(tmp_path: Path) -> None:
    """Section 3 under way while 2 is still open: the plan is on 3, so page 3 arrives — not
    page 2, which the plan moved past (ADR-0089)."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(_seeded("done", "pending", "in_progress"))
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    assert [(n["page"], n["skipped"]) for n in _notes(loop, "skill_page")] == [(3, 1)]
    assert loop._skill_pages_delivered["demo"] == {1, 3}


def test_a_section_cancelled_unseen_stays_closed(tmp_path: Path) -> None:
    """Cancelling is a decision about the title — a branch that does not apply — and costs
    no page; only DONE claims work the model never saw."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(_seeded("done", "cancelled", "cancelled"))
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    assert _notes(loop, "skill_sections_reopened") == []
    assert _notes(loop, "skill_page") == []
    assert loop.current_skill_page("demo") is None


def test_cancelling_later_sections_does_not_finish_an_unseen_one(tmp_path: Path) -> None:
    """The worker renamed the RUNNING section, marked it done, and cancelled every branch
    after it: the page matched no step and counted as "moved past" on the cancellations, so
    it never came. A cancelled later step is not progress past an earlier one — the section
    comes back and its page arrives."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(
                [
                    {"title": "Step 1: First", "status": "done", "note": MARK},
                    {"title": "The middle, in the model's own words", "status": "done"},
                    {"title": "Step 3: Third", "status": "cancelled", "note": MARK},
                ]
            )
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    (restored,) = _notes(loop, "skill_sections_restored")
    assert restored["pages"] == [2]
    (note,) = _notes(loop, "skill_page")
    assert note["page"] == 2
    assert "Step 2: Second" in [t.title for t in loop.session.task_network.tasks]


def test_held_pages_survive_a_restart_that_lost_their_headers(tmp_path: Path) -> None:
    """ADR-0034 execs a new process and a compaction drops the headers with the messages
    they rode in. The session carries which pages were held, so a section closed after the
    restart is not reopened as unseen — and the next page turns as usual."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(_seeded("done", "in_progress", "pending"))
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    assert loop.session.skill_pages_delivered == {"demo": [1, 2]}
    loop.session.messages.clear()  # the compaction took every header
    session = Session.model_validate_json(loop.session.model_dump_json())  # the store round-trip

    def script2(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _plan(_seeded("done", "done", "in_progress"), call_id="p2")
        return LLMResult(text="done")

    loop2 = _loop(_ScriptByCall(script2), tmp_path, session=session)
    asyncio.run(loop2.arun_turn("carry on"))
    assert _notes(loop2, "skill_sections_reopened") == []
    (note,) = _notes(loop2, "skill_page")
    assert note["page"] == 3
    assert session.skill_pages_delivered == {"demo": [1, 2, 3]}


def test_a_document_saved_before_the_record_takes_its_open_work_as_held(tmp_path: Path) -> None:
    """A session written by an older build carries no record: the pages it held are read
    from the headers left in the transcript plus the sections the plan has taken up — so a
    section under way when the headers were compacted away is not reopened, and the rule
    governs only what is closed from here on."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(_seeded("done", "in_progress", "pending"))
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    loop.session.messages.clear()
    loop.session.skill_pages_delivered.clear()  # the older build never wrote it
    session = Session.model_validate_json(loop.session.model_dump_json())

    def script2(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _plan(_seeded("done", "done", "pending"), call_id="p2")
        return LLMResult(text="done")

    loop2 = _loop(_ScriptByCall(script2), tmp_path, session=session)
    asyncio.run(loop2.arun_turn("carry on"))
    assert _notes(loop2, "skill_sections_reopened") == []
    (note,) = _notes(loop2, "skill_page")
    assert note["page"] == 3
    assert session.skill_pages_delivered == {"demo": [1, 2, 3]}


# ── the fourth field run's defect (2026-08-29, coach-w "awaiting section 23") ─────────


def _unattended_loop(provider: Provider, tmp_path: Path) -> AgentLoop:
    """A worker Body: ``--dangerously-skip-permissions``, no one at the prompt."""
    registry = ToolRegistry()
    registry.register(UseSkillTool())
    registry.register(UpdatePlanTool())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=10,
        skill_resolver=_Resolver({"demo": DEMO}),
        permission_policy=PermissionPolicy(PermissionMode.BYPASS),
    )


def _section_nudges(messages: list[Message]) -> int:
    return sum(
        1
        for m in messages
        if m.role == "user" and "is still open, and nothing arrives on its own" in (m.text or "")
    )


def test_an_unattended_turn_does_not_end_on_an_open_section(tmp_path: Path) -> None:
    """A worker closed its unit and ended its turn "awaiting the final park instruction
    (section 23) from the harness" — nothing is pushed, and a Body has no one to type
    (ADR-0087). The stop becomes a nudge, once per section; a second stop ends the turn."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        plans, nudges = _plans_sent(messages), _section_nudges(messages)
        if plans == 0 and nudges == 0:
            return LLMResult(text="Section 1 is finished. Awaiting section 2 from the harness.")
        if plans == 0:
            return _plan(_seeded("done", "in_progress", "pending"), call_id=f"p{n}")
        if nudges == 1:
            return LLMResult(text="Section 2 is finished. Awaiting section 3.")
        return LLMResult(text="Still waiting for section 3.")

    provider = _ScriptByCall(script)
    loop = _unattended_loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    gates = _notes(loop, "section_gate")
    assert [(g["skill"], g["page"]) for g in gates] == [("demo", 1), ("demo", 2)]
    nudge = next(
        t for msgs in provider.seen for t in _user_texts(msgs) if "Section 1 of 3 of /demo" in t
    )
    assert "'Step 1: First'" in nudge and "Do not stop to wait for the harness" in nudge
    assert [n["page"] for n in _notes(loop, "skill_page")] == [2]
    # Nudged once on section 2, stopped again: the turn ended on the model's words.
    assert loop.session.messages[-1].role == "assistant"
    assert "Still waiting" in (loop.session.messages[-1].text or "")


def test_an_attended_turn_may_end_on_an_open_section(tmp_path: Path) -> None:
    """With someone at the prompt the model may stop to talk — the guard is for Bodies."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        return LLMResult(text="Which option do you want for section 1?")

    loop = _loop(_ScriptByCall(script), tmp_path)  # no policy: the default asks
    asyncio.run(loop.arun_turn("run /demo"))
    assert _notes(loop, "section_gate") == []
    assert loop.session.messages[-1].role == "assistant"


# ── the fifth field run's defect (2026-08-29, coach-w2 on the ADR-0088 build) ──────────


def test_a_section_cancelled_then_dropped_stays_closed(tmp_path: Path) -> None:
    """The worker cancelled /start's IDLE branches; a later rewrite dropped them; the restore
    (ADR-0075) put them back PENDING and pages 4–6 arrived one per turn while the worker was
    three skills further on (coach-w2, 2026-08-29). ADR-0091: a closed section is settled —
    a rewrite that drops it neither restores nor delivers it — and the record outlives a
    restart."""

    def script(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:
            return _use()
        if n == 2:
            return _plan(_seeded("in_progress", "cancelled", "cancelled"))
        return LLMResult(text="done")

    loop = _loop(_ScriptByCall(script), tmp_path)
    asyncio.run(loop.arun_turn("run /demo"))
    assert loop.session.skill_pages_settled == {"demo": [2, 3]}
    assert _notes(loop, "skill_page") == []
    session = Session.model_validate_json(loop.session.model_dump_json())  # the store round-trip

    def script2(n: int, messages: list[Message]) -> LLMResult:
        if n == 1:  # sections 2 and 3 dropped from the plan, after being cancelled
            return _plan([_seeded("done", "cancelled", "cancelled")[0]], call_id="p2")
        return LLMResult(text="done")

    loop2 = _loop(_ScriptByCall(script2), tmp_path, session=session)
    asyncio.run(loop2.arun_turn("carry on"))
    assert _notes(loop2, "skill_sections_restored") == []
    assert _notes(loop2, "skill_page") == []
    assert [t.title for t in loop2.session.task_network.tasks] == ["Step 1: First"]
    assert loop2.current_skill_page("demo") is None

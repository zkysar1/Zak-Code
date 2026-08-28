"""Mid-turn say delivery (ADR-0051): the workspace say inbox reaches a RUNNING turn.

Every original consumer of the say contract sits BETWEEN turns (the REPL's idle wait,
the serve driver's consumer beat) — but a perpetual-loop deployment's whole session is
ONE turn (one /start, then Stop-hook vetoes without end), so a message written mid-turn
starved forever (measured on a live Mind: an operator directive sat unconsumed for 3
days). The main loop now polls the inbox at every iteration boundary and folds a
pending message in as a framed user message; sub-agent loops never do.

Hermetic: scripted providers (no network), a tiny tool registry, tmp_path workspaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.events import AgentStatus
from zakcode.hooks import TurnEndPayload, TurnEndResult
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.say_inbox import say_path, say_pending, write_say
from zakcode.session.store import Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.tools.builtins.update_plan import UpdatePlanTool


class _Recording(Provider):
    """Replays scripted results and records the messages of every call."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0
        self.seen: list[list[Message]] = []

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        self.seen.append(list(messages))
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[i]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


class _SayWhileRunning(Tool):
    """Simulates the operator sending a say WHILE the turn is executing a tool."""

    spec = ToolSpec(name="poke", description="Write a say into the workspace inbox.")

    def __init__(self, inbox: Path, text: str) -> None:
        self._inbox = inbox
        self._text = text

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        write_say(self._inbox, self._text)
        return ToolResult.ok(output="poked")


def _loop(
    provider: Provider,
    tmp_path: Path,
    *,
    consume: bool = True,
    vetoable: bool = False,
    tools: list[Tool] | None = None,
) -> tuple[AgentLoop, Session]:
    registry = ToolRegistry()
    registry.register(UpdatePlanTool())  # the ADR-0052 hold tests plan; harmless elsewhere
    for t in tools or []:
        registry.register(t)
    session = Session(cwd=str(tmp_path), model="test/model")
    loop = AgentLoop(
        provider,
        registry,
        session,
        settings=load_settings(workspace_root=tmp_path),
        max_iterations=10,
        turn_end_vetoable=vetoable,
        consume_say_inbox=consume,
    )
    return loop, session


def _tool_call(name: str) -> LLMResult:
    return LLMResult(text="", tool_calls=[ToolCall(id="t1", name=name, arguments={})])


_DONE = LLMResult(text="all done")


@pytest.mark.asyncio
async def test_say_written_mid_turn_reaches_the_next_provider_call(tmp_path: Path) -> None:
    inbox = say_path(tmp_path)
    provider = _Recording([_tool_call("poke"), _DONE])
    loop, session = _loop(
        provider, tmp_path, tools=[_SayWhileRunning(inbox, "also check the logs")]
    )
    result = await loop.arun_turn("do the thing")

    assert result.stop_reason == "completed"
    # Consumed exactly once: the slot is free again.
    assert not say_pending(inbox)
    # The framed user message is IN the conversation (persisted, survives restarts)…
    framed = [m for m in session.messages if m.role == "user" and "also check the logs" in m.text]
    assert len(framed) == 1
    assert "arrived mid-task" in framed[0].text
    # …and the provider call AFTER the tool iteration actually saw it.
    assert any("also check the logs" in m.text for m in provider.seen[1] if m.role == "user")


@pytest.mark.asyncio
async def test_no_say_pending_changes_nothing(tmp_path: Path) -> None:
    provider = _Recording([_DONE])
    loop, session = _loop(provider, tmp_path)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"
    assert [m.text for m in session.messages if m.role == "user"] == ["hi"]


@pytest.mark.asyncio
async def test_subagent_shape_never_consumes_the_inbox(tmp_path: Path) -> None:
    """A loop built without ``consume_say_inbox`` (the sub-agent construction shape)
    must not steal the user's message into a child conversation."""
    inbox = say_path(tmp_path)
    write_say(inbox, "for the main agent, not you")
    provider = _Recording([_DONE])
    loop, session = _loop(provider, tmp_path, consume=False)
    await loop.arun_turn("child task")
    assert say_pending(inbox)  # untouched — still waiting for the main loop
    assert all("for the main agent" not in m.text for m in session.messages)


@pytest.mark.asyncio
async def test_veto_continuation_picks_up_a_say_sent_during_the_stop(tmp_path: Path) -> None:
    """The perpetual-loop shape: the Stop hook vetoes the finish, and a say sent
    meanwhile is delivered on the continued iteration — not starved to a turn
    boundary that never comes."""
    inbox = say_path(tmp_path)
    vetoes: list[TurnEndPayload] = []

    def hook(payload: TurnEndPayload) -> TurnEndResult | None:
        vetoes.append(payload)
        if len(vetoes) == 1:
            write_say(inbox, "new directive: research the API")  # operator types mid-stop
            return TurnEndResult(vetoed=True, continuation_prompt="keep going")
        return None

    provider = _Recording([_DONE, _DONE])
    loop, session = _loop(provider, tmp_path, vetoable=True)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("loop iteration")

    assert result.stop_reason == "completed"
    assert len(vetoes) == 2  # one veto, one allow
    assert not say_pending(inbox)
    # The continued iteration's provider call saw BOTH the continuation and the say.
    last_call_user_texts = [m.text for m in provider.seen[1] if m.role == "user"]
    assert any("keep going" in t for t in last_call_user_texts)
    assert any("research the API" in t for t in last_call_user_texts)


@pytest.mark.asyncio
async def test_streaming_path_delivers_and_announces(tmp_path: Path) -> None:
    inbox = say_path(tmp_path)
    provider = _Recording([_tool_call("poke"), _DONE])
    loop, session = _loop(provider, tmp_path, tools=[_SayWhileRunning(inbox, "streamed directive")])
    events = [e async for e in loop.astream_turn("do the thing")]

    assert not say_pending(inbox)
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("delivered mid-turn" in s and "streamed directive" in s for s in statuses)
    framed = [m for m in session.messages if m.role == "user" and "streamed directive" in m.text]
    assert len(framed) == 1


# ── ADR-0052: task-boundary hold ──────────────────────────────────────────────


class _EchoArg(Tool):
    spec = ToolSpec(name="step", description="One unit of step work.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(output=str(args.get("n", "")))


def _plan_call(tasks: list[dict[str, Any]]) -> LLMResult:
    return LLMResult(
        text="", tool_calls=[ToolCall(id="p1", name="update_plan", arguments={"tasks": tasks})]
    )


def _step_call(n: int) -> LLMResult:
    return LLMResult(text="", tool_calls=[ToolCall(id=f"s{n}", name="step", arguments={"n": n})])


#: A strong plan-judge scorecard (ADR-0050 consumes one scripted result per structural
#: plan authoring) — high across the board, so the judge stays silent.
_JUDGE_OK = LLMResult(
    text='{"scores": {"coverage": 0.9, "granularity": 0.9, "ordering": 0.9, "soundness": 0.9},'
    ' "notes": ""}'
)


@pytest.mark.asyncio
async def test_say_holds_mid_step_and_lands_at_the_step_seam(tmp_path: Path) -> None:
    """A say arriving while a step is in flight waits; the moment a step completes
    (the seam), it is delivered — before the patience cap is anywhere near."""
    inbox = say_path(tmp_path)
    provider = _Recording(
        [
            _plan_call(
                [
                    {"title": "A", "status": "in_progress", "note": "x"},
                    {"title": "B", "note": "y"},
                ]
            ),
            _JUDGE_OK,  # judge on the structural authoring (silent)
            _tool_call("poke"),  # operator sends the say mid-step
            _step_call(1),  # still mid-step: the boundary after this HELD the say
            _plan_call(
                [
                    {"title": "A", "status": "done", "note": "x"},
                    {"title": "B", "status": "in_progress", "note": "y"},
                ]
            ),  # A completes -> the seam
            _plan_call(
                [
                    {"title": "A", "status": "done", "note": "x"},
                    {"title": "B", "status": "done", "note": "y"},
                ]
            ),
            _DONE,
        ]
    )
    loop, session = _loop(
        provider,
        tmp_path,
        tools=[_SayWhileRunning(inbox, "switch to the API question next"), _EchoArg()],
    )
    result = await loop.arun_turn("two-step job")

    assert result.stop_reason == "completed"
    assert not say_pending(inbox)
    framed = [m for m in session.messages if m.role == "user" and "switch to the API" in m.text]
    assert len(framed) == 1
    # seen: [0]=plan, [1]=judge, [2]=poke, [3]=step(held boundary), [4]=tick(held),
    # [5]=post-seam (delivered at the boundary after A completed), [6]=done.
    held_calls = provider.seen[3] + provider.seen[4]
    assert all("switch to the API" not in m.text for m in held_calls if m.role == "user")
    assert any("switch to the API" in m.text for m in provider.seen[5] if m.role == "user")


@pytest.mark.asyncio
async def test_say_patience_cap_delivers_even_when_the_step_never_ends(tmp_path: Path) -> None:
    """The hold is bounded: a step that never completes cannot starve the message —
    after _SAY_PATIENCE held boundaries it is delivered mid-step anyway."""
    inbox = say_path(tmp_path)
    provider = _Recording(
        [
            _plan_call([{"title": "A", "status": "in_progress", "note": "x"}]),
            _JUDGE_OK,
            _tool_call("poke"),  # say arrives; step A never completes
            _step_call(1),
            _step_call(2),
            _step_call(3),
            _DONE,  # sees the delivered say; plan still open -> gate nudges, then repeats
        ]
    )
    loop, session = _loop(
        provider, tmp_path, tools=[_SayWhileRunning(inbox, "are you stuck?"), _EchoArg()]
    )
    await loop.arun_turn("one stubborn step")

    assert not say_pending(inbox)
    framed = [m for m in session.messages if m.role == "user" and "are you stuck?" in m.text]
    assert len(framed) == 1
    # Boundaries after poke: 3 holds (after poke, step1, step2), delivery on the 4th
    # (after step3) — so the call at seen[6] is the first that carries it.
    for i in (3, 4, 5):
        assert all("are you stuck?" not in m.text for m in provider.seen[i] if m.role == "user")
    assert any("are you stuck?" in m.text for m in provider.seen[6] if m.role == "user")


# ── ADR-0073: a typed /<skill> say runs the skill mid-turn ──────────────────────


class _Composed:
    def __init__(self, **kw: Any) -> None:
        self.invoked = kw.get("invoked", True)
        self.name = kw.get("name", "")
        self.turn_text = kw.get("turn_text")
        self.denied_reason = kw.get("denied_reason")
        self.error = kw.get("error")


_PROBE_BODY = (
    "<command-message>probe is running</command-message>\n"
    "<command-name>/probe</command-name>\n"
    "<command-args>coach</command-args>\n\n"
    "# Probe\n\n## Phase 1: Look\n\nlook around\n\n## Phase 2: Leap\n\nleap\n"
)


def _compose_fake(calls: list[tuple[str, str]]):
    async def compose(name: str, args: str = "", *, fuzzy: bool = True) -> _Composed:
        calls.append((name, args))
        if name == "probe":
            return _Composed(name="probe", turn_text=_PROBE_BODY)
        if name == "internal":
            return _Composed(name="internal", denied_reason="internal is user-invocable: false")
        return _Composed(invoked=False)

    return compose


def _skill_loop(provider: Provider, tmp_path: Path, calls: list, tools: list[Tool]):
    registry = ToolRegistry()
    registry.register(UpdatePlanTool())
    for t in tools:
        registry.register(t)
    session = Session(cwd=str(tmp_path), model="test/model")
    loop = AgentLoop(
        provider,
        registry,
        session,
        settings=load_settings(workspace_root=tmp_path),
        max_iterations=10,
        consume_say_inbox=True,
        compose_skill=_compose_fake(calls),
    )
    return loop, session


@pytest.mark.asyncio
async def test_a_typed_skill_say_runs_the_skill_mid_turn(tmp_path: Path) -> None:
    inbox = say_path(tmp_path)
    calls: list[tuple[str, str]] = []
    provider = _Recording([_tool_call("poke"), _DONE])
    loop, session = _skill_loop(
        provider, tmp_path, calls, [_SayWhileRunning(inbox, "/probe coach")]
    )
    events = [e async for e in loop.astream_turn("do the thing")]

    assert calls == [("probe", "coach")]  # the REPL's own composition, same arguments
    assert not say_pending(inbox)
    # The composed turn text lands as-is: the command frame leads (invocation provenance),
    # with no mid-task prose wrapped around it.
    composed = [
        m for m in session.messages if m.role == "user" and m.text.startswith("<command-message>")
    ]
    assert len(composed) == 1
    assert "arrived mid-task" not in composed[0].text
    # …and the provider call after the tool iteration saw it.
    assert any(m.text.startswith("<command-message>") for m in provider.seen[1] if m.role == "user")
    # The skill's sections were seeded into the plan, like a turn-opening skill.
    titles = [t.title for t in session.task_network.tasks]
    assert any("Look" in t for t in titles) and any("Leap" in t for t in titles)
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("delivered mid-turn" in s and "/probe coach" in s for s in statuses)


@pytest.mark.asyncio
async def test_a_refused_skill_say_is_not_handed_to_the_model(tmp_path: Path) -> None:
    inbox = say_path(tmp_path)
    calls: list[tuple[str, str]] = []
    provider = _Recording([_tool_call("poke"), _DONE])
    loop, session = _skill_loop(provider, tmp_path, calls, [_SayWhileRunning(inbox, "/internal")])
    events = [e async for e in loop.astream_turn("do the thing")]

    assert calls == [("internal", "")]
    assert not say_pending(inbox)  # consumed (exactly-once), refused, not delivered
    assert not any("internal" in m.text for m in session.messages if m.role == "user")
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("/internal not run" in s and "user-invocable" in s for s in statuses)


@pytest.mark.asyncio
async def test_a_slash_say_that_is_not_a_skill_is_delivered_as_text(tmp_path: Path) -> None:
    inbox = say_path(tmp_path)
    calls: list[tuple[str, str]] = []
    provider = _Recording([_tool_call("poke"), _DONE])
    loop, session = _skill_loop(
        provider, tmp_path, calls, [_SayWhileRunning(inbox, "/nonesuch now")]
    )
    await loop.arun_turn("do the thing")

    assert calls == [("nonesuch", "now")]
    framed = [m for m in session.messages if m.role == "user" and "/nonesuch now" in m.text]
    assert len(framed) == 1 and "arrived mid-task" in framed[0].text

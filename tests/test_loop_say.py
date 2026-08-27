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
        return Capabilities()


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

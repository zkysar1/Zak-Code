"""ADR-0063: what an empty completion inside a skill turn costs, and what it says.

Field 2026-08-28 (coach on zc-03, the composed ``/start`` turn): the third completion came
back empty — 254 tokens generated, no text, no thinking, no tool call — and read as a plain
silence; the skill nudge that followed made the model call ``use_skill start`` INSIDE
``/start``, and 65 KB of instructions it already held landed a second time, because the
command path never registered its load with the per-turn reload dedup. Two fixes, pinned
here: the typed skill counts as loaded (the re-invocation gets the short pointer, and a
TURN_END veto still clears it), and an empty completion's note and status say how many
tokens were generated and delivered as nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from zakcode import Agent
from zakcode.agent.loop import AgentLoop
from zakcode.config import Settings
from zakcode.events import AgentEvent, AgentStatus
from zakcode.hooks import TurnEndPayload, TurnEndResult
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    StreamUsage,
    ToolCall,
)
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.usage import Usage


class _Replay(Provider):
    """Replays scripted results; streams each one's text and usage the way a backend does."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = list(results)
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        if not self._results:
            raise AssertionError("provider ran out of scripted results")
        return self._results.pop(0)

    async def astream(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> AsyncIterator[ProviderStreamEvent]:
        result = await self.acomplete(messages, system=system, tools=tools)
        if result.text:
            yield StreamTextDelta(text=result.text)
        yield StreamUsage(usage=result.usage)
        yield StreamDone(finish_reason=result.finish_reason)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)

    def model_id(self) -> str:
        return "scripted/test"


def _loop(provider: Provider, tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=6,
    )


SILENT_254 = LLMResult(text="", usage=Usage(completion_tokens=254, total_tokens=254))
SILENT_0 = LLMResult(text="", usage=Usage())
ANSWER = LLMResult(text="all good", usage=Usage(completion_tokens=3, total_tokens=3))


# ── the silence says what it cost ─────────────────────────────────────────────────


def test_generated_but_undelivered_tokens_are_named(tmp_path: Path) -> None:
    loop = _loop(_Replay([SILENT_254, ANSWER]), tmp_path)
    result = asyncio.run(loop.arun_turn("hello"))
    assert result.stop_reason == "completed"
    [note] = [e for e in result.trace.events if e.data.get("kind") == "empty_completion"]
    assert note.data["completion_tokens"] == 254
    assert "254 tokens generated, none delivered" in note.detail


def test_a_true_zero_token_silence_stays_plain(tmp_path: Path) -> None:
    loop = _loop(_Replay([SILENT_0, ANSWER]), tmp_path)
    result = asyncio.run(loop.arun_turn("hello"))
    [note] = [e for e in result.trace.events if e.data.get("kind") == "empty_completion"]
    assert note.data["completion_tokens"] == 0
    assert "generated" not in note.detail


def test_streaming_status_carries_the_count(tmp_path: Path) -> None:
    loop = _loop(_Replay([SILENT_254, ANSWER]), tmp_path)

    async def run() -> list[AgentEvent]:
        return [ev async for ev in loop.astream_turn("hello")]

    events = asyncio.run(run())
    statuses = [ev.message for ev in events if isinstance(ev, AgentStatus)]
    assert (
        "model went silent (254 tokens generated, none delivered); asking for a real answer"
        in statuses
    )


# ── the typed skill counts as loaded ──────────────────────────────────────────────


def _write_skill(workspace: Path, name: str, body: str) -> None:
    d = workspace / ".zakcode" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} skill.\n---\n{body}\n", encoding="utf-8"
    )


def _use(name: str, call_id: str) -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id=call_id, name="use_skill", arguments={"name": name})],
        usage=Usage(total_tokens=1),
    )


def _tool_outputs(agent: Agent) -> list[str]:
    return [
        block.output
        for message in agent.session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock)
    ]


async def test_reinvoking_the_typed_skill_gets_the_pointer(tmp_path: Path) -> None:
    _write_skill(tmp_path, "start", body="Bring the agent up, step by step.")
    agent = Agent(
        settings=Settings(default_model="scripted/test", workspace_root=tmp_path),
        enable_skills=True,
        provider=_Replay([_use("start", "t1"), LLMResult(text="up", usage=Usage(total_tokens=1))]),
    )
    invocation = await agent.compose_skill_turn("start")
    assert invocation.invoked and invocation.turn_text
    result = await agent.arun_turn(invocation.turn_text)
    assert result.stop_reason == "completed"
    [output] = _tool_outputs(agent)
    assert "[already loaded]" in output
    assert "bring the agent up" not in output.lower()  # the body did not land twice
    assert agent._skill_invocations_this_turn == 0  # the pointer costs no budget


async def test_a_different_skill_still_loads_in_full(tmp_path: Path) -> None:
    _write_skill(tmp_path, "start", body="Bring the agent up.")
    _write_skill(tmp_path, "prime", body="Prime the context.")
    agent = Agent(
        settings=Settings(default_model="scripted/test", workspace_root=tmp_path),
        enable_skills=True,
        provider=_Replay([_use("prime", "t1"), LLMResult(text="up", usage=Usage(total_tokens=1))]),
    )
    invocation = await agent.compose_skill_turn("start")
    await agent.arun_turn(invocation.turn_text)
    [output] = _tool_outputs(agent)
    assert "prime the context" in output.lower()


class _VetoOnce:
    def __init__(self, prompt: str) -> None:
        self._prompt: str | None = prompt

    def __call__(self, payload: TurnEndPayload) -> TurnEndResult | None:
        if self._prompt is None:
            return None
        prompt, self._prompt = self._prompt, None
        return TurnEndResult(vetoed=True, continuation_prompt=prompt)


async def test_a_veto_still_opens_a_fresh_skill_turn(tmp_path: Path) -> None:
    # ADR-0048 is untouched: after a TURN_END veto the typed skill's body comes back in full.
    _write_skill(tmp_path, "start", body="Bring the agent up, step by step.")
    agent = Agent(
        settings=Settings(default_model="scripted/test", workspace_root=tmp_path),
        enable_skills=True,
        provider=_Replay(
            [
                LLMResult(text="done", usage=Usage(total_tokens=1)),
                _use("start", "t1"),
                LLMResult(text="done again", usage=Usage(total_tokens=1)),
            ]
        ),
    )
    agent.hook_manager.register_turn_end(_VetoOnce("Not done: load start again and finish."))
    invocation = await agent.compose_skill_turn("start")
    result = await agent.arun_turn(invocation.turn_text)
    assert result.stop_reason == "completed"
    [output] = _tool_outputs(agent)
    assert "bring the agent up" in output.lower()
    assert "[already loaded]" not in output

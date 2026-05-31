"""The public Agent facade must expose the same turn surface the clients call.

Regression guard: the CLI drives ``agent.astream_turn(...)`` directly, so the
facade must delegate it to the loop (a prior integration left it off the facade,
which only the mocked CLI test failed to catch). These tests use the real Agent
with a scripted provider — no network.
"""

from __future__ import annotations

import inspect

import zakcode
from zakcode.agent.loop import AgentLoop
from zakcode.events import AgentDone
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    StreamDone,
    StreamTextDelta,
    StreamUsage,
)
from zakcode.usage import Usage


class _ScriptedStream(Provider):
    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        return LLMResult(text="hi")

    async def astream(self, messages, *, system=None, tools=None, **kw):
        yield StreamTextDelta(text="hello there")
        yield StreamUsage(usage=Usage(total_tokens=4))
        yield StreamDone(finish_reason="stop")

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


def test_agent_exposes_the_full_turn_surface() -> None:
    for name in ("run_turn", "arun_turn", "astream_turn"):
        assert hasattr(zakcode.Agent, name), f"Agent facade is missing {name}"


def test_agent_astream_turn_is_not_a_coroutine_function() -> None:
    # The CLI does ``async for ev in agent.astream_turn(text)`` without awaiting
    # the call first, so it must return an async iterator, not a coroutine.
    assert not inspect.iscoroutinefunction(zakcode.Agent.astream_turn)


async def test_agent_astream_turn_delegates_to_loop(tmp_path) -> None:
    agent = zakcode.Agent(workspace_root=tmp_path)
    agent.provider = _ScriptedStream()
    agent.loop = AgentLoop(agent.provider, agent.registry, agent.session, workspace_root=tmp_path)

    events = [ev async for ev in agent.astream_turn("hi")]
    assert events, "expected a non-empty event stream"
    assert events[-1].event == "done"
    done = events[-1]
    assert isinstance(done, AgentDone)
    assert done.stop_reason == "completed"
    text = "".join(e.text for e in events if e.event == "text")
    assert "hello there" in text

"""Tests that the loop injects its SubAgentSpawner into every ToolContext (M4-4a).

This is the plumbing the ``task`` tool depends on: a tool sees ``ctx.spawner`` iff
the loop was constructed with one. Sub-agent child loops are built with
``spawner=None`` (one-level nesting), which this also pins down.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from zakcode.agent.loop import AgentLoop
from zakcode.agent.subagent import SubAgentResult
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamToolCallDelta,
    ToolCall,
)
from zakcode.session.store import Session
from zakcode.tools.base import (
    SubAgentSpawner,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.usage import Usage


class _FakeSpawner:
    """A structural SubAgentSpawner (no real agents)."""

    async def spawn(self, *, type_name: str, prompt: str) -> SubAgentResult:
        return SubAgentResult(name=type_name, summary=f"ran {type_name}: {prompt}")

    def available_types(self) -> list[str]:
        return ["general-purpose"]

    def default_type(self) -> str:
        return "general-purpose"


class _CaptureTool(Tool):
    """Records the ``spawner`` it saw on the context at execution time."""

    spec = ToolSpec(name="capture", description="captures the tool context")

    def __init__(self) -> None:
        self.seen_spawner: object = "unset"

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.seen_spawner = ctx.spawner
        return ToolResult.ok("captured")


class _CallThenDoneProvider(Provider):
    """Requests ``tool_name`` on the first turn, then completes."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self._calls = 0

    async def acomplete(
        self, messages: list, *, tools: list | None = None, system: str | None = None, **kw
    ) -> LLMResult:
        self._calls += 1
        if self._calls == 1:
            return LLMResult(
                tool_calls=[ToolCall(id="c1", name=self.tool_name, arguments={})],
                usage=Usage(total_tokens=1),
            )
        return LLMResult(text="done", usage=Usage(total_tokens=1))

    async def astream(
        self, messages: list, *, tools: list | None = None, system: str | None = None, **kw
    ) -> AsyncIterator[ProviderStreamEvent]:
        result = await self.acomplete(messages, tools=tools, system=system)
        for i, call in enumerate(result.tool_calls):
            yield StreamToolCallDelta(
                index=i, id=call.id, name=call.name, arguments_delta=json.dumps(call.arguments)
            )
        yield StreamDone()

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


def _loop(tmp_path: Path, tool: _CaptureTool, spawner: SubAgentSpawner | None) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(tool)
    return AgentLoop(
        _CallThenDoneProvider("capture"),
        registry,
        Session(cwd=str(tmp_path), model="scripted/test"),
        workspace_root=tmp_path,
        spawner=spawner,
    )


async def test_loop_injects_spawner_into_tool_context(tmp_path: Path) -> None:
    tool = _CaptureTool()
    spawner = _FakeSpawner()
    await _loop(tmp_path, tool, spawner).arun_turn("go")
    assert tool.seen_spawner is spawner


async def test_loop_without_spawner_leaves_ctx_spawner_none(tmp_path: Path) -> None:
    tool = _CaptureTool()
    await _loop(tmp_path, tool, None).arun_turn("go")
    assert tool.seen_spawner is None


async def test_streaming_path_also_injects_spawner(tmp_path: Path) -> None:
    tool = _CaptureTool()
    spawner = _FakeSpawner()
    loop = _loop(tmp_path, tool, spawner)
    _ = [e async for e in loop.astream_turn("go")]
    assert tool.seen_spawner is spawner


def test_runtime_checkable_spawner_isinstance() -> None:
    # The Protocol is runtime_checkable so pydantic can validate the field.
    assert isinstance(_FakeSpawner(), SubAgentSpawner)

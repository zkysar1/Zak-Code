"""Hermetic tests for the ReAct agent loop. No network, no live models.

Coroutines are driven via ``asyncio.run`` so the suite does not depend on any
async pytest plugin being active.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop, TurnResult
from zakcode.config import PermissionTier, load_settings
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session, SessionStore
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.usage import Usage


class ScriptedProvider(Provider):
    """Returns canned LLMResults in order; the last one repeats once exhausted."""

    def __init__(self, script: Sequence[LLMResult]) -> None:
        self._script = list(script)
        self.calls = 0
        self.systems: list[str | None] = []

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> LLMResult:
        self.systems.append(system)
        idx = min(self.calls, len(self._script) - 1)
        self.calls += 1
        return self._script[idx]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


class FakeTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo back the provided text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls.append(args)
        return ToolResult.ok(f"echo: {args.get('text', '')}", data={"n": len(self.calls)})


def _registry_with(tool: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(tool)
    return reg


def _usage(p: int = 1, c: int = 1) -> Usage:
    return Usage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c, cost_usd=0.0)


def _session() -> Session:
    return Session(cwd="/tmp/work", model="ollama_chat/llama3.1")


def _make_loop(provider: Provider, registry: ToolRegistry, **kw: Any) -> AgentLoop:
    settings = load_settings(workspace_root=Path.cwd())
    return AgentLoop(provider, registry, _session(), settings=settings, **kw)


def test_no_tool_calls_completes() -> None:
    provider = ScriptedProvider([LLMResult(text="all done", finish_reason="stop", usage=_usage())])
    loop = _make_loop(provider, ToolRegistry())

    result = asyncio.run(loop.arun_turn("hello"))

    assert result.stop_reason == "completed"
    assert result.iterations == 1
    assert len(result.assistant_messages) == 1
    assert result.assistant_messages[0].text == "all done"
    assert [m.role for m in loop.session.messages] == ["user", "assistant"]


def test_tool_call_then_completes() -> None:
    tool = FakeTool()
    provider = ScriptedProvider(
        [
            LLMResult(
                text="",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
                finish_reason="tool_calls",
                usage=_usage(),
            ),
            LLMResult(text="finished", finish_reason="stop", usage=_usage()),
        ]
    )
    loop = _make_loop(provider, _registry_with(tool))

    result = asyncio.run(loop.arun_turn("please echo"))

    assert result.stop_reason == "completed"
    assert result.iterations == 2
    assert tool.calls == [{"text": "hi"}]
    assert len(result.tool_results) == 1
    assert result.tool_results[0].output == "echo: hi"
    assert result.tool_results[0].tool_use_id == "c1"
    roles = [m.role for m in loop.session.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_msg = loop.session.messages[2]
    block = tool_msg.blocks[0]
    assert isinstance(block, ToolResultBlock)
    assert block.output == "echo: hi"


def test_runaway_hits_max_iterations() -> None:
    tool = FakeTool()
    provider = ScriptedProvider(
        [
            LLMResult(
                tool_calls=[ToolCall(id="c", name="echo", arguments={"text": "x"})],
                finish_reason="tool_calls",
                usage=_usage(),
            )
        ]
    )
    loop = _make_loop(provider, _registry_with(tool), max_iterations=3)

    result = asyncio.run(loop.arun_turn("loop forever"))

    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3
    assert provider.calls == 3
    assert len(tool.calls) == 3


def test_usage_accumulates() -> None:
    provider = ScriptedProvider(
        [
            LLMResult(
                tool_calls=[ToolCall(id="c", name="echo", arguments={"text": "x"})],
                usage=_usage(2, 3),
            ),
            LLMResult(text="done", usage=_usage(5, 7)),
        ]
    )
    loop = _make_loop(provider, _registry_with(FakeTool()))

    result = asyncio.run(loop.arun_turn("go"))

    assert result.iterations == 2
    assert result.usage.prompt_tokens == 7
    assert result.usage.completion_tokens == 10
    assert result.usage.total_tokens == 17
    # usage also lands on the session for cumulative reporting
    assert loop.session.cumulative_usage().total_tokens == 17


def test_store_persists_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    provider = ScriptedProvider([LLMResult(text="ok", usage=_usage())])
    session = _session()
    settings = load_settings(workspace_root=tmp_path)
    loop = AgentLoop(provider, ToolRegistry(), session, settings=settings, store=store)

    asyncio.run(loop.arun_turn("persist me"))

    assert session.id in store.list()
    reloaded = store.load(session.id)
    assert [m.role for m in reloaded.messages] == ["user", "assistant"]
    assert reloaded.messages[0].text == "persist me"


def test_system_prompt_is_built_each_iteration() -> None:
    provider = ScriptedProvider([LLMResult(text="hi", usage=_usage())])
    loop = _make_loop(provider, _registry_with(FakeTool()))

    asyncio.run(loop.arun_turn("hello"))

    assert provider.systems and provider.systems[0]
    # the registered tool is surfaced in the system prompt
    assert "echo" in provider.systems[0]


def test_run_turn_sync_wrapper() -> None:
    provider = ScriptedProvider([LLMResult(text="sync ok", usage=_usage())])
    loop = _make_loop(provider, ToolRegistry())

    result = loop.run_turn("hello")

    assert isinstance(result, TurnResult)
    assert result.stop_reason == "completed"
    assert result.assistant_messages[0].text == "sync ok"


def test_run_turn_raises_inside_running_loop() -> None:
    provider = ScriptedProvider([LLMResult(text="x", usage=_usage())])
    loop = _make_loop(provider, ToolRegistry())

    async def _inner() -> None:
        with pytest.raises(RuntimeError, match="running event loop"):
            loop.run_turn("nope")

    asyncio.run(_inner())

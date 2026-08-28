"""End-to-end delegation tests (M4-4c-3): a real parent turn drives the `task`
tool through the loop to a real child loop, and one-level nesting holds.

These close the integration gap the M4 fresh-eyes review flagged (MINOR-1/2):
every other delegation test uses a fake spawner; here the wiring is exercised for
real — `Agent(enable_subagents=True)` → parent `AgentLoop` → `task` tool →
`SubAgentManager` → `SubAgentRunner` → child `AgentLoop` → summary back into the
parent — with only the *provider* scripted.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import zakcode.providers.litellm_provider as llm_mod
from zakcode import Agent
from zakcode.agent.budget import IterationBudget
from zakcode.agent.subagent import GENERAL_PURPOSE, SubAgentRunner
from zakcode.config import Settings
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ProviderStreamEvent, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.usage import Usage


def _first_user_text(messages: list[Message]) -> str:
    for m in messages:
        if m.role == "user":
            return m.text
    return ""


def _has_tool_result(messages: list[Message]) -> bool:
    return any(m.role == "tool" for m in messages)


class _DelegatingProvider(Provider):
    """One provider serving BOTH the parent and the child, keyed on the turn's
    first user message (the prompt each loop was started with)."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def acomplete(
        self, messages: list[Message], *, tools: list | None = None, system: str | None = None
    ) -> LLMResult:
        first = _first_user_text(messages)
        if first == "DELEGATE":  # the parent turn
            if _has_tool_result(messages):
                return LLMResult(text="PARENT-SYNTHESIS", usage=Usage(total_tokens=1))
            return LLMResult(
                tool_calls=[
                    ToolCall(
                        id="task1",
                        name="task",
                        arguments={"tasks": [{"prompt": "SUBTASK"}]},
                    )
                ],
                usage=Usage(total_tokens=1),
            )
        # the child turn (prompt == "SUBTASK")
        return LLMResult(text="CHILD-RESULT", usage=Usage(total_tokens=1))

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


async def test_parent_delegates_to_child_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_mod, "LiteLLMProvider", _DelegatingProvider)
    budget = IterationBudget(50)
    agent = Agent(
        settings=Settings(
            default_model="scripted/test",
            context_window=8192,
            workspace_root=tmp_path,
            permission_mode="allow",
        ),
        enable_subagents=True,
        budget=budget,
    )
    result = await agent.arun_turn("DELEGATE")

    # The parent finished by synthesizing after the child returned.
    assert result.stop_reason == "completed"
    assert any(m.text == "PARENT-SYNTHESIS" for m in result.assistant_messages)
    # The child's summary came back through the task tool result.
    assert any("CHILD-RESULT" in tr.output for tr in result.tool_results)
    # Exactly one child was spawned, against the shared budget.
    assert budget.children_spawned == 1
    assert budget.consumed >= 2  # at least one parent + one child iteration


def test_enable_subagents_exposes_task_tool(tmp_path: Path) -> None:
    settings = Settings(default_model="scripted/test", context_window=8192, workspace_root=tmp_path)
    on = Agent(settings=settings, enable_subagents=True)
    off = Agent(settings=settings)
    assert "task" in on.registry.names()
    assert "task" not in off.registry.names()
    # Enabling delegation places a spawner on the loop; the default does not.
    assert on.loop.spawner is not None
    assert off.loop.spawner is None


# ── one-level nesting: a child loop is built WITHOUT a spawner ────────────────────


class _CaptureSpawnerTool(Tool):
    """Records the spawner present on the tool context at execution time."""

    spec = ToolSpec(name="capture", description="captures ctx.spawner")

    def __init__(self) -> None:
        self.seen: object = "unset"

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.seen = ctx.spawner
        return ToolResult.ok("captured")


class _CallCaptureProvider(Provider):
    """Calls ``capture`` once, then completes."""

    def __init__(self) -> None:
        self._calls = 0

    async def acomplete(
        self, messages: list[Message], *, tools: list | None = None, system: str | None = None
    ) -> LLMResult:
        self._calls += 1
        if self._calls == 1:
            return LLMResult(
                tool_calls=[ToolCall(id="c1", name="capture", arguments={})],
                usage=Usage(total_tokens=1),
            )
        return LLMResult(text="child done", usage=Usage(total_tokens=1))

    async def astream(
        self, messages: list[Message], *, tools: list | None = None, system: str | None = None
    ) -> AsyncIterator[ProviderStreamEvent]:  # pragma: no cover - unused
        result = await self.acomplete(messages, tools=tools, system=system)
        from zakcode.providers.base import StreamDone

        yield StreamDone(finish_reason=result.finish_reason)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


async def test_subagent_child_tools_see_no_spawner(tmp_path: Path) -> None:
    # Even though we hand the runner a registry, the child loop it builds carries no
    # spawner — so a sub-agent structurally cannot delegate (one-level nesting).
    capture = _CaptureSpawnerTool()
    registry = ToolRegistry()
    registry.register(capture)
    runner = SubAgentRunner(
        provider=_CallCaptureProvider(),
        registry=registry,
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        budget=IterationBudget(10),
        workspace_root=tmp_path,
    )
    await runner.run(GENERAL_PURPOSE, "go")
    assert capture.seen is None


def test_facade_child_registry_excludes_task(tmp_path: Path) -> None:
    # The facade gives children a task-free registry, a second layer of nesting
    # protection on top of the absent spawner.
    agent = Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        enable_subagents=True,
    )
    manager = agent.loop.spawner
    assert manager is not None
    child_registry = manager._runner.child_registry(GENERAL_PURPOSE)  # type: ignore[attr-defined]
    assert "task" not in child_registry.names()
    session = Session(cwd=str(tmp_path), model="scripted/test")
    # And the child loop the runner builds has no spawner.
    assert agent.loop.spawner is not None and session is not None

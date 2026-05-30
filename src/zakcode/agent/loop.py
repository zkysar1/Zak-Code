"""The ReAct agent loop.

:meth:`AgentLoop.arun_turn` drives one user turn: it repeatedly asks the provider
for a completion, executes any requested tools sequentially, feeds the results
back, and stops when the model emits no further tool calls (or the iteration
budget is exhausted).

The loop is provider- and tool-agnostic: it speaks only the frozen contracts in
:mod:`zakcode.messages`, :mod:`zakcode.providers.base`, and
:mod:`zakcode.tools.base`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.config import Settings, load_settings
from zakcode.messages import ContentBlock, Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import Provider
from zakcode.session.store import Session, SessionStore
from zakcode.tools.base import ToolContext, ToolRegistry, ToolSpec
from zakcode.usage import Usage

#: Fallback iteration budget when neither an explicit value nor settings provide one.
DEFAULT_MAX_ITERATIONS = 50


class TurnResult(BaseModel):
    """Outcome of a single :meth:`AgentLoop.arun_turn` call."""

    assistant_messages: list[Message] = Field(default_factory=list)
    tool_results: list[ToolResultBlock] = Field(default_factory=list)
    iterations: int = 0
    usage: Usage = Field(default_factory=Usage)
    stop_reason: str = "completed"


class AgentLoop:
    """Stateful driver that advances a :class:`Session` one user turn at a time."""

    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        session: Session,
        *,
        prompt_builder: SystemPromptBuilder | None = None,
        settings: Settings | None = None,
        store: SessionStore | None = None,
        max_iterations: int | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.session = session
        self.prompt_builder = prompt_builder or SystemPromptBuilder()
        self.settings = settings or load_settings()
        self.store = store
        self.workspace_root = workspace_root or self.settings.workspace_root
        if max_iterations is not None:
            self.max_iterations = max_iterations
        else:
            self.max_iterations = self.settings.max_iterations or DEFAULT_MAX_ITERATIONS

    # ── internals ────────────────────────────────────────────────────────────

    def _persist(self) -> None:
        if self.store is not None:
            self.store.save(self.session)

    def _tool_specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for name in self.registry.names():
            tool = self.registry.get(name)
            if tool is not None:
                specs.append(tool.spec)
        return specs

    def _build_system(self) -> str:
        return self.prompt_builder.build(self.settings, tools=self._tool_specs())

    # ── public API ───────────────────────────────────────────────────────────

    async def arun_turn(self, user_text: str) -> TurnResult:
        """Run one user turn to completion (or until the iteration budget runs out)."""
        self.session.add_message(Message.user(user_text))
        self._persist()

        turn_assistant: list[Message] = []
        turn_tool_results: list[ToolResultBlock] = []
        turn_usage = Usage()
        iterations = 0
        stop_reason = "max_iterations"

        tool_defs = self.registry.definitions()
        ctx = ToolContext(workspace_root=self.workspace_root)

        while iterations < self.max_iterations:
            iterations += 1
            system = self._build_system()

            result = await self.provider.acomplete(
                self.session.messages,
                system=system,
                tools=tool_defs or None,
            )

            blocks: list[ContentBlock] = []
            if result.text:
                blocks.append(TextBlock(text=result.text))
            for call in result.tool_calls:
                blocks.append(ToolUseBlock(id=call.id, name=call.name, input=call.arguments))

            assistant_msg = Message(role="assistant", blocks=blocks)
            self.session.add_message(assistant_msg)
            self.session.add_usage(result.usage)
            turn_assistant.append(assistant_msg)
            turn_usage = turn_usage + result.usage
            self._persist()

            if not result.has_tool_calls:
                stop_reason = "completed"
                break

            result_blocks: list[ToolResultBlock] = []
            for call in result.tool_calls:
                tool_res = await self.registry.execute(call.name, call.arguments, ctx)
                block = ToolResultBlock(
                    tool_use_id=call.id,
                    output=tool_res.output,
                    is_error=tool_res.is_error,
                    data=tool_res.data,
                )
                result_blocks.append(block)
                turn_tool_results.append(block)

            self.session.add_message(Message.tool_results(result_blocks))
            self._persist()

        return TurnResult(
            assistant_messages=turn_assistant,
            tool_results=turn_tool_results,
            iterations=iterations,
            usage=turn_usage,
            stop_reason=stop_reason,
        )

    def run_turn(self, user_text: str) -> TurnResult:
        """Synchronous wrapper around :meth:`arun_turn`.

        Refuses to run if an event loop is already active in this thread, since
        ``asyncio.run`` would raise; call ``arun_turn`` directly from async code.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun_turn(user_text))
        raise RuntimeError(
            "run_turn() cannot be called from a running event loop; await arun_turn() instead."
        )

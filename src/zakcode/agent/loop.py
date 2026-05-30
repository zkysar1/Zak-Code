"""The ReAct agent loop.

:meth:`AgentLoop.arun_turn` drives one user turn: it repeatedly asks the provider
for a completion, executes any requested tools sequentially, feeds the results
back, and stops when the model emits no further tool calls (or a stop condition
fires — the iteration budget, a doom loop, or cancellation).

The loop is provider- and tool-agnostic: it speaks only the frozen contracts in
:mod:`zakcode.messages`, :mod:`zakcode.providers.base`, and
:mod:`zakcode.tools.base`.

Stop conditions
---------------
``TurnResult.stop_reason`` is one of:

* ``"completed"`` — the model emitted no tool calls (the normal end of a turn).
  Also used when the model returns neither text nor tool calls (an empty
  completion): the turn ends cleanly with a (possibly empty) assistant message.
* ``"max_iterations"`` — the per-turn iteration budget was exhausted. This is the
  hard outer bound and takes precedence over the doom-loop guard when the two
  would fire on the same (final) iteration.
* ``"doom_loop"`` — the model requested the *same* tool with *identical*
  arguments :data:`DOOM_LOOP_THRESHOLD` times in a row, so the loop stops early
  rather than burning the whole iteration budget on a no-progress cycle. Fires
  only while iteration budget remains to be saved.

Cancellation (``asyncio.CancelledError``) is never treated as a normal stop: it
propagates out of the turn after the session has been persisted in a consistent
state, so a cancelled turn never leaves a half-written/corrupt session.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.config import Settings, load_settings
from zakcode.messages import ContentBlock, Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import LLMResult, Provider, ToolCall
from zakcode.session.store import Session, SessionStore
from zakcode.tools.base import ToolContext, ToolRegistry, ToolSpec
from zakcode.usage import Usage

#: Fallback iteration budget when neither an explicit value nor settings provide one.
DEFAULT_MAX_ITERATIONS = 50

#: How many consecutive iterations may request the *same* tool with *identical*
#: arguments before the loop gives up with ``stop_reason="doom_loop"``. The model
#: repeating the exact same call is making no progress, so we stop early rather
#: than spend the whole iteration budget on it.
DOOM_LOOP_THRESHOLD = 3


class TurnResult(BaseModel):
    """Outcome of a single :meth:`AgentLoop.arun_turn` call."""

    assistant_messages: list[Message] = Field(default_factory=list)
    tool_results: list[ToolResultBlock] = Field(default_factory=list)
    iterations: int = 0
    usage: Usage = Field(default_factory=Usage)
    stop_reason: str = "completed"


def _call_signature(call: ToolCall) -> tuple[str, str]:
    """A stable, hashable identity for a tool call (name + canonical arguments).

    Arguments are serialized with sorted keys so two logically-identical calls
    compare equal regardless of dict ordering. Falls back to ``repr`` for the
    (vanishingly rare) non-JSON-serializable argument value.
    """
    try:
        args = json.dumps(call.arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args = repr(sorted(call.arguments.items()))
    return (call.name, args)


def _batch_signature(calls: list[ToolCall]) -> tuple[tuple[str, str], ...]:
    """Signature for a whole batch of tool calls requested in one iteration."""
    return tuple(_call_signature(c) for c in calls)


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

    @staticmethod
    def _assistant_message(result: LLMResult) -> Message:
        """Build the assistant message for one completion.

        A completion with neither text nor tool calls yields an assistant message
        with no blocks (rather than a crash): the turn still ends cleanly.
        """
        blocks: list[ContentBlock] = []
        if result.text:
            blocks.append(TextBlock(text=result.text))
        for call in result.tool_calls:
            blocks.append(ToolUseBlock(id=call.id, name=call.name, input=call.arguments))
        return Message(role="assistant", blocks=blocks)

    # ── public API ───────────────────────────────────────────────────────────

    async def arun_turn(self, user_text: str) -> TurnResult:
        """Run one user turn to completion (or until a stop condition fires).

        Stop conditions are documented on this module. ``asyncio.CancelledError``
        is re-raised (never reported as a normal stop) after the session is left
        in a consistent, persisted state.
        """
        try:
            return await self._run_turn(user_text)
        except asyncio.CancelledError:
            # Cancellation is a control signal, not a stop reason. The session has
            # only ever been mutated + persisted at message boundaries (see
            # _run_turn), so on-disk state is consistent here. Best-effort persist
            # once more, swallowing a save error so the original CancelledError is
            # what propagates, then re-raise.
            with contextlib.suppress(Exception):
                self._persist()
            raise

    async def _run_turn(self, user_text: str) -> TurnResult:
        self.session.add_message(Message.user(user_text))
        self._persist()

        turn_assistant: list[Message] = []
        turn_tool_results: list[ToolResultBlock] = []
        turn_usage = Usage()
        iterations = 0
        stop_reason = "max_iterations"

        # Doom-loop tracking: the signature of the previous iteration's tool-call
        # batch and how many times in a row we have now seen it.
        last_signature: tuple[tuple[str, str], ...] | None = None
        repeat_count = 0

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

            assistant_msg = self._assistant_message(result)
            self.session.add_message(assistant_msg)
            self.session.add_usage(result.usage)
            turn_assistant.append(assistant_msg)
            turn_usage = turn_usage + result.usage
            self._persist()

            # An empty completion (no text, no tool calls) ends the turn cleanly.
            if not result.has_tool_calls:
                stop_reason = "completed"
                break

            # Doom-loop guard: if this iteration's tool-call batch is byte-for-byte
            # identical to the previous one, count the repeat. Once it hits the
            # threshold we stop early with "doom_loop" — but only while there is
            # still iteration budget left to save. If the threshold coincides with
            # the final allowed iteration, the loop would have stopped anyway, so
            # "max_iterations" stays the accurate (and outer-bound) stop reason.
            signature = _batch_signature(result.tool_calls)
            if signature == last_signature:
                repeat_count += 1
            else:
                repeat_count = 1
                last_signature = signature
            if repeat_count >= DOOM_LOOP_THRESHOLD and iterations < self.max_iterations:
                stop_reason = "doom_loop"
                break

            result_blocks: list[ToolResultBlock] = []
            for call in result.tool_calls:
                # A tool that returns is_error does NOT abort the turn: the error
                # result is fed back so the model can see it and recover.
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

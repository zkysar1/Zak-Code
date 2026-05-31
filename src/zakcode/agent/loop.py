"""The ReAct agent loop.

:meth:`AgentLoop.arun_turn` drives one user turn: it repeatedly asks the provider
for a completion, executes any requested tools sequentially, feeds the results
back, and stops when the model emits no further tool calls (or a stop condition
fires — the iteration budget, a doom loop, or cancellation).

:meth:`AgentLoop.astream_turn` is the *incremental* view of the same turn: it
consumes the provider's token stream (:meth:`Provider.astream`) instead of one
buffered completion and yields a sequence of :data:`~zakcode.events.AgentEvent`
(live text deltas, tool calls, tool results, a cumulative usage event, and a
terminal ``AgentDone``). It reuses every other piece of the buffered path — the
system-prompt build, sequential tool execution, the doom-loop guard, the
iteration budget, persistence, and the cancellation contract — so its
stop-reason / iteration semantics match :meth:`arun_turn` exactly.

The loop is provider- and tool-agnostic: it speaks only the frozen contracts in
:mod:`zakcode.messages`, :mod:`zakcode.providers.base`,
:mod:`zakcode.tools.base`, and the client-facing :mod:`zakcode.events`.

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
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import BaseModel, Field

from zakcode.agent._stream import ToolCallAccumulator
from zakcode.agent.budget import IterationBudget
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.config import Settings, load_settings
from zakcode.events import (
    AgentDone,
    AgentEvent,
    AgentStatus,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.hooks import HookEvent, HookManager, HookPayload
from zakcode.messages import ContentBlock, Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.permissions import PermissionPolicy
from zakcode.providers.base import (
    LLMResult,
    Provider,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsage,
    ToolCall,
)
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
        permission_policy: PermissionPolicy | None = None,
        hook_manager: HookManager | None = None,
        budget: IterationBudget | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.session = session
        self.prompt_builder = prompt_builder or SystemPromptBuilder()
        self.settings = settings or load_settings()
        self.store = store
        self.workspace_root = workspace_root or self.settings.workspace_root
        # Optional shared iteration budget (M4). When injected, it is an ADDITIONAL
        # bound on top of the per-turn ``max_iterations`` cap: each iteration draws
        # one unit from the shared pool, and the turn stops with
        # ``stop_reason="max_iterations"`` when the pool is empty. A parent and its
        # sub-agents share one budget instance so the whole delegation tree's
        # iteration count is bounded by a single pool. ``None`` ⇒ unchanged
        # behavior (the local cap is the only bound).
        self.budget = budget
        # The security gate is INJECTED, not assumed. A bare AgentLoop with no
        # policy is ungated (a pure mechanism, convenient for library/tests); the
        # Agent facade — the real entry point — always injects a policy built from
        # settings.permission_mode (deny-first). ``hook_manager`` defaults to an
        # empty (no-op) manager so the hook calls are always safe to make.
        self.permission_policy = permission_policy
        self.hook_manager = hook_manager or HookManager()
        if max_iterations is not None:
            self.max_iterations = max_iterations
        else:
            self.max_iterations = self.settings.max_iterations or DEFAULT_MAX_ITERATIONS

    # ── internals ────────────────────────────────────────────────────────────

    def _persist(self) -> None:
        if self.store is not None:
            self.store.save(self.session)

    def _grant_iteration(self, iterations_done: int) -> bool:
        """Whether the loop may run another iteration (and reserve it if so).

        Two independent bounds, both of which must allow the iteration:

        1. The per-turn ``max_iterations`` cap (always applies).
        2. The shared :class:`IterationBudget`, if one was injected — one unit is
           consumed from the shared pool here, so a parent and its sub-agents
           cannot collectively exceed it. When the pool is empty this returns
           ``False`` without consuming anything.

        Returning ``False`` is the loop's signal to stop with
        ``stop_reason="max_iterations"``.
        """
        if iterations_done >= self.max_iterations:
            return False
        if self.budget is not None:
            return self.budget.try_consume(1)
        return True

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

    async def _execute_tool_call(self, call: ToolCall, ctx: ToolContext) -> ToolResultBlock:
        """Run one tool call through the full gate, returning its result block.

        The single seam both the buffered (:meth:`_run_turn`) and streaming
        (:meth:`astream_turn`) paths funnel through, so they gate identically. The
        stages, in order:

        1. **Permission** — :meth:`PermissionPolicy.authorize` (deny-first, decided
           here where the model cannot reach it). Only runs if a policy was injected.
        2. **PreToolUse hooks** — may veto the call or rewrite its arguments.
        3. **Execute** — :meth:`ToolRegistry.execute` (which itself never raises).
        4. **PostToolUse hooks** — observe-only; any note is appended as feedback.

        A permission denial or hook veto is returned as an *error*
        :class:`ToolResultBlock` (never an exception), so the turn continues and the
        model sees the feedback and can adapt.
        """
        tool = self.registry.get(call.name)
        spec = tool.spec if tool is not None else None
        cwd = str(self.workspace_root)

        # 1. Permission gate (only when a policy is injected; see __init__).
        if self.permission_policy is not None:
            allowed, reason = await self.permission_policy.authorize(spec, call.arguments)
            if not allowed:
                return ToolResultBlock(
                    tool_use_id=call.id,
                    output=f"Permission denied for {call.name!r}: {reason}",
                    is_error=True,
                    data={"permission_denied": True, "reason": reason},
                )

        arguments = call.arguments

        # 2. PreToolUse hooks (veto or argument rewrite).
        pre = await self.hook_manager.run(
            HookPayload(
                event=HookEvent.PRE_TOOL_USE,
                tool_name=call.name,
                arguments=arguments,
                cwd=cwd,
            )
        )
        if pre.blocked:
            return ToolResultBlock(
                tool_use_id=call.id,
                output=f"Blocked by hook for {call.name!r}: {pre.message}",
                is_error=True,
                data={"hook_blocked": True, "reason": pre.message},
            )
        if pre.mutated_arguments is not None:
            arguments = pre.mutated_arguments

        # 3. Execute (registry.execute wraps any failure into an error ToolResult).
        tool_res = await self.registry.execute(call.name, arguments, ctx)

        # 4. PostToolUse hooks (observe-only; their notes are appended as feedback).
        post = await self.hook_manager.run(
            HookPayload(
                event=HookEvent.POST_TOOL_USE,
                tool_name=call.name,
                arguments=arguments,
                cwd=cwd,
                output=tool_res.output,
                is_error=tool_res.is_error,
            )
        )
        output = tool_res.output
        if post.message:
            output = f"{output}\n[hook] {post.message}" if output else f"[hook] {post.message}"

        return ToolResultBlock(
            tool_use_id=call.id,
            output=output,
            is_error=tool_res.is_error,
            data=tool_res.data,
        )

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
                # Each call runs through the permission + hook gate. A denial,
                # veto, or tool error becomes an error result that is fed back so
                # the model can recover — it never aborts the turn.
                block = await self._execute_tool_call(call, ctx)
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

    # ── streaming API ──────────────────────────────────────────────────────────

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Run one user turn, yielding :data:`AgentEvent`s as the turn unfolds.

        This is the incremental twin of :meth:`arun_turn`: it consumes
        :meth:`Provider.astream` and re-emits a client-facing event stream while
        driving the exact same cycle (system prompt, sequential tool execution,
        doom-loop guard, iteration budget, persistence). Stop-reason and iteration
        semantics match the buffered path.

        Event order per turn:

        * a live ``AgentTextDelta`` for each streamed text chunk;
        * one ``AgentToolCall`` then one ``AgentToolResult`` for each tool the
          model requested, in order;
        * possibly an ``AgentStatus`` notice (e.g. the doom-loop stop);
        * always a final ``AgentUsage`` (cumulative) immediately followed by an
          ``AgentDone`` carrying the same usage plus ``stop_reason``/``iterations``.

        ``asyncio.CancelledError`` is never converted to an ``AgentDone``: it is
        re-raised after a best-effort persist (matching :meth:`arun_turn`), so a
        cancelled stream never reports a normal stop.
        """
        self.session.add_message(Message.user(user_text))
        self._persist()

        turn_usage = Usage()
        iterations = 0
        stop_reason = "max_iterations"

        # Doom-loop tracking (identical semantics to the buffered path).
        last_signature: tuple[tuple[str, str], ...] | None = None
        repeat_count = 0

        tool_defs = self.registry.definitions()
        ctx = ToolContext(workspace_root=self.workspace_root)

        try:
            while iterations < self.max_iterations:
                iterations += 1
                system = self._build_system()

                text_parts: list[str] = []
                accumulator = ToolCallAccumulator()

                async for ev in self.provider.astream(
                    self.session.messages,
                    system=system,
                    tools=tool_defs or None,
                ):
                    if isinstance(ev, StreamTextDelta):
                        text_parts.append(ev.text)
                        yield AgentTextDelta(text=ev.text)
                    elif isinstance(ev, StreamToolCallDelta):
                        accumulator.add(ev)
                    elif isinstance(ev, StreamUsage):
                        turn_usage = turn_usage + ev.usage
                        self.session.add_usage(ev.usage)
                    elif isinstance(ev, StreamDone):
                        # finish_reason is advisory here; the loop's own stop
                        # conditions decide the turn's stop_reason. Break the inner
                        # stream and assemble the assistant message.
                        break

                tool_calls = accumulator.finalize()
                assistant_text = "".join(text_parts)

                assistant_msg = self._stream_assistant_message(assistant_text, tool_calls)
                self.session.add_message(assistant_msg)
                self._persist()

                # No tool calls → the turn is complete.
                if not tool_calls:
                    stop_reason = "completed"
                    break

                # Doom-loop guard — identical to the buffered path.
                signature = _batch_signature(tool_calls)
                if signature == last_signature:
                    repeat_count += 1
                else:
                    repeat_count = 1
                    last_signature = signature
                if repeat_count >= DOOM_LOOP_THRESHOLD and iterations < self.max_iterations:
                    stop_reason = "doom_loop"
                    yield AgentStatus(message="stopping: repeated identical tool calls")
                    break

                # Execute each call sequentially through the SAME gate as the
                # buffered path (_execute_tool_call), surfacing call + result events.
                result_blocks: list[ToolResultBlock] = []
                for call in tool_calls:
                    yield AgentToolCall(id=call.id, name=call.name, arguments=call.arguments)
                    block = await self._execute_tool_call(call, ctx)
                    result_blocks.append(block)
                    yield AgentToolResult(
                        tool_use_id=block.tool_use_id,
                        output=block.output,
                        is_error=block.is_error,
                    )

                self.session.add_message(Message.tool_results(result_blocks))
                self._persist()
        except asyncio.CancelledError:
            # Cancellation is a control signal, not a stop reason. State has only
            # been mutated + persisted at message boundaries, so it is consistent.
            # Best-effort persist (swallow save errors) then re-raise so the
            # CancelledError propagates rather than becoming a normal AgentDone.
            with contextlib.suppress(Exception):
                self._persist()
            raise

        yield AgentUsage(usage=turn_usage)
        yield AgentDone(stop_reason=stop_reason, iterations=iterations, usage=turn_usage)

    @staticmethod
    def _stream_assistant_message(text: str, tool_calls: list[ToolCall]) -> Message:
        """Build the assistant message from streamed text + accumulated tool calls.

        Mirrors :meth:`_assistant_message` (the buffered builder): a leading
        :class:`TextBlock` when any text streamed, then one :class:`ToolUseBlock`
        per finalized call. A response with neither yields an empty-blocks message
        (the turn still ends cleanly).
        """
        blocks: list[ContentBlock] = []
        if text:
            blocks.append(TextBlock(text=text))
        for call in tool_calls:
            blocks.append(ToolUseBlock(id=call.id, name=call.name, input=call.arguments))
        return Message(role="assistant", blocks=blocks)

"""Edge-case / hardening tests for the ReAct agent loop (:mod:`zakcode.agent.loop`).

These complement ``test_loop.py`` and exercise the control-flow and stop-condition
hardening: cancellation propagation, empty completions, the ``max_iterations``
boundary, the doom-loop guard, error-tool recovery, and usage accumulation.

Everything is hermetic: scripted in-memory providers (no network, no model), a
tiny in-memory tool registry, and a ``tmp_path`` workspace.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import (
    _EMPTY_COMPLETION_PLACEHOLDER,
    _MAX_DOOM_RECOVERIES,
    DEFAULT_MAX_ITERATIONS,
    DOOM_LOOP_THRESHOLD,
    AgentLoop,
    TurnResult,
)
from zakcode.config import load_settings
from zakcode.messages import Message, ToolUseBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session, SessionStore
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.usage import Usage

#: Iterations a forever-repeating model burns before doom_loop: THRESHOLD repeats, ONE recovery
#: attempt, then THRESHOLD more (the recovery re-enters once before the guard gives up).
_DOOM_ITERS = DOOM_LOOP_THRESHOLD * (1 + _MAX_DOOM_RECOVERIES)

# ── scripted providers ───────────────────────────────────────────────────────


class ScriptedProvider(Provider):
    """Replays a fixed list of :class:`LLMResult`s, one per ``acomplete`` call."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = list(results)
        self.calls: list[list[Message]] = []

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        self.calls.append(list(messages))
        if not self._results:
            raise AssertionError("provider ran out of scripted results")
        return self._results.pop(0)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return sum(len(m.text) for m in messages)

    def capabilities(self) -> Capabilities:
        return Capabilities()


class LoopingProvider(Provider):
    """Always requests the *same* tool call, forever (drives the doom-loop guard)."""

    def __init__(
        self,
        name: str = "echo",
        arguments: dict[str, Any] | None = None,
        *,
        usage: Usage | None = None,
    ) -> None:
        self.name = name
        self.arguments = {} if arguments is None else arguments
        self.usage = usage or Usage()
        self.call_count = 0

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        self.call_count += 1
        # A fresh id each call: identity must be (name, arguments), never the id.
        return LLMResult(
            tool_calls=[
                ToolCall(id=f"c{self.call_count}", name=self.name, arguments=dict(self.arguments))
            ],
            usage=self.usage,
        )

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


class CancelOnNthCallProvider(Provider):
    """Raises :class:`asyncio.CancelledError` on the ``n``-th ``acomplete`` call."""

    def __init__(self, cancel_on: int = 1) -> None:
        self.cancel_on = cancel_on
        self.call_count = 0

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        self.call_count += 1
        if self.call_count >= self.cancel_on:
            # Suspend once so cancellation flows through the normal await
            # machinery (mirrors a real provider being cancelled mid-await)
            # rather than raising synchronously from the coroutine body.
            await asyncio.sleep(0)
            raise asyncio.CancelledError
        # Before the cancel point, request a tool with distinct args each call so
        # the loop keeps iterating (a text-only completion would end the turn) and
        # the doom-loop guard never trips on the way to the cancellation.
        return LLMResult(
            tool_calls=[
                ToolCall(id=f"c{self.call_count}", name="echo", arguments={"n": self.call_count})
            ]
        )

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


# ── tools ────────────────────────────────────────────────────────────────────


class EchoTool(Tool):
    spec = ToolSpec(name="echo", description="Echo back the provided text.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(output=str(args.get("text", "")))


class CancelTool(Tool):
    """A tool whose handler raises ``CancelledError`` (e.g. its own await is cut)."""

    spec = ToolSpec(name="cancel", description="Raises CancelledError when run.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        await asyncio.sleep(0)
        raise asyncio.CancelledError


class AlwaysErrorTool(Tool):
    spec = ToolSpec(name="boom", description="Always returns an error result.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.error("kaboom")


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(CancelTool())
    reg.register(AlwaysErrorTool())
    return reg


def _settings(tmp_path: Path, **over: Any):
    base: dict[str, Any] = {"workspace_root": tmp_path, "max_iterations": 10}
    base.update(over)
    return load_settings(**base)


def _session(tmp_path: Path) -> Session:
    # The Session model carries cwd + model alongside its message history.
    return Session(cwd=str(tmp_path), model="ollama_chat/llama3.1")


def _make_loop(
    provider: Provider,
    tmp_path: Path,
    *,
    session: Session | None = None,
    store: SessionStore | None = None,
    **settings_over: Any,
) -> AgentLoop:
    return AgentLoop(
        provider,
        _registry(),
        session if session is not None else _session(tmp_path),
        settings=_settings(tmp_path, **settings_over),
        store=store,
    )


# ── empty completion ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_completion_completes_cleanly(tmp_path: Path) -> None:
    # Neither text nor tool calls: must end with stop_reason="completed".
    provider = ScriptedProvider([LLMResult()])
    loop = _make_loop(provider, tmp_path)
    result = await loop.arun_turn("say nothing")
    assert isinstance(result, TurnResult)
    assert result.stop_reason == "completed"
    assert result.iterations == 1
    # The assistant message carries a PLACEHOLDER block (contract changed 2026-08-22):
    # an empty-blocks assistant message poisons the stored history — OpenAI-compat
    # providers reject the whole transcript on every later call.
    assert len(result.assistant_messages) == 1
    assert result.assistant_messages[-1].blocks
    assert result.assistant_messages[-1].text == _EMPTY_COMPLETION_PLACEHOLDER


@pytest.mark.asyncio
async def test_empty_completion_does_not_loop(tmp_path: Path) -> None:
    # Even with a generous budget, an empty completion stops after one iteration
    # (no infinite loop). Only one scripted result is provided on purpose.
    provider = ScriptedProvider([LLMResult()])
    loop = _make_loop(provider, tmp_path, max_iterations=50)
    result = await loop.arun_turn("hi")
    assert result.iterations == 1
    assert len(provider.calls) == 1  # exactly one model call, no re-querying


# ── max_iterations boundary ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_iterations_cap_of_one_runs_once(tmp_path: Path) -> None:
    # A cap of 1 still runs exactly one iteration.
    provider = LoopingProvider(arguments={"text": "x"})
    loop = _make_loop(provider, tmp_path, max_iterations=1)
    result = await loop.arun_turn("go")
    assert result.iterations == 1
    assert result.stop_reason == "max_iterations"
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_max_iterations_exact_boundary(tmp_path: Path) -> None:
    # Distinct tool calls each iteration so the doom-loop guard never fires; the
    # only thing that can stop us is the iteration cap, hit exactly.
    cap = 5
    scripted = [
        LLMResult(tool_calls=[ToolCall(id=f"c{i}", name="echo", arguments={"text": str(i)})])
        for i in range(cap + 3)
    ]
    provider = ScriptedProvider(scripted)
    loop = _make_loop(provider, tmp_path, max_iterations=cap)
    result = await loop.arun_turn("count")
    assert result.stop_reason == "max_iterations"
    assert result.iterations == cap


@pytest.mark.asyncio
async def test_explicit_max_iterations_overrides_settings(tmp_path: Path) -> None:
    provider = LoopingProvider(arguments={"text": "y"})
    # settings says 10, explicit kwarg says 2 -> explicit wins.
    loop = AgentLoop(
        provider,
        _registry(),
        _session(tmp_path),
        settings=_settings(tmp_path, max_iterations=10),
        max_iterations=2,
    )
    result = await loop.arun_turn("go")
    assert loop.max_iterations == 2
    assert result.iterations == 2


def test_default_max_iterations_constant() -> None:
    # 0 = UNLIMITED (2026-08-25 field decision: minds run for days; the real
    # runaway guards are the doom-loop detector and the cost budget).
    assert DEFAULT_MAX_ITERATIONS == 0


# ── doom-loop guard ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_doom_loop_stops_before_budget(tmp_path: Path) -> None:
    # Identical call every iteration; a large budget would otherwise burn forever. The guard tries
    # ONE recovery first, so it fires after THRESHOLD repeats + the recovery + THRESHOLD more.
    provider = LoopingProvider(arguments={"text": "same"})
    loop = _make_loop(provider, tmp_path, max_iterations=100)
    result = await loop.arun_turn("repeat")
    assert result.stop_reason == "doom_loop"
    assert result.iterations == _DOOM_ITERS
    # We stopped BEFORE the iteration budget (and far below it).
    assert result.iterations < 100


@pytest.mark.asyncio
async def test_doom_loop_identity_ignores_call_id(tmp_path: Path) -> None:
    # LoopingProvider hands a fresh id each call; identity is (name, arguments),
    # so the doom-loop guard must still fire.
    provider = LoopingProvider(name="echo", arguments={"text": "k"})
    loop = _make_loop(provider, tmp_path, max_iterations=50)
    result = await loop.arun_turn("repeat")
    assert result.stop_reason == "doom_loop"
    assert result.iterations == _DOOM_ITERS


@pytest.mark.asyncio
async def test_doom_loop_arg_order_insensitive(tmp_path: Path) -> None:
    # Same call but arguments built with different key orders -> still a repeat.
    # _DOOM_ITERS identical calls (alternating key order) so the guard fires even after its one
    # recovery attempt -> arg-order-insensitive identity holds across the recovery too.
    calls = [
        LLMResult(
            tool_calls=[
                ToolCall(
                    id=f"c{i}",
                    name="echo",
                    arguments={"x": 1, "y": 2} if i % 2 else {"y": 2, "x": 1},
                )
            ]
        )
        for i in range(_DOOM_ITERS)
    ]
    calls.append(LLMResult(text="should not get here"))
    provider = ScriptedProvider(calls)
    loop = _make_loop(provider, tmp_path, max_iterations=50)
    result = await loop.arun_turn("repeat")
    assert result.stop_reason == "doom_loop"
    assert result.iterations == _DOOM_ITERS


@pytest.mark.asyncio
async def test_doom_loop_resets_on_different_call(tmp_path: Path) -> None:
    # Two identical calls, then a *different* call resets the counter, then the
    # model completes. The guard must NOT fire (no 3-in-a-row run).
    calls = [
        LLMResult(tool_calls=[ToolCall(id="a", name="echo", arguments={"text": "same"})]),
        LLMResult(tool_calls=[ToolCall(id="b", name="echo", arguments={"text": "same"})]),
        LLMResult(tool_calls=[ToolCall(id="c", name="echo", arguments={"text": "different"})]),
        LLMResult(text="done"),
    ]
    provider = ScriptedProvider(calls)
    loop = _make_loop(provider, tmp_path, max_iterations=50)
    result = await loop.arun_turn("vary")
    assert result.stop_reason == "completed"
    assert result.iterations == 4


@pytest.mark.asyncio
async def test_doom_loop_different_args_not_triggered(tmp_path: Path) -> None:
    # Same tool, but each call has different args -> never a repeat -> completes.
    calls = [
        LLMResult(tool_calls=[ToolCall(id=f"c{i}", name="echo", arguments={"text": str(i)})])
        for i in range(DOOM_LOOP_THRESHOLD + 2)
    ]
    calls.append(LLMResult(text="finished"))
    provider = ScriptedProvider(calls)
    loop = _make_loop(provider, tmp_path, max_iterations=50)
    result = await loop.arun_turn("vary args")
    assert result.stop_reason == "completed"


@pytest.mark.asyncio
async def test_doom_recovery_nudge_lets_model_recover(tmp_path: Path) -> None:
    # Three identical calls trip the doom guard; instead of giving up, the loop injects ONE
    # recovery nudge and re-enters -> the model breaks the loop on the next reply and completes.
    # Without the recovery this would stop at doom_loop on the 3rd repeat and never reach the 4th.
    calls = [
        LLMResult(tool_calls=[ToolCall(id="a", name="echo", arguments={"text": "same"})]),
        LLMResult(tool_calls=[ToolCall(id="b", name="echo", arguments={"text": "same"})]),
        LLMResult(tool_calls=[ToolCall(id="c", name="echo", arguments={"text": "same"})]),
        LLMResult(text="different approach -> done"),
    ]
    provider = ScriptedProvider(calls)
    loop = _make_loop(provider, tmp_path, max_iterations=50)
    result = await loop.arun_turn("repeat")
    assert result.stop_reason == "completed"  # the recovery rescued the turn
    assert result.iterations == 4


def test_doom_recovery_cap_constant() -> None:
    assert _MAX_DOOM_RECOVERIES == 1


def test_doom_loop_threshold_constant() -> None:
    assert DOOM_LOOP_THRESHOLD == 3


# ── error tool does not abort the turn ────────────────────────────────────────


@pytest.mark.asyncio
async def test_error_tool_does_not_abort_turn(tmp_path: Path) -> None:
    # Tool errors twice (different args so doom-loop stays out of it), then the
    # model recovers and completes. The error results are surfaced, not raised.
    calls = [
        LLMResult(tool_calls=[ToolCall(id="e1", name="boom", arguments={"n": 1})]),
        LLMResult(tool_calls=[ToolCall(id="e2", name="boom", arguments={"n": 2})]),
        LLMResult(text="recovered after errors"),
    ]
    provider = ScriptedProvider(calls)
    loop = _make_loop(provider, tmp_path, max_iterations=50)
    result = await loop.arun_turn("try the broken tool")
    assert result.stop_reason == "completed"
    assert result.iterations == 3
    assert len(result.tool_results) == 2
    assert all(b.is_error for b in result.tool_results)
    assert all(b.output == "kaboom" for b in result.tool_results)
    assert result.assistant_messages[-1].text == "recovered after errors"


# ── usage accumulation across iterations ──────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_accumulates_over_many_iterations(tmp_path: Path) -> None:
    calls = [
        LLMResult(
            tool_calls=[ToolCall(id=f"c{i}", name="echo", arguments={"text": str(i)})],
            usage=Usage(prompt_tokens=i, completion_tokens=2 * i, total_tokens=3 * i, cost_usd=0.5),
        )
        for i in range(1, 5)  # 4 tool-calling iterations: i = 1..4
    ]
    calls.append(
        LLMResult(
            text="final",
            usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110, cost_usd=1.0),
        )
    )
    provider = ScriptedProvider(calls)
    loop = _make_loop(provider, tmp_path, max_iterations=50)
    result = await loop.arun_turn("accumulate")
    assert result.stop_reason == "completed"
    assert result.iterations == 5
    # prompt: 1+2+3+4+100 = 110 ; completion: 2+4+6+8+10 = 30 ; total: 110*? -> sum
    assert result.usage.prompt_tokens == 110
    assert result.usage.completion_tokens == 30
    assert result.usage.total_tokens == (3 + 6 + 9 + 12 + 110)
    assert result.usage.cost_usd == pytest.approx(0.5 * 4 + 1.0)


@pytest.mark.asyncio
async def test_usage_matches_session_cumulative(tmp_path: Path) -> None:
    # The turn's accumulated usage should equal the session's recorded total.
    calls = [
        LLMResult(
            tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "a"})],
            usage=Usage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
        ),
        LLMResult(text="end", usage=Usage(prompt_tokens=4, completion_tokens=1, total_tokens=5)),
    ]
    provider = ScriptedProvider(calls)
    session = _session(tmp_path)
    loop = _make_loop(provider, tmp_path, session=session)
    result = await loop.arun_turn("go")
    cumulative = session.cumulative_usage()
    assert result.usage.total_tokens == cumulative.total_tokens == 15


# ── cancellation ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancellation_propagates_not_swallowed(tmp_path: Path) -> None:
    provider = CancelOnNthCallProvider(cancel_on=1)
    loop = _make_loop(provider, tmp_path)
    with pytest.raises(asyncio.CancelledError):
        await loop.arun_turn("cancel me")


@pytest.mark.asyncio
async def test_cancellation_midturn_propagates(tmp_path: Path) -> None:
    # Several normal completions, then cancellation on a later call: still raises.
    provider = CancelOnNthCallProvider(cancel_on=3)
    loop = _make_loop(provider, tmp_path)
    with pytest.raises(asyncio.CancelledError):
        await loop.arun_turn("go a while then cancel")
    assert provider.call_count == 3


@pytest.mark.asyncio
async def test_cancellation_from_tool_propagates(tmp_path: Path) -> None:
    # A tool handler raising CancelledError must propagate as cancellation, not be
    # caught and reported as a normal stop. (Note: ToolRegistry.execute only
    # catches Exception, so BaseException-derived CancelledError escapes.)
    provider = ScriptedProvider(
        [LLMResult(tool_calls=[ToolCall(id="c1", name="cancel", arguments={})])]
    )
    loop = _make_loop(provider, tmp_path)
    with pytest.raises(asyncio.CancelledError):
        await loop.arun_turn("run the cancelling tool")


@pytest.mark.asyncio
async def test_cancellation_leaves_consistent_session(tmp_path: Path) -> None:
    # After cancellation, the persisted session must round-trip cleanly (no
    # half-written/corrupt document) and contain only complete messages.
    store = SessionStore(tmp_path / "sessions")
    provider = CancelOnNthCallProvider(cancel_on=1)
    session = _session(tmp_path)
    loop = _make_loop(provider, tmp_path, session=session, store=store)
    with pytest.raises(asyncio.CancelledError):
        await loop.arun_turn("cancel and persist")

    loaded = store.load(session.id)  # raises if the file is corrupt
    assert loaded.id == session.id
    # The user message was persisted before the (cancelled) first model call.
    assert any(m.role == "user" and m.text == "cancel and persist" for m in loaded.messages)
    # Every persisted message validates as a proper Message (consistency check).
    for m in loaded.messages:
        assert isinstance(m, Message)


# ── interaction: scripted tool-use shapes the assistant message ───────────────


@pytest.mark.asyncio
async def test_assistant_message_carries_tool_use_blocks(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            LLMResult(
                text="calling now",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
            ),
            LLMResult(text="done"),
        ]
    )
    loop = _make_loop(provider, tmp_path)
    result = await loop.arun_turn("echo please")
    first = result.assistant_messages[0]
    tool_uses = [b for b in first.blocks if isinstance(b, ToolUseBlock)]
    assert len(tool_uses) == 1
    assert tool_uses[0].name == "echo"
    assert tool_uses[0].input == {"text": "hi"}

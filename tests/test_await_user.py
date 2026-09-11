"""Waiting for the operator is a turn-ENDING state (ADR-0121, Zak-Code #181).

Field incident 2026-08-22, a live agent on slow local inference: the model reached a
non-skippable user-input gate mid-plan, said so correctly ("I can't advance without your
explicit answer"), and the harness's plan-continuation re-invoked it anyway — 27 of the
session's 50 iterations spent restating "still waiting" at ~15-17 minutes a call, each
restatement growing the turn's context. Prose cannot stop a loop. These pin the tool that
can, and the two properties that make it worth having: the turn ENDS, and the plan SURVIVES.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from zakcode.agent.loop import _BLOCKER_NUDGE, AgentLoop
from zakcode.permissions import PermissionMode, PermissionPolicy
from zakcode.cli.render import _STOP_LABEL
from zakcode.events import AgentStatus
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolCall,
)
from zakcode.session.store import Session
from zakcode.tasks import Task
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.await_user import AwaitUserTool
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.usage import Usage

QUESTION = "Should I deploy to staging or production first?"


class _Scripted(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(self, messages: list[Any], **kw: Any) -> LLMResult:
        self.calls += 1
        index = min(self.calls - 1, len(self._results) - 1)
        return self._results[index]

    def count_tokens(self, messages: list[Any], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=200_000)


class _ScriptedStream(_Scripted):
    async def acomplete(self, messages: list[Any], **kw: Any) -> LLMResult:  # pragma: no cover
        raise AssertionError("streaming path must not use the buffered call")

    async def astream(self, messages: list[Any], **kw: Any) -> AsyncIterator[ProviderStreamEvent]:
        self.calls += 1
        result = self._results[min(self.calls - 1, len(self._results) - 1)]
        if result.text:
            yield StreamTextDelta(text=result.text)
        for index, call in enumerate(result.tool_calls):
            yield StreamToolCallDelta(
                index=index, id=call.id, name=call.name, arguments_delta=_json(call.arguments)
            )
        yield StreamDone(finish_reason="tool_calls" if result.tool_calls else "stop")


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj)


def _judge_ok() -> LLMResult:
    """The always-on decomposition judge (ADR-0050) fires once per turn on the first
    structural plan authoring, so any script that authors a plan feeds it one verdict."""
    import json

    return LLMResult(
        text=json.dumps(
            {"scores": {"coverage": 0.9, "granularity": 0.9, "ordering": 0.9, "soundness": 0.9}}
        ),
        usage=Usage(total_tokens=2),
    )


def _plan(tasks: list[dict]) -> LLMResult:
    return LLMResult(
        text="",
        tool_calls=[ToolCall(id="p1", name="update_plan", arguments={"tasks": tasks})],
        usage=Usage(total_tokens=1),
    )


def _await(question: str = QUESTION, *, name: str = "await_user", **extra: Any) -> LLMResult:
    args: dict[str, Any] = {"question": question} if question is not None else {}
    args.update(extra)
    return LLMResult(
        text="I need your call on this before I can go on.",
        tool_calls=[ToolCall(id="a1", name=name, arguments=args)],
        usage=Usage(total_tokens=1),
    )


def _ctx() -> ToolContext:
    return ToolContext(workspace_root=Path("/tmp"))


def _loop(provider: Provider) -> tuple[AgentLoop, Session]:
    session = Session(cwd="/tmp", model="test/model")
    return AgentLoop(provider, default_registry(), session, max_iterations=20), session


# ── the tool ──────────────────────────────────────────────────────────────────


async def test_the_tool_reports_the_question_and_that_the_plan_is_kept() -> None:
    result = await AwaitUserTool().execute({"question": QUESTION}, _ctx())
    assert not result.is_error
    assert QUESTION in result.output
    assert "plan is kept" in result.output
    assert result.data is not None and result.data["question"] == QUESTION


async def test_the_tool_refuses_an_empty_question() -> None:
    for args in ({}, {"question": "   "}, {"question": 7}):
        result = await AwaitUserTool().execute(args, _ctx())
        assert result.is_error and "'question' is required" in result.output


# ── the terminal (buffered) ───────────────────────────────────────────────────


async def test_await_user_ends_the_turn_and_leaves_the_open_step_alone() -> None:
    provider = _Scripted(
        [
            _plan([{"title": "gather", "status": "in_progress"}, {"title": "deploy"}]),
            _judge_ok(),
            _await(),
            LLMResult(text="still waiting", tool_calls=[], usage=Usage(total_tokens=1)),
        ]
    )
    loop, session = _loop(provider)
    result = await loop.arun_turn("ship it")

    assert result.stop_reason == "awaiting_user"
    # The whole point: the model is asked ONCE more after the plan, not 27 times.
    assert provider.calls == 3
    # …and the plan survives untouched, so the operator's answer resumes it.
    assert [t.status for t in session.task_network.tasks] == ["in_progress", "pending"]
    assert len(session.task_network.actionable_remaining()) == 2


async def test_the_question_is_recorded_for_the_operator() -> None:
    provider = _Scripted([_await()])
    loop, _ = _loop(provider)
    await loop.arun_turn("ship it")
    notes = [
        e for e in loop._trace.of_kind("intervention") if e.data.get("kind") == "awaiting_user"
    ]
    assert notes and QUESTION in notes[-1].detail


async def test_a_malformed_call_never_silently_ends_the_turn() -> None:
    # A question-less call is an ERROR, so the terminal is not armed and the model gets
    # its error result and continues — a bad argument must not look like a decision to stop.
    provider = _Scripted([_await(None), LLMResult(text="fixed it", tool_calls=[])])
    loop, _ = _loop(provider)
    result = await loop.arun_turn("ship it")
    assert result.stop_reason == "completed"
    assert provider.calls == 2


async def test_the_aliases_end_the_turn_too() -> None:
    for alias in ("ask_user", "wait_for_user"):
        provider = _Scripted([_await(name=alias)])
        loop, _ = _loop(provider)
        assert (await loop.arun_turn("ship it")).stop_reason == "awaiting_user", alias


# ── the terminal (streaming twin) ─────────────────────────────────────────────


async def test_streaming_announces_the_wait_and_ends_the_turn() -> None:
    provider = _ScriptedStream(
        [_plan([{"title": "gather", "status": "in_progress"}]), _judge_ok(), _await()]
    )
    loop, session = _loop(provider)
    events = [event async for event in loop.astream_turn("ship it")]
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any(s.startswith("waiting for you") and QUESTION in s for s in statuses)
    assert provider.calls == 3
    assert session.task_network.tasks[0].status == "in_progress"


# -- the unattended terminal: no operator to wait for (ADR-0133) --------------


def _loop_with_mode(provider: Provider, mode: PermissionMode) -> tuple[AgentLoop, Session]:
    session = Session(cwd="/tmp", model="test/model")
    loop = AgentLoop(
        provider,
        default_registry(),
        session,
        max_iterations=20,
        permission_policy=PermissionPolicy(mode),
    )
    return loop, session


async def test_await_user_fails_closed_when_unattended() -> None:
    # Autonomous, and a worker Body's bypass, both mean "no one at the prompt" (loop.unattended):
    # await_user then has no terminus, so it must NOT end the turn -- the loop continues instead
    # of stranding on a question no one will answer (the bobby /start-that-asked-to-boot class).
    for mode in (PermissionMode.AUTONOMOUS, PermissionMode.BYPASS):
        provider = _Scripted([_await(), LLMResult(text="decided it myself", tool_calls=[])])
        loop, _ = _loop_with_mode(provider, mode)
        result = await loop.arun_turn("ship it")
        assert result.stop_reason == "completed", mode
        assert provider.calls == 2, mode  # re-invoked, not stranded on the one await call
        armed = [
            e for e in loop._trace.of_kind("intervention") if e.data.get("kind") == "awaiting_user"
        ]
        assert not armed, mode
        refused = [
            e
            for e in loop._trace.of_kind("intervention")
            if e.data.get("kind") == "awaiting_refused"
        ]
        assert refused, mode


async def test_streaming_await_user_fails_closed_when_unattended() -> None:
    provider = _ScriptedStream([_await(), LLMResult(text="decided it myself", tool_calls=[])])
    loop, _ = _loop_with_mode(provider, PermissionMode.AUTONOMOUS)
    statuses = [
        e.message async for e in loop.astream_turn("ship it") if isinstance(e, AgentStatus)
    ]
    assert not any(s.startswith("waiting for you") for s in statuses)
    assert provider.calls == 2  # continued, not stranded


async def test_await_user_still_waits_when_an_operator_is_present() -> None:
    # ask / acceptEdits / allow all have someone at the prompt: the ADR-0121 terminal is unchanged.
    for mode in (PermissionMode.ASK, PermissionMode.ACCEPT_EDITS, PermissionMode.ALLOW):
        provider = _Scripted([_await()])
        loop, _ = _loop_with_mode(provider, mode)
        result = await loop.arun_turn("ship it")
        assert result.stop_reason == "awaiting_user", mode
        assert provider.calls == 1, mode


# ── the rails that point at it ────────────────────────────────────────────────


def test_the_blocker_nudge_offers_the_tool() -> None:
    assert "await_user" in _BLOCKER_NUDGE


def test_the_plan_gate_nudge_offers_the_tool() -> None:
    session = Session(cwd="/tmp", model="test/model")
    loop = AgentLoop(_Scripted([]), default_registry(), session, max_iterations=2)
    session.task_network.replace_from_author([Task(title="wait on a person", status="in_progress")])
    nudge = loop._plan_gate_nudge()
    assert nudge is not None and "await_user" in nudge


def test_the_stop_label_reads_as_a_pause_not_a_failure() -> None:
    assert "waiting for you" in _STOP_LABEL["awaiting_user"]


def test_the_tool_is_registered_with_its_aliases() -> None:
    registry = default_registry()
    for name in ("await_user", "ask_user", "wait_for_user"):
        assert registry.get(name) is not None, name

"""Hermetic tests for the streaming agent turn and the tool-call accumulator.

No network and no real model: a :class:`ScriptedStreamProvider` yields canned
:data:`~zakcode.providers.base.ProviderStreamEvent`s, and tools run through a real
:class:`~zakcode.tools.base.ToolRegistry`. The accumulator is also unit-tested
directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from zakcode.agent._stream import ToolCallAccumulator, parse_arguments
from zakcode.agent.loop import AgentLoop
from zakcode.config import Settings
from zakcode.events import (
    AgentDone,
    AgentStatus,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsage,
)
from zakcode.session.store import Session
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.usage import Usage

# ── test doubles ──────────────────────────────────────────────────────────────


class ScriptedStreamProvider(Provider):
    """A provider whose ``astream`` replays canned events per call.

    ``scripts`` is a list of per-iteration event lists; the Nth ``astream`` call
    replays ``scripts[N]`` (the last script repeats once exhausted). The
    non-streaming methods are stubs sufficient for the loop's needs.
    """

    def __init__(self, scripts: list[list[ProviderStreamEvent]]) -> None:
        self._scripts = scripts
        self.calls = 0
        self.seen_systems: list[str | None] = []

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:  # pragma: no cover - streaming tests never call this
        return LLMResult()

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.seen_systems.append(system)
        idx = min(self.calls, len(self._scripts) - 1)
        self.calls += 1
        for event in self._scripts[idx]:
            yield event


class EchoTool(Tool):
    """A trivial tool that echoes its ``value`` argument and records calls."""

    spec = ToolSpec(
        name="echo",
        description="Echo the provided value back.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls.append(args)
        return ToolResult.ok(f"echo:{args.get('value', '')}")


# ── fixtures / helpers ────────────────────────────────────────────────────────


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _settings(tmp_path: Any, max_iterations: int = 10) -> Settings:
    return Settings(workspace_root=tmp_path, max_iterations=max_iterations)


def _session() -> Session:
    return Session(cwd="/tmp", model="test-model")


def _build_loop(
    tmp_path: Any,
    provider: Provider,
    *,
    registry: ToolRegistry | None = None,
    max_iterations: int | None = None,
) -> AgentLoop:
    reg = registry if registry is not None else ToolRegistry()
    return AgentLoop(
        provider,
        reg,
        _session(),
        settings=_settings(tmp_path),
        max_iterations=max_iterations,
        workspace_root=tmp_path,
    )


async def _collect(loop: AgentLoop, text: str) -> list[Any]:
    return [ev async for ev in loop.astream_turn(text)]


# ── accumulator unit tests ────────────────────────────────────────────────────


def test_accumulator_concatenates_fragments_and_parses_args() -> None:
    acc = ToolCallAccumulator()
    acc.add(StreamToolCallDelta(index=0, id="c1", name="echo"))
    acc.add(StreamToolCallDelta(index=0, arguments_delta='{"val'))
    acc.add(StreamToolCallDelta(index=0, arguments_delta='ue": "hi"}'))
    calls = acc.finalize()
    assert len(calls) == 1
    assert calls[0].id == "c1"
    assert calls[0].name == "echo"
    assert calls[0].arguments == {"value": "hi"}


def test_accumulator_orders_by_index() -> None:
    acc = ToolCallAccumulator()
    # Fragments arrive out of index order; finalize() must sort by index.
    acc.add(StreamToolCallDelta(index=1, id="b", name="second", arguments_delta="{}"))
    acc.add(StreamToolCallDelta(index=0, id="a", name="first", arguments_delta="{}"))
    calls = acc.finalize()
    assert [c.id for c in calls] == ["a", "b"]
    assert [c.name for c in calls] == ["first", "second"]


def test_accumulator_first_id_name_wins() -> None:
    acc = ToolCallAccumulator()
    acc.add(StreamToolCallDelta(index=0, id="first", name="echo"))
    # A later fragment with a different id/name must NOT overwrite the first.
    acc.add(StreamToolCallDelta(index=0, id="second", name="other", arguments_delta="{}"))
    calls = acc.finalize()
    assert calls[0].id == "first"
    assert calls[0].name == "echo"


def test_accumulator_bad_json_falls_back_to_raw() -> None:
    acc = ToolCallAccumulator()
    acc.add(StreamToolCallDelta(index=0, id="c1", name="echo", arguments_delta="{not json"))
    calls = acc.finalize()
    assert calls[0].arguments == {"_raw": "{not json"}


def test_accumulator_empty_args_is_empty_dict() -> None:
    acc = ToolCallAccumulator()
    acc.add(StreamToolCallDelta(index=0, id="c1", name="echo"))
    calls = acc.finalize()
    assert calls[0].arguments == {}


def test_parse_arguments_edge_cases() -> None:
    assert parse_arguments("") == {}
    assert parse_arguments("   ") == {}
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments("not json") == {"_raw": "not json"}
    # A valid JSON non-object also falls back to _raw (callers want a dict).
    assert parse_arguments("[1, 2]") == {"_raw": "[1, 2]"}
    assert parse_arguments("42") == {"_raw": "42"}


def test_accumulator_truthiness() -> None:
    acc = ToolCallAccumulator()
    assert not acc
    assert len(acc) == 0
    acc.add(StreamToolCallDelta(index=0, id="c1", name="echo"))
    assert acc
    assert len(acc) == 1


# ── streaming turn tests ──────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_text_only_turn_completes(tmp_path: Any) -> None:
    provider = ScriptedStreamProvider(
        [
            [
                StreamTextDelta(text="Hello, "),
                StreamTextDelta(text="world!"),
                StreamUsage(usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)),
                StreamDone(finish_reason="stop"),
            ]
        ]
    )
    loop = _build_loop(tmp_path, provider)
    events = await _collect(loop, "hi")

    text = "".join(e.text for e in events if isinstance(e, AgentTextDelta))
    assert text == "Hello, world!"

    done = events[-1]
    assert isinstance(done, AgentDone)
    assert done.stop_reason == "completed"
    assert done.iterations == 1

    # A cumulative usage event precedes done and carries the streamed usage.
    usages = [e for e in events if isinstance(e, AgentUsage)]
    assert len(usages) == 1
    assert usages[0].usage.total_tokens == 5
    assert done.usage.total_tokens == 5

    # No tool calls in a text-only turn.
    assert not any(isinstance(e, AgentToolCall) for e in events)


@pytest.mark.anyio
async def test_streamed_tool_call_is_reassembled_and_executed(tmp_path: Any) -> None:
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)

    provider = ScriptedStreamProvider(
        [
            # Iteration 1: a tool call streamed in fragments.
            [
                StreamToolCallDelta(index=0, id="call-1", name="echo"),
                StreamToolCallDelta(index=0, arguments_delta='{"value":'),
                StreamToolCallDelta(index=0, arguments_delta=' "ping"}'),
                StreamUsage(usage=Usage(total_tokens=4)),
                StreamDone(finish_reason="tool_calls"),
            ],
            # Iteration 2: model responds with text -> turn completes.
            [
                StreamTextDelta(text="done"),
                StreamUsage(usage=Usage(total_tokens=2)),
                StreamDone(finish_reason="stop"),
            ],
        ]
    )
    loop = _build_loop(tmp_path, provider, registry=registry)
    events = await _collect(loop, "please echo")

    calls = [e for e in events if isinstance(e, AgentToolCall)]
    assert len(calls) == 1
    assert calls[0].name == "echo"
    assert calls[0].id == "call-1"
    assert calls[0].arguments == {"value": "ping"}

    results = [e for e in events if isinstance(e, AgentToolResult)]
    assert len(results) == 1
    assert results[0].tool_use_id == "call-1"
    assert results[0].output == "echo:ping"
    assert results[0].is_error is False

    # The real tool actually ran with the reassembled args.
    assert tool.calls == [{"value": "ping"}]

    # AgentToolCall is emitted before its AgentToolResult.
    call_idx = events.index(calls[0])
    res_idx = events.index(results[0])
    assert call_idx < res_idx

    done = events[-1]
    assert isinstance(done, AgentDone)
    assert done.stop_reason == "completed"
    assert done.iterations == 2
    # Cumulative usage spans both iterations.
    assert done.usage.total_tokens == 6


@pytest.mark.anyio
async def test_malformed_streamed_args_use_raw_fallback(tmp_path: Any) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    provider = ScriptedStreamProvider(
        [
            [
                StreamToolCallDelta(index=0, id="call-x", name="echo"),
                StreamToolCallDelta(index=0, arguments_delta="{broken"),
                StreamDone(finish_reason="tool_calls"),
            ],
            [
                StreamTextDelta(text="ok"),
                StreamDone(finish_reason="stop"),
            ],
        ]
    )
    loop = _build_loop(tmp_path, provider, registry=registry)
    events = await _collect(loop, "go")

    calls = [e for e in events if isinstance(e, AgentToolCall)]
    assert len(calls) == 1
    assert calls[0].arguments == {"_raw": "{broken"}

    # No crash; the turn still completes after the follow-up text iteration.
    done = events[-1]
    assert isinstance(done, AgentDone)
    assert done.stop_reason == "completed"


@pytest.mark.anyio
async def test_doom_loop_stops_turn(tmp_path: Any) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    # The identical call is streamed every iteration -> doom loop trips.
    def _repeated() -> list[ProviderStreamEvent]:
        return [
            StreamToolCallDelta(index=0, id="dup", name="echo", arguments_delta='{"value": "x"}'),
            StreamDone(finish_reason="tool_calls"),
        ]

    provider = ScriptedStreamProvider([_repeated() for _ in range(10)])
    loop = _build_loop(tmp_path, provider, registry=registry, max_iterations=20)
    events = await _collect(loop, "loop")

    done = events[-1]
    assert isinstance(done, AgentDone)
    assert done.stop_reason == "doom_loop"

    # Two advisories now: the bounded recovery nudge first, then the final doom stop.
    statuses = [e for e in events if isinstance(e, AgentStatus)]
    assert len(statuses) == 2
    assert "recovering" in statuses[0].message  # the recovery attempt
    assert "repeated identical tool calls" in statuses[1].message  # the give-up


@pytest.mark.anyio
async def test_max_iterations_stops_turn(tmp_path: Any) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    # Each iteration asks for a DISTINCT tool call so the doom guard never fires;
    # only the iteration budget can stop it.
    scripts: list[list[ProviderStreamEvent]] = [
        [
            StreamToolCallDelta(
                index=0, id=f"c{i}", name="echo", arguments_delta=f'{{"value": "{i}"}}'
            ),
            StreamDone(finish_reason="tool_calls"),
        ]
        for i in range(10)
    ]
    provider = ScriptedStreamProvider(scripts)
    loop = _build_loop(tmp_path, provider, registry=registry, max_iterations=2)
    events = await _collect(loop, "keep going")

    done = events[-1]
    assert isinstance(done, AgentDone)
    assert done.stop_reason == "max_iterations"
    assert done.iterations == 2


@pytest.mark.anyio
async def test_usage_event_precedes_done(tmp_path: Any) -> None:
    provider = ScriptedStreamProvider(
        [
            [
                StreamTextDelta(text="hi"),
                StreamUsage(usage=Usage(total_tokens=7)),
                StreamDone(finish_reason="stop"),
            ]
        ]
    )
    loop = _build_loop(tmp_path, provider)
    events = await _collect(loop, "x")

    assert isinstance(events[-1], AgentDone)
    assert isinstance(events[-2], AgentUsage)
    assert events[-2].usage.total_tokens == 7

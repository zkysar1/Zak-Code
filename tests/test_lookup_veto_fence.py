"""ADR-0422: a rule-lookup loop that a turn-end hook keeps sending back is compacted, then ended.

The doom guard and the lookup-only bound stop a loop of ``read_rule`` asks, and a hook that vetoes
every stop sends it back each time with those guards reset, so nothing else ever ends it. The
fence counts the stops such a loop has had sent back with nothing but lookups between them: at
the second, the next call's compaction check compacts whatever the size; a second trip with still
no work between ends the turn as ``veto_stall``, the ADR-0187 fence's ending. A work call that is
not a lookup starts the count over, and a doom loop of anything but lookups is not counted.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from zakcode.agent.loop import DOOM_LOOP_THRESHOLD, AgentLoop
from zakcode.agent.prompt import SystemPromptBuilder
from zakcode.events import AgentDone, AgentStatus
from zakcode.hooks import TurnEndPayload, TurnEndResult
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.rules import Rule, RuleRegistry
from zakcode.session.store import Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.tools.builtins.read_rule import ReadRuleTool
from zakcode.usage import Usage
from zakcode.wakeup import LOOP_SENTINEL


class _Scripted(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = list(results)
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        self.calls += 1
        if not self._results:
            raise AssertionError("provider ran out of scripted results")
        return self._results.pop(0)

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=32768)


class _Compactor:
    """Never due by size; records each compaction and what the transcript ended with then."""

    def __init__(self) -> None:
        self.last_texts: list[str] = []

    def should_compact(self, messages: list[Message], **kw: Any) -> bool:
        return False

    async def compact(self, messages: list[Message], **kw: Any) -> Any:
        self.last_texts.append(messages[-1].text)
        return SimpleNamespace(compacted=True, messages=list(messages[-4:]))

    def elide(self, messages: list[Message], **kw: Any) -> Any:
        return SimpleNamespace(compacted=False, messages=list(messages))


class _EchoTool(Tool):
    spec = ToolSpec(name="echo", description="Echo back the provided text.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(output=str(args.get("text", "")))


class _FailTool(Tool):
    """A call that fails every time: a failed call is no work, so only the stop's own kind
    can keep a loop of it out of the count."""

    spec = ToolSpec(name="probe", description="Probe a service.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.error("connection refused")


def _ask(i: int, name: str = "error-response") -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id=f"r{i}", name="read_rule", arguments={"name": name})],
        usage=Usage(total_tokens=1),
    )


def _echo(i: int, text: str = "work") -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id=f"e{i}", name="echo", arguments={"text": text})],
        usage=Usage(total_tokens=1),
    )


def _probe(i: int) -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id=f"p{i}", name="probe", arguments={"host": "db"})],
        usage=Usage(total_tokens=1),
    )


_DONE = LLMResult(text="done", usage=Usage(total_tokens=1))


def _registry() -> RuleRegistry:
    reg = RuleRegistry()
    body = "# Errors\nAfter any infrastructure call, check the error alerts."
    reg.add(Rule("error-response", body, Path("/x/error-response.md"), always_apply=True))
    body = "# Returns\nEnd every step with a tool call."
    reg.add(Rule("return-protocol", body, Path("/x/return-protocol.md"), always_apply=True))
    return reg


class _SendsBack:
    """A loop's turn-end hook: every stop but a finished answer is sent back."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def __call__(self, payload: TurnEndPayload) -> TurnEndResult | None:
        self.asked.append(payload.stop_reason)
        if payload.stop_reason == "completed":
            return None
        return TurnEndResult(vetoed=True, continuation_prompt="Keep going with the loop.")


def _loop(
    tmp_path: Path, results: list[LLMResult], *, compact: bool = True
) -> tuple[AgentLoop, _Compactor, _SendsBack]:
    tools = ToolRegistry()
    tools.register(ReadRuleTool())
    tools.register(_EchoTool())
    tools.register(_FailTool())
    registry = _registry()
    compactor = _Compactor()
    loop = AgentLoop(
        _Scripted(results),
        tools,
        Session(cwd=str(tmp_path), model="t"),
        prompt_builder=SystemPromptBuilder(rules=registry.render_index()),
        rule_registry=registry,
        compactor=compactor if compact else None,  # type: ignore[arg-type]
        max_iterations=60,
        turn_end_vetoable=True,
    )
    hook = _SendsBack()
    loop.hook_manager.register_turn_end(hook)
    return loop, compactor, hook


async def _turn(loop: AgentLoop, *, streamed: bool) -> tuple[str, list[str]]:
    if not streamed:
        result = await loop.arun_turn("go")
        return result.stop_reason, []
    statuses: list[str] = []
    stop_reason = ""
    async for event in loop.astream_turn("go"):
        if isinstance(event, AgentStatus):
            statuses.append(event.message)
        elif isinstance(event, AgentDone):
            stop_reason = event.stop_reason
    return stop_reason, statuses


#: One doom-loop stop of a lookup loop: the same ask until the doom guard stops it.
_STREAK = DOOM_LOOP_THRESHOLD


def _asks(start: int, streaks: int) -> list[LLMResult]:
    return [_ask(start + i) for i in range(streaks * _STREAK)]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
async def test_a_lookup_loop_sent_back_twice_is_compacted_before_the_next_call(
    tmp_path: Path, streamed: bool
) -> None:
    """The measured loop: two pointer answers and a doom-loop stop, sent back, over and over.
    The first stop is the doom guard's own recovery; the next two reach the hook."""
    loop, compactor, hook = _loop(tmp_path, [*_asks(0, 3), _DONE])
    stop_reason, statuses = await _turn(loop, streamed=streamed)
    assert stop_reason == "completed"
    assert hook.asked == ["doom_loop", "doom_loop", "completed"]
    # One compaction, and it ran on the transcript the second veto left: its re-entry is last.
    assert len(compactor.last_texts) == 1
    assert "Keep going with the loop." in compactor.last_texts[0]
    if streamed:
        assert any("rule-lookup loop was sent back 2 times" in s for s in statuses), statuses


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
async def test_a_second_trip_with_no_work_between_ends_the_turn_as_veto_stall(
    tmp_path: Path, streamed: bool
) -> None:
    """Compaction did not end the loop: two more stops sent back with only lookups between, and
    the turn ends as the ADR-0187 fence ends one, with a wake-up left behind."""
    loop, compactor, hook = _loop(tmp_path, [*_asks(0, 5), _DONE])
    stop_reason, _ = await _turn(loop, streamed=streamed)
    assert stop_reason == "veto_stall"
    assert len(compactor.last_texts) == 1
    assert hook.asked == ["doom_loop"] * 4
    pending = loop.wakeup_slot.pending()
    assert pending is not None and pending.prompt == LOOP_SENTINEL


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
@pytest.mark.parametrize(
    ("asks", "ends", "asked"),
    [(13, "completed", ["stuck", "stuck", "completed"]), (25, "veto_stall", ["stuck"] * 4)],
    ids=["compacts", "ends"],
)
async def test_a_lookup_loop_the_stuck_ladder_stops_is_counted_too(
    tmp_path: Path, asks: int, ends: str, asked: list[str], streamed: bool
) -> None:
    """Two rules asked for in turn never repeat a batch, so the doom guard never stops them: the
    lookup-only bound refuses them and the ladder stops the turn, and the hook sends it back."""
    names = ("error-response", "return-protocol")
    loop, compactor, hook = _loop(tmp_path, [*(_ask(i, names[i % 2]) for i in range(asks)), _DONE])
    stop_reason, _ = await _turn(loop, streamed=streamed)
    assert stop_reason == ends
    assert hook.asked == asked
    assert len(compactor.last_texts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
async def test_a_work_call_between_the_stops_starts_the_count_over(
    tmp_path: Path, streamed: bool
) -> None:
    """Between the two stops the model ran a command: the loop is not one loop, nothing compacts."""
    script = [*_asks(0, 2), _echo(0), *_asks(10, 1), _DONE]
    loop, compactor, hook = _loop(tmp_path, script)
    stop_reason, _ = await _turn(loop, streamed=streamed)
    assert stop_reason == "completed"
    assert hook.asked == ["doom_loop", "doom_loop", "completed"]
    assert compactor.last_texts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
async def test_a_work_call_after_the_compaction_earns_the_loop_another(
    tmp_path: Path, streamed: bool
) -> None:
    """A command after the compaction is progress: the next loop is compacted too, not ended."""
    script = [*_asks(0, 3), _echo(0), *_asks(10, 2), _DONE]
    loop, compactor, hook = _loop(tmp_path, script)
    stop_reason, _ = await _turn(loop, streamed=streamed)
    assert stop_reason == "completed"
    assert hook.asked == ["doom_loop"] * 4 + ["completed"]
    assert len(compactor.last_texts) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
async def test_a_new_turn_compacts_its_own_loop_again(tmp_path: Path, streamed: bool) -> None:
    """The compaction one turn spent does not count against the next turn's loop."""
    loop, compactor, hook = _loop(tmp_path, [*_asks(0, 3), _DONE, *_asks(20, 3), _DONE])
    first, _ = await _turn(loop, streamed=streamed)
    second, _ = await _turn(loop, streamed=streamed)
    assert (first, second) == ("completed", "completed")
    assert len(compactor.last_texts) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
async def test_with_no_compactor_the_second_trip_still_ends_the_turn(
    tmp_path: Path, streamed: bool
) -> None:
    """The fence counts the compaction it asked for: with none to run, the context the loop
    runs on is unchanged, so the turn ends on the same count."""
    loop, _, hook = _loop(tmp_path, [*_asks(0, 5), _DONE], compact=False)
    stop_reason, _ = await _turn(loop, streamed=streamed)
    assert stop_reason == "veto_stall"
    assert hook.asked == ["doom_loop"] * 4


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
async def test_a_compaction_asked_for_does_not_outlive_its_turn(
    tmp_path: Path, streamed: bool
) -> None:
    """A turn that ended before its next call leaves the asked-for compaction behind; the next
    turn starts without it."""
    loop, compactor, _ = _loop(tmp_path, [_DONE])
    loop._compact_due = "a rule-lookup loop was sent back 2 times with no work call between"
    stop_reason, _ = await _turn(loop, streamed=streamed)
    assert stop_reason == "completed"
    assert compactor.last_texts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
async def test_a_doom_loop_of_anything_but_lookups_is_not_counted(
    tmp_path: Path, streamed: bool
) -> None:
    """The fence is for lookups: a failing probe repeated and sent back twice compacts nothing,
    though no work ran between its stops either."""
    script = [_probe(i) for i in range(3 * _STREAK)] + [_DONE]
    loop, compactor, hook = _loop(tmp_path, script)
    stop_reason, _ = await _turn(loop, streamed=streamed)
    assert stop_reason == "completed"
    assert hook.asked[:2] == ["doom_loop", "doom_loop"]
    assert compactor.last_texts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
async def test_each_turn_starts_the_count_over(tmp_path: Path, streamed: bool) -> None:
    """A stop sent back at the end of one turn and one in the next are not two in a row."""
    loop, compactor, hook = _loop(tmp_path, [*_asks(0, 2), _DONE, *_asks(20, 2), _DONE])
    first, _ = await _turn(loop, streamed=streamed)
    second, _ = await _turn(loop, streamed=streamed)
    assert (first, second) == ("completed", "completed")
    assert hook.asked == ["doom_loop", "completed", "doom_loop", "completed"]
    assert compactor.last_texts == []

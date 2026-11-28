"""Prose no one reads, in a loop session (ADR-0421).

A session runs a loop skill, a reply tried to end the turn, something sent it back and no person
has spoken since: the loop holds the floor, and no person reads the prose the model writes. The
turn-end hook's refusal passes the floor on any reply with text, a gate's (the plan gate's) only
on a full answer, longer than the cap or continued at the output limit: a shorter reply may be a
preface. A streamed reply that writes more than the cap of prose with no tool call started is cut
where it passes the cap, and a reply of that kind that reaches the output limit is not continued.
The first answer of a turn, an answer after a person's mid-task message, and any reply outside a
loop session are never touched; nor is a stream whose usage a turn ceiling or the spend gate
counts, since a cut stream reports none. Hermetic.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.budget import IterationBudget
from zakcode.agent.loop import _LOOP_PROSE_CAP, AgentLoop
from zakcode.config import load_settings
from zakcode.hooks import TurnEndPayload, TurnEndResult
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCallDelta,
    ToolCall,
)
from zakcode.providers.spend_gate import SpendGate, reset_gate_for_tests
from zakcode.session.say_inbox import say_path, write_say
from zakcode.session.store import Session
from zakcode.tasks import Task
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec

_CONTINUE = "Continue exactly where you left off."
_CARRY_ON = "Not a stop: carry on with the work."


@dataclass
class Reply:
    """One scripted model reply: thinking, then text in chunks, then tool calls."""

    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    finish: str = "stop"
    thinking: str = ""
    chunk: int = 50  # chars per text delta; 0 sends the whole text as one delta


class Scripted(Provider):
    """Streams (or returns) scripted replies in order, recording how far each stream got."""

    def __init__(self, *script: Reply) -> None:
        self._script = list(script)
        self.calls = 0
        self.text_deltas_sent: dict[int, int] = {}
        self.finished: dict[int, bool] = {}

    def _next(self) -> tuple[int, Reply]:
        n = self.calls
        self.calls += 1
        return n, self._script[min(n, len(self._script) - 1)]

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        _, r = self._next()
        return LLMResult(
            text=r.text, tool_calls=list(r.calls), finish_reason=r.finish, thinking=r.thinking
        )

    async def astream(
        self, messages, *, system=None, tools=None, **kw
    ) -> AsyncIterator[ProviderStreamEvent]:
        n, r = self._next()
        self.text_deltas_sent[n] = 0
        self.finished[n] = False
        for i in range(0, len(r.thinking), 50):
            yield StreamThinkingDelta(text=r.thinking[i : i + 50])
        step = r.chunk or max(len(r.text), 1)
        for i in range(0, len(r.text), step):
            self.text_deltas_sent[n] += 1
            yield StreamTextDelta(text=r.text[i : i + step])
        for i, call in enumerate(r.calls):
            yield StreamToolCallDelta(
                index=i, id=call.id, name=call.name, arguments_delta=json.dumps(call.arguments)
            )
        yield StreamDone(finish_reason=r.finish)
        self.finished[n] = True

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


class Probe(Tool):
    """A read that returns a line; ``say`` makes it leave a person's message in the inbox."""

    def __init__(self, name: str, say: tuple[Path, str] | None = None) -> None:
        self.spec = ToolSpec(name=name, description="Read something.")
        self._say = say

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if self._say is not None:
            write_say(*self._say)
        return ToolResult.ok(output="read 3 lines")


class CarryOn:
    """A turn-end hook that refuses the first ``n`` stops, as a loop framework's does, and
    keeps the reason of every stop it was asked about."""

    def __init__(self, n: int = 1) -> None:
        self.left = n
        self.asked: list[str] = []

    def __call__(self, payload: TurnEndPayload) -> TurnEndResult | None:
        self.asked.append(payload.stop_reason)
        if self.left <= 0:
            return None
        self.left -= 1
        return TurnEndResult(vetoed=True, continuation_prompt=_CARRY_ON)


@pytest.fixture(autouse=True)
def _no_spend_gate() -> Iterator[None]:
    reset_gate_for_tests(SpendGate())
    yield
    reset_gate_for_tests(None)


def _loop(
    provider: Provider,
    tmp_path: Path,
    *,
    loop_skill: str = "patrol loop",
    vetoes: int = 1,
    tools: list[Tool] | None = None,
    budget: IterationBudget | None = None,
    hook: CarryOn | None = None,
) -> AgentLoop:
    registry = ToolRegistry()
    for tool in tools or [Probe("probe"), Probe("next")]:
        registry.register(tool)
    session = Session(cwd=str(tmp_path), model="test/model")
    session.loop_skill = loop_skill
    loop = AgentLoop(
        provider,
        registry,
        session,
        settings=load_settings(workspace_root=tmp_path),
        max_iterations=20,
        turn_end_vetoable=True,
        consume_say_inbox=True,
        budget=budget,
    )
    loop.hook_manager.register_turn_end(hook or CarryOn(vetoes))
    return loop


def _stream(loop: AgentLoop, text: str = "begin") -> list[Any]:
    async def go() -> list[Any]:
        return [ev async for ev in loop.astream_turn(text)]

    return asyncio.run(go())


def _prose(n: int, tag: str = "step") -> str:
    """``n`` characters of prose that never repeats (so the degeneration probe stays quiet);
    a different ``tag`` gives different prose."""
    words = [f"{tag}{i} measured {i * 7 % 13} then {i * 3 % 11};" for i in range(n)]
    return " ".join(words)[:n]


def _call(name: str, cid: str = "c1") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments={})


#: The turn's first answer, which the hook refuses as a stop: from here the loop holds the floor.
_FIRST = Reply(text="Checked the queue.")


def _replies(loop: AgentLoop) -> list[Any]:
    return [m for m in loop.session.messages if m.role == "assistant"]


def _asked_to_continue(loop: AgentLoop) -> bool:
    return any(_CONTINUE in m.text for m in loop.session.messages if m.role == "user")


def _called(loop: AgentLoop) -> list[str]:
    return [m.tool_uses[0].name for m in _replies(loop) if m.tool_uses]


# ── the cut ──────────────────────────────────────────────────────────────────


def test_prose_is_cut_once_the_loop_holds_the_floor(tmp_path: Path) -> None:
    provider = Scripted(_FIRST, Reply(calls=[_call("probe")]), Reply(text=_prose(9000)))
    loop = _loop(provider, tmp_path)
    events = _stream(loop)
    reply = _replies(loop)[-1].text
    assert _LOOP_PROSE_CAP < len(reply) <= _LOOP_PROSE_CAP + 50
    assert provider.finished[2] is False  # the reply's stream was left, not read to its end
    assert provider.text_deltas_sent[2] < 9000 // 50
    assert events[-1].degraded
    assert events[-1].stop_reason == "completed"  # the turn ends on the text as written
    assert not _asked_to_continue(loop)
    statuses = [ev.message for ev in events if type(ev).__name__ == "AgentStatus"]
    assert any("prose reply cut" in s for s in statuses)


def test_prose_at_the_cap_is_kept_whole(tmp_path: Path) -> None:
    text = _prose(_LOOP_PROSE_CAP)
    loop = _loop(Scripted(_FIRST, Reply(text=text)), tmp_path)
    events = _stream(loop)
    assert _replies(loop)[-1].text == text
    statuses = [ev.message for ev in events if type(ev).__name__ == "AgentStatus"]
    assert not any("prose reply cut" in s for s in statuses)


def test_long_prose_before_a_tool_call_is_cut_with_its_call(tmp_path: Path) -> None:
    """The cut is decided on the prose alone: a call that would have come after more than the
    cap of prose is not waited for. The turn ends on the text; nothing ran."""
    loop = _loop(Scripted(_FIRST, Reply(text=_prose(4000), calls=[_call("next")])), tmp_path)
    _stream(loop)
    assert len(_replies(loop)[-1].text) <= _LOOP_PROSE_CAP + 50
    assert _called(loop) == []


# ── who holds the floor ──────────────────────────────────────────────────────


def test_the_first_answer_of_a_turn_is_not_cut(tmp_path: Path) -> None:
    """Whoever opened the turn may be waiting for its answer, after tool results too."""
    text = _prose(6000)
    loop = _loop(Scripted(Reply(calls=[_call("probe")]), Reply(text=text)), tmp_path, vetoes=0)
    _stream(loop, "please find out why the build failed")
    assert _replies(loop)[-1].text == text


def test_a_person_speaking_mid_task_takes_the_floor_back(tmp_path: Path) -> None:
    text = _prose(6000)
    poke = Probe("probe", say=(say_path(tmp_path), "what did you find? explain it to me"))
    loop = _loop(
        Scripted(_FIRST, Reply(calls=[_call("probe")]), Reply(text=text)),
        tmp_path,
        tools=[poke],
    )
    _stream(loop)
    assert any("arrived mid-task" in m.text for m in loop.session.messages if m.role == "user")
    assert _replies(loop)[-1].text == text


def test_a_veto_after_silence_does_not_take_the_floor(tmp_path: Path) -> None:
    """The hook refusing a stop the model never answered (silence here) leaves the person who
    opened the turn waiting, so their answer is not cut."""
    text = _prose(6000)
    silent = Reply(text="")
    hook = CarryOn(1)
    loop = _loop(Scripted(silent, silent, silent, Reply(text=text)), tmp_path, hook=hook)
    _stream(loop, "please explain why the build failed")
    assert hook.asked[0] == "gave_up"  # the refused stop was the silence
    assert _replies(loop)[-1].text == text


def test_a_silent_end_after_a_preface_does_not_take_the_floor(tmp_path: Path) -> None:
    """A turn whose last reply said nothing ends as completed when an earlier reply had text.
    It has still not answered whoever opened it."""
    text = _prose(6000)
    hook = CarryOn(1)
    provider = Scripted(
        Reply(text="let me look", calls=[_call("probe")]), Reply(text=""), Reply(text=text)
    )
    loop = _loop(provider, tmp_path, hook=hook)
    _stream(loop, "please explain why the build failed")
    assert hook.asked[0] == "completed"  # the refused stop was the empty end
    assert _replies(loop)[-1].text == text


def test_a_buffered_silent_end_after_a_preface_does_not_take_the_floor(tmp_path: Path) -> None:
    hook = CarryOn(1)
    provider = Scripted(
        Reply(text="let me look", calls=[_call("probe")]),
        Reply(text=""),
        Reply(text=_prose(800), finish="length"),
        Reply(text="the rest"),
    )
    loop = _loop(provider, tmp_path, hook=hook)
    asyncio.run(loop.arun_turn("please explain what you found"))
    assert hook.asked[0] == "completed"
    assert provider.calls == 4  # the cut-off answer was continued
    assert _asked_to_continue(loop)


def test_the_floor_holds_after_a_cut(tmp_path: Path) -> None:
    provider = Scripted(_FIRST, Reply(text=_prose(6000)), Reply(text=_prose(6000, "next")))
    loop = _loop(provider, tmp_path, vetoes=2)
    _stream(loop)
    cut = [m.text for m in _replies(loop)][1:]
    assert len(cut) == 2 and all(len(t) <= _LOOP_PROSE_CAP + 50 for t in cut)


def test_a_new_turn_starts_with_the_floor_back(tmp_path: Path) -> None:
    text = _prose(6000)
    provider = Scripted(_FIRST, Reply(text="done"), Reply(calls=[_call("probe")]), Reply(text=text))
    loop = _loop(provider, tmp_path)
    _stream(loop)  # the hook continues it once: the loop held the floor in this turn
    _stream(loop, "please explain what you found")
    assert _replies(loop)[-1].text == text


def test_a_new_buffered_turn_starts_with_the_floor_back(tmp_path: Path) -> None:
    provider = Scripted(
        _FIRST,
        Reply(text="done"),
        Reply(text=_prose(800), finish="length"),
        Reply(text="the rest"),
    )
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("begin"))
    asyncio.run(loop.arun_turn("please explain what you found"))
    assert provider.calls == 4  # the second turn's cut-off answer was continued
    assert _asked_to_continue(loop)


def test_outside_a_loop_session_nothing_is_cut(tmp_path: Path) -> None:
    text = _prose(6000)
    loop = _loop(Scripted(_FIRST, Reply(text=text)), tmp_path, loop_skill="")
    _stream(loop)
    assert _replies(loop)[-1].text == text


def _planned(loop: AgentLoop, status: str = "in_progress") -> AgentLoop:
    """Give the session a plan with a step still open, so the plan gate sends a reply back. A
    step in progress holds a person's message for its seam (ADR-0052); a pending one does not."""
    loop.session.task_network.replace_from_author([Task(title="file the report", status=status)])
    return loop


def test_a_reply_the_plan_gate_sends_back_passes_the_floor(tmp_path: Path) -> None:
    """The measured turn: a person asked, the answer reached the output limit and was continued,
    and the plan gate, not the hook, sent each reply back. The continued answer is still the
    first answer and stays whole, past the cap too; its short last piece still ends a full
    answer, and the replies after it are the loop's."""
    piece = _prose(3000, "rest")
    provider = Scripted(
        Reply(text=_prose(800), finish="length"),
        Reply(text=piece, finish="length"),
        Reply(text="the end."),
        Reply(text=_prose(9000, "x")),
        Reply(text=_prose(9000, "y")),
    )
    hook = CarryOn(0)
    loop = _planned(_loop(provider, tmp_path, hook=hook))
    events = _stream(loop, "please explain what you found")
    texts = [m.text for m in _replies(loop)]
    assert texts[1] == piece
    assert len(texts) == 5 and all(len(t) <= _LOOP_PROSE_CAP + 50 for t in texts[3:])
    assert hook.asked == ["completed"]  # the hook refused no stop: the gate sent them back
    assert events[-1].stop_reason == "completed"


def test_a_buffered_reply_the_plan_gate_sends_back_passes_the_floor(tmp_path: Path) -> None:
    """Buffered, the floor decides whether a reply cut off at the output limit is continued: the
    first answer is, each time; the loop's prose after the gate sent the answer back is not."""
    provider = Scripted(
        Reply(text=_prose(800), finish="length"),
        Reply(text=_prose(800, "rest"), finish="length"),
        Reply(text="That is all I found."),
        Reply(text=_prose(800, "x"), finish="length"),
        Reply(text=_prose(800, "y"), finish="length"),
    )
    loop = _planned(_loop(provider, tmp_path, hook=CarryOn(0)))
    result = asyncio.run(loop.arun_turn("please explain what you found"))
    asked = [m for m in loop.session.messages if m.role == "user" and _CONTINUE in m.text]
    assert len(asked) == 2  # the answer, twice; the loop's prose never
    assert provider.calls == 5  # the answer in three parts, then the gate's two send-backs
    assert result.stop_reason == "completed"


def test_a_short_reply_a_gate_sends_back_leaves_the_floor_with_the_person(tmp_path: Path) -> None:
    """A preface with no tool call, sent back by the plan gate, answered no one: the answer that
    follows the work stays whole. Once that answer tries to end the turn, the loop has the floor."""
    answer = _prose(6000, "rest")
    provider = Scripted(
        Reply(text="I will start by looking."),
        Reply(calls=[_call("probe")]),
        Reply(text=answer),
        Reply(text=_prose(9000, "x")),
    )
    loop = _planned(_loop(provider, tmp_path, hook=CarryOn(0)))
    _stream(loop, "please explain what you found")
    texts = [m.text for m in _replies(loop)]
    assert texts[2] == answer
    assert len(texts) == 4 and len(texts[3]) <= _LOOP_PROSE_CAP + 50


def test_a_buffered_short_reply_a_gate_sends_back_leaves_the_floor_with_the_person(
    tmp_path: Path,
) -> None:
    provider = Scripted(
        Reply(text="I will start by looking."),
        Reply(calls=[_call("probe")]),
        Reply(text=_prose(800), finish="length"),
        Reply(text="the rest"),
        Reply(text=_prose(800, "x"), finish="length"),
        Reply(text=_prose(800, "y"), finish="length"),
    )
    loop = _planned(_loop(provider, tmp_path, hook=CarryOn(0)))
    asyncio.run(loop.arun_turn("please explain what you found"))
    asked = [m for m in loop.session.messages if m.role == "user" and _CONTINUE in m.text]
    assert len(asked) == 1  # the answer was continued; the loop's prose after it was not
    assert provider.calls == 5


def test_a_buffered_long_answer_a_gate_sends_back_passes_the_floor(tmp_path: Path) -> None:
    provider = Scripted(
        Reply(text=_prose(3000)),
        Reply(text=_prose(800, "x"), finish="length"),
        Reply(text=_prose(800, "y"), finish="length"),
    )
    loop = _planned(_loop(provider, tmp_path, hook=CarryOn(0)))
    asyncio.run(loop.arun_turn("please explain what you found"))
    assert not _asked_to_continue(loop)


def _speaks(tmp_path: Path) -> Probe:
    return Probe("probe", say=(say_path(tmp_path), "what did you find? explain it to me"))


def test_after_a_person_speaks_a_short_reply_a_gate_sends_back_is_a_preface_again(
    tmp_path: Path,
) -> None:
    """The loop had the floor from a continued answer; a person speaks mid-task, and a short
    reply the gate sends back after that is a preface again: their answer stays whole."""
    answer = _prose(6000, "again")
    provider = Scripted(
        Reply(text=_prose(800), finish="length"),
        Reply(text="the end."),
        Reply(calls=[_call("probe")]),
        Reply(text="Let me look again."),
        Reply(text=answer),
    )
    loop = _planned(
        _loop(provider, tmp_path, hook=CarryOn(0), tools=[_speaks(tmp_path)]), "pending"
    )
    _stream(loop, "please explain what you found")
    assert _replies(loop)[-1].text == answer


def test_a_buffered_short_reply_after_a_person_speaks_is_a_preface_again(tmp_path: Path) -> None:
    provider = Scripted(
        Reply(text=_prose(800), finish="length"),
        Reply(text="the end."),
        Reply(calls=[_call("probe")]),
        Reply(text="Let me look again."),
        Reply(text=_prose(800, "again"), finish="length"),
        Reply(text="the rest"),
    )
    loop = _planned(
        _loop(provider, tmp_path, hook=CarryOn(0), tools=[_speaks(tmp_path)]), "pending"
    )
    asyncio.run(loop.arun_turn("please explain what you found"))
    asked = [m for m in loop.session.messages if m.role == "user" and _CONTINUE in m.text]
    assert len(asked) == 2  # the first answer, and the answer after the person spoke


# ── what keeps streaming ─────────────────────────────────────────────────────


def test_a_short_preface_then_a_tool_call_is_untouched(tmp_path: Path) -> None:
    provider = Scripted(
        _FIRST, Reply(text=_prose(300), calls=[_call("next", "c2")]), Reply(text="done")
    )
    loop = _loop(provider, tmp_path)
    _stream(loop)
    assert _called(loop) == ["next"]


def test_a_tool_call_that_started_keeps_the_stream(tmp_path: Path) -> None:
    """Text after a native tool call has started does not count toward a cut."""

    class CallFirst(Scripted):
        async def astream(self, messages, *, system=None, tools=None, **kw):
            n, _ = self._next()
            if n == 0:
                yield StreamTextDelta(text=_FIRST.text)
                yield StreamDone(finish_reason="stop")
                return
            if n == 1:
                yield StreamToolCallDelta(index=0, id="c1", name="probe", arguments_delta="{}")
                for i in range(0, 6000, 50):
                    yield StreamTextDelta(text=_prose(6000)[i : i + 50])
                yield StreamDone(finish_reason="tool_calls")
                return
            yield StreamTextDelta(text="done")
            yield StreamDone(finish_reason="stop")

    loop = _loop(CallFirst(Reply()), tmp_path)
    _stream(loop)
    calling = [m for m in _replies(loop) if m.tool_uses]
    assert len(calling) == 1 and len(calling[0].text) == 6000


def test_thinking_does_not_count(tmp_path: Path) -> None:
    loop = _loop(Scripted(_FIRST, Reply(text=_prose(400), thinking=_prose(20000))), tmp_path)
    _stream(loop)
    assert _replies(loop)[-1].text == _prose(400)


def test_a_reply_that_arrives_whole_is_not_cut(tmp_path: Path) -> None:
    """A provider that replays a finished result sends its text as one delta and its calls
    after it. Nothing is left to save, and a cut would drop the calls."""
    provider = Scripted(
        _FIRST,
        Reply(text=_prose(5000), calls=[_call("next", "c2")], chunk=0),
        Reply(text="done"),
    )
    loop = _loop(provider, tmp_path)
    _stream(loop)
    assert _called(loop) == ["next"]


def test_a_reply_writing_a_text_tool_call_is_not_cut(tmp_path: Path) -> None:
    body = json.dumps({"name": "write", "arguments": {"content": _prose(5000)}})
    text = f"Writing it now.\n<tool_call>\n{body}\n</tool_call>"
    loop = _loop(Scripted(_FIRST, Reply(text=text)), tmp_path)
    _stream(loop)
    assert _replies(loop)[-1].text == text


def test_a_turn_ceiling_keeps_the_stream(tmp_path: Path) -> None:
    """A ceiling reads every call's usage, and a cut stream reports none."""
    text = _prose(6000)
    budget = IterationBudget(50, max_tokens=10**9)
    loop = _loop(Scripted(_FIRST, Reply(text=text)), tmp_path, budget=budget)
    _stream(loop)
    assert _replies(loop)[-1].text == text


def test_an_armed_spend_gate_keeps_the_stream(tmp_path: Path) -> None:
    gate = SpendGate()
    gate.set_budget(10.0, 1.0, 0.0)
    reset_gate_for_tests(gate)
    text = _prose(6000)
    loop = _loop(Scripted(_FIRST, Reply(text=text)), tmp_path)
    _stream(loop)
    assert _replies(loop)[-1].text == text


# ── the output limit ─────────────────────────────────────────────────────────


def test_prose_cut_off_at_the_output_limit_is_not_continued(tmp_path: Path) -> None:
    provider = Scripted(_FIRST, Reply(text=_prose(800), finish="length"), Reply(text="more"))
    loop = _loop(provider, tmp_path)
    events = _stream(loop)
    assert provider.calls == 2  # the first answer and the cut-off reply; no continuation
    assert not _asked_to_continue(loop)
    assert events[-1].degraded


def test_buffered_prose_cut_off_at_the_output_limit_is_not_continued(tmp_path: Path) -> None:
    provider = Scripted(_FIRST, Reply(text=_prose(800), finish="length"), Reply(text="more"))
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("begin"))
    assert provider.calls == 2
    assert not _asked_to_continue(loop)
    assert result.degraded


def test_a_cut_off_first_answer_is_still_continued(tmp_path: Path) -> None:
    provider = Scripted(Reply(text=_prose(800), finish="length"), Reply(text="the rest"))
    loop = _loop(provider, tmp_path, vetoes=0)
    _stream(loop, "please explain what you found")
    assert provider.calls == 2
    assert _asked_to_continue(loop)

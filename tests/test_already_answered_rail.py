"""ADR-0217: the call that exists only to hand back a tool result is told the answer already
stands, so the model stops instead of saying it again.

The shape, measured 2026-09-22 on a served Mind and reported by the user from their terminal.
Inside ONE turn: the model wrote a full verdict, called a tool (a bare ``echo``, made to satisfy
a framework rule that every turn end be a tool call), and then — asked again because a tool had
run and its result must go back — wrote the same verdict a second time. The user read the same
answer twice and paid for the call that produced it.

Nothing could catch it after the fact. The repeat is the model's own prose, and the stuck ladder
compares tool OUTPUTS: it is handed each iteration's assistant text and uses it in one place,
only to ask whether it is empty. And by the time the second answer could be recognised as a
repeat it is already streaming onto the screen, where it cannot be unsaid. So this acts BEFORE,
on the one fact that is certain at the time: the previous completion had already answered.

WHAT IS PINNED HERE IS THE WIRE, NEVER THE SOURCE (guard-6333). Every assertion reads the rails
the loop recorded for each real request, so a test cannot pass because a constant exists.

THE CONTROL IS THE SECOND TEST and it is the point of the file: the line must ride ONLY on the
call that follows an answer. A rail that rides every call would pass every "does it ride?" test
ever written while quietly costing tokens on every request of every turn.

Hermetic: a scripted provider, a real read of a file in tmp_path, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import (
    _SUBSTANTIAL_ANSWER,
    AgentLoop,
    _answered_then_called_a_tool,
)
from zakcode.messages import Message, TextBlock, ToolUseBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools import default_registry
from zakcode.usage import Usage

MODEL = "fake/scripted"
#: An ANSWER: the shape sera repeated, padded to the length that separates an answer from
#: narration. Its content is irrelevant; its LENGTH is the thing under test.
ANSWER = (
    "The loop will not start because the agent is IDLE. This is the designed behavior: the "
    "graceful stop completed, the state is IDLE and the mode is assistant, and loop re-entry "
    "is blocked whenever the state is not RUNNING. "
) * 2
#: NARRATION: what a model says on its way to doing something. Below the line, so no rail.
NARRATION = "Reading the state file now."


class _Scripted(Provider):
    """Hands back scripted results in order; the last one repeats."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(self, messages: Any, *, system: Any = None, tools: Any = None, **kw: Any):  # type: ignore[override]
        index = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[index]

    def model_id(self) -> str:
        return MODEL

    def count_tokens(self, messages: Any, *, system: Any = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _read(call_id: str, path: Path) -> ToolCall:
    return ToolCall(id=call_id, name="read_file", arguments={"path": str(path)})


def _said(text: str, call: ToolCall | None = None) -> LLMResult:
    return LLMResult(text=text, tool_calls=[call] if call else [], usage=Usage(total_tokens=1))


def _loop(provider: Provider, tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        provider,
        default_registry(),
        Session(cwd=str(tmp_path), model=MODEL),
        workspace_root=tmp_path,
        max_iterations=10,
    )


def _rails(loop: AgentLoop) -> list[list[str]]:
    """The ephemeral rails the loop recorded for each real request, in order — the WIRE."""
    return [event.data["rails"] for event in loop._trace.of_kind("usage")]


async def _run(loop: AgentLoop, text: str, *, streamed: bool) -> str:
    if not streamed:
        return (await loop.arun_turn(text)).stop_reason
    async for _ in loop.astream_turn(text):
        pass
    return str(loop.session.last_stop_reason)


@pytest.fixture
def target(tmp_path: Path) -> Path:
    path = tmp_path / "state.txt"
    path.write_text("IDLE\n", encoding="utf-8")
    return path


# ── the rule ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_the_line_rides_the_call_that_hands_back_a_tool_result_after_an_answer(
    tmp_path: Path, target: Path, streamed: bool
) -> None:
    provider = _Scripted([_said(ANSWER, _read("c1", target)), _said("done")])
    loop = _loop(provider, tmp_path)
    assert await _run(loop, "what is the state?", streamed=streamed) == "completed"

    rails = _rails(loop)
    assert len(rails) == 2, rails
    assert "already_answered" not in rails[0]  # nothing had been answered yet
    assert "already_answered" in rails[1]  # the call that exists only to hand back the result


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_turn_that_only_narrated_before_its_tool_call_never_pays_for_the_line(
    tmp_path: Path, target: Path, streamed: bool
) -> None:
    """THE CONTROL. One thing differs from the test above: the length of what was said first.

    A model narrating its way to a tool call has answered nothing, so there is nothing for the
    line to be about, and a turn like this must carry it on no call at all. Without this the
    suite would pass just as well for a rail that rides unconditionally.
    """
    provider = _Scripted([_said(NARRATION, _read("c1", target)), _said(ANSWER)])
    loop = _loop(provider, tmp_path)
    assert await _run(loop, "what is the state?", streamed=streamed) == "completed"

    rails = _rails(loop)
    assert len(rails) == 2, rails
    assert all("already_answered" not in call for call in rails), rails


@pytest.mark.asyncio
async def test_the_line_stops_riding_once_the_model_moves_on(tmp_path: Path, target: Path) -> None:
    """It is recomputed from every completion, so it can never be stale by an iteration: an
    answer, then a tool, then a SHORT completion that calls another tool, and the line is gone."""
    provider = _Scripted(
        [
            _said(ANSWER, _read("c1", target)),  # call 1: answers and calls a tool
            _said(NARRATION, _read("c2", target)),  # call 2: carries the line, moves on
            _said("done"),  # call 3: must NOT carry it
        ]
    )
    loop = _loop(provider, tmp_path)
    assert await _run(loop, "what is the state?", streamed=False) == "completed"

    rails = _rails(loop)
    assert len(rails) == 3, rails
    assert "already_answered" not in rails[0]
    assert "already_answered" in rails[1]
    assert "already_answered" not in rails[2]


@pytest.mark.asyncio
async def test_an_answer_with_no_tool_call_leaves_nothing_behind_for_the_next_turn(
    tmp_path: Path, target: Path
) -> None:
    """An answer alone ENDS a turn, so no call follows it to be told anything. What is pinned
    is that it also leaves nothing armed for the NEXT turn, which a person opened."""
    provider = _Scripted([_said(ANSWER), _said(ANSWER, _read("c1", target)), _said("done")])
    loop = _loop(provider, tmp_path)

    assert await _run(loop, "what is the state?", streamed=False) == "completed"
    assert loop._answer_already_given is False

    assert await _run(loop, "and now?", streamed=False) == "completed"
    rails = _rails(loop)
    assert "already_answered" not in rails[0], rails  # the second turn's first call
    assert "already_answered" in rails[-1], rails  # and its own answer-then-tool pair


@pytest.mark.asyncio
async def test_a_turn_cut_off_mid_answer_leaves_nothing_armed_for_the_next_one(
    tmp_path: Path, target: Path
) -> None:
    """The only way a turn ENDS with an answer standing, and the reason the flag is cleared at
    every turn start rather than only when it stops being true.

    Normally a turn ends on a completion with no tool call, which leaves the flag false on its
    own. But a turn cut off by its iteration budget ends right after a completion that answered
    AND called a tool — so the flag is still set when the next turn begins, and without the
    reset a person's fresh question would be answered under a line about an answer they never saw.
    """
    provider = _Scripted([_said(ANSWER, _read("c1", target)), _said("done")])
    loop = _loop(provider, tmp_path)
    loop.max_iterations = 1  # the budget ends the turn immediately after the first completion

    assert await _run(loop, "what is the state?", streamed=False) == "max_iterations"
    # The hazard is REAL and asserted before the remedy: the turn ended with an answer standing.
    assert loop._answer_already_given is True

    loop.max_iterations = 10
    assert await _run(loop, "a fresh question", streamed=False) == "completed"
    # The trace is per-turn, so index 0 here is the FIRST call of the second turn.
    assert "already_answered" not in _rails(loop)[0], _rails(loop)


def test_an_answer_alone_is_not_an_answer_followed_by_a_tool() -> None:
    """Both halves of the predicate, asserted on the predicate itself.

    An answer with no tool call ENDS the turn, so no call follows it to be told anything, and
    nothing on the wire can distinguish a predicate that requires the tool call from one that
    does not. Read directly, then, rather than left to an integration test that cannot reach it.
    """

    def built(text: str, *, with_tool: bool) -> Message:
        """An assistant message in the same shape the loop stores one, built here rather than
        driven through a turn: an answer with no tool call ends a turn, so no run can produce
        the middle case at a moment where anything reads it."""
        blocks: list[Any] = [TextBlock(text=text)]
        if with_tool:
            blocks.append(ToolUseBlock(id="c1", name="read_file", arguments={"path": "x"}))
        return Message(role="assistant", blocks=blocks)

    answer_and_tool = built(ANSWER, with_tool=True)
    answer_alone = built(ANSWER, with_tool=False)
    narration_and_tool = built(NARRATION, with_tool=True)

    assert _answered_then_called_a_tool(answer_and_tool) is True
    assert _answered_then_called_a_tool(answer_alone) is False  # no tool ran; no call follows
    assert _answered_then_called_a_tool(narration_and_tool) is False  # nothing was answered


def test_the_line_between_an_answer_and_narration_is_a_length_and_it_is_documented() -> None:
    """The threshold is the whole predicate, so it is asserted rather than left implicit."""
    assert len(NARRATION) < _SUBSTANTIAL_ANSWER <= len(ANSWER.strip())

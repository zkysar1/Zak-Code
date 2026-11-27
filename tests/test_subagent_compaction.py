"""A delegated sub-agent whose context fills answers its parent instead of dying at the window.

Compaction used to stop at the delegation seam. The loop a sub-agent runs on was built with no
compactor, so a child whose transcript outgrew the model's window had no way to recover: the
overflow ended its turn as ``provider_error`` with ``(recovery: no compactor; 0/2 attempts)``.
Measured on a self-hosted worker, a sub-agent's prompt grew from 22.5k to 130.6k tokens over
three hours of reading and died at the 131,072-token window, and the parent got a stop back
instead of the child's answer.

The child then got the parent's compactor and was summarized like the parent, and read on.
Sub-agents that compacted ran on for up to 172 minutes more while their parents waited, so a
child is no longer summarized at the threshold (ADR-0412): it is asked for its report, and its
turn ends with that reply. When the loop cannot see the size coming, the in-call overflow
recovery compacts once, which makes room for that one reply, and the child is asked then.

These tests drive the real seam end to end. ``Agent(enable_subagents=True,
enable_compaction=...)`` runs the parent loop, and its ``task`` call goes through
``SubAgentManager`` and ``SubAgentRunner`` into a real child loop. Only the model is scripted. It
stands in for a server with a small window and refuses any request larger than that window, the
way a llama.cpp server does. The child reads one large file per step until its reads add up to
about twice the window, then answers. With compaction off the child dies at the window, which is
the control showing the scenario really overflows. With compaction on the child reports when its
context fills, and a parent that reads the same files itself compacts and reads on. A sub-agent
runs on the buffered loop and the streaming loop carries the same stop, so the twin tests drive
both loops directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zakcode import Agent
from zakcode.agent.budget import IterationBudget
from zakcode.agent.compact import SUMMARY_MARKER, Compactor
from zakcode.agent.loop import (
    _CONTEXT_FULL_NOT_RUN,
    _CONTEXT_FULL_NOT_RUN_REASK,
    _CONTEXT_FULL_RAIL,
    _CONTEXT_FULL_REASK,
    AgentLoop,
)
from zakcode.agent.subagent import _HANDOFF
from zakcode.config import Settings
from zakcode.events import AgentDone, AgentStatus
from zakcode.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import (
    Capabilities,
    ContextWindowExceeded,
    LLMResult,
    Provider,
    ToolCall,
)
from zakcode.providers.routing import thinking_extra_body
from zakcode.session.store import Session
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.tools.builtins.task import CONTEXT_FULL_NO_REPORT_NOTE, CONTEXT_FULL_NOTE
from zakcode.usage import Usage

#: The scripted server's context window, in tokens. Small, so a few reads overflow it.
WINDOW = 8192
#: Files the child reads before it answers. The loop clamps each result to a quarter of the
#: window, so the eight together come to about twice what the window holds.
READS = 8
#: Long enough that the summarizer's minimum-length check never rejects it.
_SUMMARY = "The sub-agent is reading the numbered files one at a time and has not finished. " * 8


def _tokens(messages: list[Message]) -> int:
    """The scripted server's size for a request: every character it carries, 3 to a token."""
    chars = 0
    for message in messages:
        for block in message.blocks:
            if isinstance(block, TextBlock):
                chars += len(block.text)
            elif isinstance(block, ToolResultBlock):
                chars += len(block.output)
            elif isinstance(block, ToolUseBlock):
                chars += len(json.dumps(block.input))
    return chars // 3


class _WindowedModel(Provider):
    """One small-window model for the parent, its child and the compaction summarizer.

    The parent delegates once and wraps up when the result comes back. A request is the child's
    when its system prompt carries the handoff instruction every sub-agent gets. The summarizer
    is recognised by its own instruction.

    With ``reports_usage=False`` the model reports no prompt size and its estimator counts
    nothing, so the loop never sees the transcript grow. Then only the server's refusal can
    start a compaction, which tests the in-turn overflow recovery on its own.

    A child asked for its report (ADR-0412) gives one, and with ``report_calls_tool=True`` it
    also asks for a Write in that reply, which must not run. With ``parent_reads=True`` the
    parent does the reads itself instead of delegating: the parent's own loop is not a
    sub-agent, so it compacts and reads on. With ``as_child=True`` every request but the
    summarizer's is the child's, for a loop built directly with no handoff in its prompt.

    ``report_shapes`` scripts the replies to the requests that carry the ask, in order, the last
    one repeating: ``"text"`` is the report, ``"empty"`` a reply with nothing in it (the shape
    of thinking that took the room the window left), ``"blank"`` one with only whitespace,
    ``"overflow"`` the same cut at a cap the window held down, and ``"tool"`` a Write with no
    text. With ``plans_first=True`` the child's
    first act is a plan, so the loop has a plan to remind it of.
    """

    def __init__(
        self,
        *,
        reports_usage: bool = True,
        report_calls_tool: bool = False,
        parent_reads: bool = False,
        as_child: bool = False,
        report_shapes: tuple[str, ...] = ("text",),
        plans_first: bool = False,
    ) -> None:
        self.reports_usage = reports_usage
        self.report_calls_tool = report_calls_tool
        self.parent_reads = parent_reads
        self.as_child = as_child
        self.report_shapes = report_shapes
        self.plans_first = plans_first
        self.reads = 0
        self.refusals = 0
        self.refusals_after_ask = 0  # refused requests that already carried the ask
        self.summaries = 0
        self.reports = 0
        self.rails_in_report_request = 0
        self.report_kw: list[dict[str, Any]] = []  # the call options of each answered ask
        self.ask_last: list[bool] = []  # whether each answered ask was the request's last message
        self.summaries_at_ask: list[int] = []  # summaries made before each answered ask
        self.child_saw_summary = False
        self.parent_saw_summary = False

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> LLMResult:
        system = system or ""
        if system.startswith("You are compacting"):
            self.summaries += 1
            return LLMResult(text=f"<summary>{_SUMMARY}</summary>", usage=Usage(total_tokens=1))
        size = _tokens(messages)
        if size > WINDOW:
            self.refusals += 1
            if any(_CONTEXT_FULL_RAIL in m.text for m in messages):
                self.refusals_after_ask += 1
            raise ContextWindowExceeded(
                f"request ({size} tokens) exceeds the available context size ({WINDOW} tokens)"
            )
        usage = Usage(prompt_tokens=size if self.reports_usage else 0, total_tokens=size + 1)
        if _HANDOFF not in system and not self.as_child:
            if self.parent_reads:
                if any(SUMMARY_MARKER in m.text for m in messages):
                    self.parent_saw_summary = True
                return self._read_or(usage, answer="PARENT-RESULT")
            if any(m.role == "tool" for m in messages):
                return LLMResult(text="PARENT-SYNTHESIS", usage=usage)
            call = ToolCall(
                id="task1", name="task", arguments={"tasks": [{"prompt": "Read the files."}]}
            )
            return LLMResult(tool_calls=[call], usage=usage)
        if any(SUMMARY_MARKER in m.text for m in messages):
            self.child_saw_summary = True
        rails = sum(_CONTEXT_FULL_RAIL in m.text for m in messages)
        if rails:
            shape = self.report_shapes[min(self.reports, len(self.report_shapes) - 1)]
            self.reports += 1
            self.rails_in_report_request = rails
            self.report_kw.append(dict(kw))
            asks = (_CONTEXT_FULL_RAIL, _CONTEXT_FULL_REASK)
            self.ask_last.append(any(a in messages[-1].text for a in asks))
            self.summaries_at_ask.append(self.summaries)
            if shape == "empty":
                return LLMResult(text="", usage=usage)
            if shape == "blank":
                return LLMResult(text=" \n", usage=usage)
            if shape == "overflow":
                return LLMResult(
                    thinking="still thinking",
                    finish_reason="length",
                    clamped_output_cap=64,
                    usage=usage,
                )
            write = {"path": "late.txt", "content": "x"}
            if shape == "tool":
                return LLMResult(
                    tool_calls=[ToolCall(id="late-tool", name="Write", arguments=write)],
                    usage=usage,
                )
            late = ToolCall(id="late", name="Write", arguments=write)
            calls = [late] if self.report_calls_tool else []
            return LLMResult(text="CHILD-REPORT", tool_calls=calls, usage=usage)
        if self.plans_first and not any(m.role == "tool" for m in messages):
            steps = [{"title": "Read the numbered files"}, {"title": "Report what they say"}]
            plan = ToolCall(id="plan", name="update_plan", arguments={"tasks": steps})
            return LLMResult(tool_calls=[plan], usage=usage)
        return self._read_or(usage, answer="CHILD-RESULT")

    def _read_or(self, usage: Usage, *, answer: str) -> LLMResult:
        """The next read while any is left, then the answer."""
        if self.reads < READS:
            self.reads += 1
            call = ToolCall(
                id=f"read{self.reads}", name="Read", arguments={"path": f"part{self.reads}.txt"}
            )
            return LLMResult(tool_calls=[call], usage=usage)
        return LLMResult(text=answer, usage=usage)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        # Characters over four, under the server's count the way a real estimate runs under it.
        return _tokens(messages) * 3 // 4 if self.reports_usage else 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=WINDOW)


def _settings(tmp_path: Path) -> Settings:
    """The files to read, and settings over them."""
    for i in range(1, READS + 1):
        # 24,000 characters a file, so every read comes back at the loop's clamp.
        (tmp_path / f"part{i}.txt").write_text(f"part {i}: {'x' * 50}\n" * 400)
    return Settings(
        default_model="scripted/test",
        context_window=WINDOW,
        workspace_root=tmp_path,
        permission_mode="allow",
    )


def _agent(tmp_path: Path, model: _WindowedModel, *, compaction: bool) -> Agent:
    """The files to read, and a parent agent over the scripted model."""
    return Agent(
        settings=_settings(tmp_path),
        provider=model,
        enable_subagents=True,
        enable_compaction=compaction,
    )


async def _delegate(tmp_path: Path, model: _WindowedModel, *, compaction: bool) -> str:
    """Run one parent turn that delegates the reads, and return the task tool's output."""
    agent = _agent(tmp_path, model, compaction=compaction)
    result = await agent.arun_turn("DELEGATE")
    assert result.stop_reason == "completed"
    assert any(m.text == "PARENT-SYNTHESIS" for m in result.assistant_messages)
    (output,) = [tr.output for tr in result.tool_results]
    return output


async def test_without_compaction_the_sub_agent_dies_at_the_window(tmp_path: Path) -> None:
    # The control: the same delegation with compaction off overflows and has no recovery.
    model = _WindowedModel()
    output = await _delegate(tmp_path, model, compaction=False)
    assert "(stopped: provider_error)" in output
    assert "CHILD-RESULT" not in output
    assert model.refusals == 1
    assert model.summaries == 0
    assert model.reads < READS


@pytest.mark.parametrize("reports_usage", [True, False], ids=["threshold", "overflow-recovery"])
async def test_a_sub_agent_reports_when_its_context_fills(
    tmp_path: Path, reports_usage: bool
) -> None:
    model = _WindowedModel(reports_usage=reports_usage)
    output = await _delegate(tmp_path, model, compaction=True)
    # The child's report came back as its answer, from where its context filled: not a stop
    # with no answer, and not the end of reads it never got to.
    assert output == f"[task 1] general-purpose (stopped: context_full; {CONTEXT_FULL_NOTE}):\n" + (
        "CHILD-REPORT"
    )
    assert 0 < model.reads < READS
    # It was asked once, and that request was its last.
    assert model.reports == 1
    assert model.rails_in_report_request == 1
    if reports_usage:
        # The loop saw the transcript grow and asked before the server refused anything: the
        # threshold check asked, not a summary. Its estimate runs under this server's count and
        # an 8k window leaves little room past the threshold, so the ask itself may be refused
        # once; the overflow recovery then makes room for the report, and nothing else.
        assert model.refusals == model.refusals_after_ask <= 1
        assert model.summaries == model.refusals
    else:
        # Blind to the size, the child ran into the window. The recovery compacted to make
        # room, and the retried call carried the summary and the ask.
        assert model.refusals >= 1
        assert model.refusals_after_ask == 0
        assert model.summaries >= 1
        assert model.child_saw_summary


async def test_a_tool_called_in_the_report_does_not_run(tmp_path: Path) -> None:
    model = _WindowedModel(report_calls_tool=True)
    output = await _delegate(tmp_path, model, compaction=True)
    assert output.endswith("):\nCHILD-REPORT")
    assert model.reports == 1
    assert not (tmp_path / "late.txt").exists()


async def test_a_parent_loop_is_summarized_and_reads_on(tmp_path: Path) -> None:
    # The control for the rule's scope: the parent's own loop is no sub-agent. Reading the same
    # files itself, it is summarized at the threshold and finishes every read.
    model = _WindowedModel(parent_reads=True)
    result = await _agent(tmp_path, model, compaction=True).arun_turn("READ")
    assert result.stop_reason == "completed"
    assert any(m.text == "PARENT-RESULT" for m in result.assistant_messages)
    assert model.reads == READS
    assert model.summaries >= 1
    assert model.parent_saw_summary
    assert model.reports == 0


async def _run_reporting_loop(
    tmp_path: Path,
    model: _WindowedModel,
    *,
    streams: bool,
    expect: str = "context_full",
    events: list[object] | None = None,
) -> AgentLoop:
    """Run one turn of a loop built directly to report at the window; it must end ``expect``.
    A streamed turn's events are added to ``events`` when one is given."""
    settings = _settings(tmp_path)
    loop = AgentLoop(
        model,
        default_registry(settings),
        Session(cwd=str(tmp_path), model="scripted/test"),
        settings=settings,
        workspace_root=tmp_path,
        compactor=Compactor(),
        report_at_window=True,
    )
    if streams:
        seen = [ev async for ev in loop.astream_turn("READ")]
        if events is not None:
            events.extend(seen)
        (done,) = [ev for ev in seen if isinstance(ev, AgentDone)]
        stop_reason = done.stop_reason
    else:
        stop_reason = (await loop.arun_turn("READ")).stop_reason
    assert stop_reason == expect
    return loop


@pytest.mark.parametrize("streams", [False, True], ids=["buffered", "streaming"])
async def test_a_loop_that_reports_at_the_window_ends_its_turn_on_the_report(
    tmp_path: Path, streams: bool
) -> None:
    # A sub-agent runs on the buffered loop, and the streaming twin carries the same stop. Either
    # loop, reporting at the window, reads until its context fills, is asked once, and its turn
    # ends with that reply. The Write called in the reply does not run, and that call is answered
    # in the session, so the transcript stays resumable.
    model = _WindowedModel(report_calls_tool=True, as_child=True)
    loop = await _run_reporting_loop(tmp_path, model, streams=streams)
    assert 0 < model.reads < READS
    assert model.reports == 1
    assert model.rails_in_report_request == 1
    assert model.refusals == model.refusals_after_ask <= 1
    assert not (tmp_path / "late.txt").exists()
    (answer,) = loop.session.messages[-1].blocks
    assert isinstance(answer, ToolResultBlock)
    assert (answer.tool_use_id, answer.output) == ("late", _CONTEXT_FULL_NOT_RUN)


@pytest.mark.parametrize("first", ["empty", "tool"])
async def test_a_reply_with_no_report_is_asked_for_once_more(tmp_path: Path, first: str) -> None:
    # Thinking can take all the room the window leaves, and a model can answer the ask with a
    # tool call. Neither reply is a report: the child is asked once more, with thinking off, and
    # its reply to that is its answer. The Write it called does not run.
    model = _WindowedModel(report_shapes=(first, "text"))
    output = await _delegate(tmp_path, model, compaction=True)
    assert output.startswith(
        f"[task 1] general-purpose (stopped: context_full; {CONTEXT_FULL_NOTE}):"
    )
    assert output.endswith("CHILD-REPORT")
    assert model.reports == 2
    assert model.report_kw[0].get("extra_body") is None
    assert model.report_kw[1].get("extra_body") == thinking_extra_body(False)
    assert not (tmp_path / "late.txt").exists()


@pytest.mark.parametrize("shape", ["empty", "tool"])
async def test_the_second_ask_ends_the_turn_whatever_the_reply_holds(
    tmp_path: Path, shape: str
) -> None:
    # The second ask is the last: a child that answers it with nothing, or with a tool call again,
    # still ends its turn, so the parent gets the turn back instead of a loop of asks. The parent
    # is told the child gave no report, not handed what it last read as if it were one.
    model = _WindowedModel(report_shapes=(shape,))
    output = await _delegate(tmp_path, model, compaction=True)
    assert output.startswith(
        "[task 1] general-purpose "
        f"(stopped: context_full_no_report; {CONTEXT_FULL_NO_REPORT_NOTE}):"
    )
    assert "CHILD-REPORT" not in output
    assert model.reports == 2
    assert not (tmp_path / "late.txt").exists()


async def test_a_child_with_a_plan_reads_the_ask_last(tmp_path: Path) -> None:
    # A child with an open plan is reminded of it on every call, after the conversation. Once it
    # is asked for its report, nothing follows the ask: a reminder to advance the plan would
    # outrank it on the call that ends the turn.
    model = _WindowedModel(plans_first=True)
    output = await _delegate(tmp_path, model, compaction=True)
    assert output.endswith("CHILD-REPORT")
    assert model.ask_last == [True]


@pytest.mark.parametrize("streams", [False, True], ids=["buffered", "streaming"])
async def test_both_loops_ask_once_more_after_a_tool_only_reply(
    tmp_path: Path, streams: bool
) -> None:
    # The second ask in each twin: the tool called in the first reply is answered and not run,
    # the second ask goes out with thinking off and is the last thing read, and the reply to it
    # ends the turn.
    model = _WindowedModel(report_shapes=("tool", "text"), as_child=True)
    loop = await _run_reporting_loop(tmp_path, model, streams=streams)
    assert model.reports == 2
    assert model.report_kw[1].get("extra_body") == thinking_extra_body(False)
    assert model.ask_last == [True, True]
    assert not (tmp_path / "late.txt").exists()
    answered = {
        b.tool_use_id: b.output
        for m in loop.session.messages
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }
    assert answered["late-tool"] == _CONTEXT_FULL_NOT_RUN_REASK
    assert loop.session.messages[-1].text == "CHILD-REPORT"


async def test_a_report_the_window_held_down_is_compacted_and_asked_again(tmp_path: Path) -> None:
    # A reply cut at a cap the window held down, with only thinking in it, is no report and has
    # no room to be one: the child is compacted (ADR-0310) and asked once more, thinking off.
    model = _WindowedModel(report_shapes=("overflow", "text"))
    output = await _delegate(tmp_path, model, compaction=True)
    assert output.endswith("CHILD-REPORT")
    assert model.reports == 2
    assert model.summaries_at_ask[1] == model.summaries_at_ask[0] + 1
    assert model.report_kw[1].get("extra_body") == thinking_extra_body(False)


@pytest.mark.parametrize("streams", [False, True], ids=["buffered", "streaming"])
async def test_both_loops_make_room_before_the_second_ask(tmp_path: Path, streams: bool) -> None:
    # The room in each twin: a report the window held down is compacted before the second ask,
    # which goes out with thinking off and is the last thing read, and the reply to it ends the
    # turn.
    model = _WindowedModel(report_shapes=("overflow", "text"), as_child=True)
    loop = await _run_reporting_loop(tmp_path, model, streams=streams)
    assert model.reports == 2
    assert model.summaries_at_ask[1] == model.summaries_at_ask[0] + 1
    assert model.report_kw[1].get("extra_body") == thinking_extra_body(False)
    assert model.ask_last == [True, True]
    assert loop.session.messages[-1].text == "CHILD-REPORT"


@pytest.mark.parametrize("second", ["empty", "blank", "tool"])
@pytest.mark.parametrize("streams", [False, True], ids=["buffered", "streaming"])
async def test_both_loops_end_the_turn_on_the_second_ask(
    tmp_path: Path, streams: bool, second: str
) -> None:
    # The bound in each twin: a second reply with no text in it (nothing, only whitespace, or a
    # tool call) still ends the turn, the tool does not run, and the turn says it ended with no
    # report.
    model = _WindowedModel(report_shapes=("empty", second), as_child=True)
    events: list[object] = []
    await _run_reporting_loop(
        tmp_path, model, streams=streams, expect="context_full_no_report", events=events
    )
    assert model.reports == 2
    assert not (tmp_path / "late.txt").exists()
    if streams:
        statuses = [ev.message for ev in events if isinstance(ev, AgentStatus)]
        assert statuses[-1] == "stopping: sub-agent context full, it gave no report"


@pytest.mark.parametrize("streams", [False, True], ids=["buffered", "streaming"])
async def test_both_loops_ask_once_more_after_a_blank_reply(tmp_path: Path, streams: bool) -> None:
    # A reply that is only whitespace holds no report either: each twin asks once more.
    model = _WindowedModel(report_shapes=("blank", "text"), as_child=True)
    loop = await _run_reporting_loop(tmp_path, model, streams=streams)
    assert model.reports == 2
    assert model.report_kw[1].get("extra_body") == thinking_extra_body(False)
    assert loop.session.messages[-1].text == "CHILD-REPORT"


@pytest.mark.parametrize("streams", [False, True], ids=["buffered", "streaming"])
async def test_the_second_ask_costs_the_shared_pool_nothing(tmp_path: Path, streams: bool) -> None:
    # A reply with no report did no work, so its unit goes back to the pool a parent and its
    # children share, as a discarded completion's does. The turn's own count still counts it, so
    # the refund cannot loop a turn. The reply that ends the turn is paid for, report or not.
    runs = []
    for shapes, ending in (
        (("text",), "context_full"),
        (("empty", "text"), "context_full"),
        (("empty",), "context_full_no_report"),
    ):
        settings = _settings(tmp_path)
        pool = IterationBudget(0)  # no ceiling; it counts what the turn consumes
        loop = AgentLoop(
            _WindowedModel(report_shapes=shapes, as_child=True),
            default_registry(settings),
            Session(cwd=str(tmp_path), model="scripted/test"),
            settings=settings,
            workspace_root=tmp_path,
            compactor=Compactor(),
            report_at_window=True,
            budget=pool,
        )
        if streams:
            (end,) = [ev async for ev in loop.astream_turn("READ") if isinstance(ev, AgentDone)]
        else:
            end = await loop.arun_turn("READ")
        assert end.stop_reason == ending
        runs.append((end.iterations, pool.consumed))
    (one_ask, one_pool), (two_asks, two_pool), (no_report, no_report_pool) = runs
    assert two_asks == no_report == one_ask + 1
    # The pool pays for every reply but a first one that held no report.
    assert (one_pool, two_pool, no_report_pool) == (one_ask, one_ask, one_ask)


@pytest.mark.parametrize("streams", [False, True], ids=["buffered", "streaming"])
async def test_the_second_ask_fits_a_pool_with_room_for_one(tmp_path: Path, streams: bool) -> None:
    # The unit goes back before the second ask is made, not when the turn ends: a pool sized for a
    # turn that is asked once still takes the second ask, and the report still comes back.
    async def run(shapes: tuple[str, ...], pool: IterationBudget) -> tuple[Any, AgentLoop]:
        settings = _settings(tmp_path)
        loop = AgentLoop(
            _WindowedModel(report_shapes=shapes, as_child=True),
            default_registry(settings),
            Session(cwd=str(tmp_path), model="scripted/test"),
            settings=settings,
            workspace_root=tmp_path,
            compactor=Compactor(),
            report_at_window=True,
            budget=pool,
        )
        if streams:
            (end,) = [ev async for ev in loop.astream_turn("READ") if isinstance(ev, AgentDone)]
            return end, loop
        return await loop.arun_turn("READ"), loop

    once, _ = await run(("text",), IterationBudget(0))
    assert once.stop_reason == "context_full"  # the pool is sized by a turn that ended as asked
    twice, loop = await run(("empty", "text"), IterationBudget(once.iterations))
    assert twice.stop_reason == "context_full"
    assert loop.provider.reports == 2  # the second ask was made
    assert loop.session.messages[-1].text == "CHILD-REPORT"

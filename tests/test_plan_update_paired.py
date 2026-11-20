"""A completion that only updated the plan is sent paired with the completion after it (ADR-0401).

The session keeps every completion as the model wrote it. On a self-hosted model the request
carries a plan-only completion, its answer, the next completion and that one's answer as one
completion holding the update first and then the next calls, and one tool message holding all
their results: the shape ADR-0237 asks the model to write. A failed update, an update with text
or another call beside it, a next completion that updates the plan itself, anything between them,
an exchange not yet answered and a signed thinking block are sent as they are. What was measured
is recorded in ADR-0401.

Projection-level tests pin what is paired and what is not, and where the projection changes;
the gate's tests pin which models pair; loop-level tests pin, on BOTH turn paths, that a
self-hosted model is handed the pair while the session keeps both completions, and that a
metered model is handed them as they came.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_loop_planning import _done, _judge_ok, _plan_call, _review_ok, _Scripted
from tests.test_plan_only_bound import _Probe, _probe_call, _steps
from zakcode.agent.loop import AgentLoop, _pair_plan_updates
from zakcode.config import Settings
from zakcode.messages import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.usage import Usage


def _blocks(uses: list[ToolUseBlock], text: str, thinking: str, signature: str | None) -> list[Any]:
    blocks: list[Any] = []
    if thinking:
        blocks.append(ThinkingBlock(text=thinking, signature=signature))
    if text:
        blocks.append(TextBlock(text=text))
    return [*blocks, *uses]


def _update(
    n: int,
    *,
    error: bool = False,
    text: str = "",
    thinking: str = "",
    signature: str | None = None,
    other: str | None = None,
) -> list[Message]:
    """A completion calling update_plan's short form, and the tool message answering it; with
    ``other``, a second call to that tool beside it."""
    uses = [
        ToolUseBlock(id=f"plan-{n}", name="update_plan", input={"advance": {"outcome": "found"}})
    ]
    if other is not None:
        uses.append(ToolUseBlock(id=f"other-{n}", name=other, input={}))
    output = "failed" if error else "Plan updated: 1/2 steps done"
    results = [ToolResultBlock(tool_use_id=u.id, output=output, is_error=error) for u in uses]
    return [
        Message(role="assistant", blocks=_blocks(uses, text, thinking, signature)),
        Message.tool_results(results),
    ]


def _work(
    n: int, *, error: bool = False, text: str = "", thinking: str = "", signature: str | None = None
) -> list[Message]:
    """A completion running one command, and the tool message answering it."""
    use = ToolUseBlock(id=f"work-{n}", name="Bash", input={"command": f"step {n}"})
    result = ToolResultBlock(
        tool_use_id=use.id, output="failed" if error else f"ran step {n}", is_error=error
    )
    return [
        Message(role="assistant", blocks=_blocks([use], text, thinking, signature)),
        Message.tool_results([result]),
    ]


def _ids(message: Message) -> list[str]:
    if message.role == "assistant":
        return [use.id for use in message.tool_uses]
    return [b.tool_use_id for b in message.blocks if isinstance(b, ToolResultBlock)]


def _wire(messages: list[Message]) -> list[dict[str, Any]]:
    """Each message whole, its time stamp included."""
    return [m.model_dump() for m in messages]


# ── the projection ────────────────────────────────────────────────────


def test_a_plan_only_completion_goes_out_inside_the_next_one_update_first() -> None:
    head = Message.user("do the thing")
    plan, plan_answer = _update(1)
    work, work_answer = _work(2, text="Running the next step.")
    history = [head, plan, plan_answer, work, work_answer]
    sent = _pair_plan_updates(history)
    assert len(sent) == 3 and sent[0] is head
    call, answer = sent[1], sent[2]
    assert call.role == "assistant" and _ids(call) == ["plan-1", "work-2"]
    assert call.text == "Running the next step."
    assert answer.role == "tool" and _ids(answer) == ["plan-1", "work-2"]
    assert answer.blocks[0] is plan_answer.blocks[0] and answer.blocks[1] is work_answer.blocks[0]
    assert call.created_at == plan.created_at and answer.created_at == work_answer.created_at
    assert len(history) == 5 and history[1] is plan  # the history itself is untouched


def test_each_update_of_a_run_of_steps_pairs_with_the_completion_after_it() -> None:
    history = [Message.user("go"), *_update(1), *_work(2), *_update(3), *_work(4)]
    sent = _pair_plan_updates(history)
    assert [_ids(m) for m in sent[1:]] == [
        ["plan-1", "work-2"],
        ["plan-1", "work-2"],
        ["plan-3", "work-4"],
        ["plan-3", "work-4"],
    ]


def test_of_two_updates_in_a_row_the_second_pairs() -> None:
    history = [Message.user("go"), *_update(1), *_update(2), *_work(3)]
    sent = _pair_plan_updates(history)
    assert len(sent) == 5 and sent[1] is history[1] and sent[2] is history[2]
    assert _ids(sent[3]) == ["plan-2", "work-3"] and _ids(sent[4]) == ["plan-2", "work-3"]


def test_a_command_that_failed_still_takes_the_update_before_it() -> None:
    sent = _pair_plan_updates([Message.user("go"), *_update(1), *_work(2, error=True)])
    assert len(sent) == 3 and _ids(sent[1]) == ["plan-1", "work-2"]
    assert [b.is_error for b in sent[2].blocks if isinstance(b, ToolResultBlock)] == [False, True]


def test_an_unsigned_thinking_block_goes_with_the_completion_that_is_sent() -> None:
    """An unsigned block (a pod's reasoning) is never sent back; the update's is left out and the
    next completion's stays where it was, ahead of the calls."""
    history = [Message.user("go"), *_update(1, thinking="plan"), *_work(2, thinking="next")]
    sent = _pair_plan_updates(history)
    assert len(sent) == 3
    thinking = [b.text for b in sent[1].blocks if isinstance(b, ThinkingBlock)]
    assert thinking == ["next"] and isinstance(sent[1].blocks[0], ThinkingBlock)


def test_with_nothing_to_pair_the_history_itself_is_sent() -> None:
    history = [Message.user("go"), *_work(1), *_work(2)]
    assert _pair_plan_updates(history) is history
    assert _pair_plan_updates([]) == []


@pytest.mark.parametrize(
    "exchange",
    [
        [*_update(1, error=True), *_work(2)],  # the update failed: its answer is news
        [*_update(1, text="Closing step 1."), *_work(2)],  # the model said something
        [*_update(1, other="probe"), *_work(2)],  # another call rode with the update
        [*_update(1), *_update(2, other="Bash")],  # the next completion updated the plan itself
        [*_update(1, thinking="t", signature="s"), *_work(2)],  # replayed exactly as returned
        [*_update(1), *_work(2, thinking="t", signature="s")],  # replayed exactly as returned
        [*_update(1), _work(2)[0]],  # the next completion is not answered yet
        [*_update(1), Message.user("[/a-skill — page 2 of 3: next]"), *_work(2)],  # a page came
        [*_update(1), Message.assistant_text("All done.")],  # no call followed
    ],
    ids=[
        "update-failed",
        "update-with-text",
        "update-beside-a-call",
        "next-updates-the-plan",
        "signed-update",
        "signed-next",
        "next-unanswered",
        "page-between",
        "text-after",
    ],
)
def test_an_exchange_the_pair_would_misstate_is_sent_as_it_is(exchange: list[Message]) -> None:
    history = [Message.user("go"), *exchange]
    assert _pair_plan_updates(history) is history


def test_the_projection_moves_only_where_an_update_meets_the_completion_after_it() -> None:
    """Built from each four-message exchange alone: what was sent is the start of what is sent
    next, except once per pairing, when the next completion's answer arrives and the update joins
    it. Then the two agree up to the update, so a cache that keeps every prefix it read reuses all
    but the update, its answer and the next call."""
    parts = [
        _update(1),
        _work(2),
        [Message.assistant_text("noted")],
        _work(3),
        _update(4),
        _update(5),
        _work(6),
        [Message.assistant_text("done")],
    ]
    history: list[Message] = [Message.user("go")]
    before = _wire(_pair_plan_updates(history))
    moved = 0
    for message in [m for part in parts for m in part]:
        history = [*history, message]
        after = _wire(_pair_plan_updates(history))
        if after[: len(before)] != before:
            moved += 1
            assert before[:-3] == after[:-2]
        before = after
    assert moved == 2  # once per pairing: 1 with 2, and 5 with 6


# ── which models pair ────────────────────────────────────────────────────

_BASE = "http://127.0.0.1:9090/v1"


class _SelfHosted(_Scripted):
    """The scripted provider as an OpenAI-compatible model on a listed self-hosted base."""

    api_base = _BASE
    local_api_bases = [_BASE]

    def model_id(self) -> str:
        return "openai/pod-model"


class _Unlisted(_SelfHosted):
    """A generic model on a base the operator did not list as theirs (it may be a gateway)."""

    local_api_bases = ["http://10.0.0.9:9090/v1"]


class _Cloud(_SelfHosted):
    """A named cloud model: its prefix carries its own host, whatever the base says."""

    def model_id(self) -> str:
        return "anthropic/some-model"


def _loop(provider: Provider) -> AgentLoop:
    session = Session(cwd="/tmp", model=provider.model_id() or "test/model")
    return AgentLoop(provider, default_registry(), session, max_iterations=30, settings=Settings())


def test_only_a_self_hosted_model_not_measured_to_reuse_whole_prompts_pairs() -> None:
    loop = _loop(_SelfHosted([_done()]))
    assert loop._pairs_plan_updates()
    loop.session.tail_sparse_models.append("openai/pod-model")  # ADR-0193 measured it
    assert not loop._pairs_plan_updates()
    for provider in (_Scripted([_done()]), _Unlisted([_done()]), _Cloud([_done()])):
        assert not _loop(provider)._pairs_plan_updates()


# ── the loop, both turn paths ────────────────────────────────────────────


def _advance_call(n: int) -> LLMResult:
    call = ToolCall(
        id=f"advance-{n}", name="update_plan", arguments={"advance": {"outcome": "it answered"}}
    )
    return LLMResult(text="", tool_calls=[call], usage=Usage(total_tokens=1))


def _script() -> list[LLMResult]:
    return [
        _plan_call(_steps("a", "b")),
        _judge_ok(),
        _probe_call(1),
        _advance_call(1),
        _probe_call(2),
        _done(),
        _review_ok(),
    ]


async def _turn(provider: _Scripted, streamed: bool) -> Session:
    registry = default_registry()
    registry.register(_Probe())
    session = Session(cwd="/tmp", model=provider.model_id() or "test/model")
    loop = AgentLoop(provider, registry, session, max_iterations=30, settings=Settings())
    if streamed:
        async for _event in loop.astream_turn("do a multi-step thing"):
            pass
    else:
        await loop.arun_turn("do a multi-step thing")
    return session


def _calls(messages: list[Message]) -> list[list[str]]:
    return [_ids(m) for m in messages if m.role == "assistant" and m.tool_uses]


def _answers(messages: list[Message]) -> list[list[str]]:
    return [_ids(m) for m in messages if m.role == "tool"]


def _request_after(provider: _Scripted, call_id: str) -> list[Message]:
    """The first request that carried the answer to ``call_id``."""
    return next(
        request
        for request in provider.seen
        if any(
            isinstance(b, ToolResultBlock) and b.tool_use_id == call_id
            for m in request
            for b in m.blocks
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_self_hosted_model_is_handed_the_update_paired_but_the_session_keeps_both(
    streamed: bool,
) -> None:
    provider = _SelfHosted(_script())
    session = await _turn(provider, streamed)
    assert ["advance-1"] in _calls(session.messages) and ["probe-2"] in _calls(session.messages)
    request = _request_after(provider, "probe-2")
    assert ["advance-1", "probe-2"] in _calls(request)
    assert ["advance-1"] not in _calls(request) and ["probe-2"] not in _calls(request)
    assert ["advance-1", "probe-2"] in _answers(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_metered_model_is_handed_each_completion_as_it_came(streamed: bool) -> None:
    provider = _Cloud(_script())
    await _turn(provider, streamed)
    request = _request_after(provider, "probe-2")
    assert ["advance-1"] in _calls(request) and ["probe-2"] in _calls(request)
    assert ["advance-1", "probe-2"] not in _calls(request)

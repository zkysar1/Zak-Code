"""The bound on consecutive completions that only ask for rules the model holds (ADR-0399).

A completion of plain ``read_rule`` lookups whose every answer is a pointer (the text is in the
conversation, or pinned in the system prompt) reads nothing new, and each can name another rule,
so the doom guard's repeat test (ADR-0391) never sees a repeat. Measured 2026-11-20 on a served
fleet: one worker session sent 374 such single-call completions in 90 minutes, 344 answered by a
pointer, and no guard fired. The third such completion in a row is answered unexecuted, but only
when every rule it asks for would get a pointer: a first read and ``refresh=true`` always run.

Tracker-level tests pin the counting, read_rule-level tests pin that the loop's prediction is the
tool's own decision, and loop-level tests pin the refusal on BOTH turn paths.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.test_loop_planning import _done, _judge_ok, _plan_call, _review_ok, _Scripted
from tests.test_plan_only_bound import _Probe, _probe_call, _steps
from tests.test_rule_reask import _ctx, _registry, _text
from zakcode.agent.loop import _PLAN_TOOLS, _SHELL_TOOLS, AgentLoop
from zakcode.agent.stuck import (
    SIG_LOOKUP_ONLY,
    StuckAction,
    StuckTracker,
)
from zakcode.config import Settings
from zakcode.events import AgentDone, AgentStatus
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import LLMResult, ToolCall
from zakcode.rules import Rule, RuleRegistry
from zakcode.session.store import Session
from zakcode.tools.base import RULE_POINTER, ToolContext
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.tools.builtins.read_rule import REFRESH_HEAD, ReadRuleTool, would_point
from zakcode.usage import Usage

# ── the tracker ─────────────────────────────────────────────────────────────────


def _ask(*names: str, n: int = 0, refresh: bool = False) -> list[ToolCall]:
    extra: dict[str, Any] = {"refresh": True} if refresh else {}
    return [
        ToolCall(id=f"r{n}-{i}", name="read_rule", arguments={"name": name, **extra})
        for i, name in enumerate(names)
    ]


def _tracker() -> StuckTracker:
    return StuckTracker(plan_tools=_PLAN_TOOLS, shell_tools=_SHELL_TOOLS)


def _answered(
    tracker: StuckTracker, calls: list[ToolCall], *, pointers: tuple[bool, ...] | None = None
) -> None:
    """Observe ``calls`` answered by pointers (all, or as ``pointers`` says, call by call)."""
    flags = pointers if pointers is not None else (True,) * len(calls)
    tracker.observe(
        calls,
        [
            ToolResultBlock(
                tool_use_id=c.id,
                output="[already read; ask 2] Rule ..." if pointed else "# a rule\nIts text.",
                data={RULE_POINTER: True} if pointed else None,
            )
            for c, pointed in zip(calls, flags, strict=True)
        ],
    )


def _refused(tracker: StuckTracker, calls: list[ToolCall]) -> None:
    tracker.observe(
        calls,
        [
            ToolResultBlock(
                tool_use_id=c.id,
                output=tracker.lookup_only_message(None),
                is_error=True,
                data={"plan_only_bound": True},
            )
            for c in calls
        ],
    )


def test_the_third_pointer_answered_lookup_in_a_row_is_due_though_every_rule_differs() -> None:
    tracker = _tracker()
    assert not tracker.lookup_only_due(_ask("a"))
    _answered(tracker, _ask("a", n=1))
    assert not tracker.lookup_only_due(_ask("b"))
    _answered(tracker, _ask("b", n=2))
    assert tracker.lookup_only_run == 2
    assert tracker.lookup_only_due(_ask("c", n=3))
    assert tracker.lookup_only_due(_ask("c", "d", n=3))  # several rules in one completion too


def test_a_first_read_ends_the_run() -> None:
    tracker = _tracker()
    _answered(tracker, _ask("a", n=1))
    _answered(tracker, _ask("b", n=2))
    _answered(tracker, _ask("c", n=3), pointers=(False,))  # its text came back: a new rule
    assert tracker.lookup_only_run == 0
    assert not tracker.lookup_only_due(_ask("a", n=4))


def test_one_text_answer_beside_pointers_ends_the_run() -> None:
    tracker = _tracker()
    _answered(tracker, _ask("a", n=1))
    _answered(tracker, _ask("b", n=2))
    _answered(tracker, _ask("a", "c", n=3), pointers=(True, False))
    assert tracker.lookup_only_run == 0


def test_an_error_answer_ends_the_run() -> None:
    tracker = _tracker()
    _answered(tracker, _ask("a", n=1))
    _answered(tracker, _ask("b", n=2))
    calls = _ask("no-such-rule", n=3)
    tracker.observe(
        calls, [ToolResultBlock(tool_use_id=calls[0].id, output="no rule", is_error=True)]
    )
    assert tracker.lookup_only_run == 0


def test_a_refresh_is_never_due_and_never_counted() -> None:
    tracker = _tracker()
    _answered(tracker, _ask("a", n=1))
    _answered(tracker, _ask("b", n=2))
    assert not tracker.lookup_only_due(_ask("a", n=3, refresh=True))
    _answered(tracker, _ask("a", n=3, refresh=True))  # even flagged, it is an ask on purpose
    assert tracker.lookup_only_run == 0


def test_any_other_tool_in_the_completion_ends_the_run_and_is_never_due() -> None:
    tracker = _tracker()
    _answered(tracker, _ask("a", n=1))
    _answered(tracker, _ask("b", n=2))
    mixed = [*_ask("c", n=3), ToolCall(id="x", name="read_file", arguments={"path": "x"})]
    assert not tracker.lookup_only_due(mixed)
    _answered(tracker, mixed)
    assert tracker.lookup_only_run == 0


def test_a_plan_call_ends_the_run() -> None:
    tracker = _tracker()
    _answered(tracker, _ask("a", n=1))
    _answered(tracker, _ask("b", n=2))
    plan = [ToolCall(id="p1", name="update_plan", arguments={"tasks": []})]
    tracker.observe(plan, [ToolResultBlock(tool_use_id="p1", output="Plan updated: 0/1 done")])
    assert tracker.lookup_only_run == 0
    assert tracker.plan_only_run == 1


def test_the_flag_counts_only_on_a_read_rule_lookup() -> None:
    tracker = _tracker()
    other = [ToolCall(id="u", name="use_skill", arguments={"name": "a"})]
    for _ in range(3):
        _answered(tracker, other)
    assert tracker.lookup_only_run == 0


def test_a_reset_clears_the_run() -> None:
    tracker = _tracker()
    _answered(tracker, _ask("a", n=1))
    _answered(tracker, _ask("b", n=2))
    tracker.reset()  # a TURN_END veto sent the model round again
    assert tracker.lookup_only_run == 0
    assert not tracker.lookup_only_due(_ask("c", n=3))


def test_a_refusal_is_its_own_signal_and_climbs_the_ladder_to_a_stop() -> None:
    tracker = _tracker()
    _answered(tracker, _ask("a", n=1))
    _answered(tracker, _ask("b", n=2))
    actions = []
    for n, name in zip(range(3, 8), "cabca", strict=True):
        _refused(tracker, _ask(name, n=n))
        assert tracker.last_signals == [SIG_LOOKUP_ONLY]
        assert tracker.evidence() == {"signals": SIG_LOOKUP_ONLY, "lookup_only": n}
        assert tracker.last_was_lookup_only
        assert not tracker.last_was_plan_only and not tracker.last_was_print_only
        actions.append(tracker.next_action())
    assert tracker.failing_tools() == []  # never "read_rule has failed N times"
    assert tracker.plan_only_run == 0 and tracker.print_only_run == 0
    assert actions == [
        StuckAction.CONTINUE,
        StuckAction.NUDGE,
        StuckAction.NARROW,
        StuckAction.STEP_BACK,
        StuckAction.STOP,
    ]


def test_what_the_rails_say_is_true_of_what_was_counted() -> None:
    tracker = _tracker()
    _answered(tracker, _ask("a", n=1))
    _answered(tracker, _ask("b", n=2))
    text = tracker.lookup_only_message("2 'Run the tests'")
    assert text.startswith(
        "Not executed: this is response 3 in a row that only asks read_rule for rules already "
        "in your context"
    )
    assert "current step is 2 'Run the tests': apply the rules you have to it" in text
    assert "or answer in text if the work is done" in text
    assert "refresh=true returns a rule's text only if you need its exact wording" in text
    assert "plan tools" not in text and "print" not in text
    assert tracker.lookup_only_message(None).count("Apply the rules you have") == 1
    _refused(tracker, _ask("c", n=3))
    nudge = tracker.nudge_message()
    assert nudge.startswith(
        "You have now sent 3 responses in a row that only ask read_rule for rules already"
    )
    assert "plan tools" not in nudge and "print" not in nudge


# ── read_rule: the loop's prediction is the tool's own decision ─────────────────

_HELD_HEAD = "[already read; ask 1] Rule 'alpha-rule' is in your context"


def _case(name: str) -> tuple[ToolContext, dict[str, Any], bool]:
    """A named state: the context, the call's arguments, and whether a pointer answers it."""
    registry = _registry()
    held = (_text(registry, "alpha-rule"),)
    cases: dict[str, tuple[ToolContext, dict[str, Any], bool]] = {
        "first-read": (_ctx(registry), {"name": "alpha-rule"}, False),
        "held": (_ctx(registry, held=held), {"name": "alpha-rule"}, True),
        "right-after-its-pointer": (
            _ctx(registry, held=(*held, _HELD_HEAD), last=_HELD_HEAD),
            {"name": "alpha-rule"},
            True,
        ),
        "pinned": (_ctx(registry), {"name": "error-response"}, True),
        "pinned-refresh": (_ctx(registry), {"name": "error-response", "refresh": True}, False),
        "held-refresh": (
            _ctx(registry, held=held),
            {"name": "alpha-rule", "refresh": True},
            False,
        ),
        "held-refresh-after-a-refresh": (
            _ctx(registry, held=(*held, REFRESH_HEAD.format(name="alpha-rule") + held[0])),
            {"name": "alpha-rule", "refresh": True},
            True,
        ),
        "held-case-slip": (_ctx(registry, held=held), {"name": " ALPHA-RULE "}, True),
        "no-such-rule": (_ctx(registry, held=held), {"name": "alpha-rules"}, False),
        "empty-name": (_ctx(registry, held=held), {"name": ""}, False),
        "rules-disabled": (ToolContext(workspace_root=Path("/tmp")), {"name": "x"}, False),
    }
    return cases[name]


@pytest.mark.parametrize(
    "case",
    [
        "first-read",
        "held",
        "right-after-its-pointer",
        "pinned",
        "pinned-refresh",
        "held-refresh",
        "held-refresh-after-a-refresh",
        "held-case-slip",
        "no-such-rule",
        "empty-name",
        "rules-disabled",
    ],
)
def test_would_point_is_true_exactly_when_the_tool_answers_with_a_flagged_pointer(
    case: str,
) -> None:
    ctx, args, pointed = _case(case)
    predicted = would_point(args, ctx)  # asked first: it must change nothing the tool reads
    res = asyncio.run(ReadRuleTool().execute(args, ctx))
    flagged = bool((res.data or {}).get(RULE_POINTER))
    assert predicted is pointed
    assert flagged is pointed
    assert res.output.startswith("[already read; ask") is pointed
    if pointed:
        assert not res.verbatim and not res.is_error


def test_would_point_does_not_count_an_ask() -> None:
    """The ask count a compaction reads (ADR-0386) is the tool's one side effect."""
    asked: list[str] = []
    registry = _registry()
    ctx = ToolContext(
        workspace_root=Path("/tmp"),
        rule_registry=registry,
        in_context=lambda text: text == _text(registry, "alpha-rule"),
        note_rule_read=asked.append,
    )
    assert would_point({"name": "alpha-rule"}, ctx)
    assert asked == []
    asyncio.run(ReadRuleTool().execute({"name": "alpha-rule"}, ctx))
    assert asked == ["alpha-rule"]


# ── the loop, both turn paths ───────────────────────────────────────────────────

_RULES = ("a-rule", "b-rule", "c-rule", "d-rule")


def _rule_call(n: int, name: str, *, refresh: bool = False) -> LLMResult:
    args: dict[str, Any] = {"name": name, **({"refresh": True} if refresh else {})}
    call = ToolCall(id=f"rule-{n}", name="read_rule", arguments=args)
    return LLMResult(text="", tool_calls=[call], usage=Usage(total_tokens=1))


_STATUSES: list[str] = []


async def _turn(script: list[LLMResult], streamed: bool) -> tuple[Session, AgentLoop, _Probe, str]:
    _STATUSES.clear()
    rules = RuleRegistry()
    for name in _RULES:
        rules.add(Rule(name, f"# {name}\nDo the {name} thing.", Path(f"/x/{name}.md")))
    probe = _Probe()
    registry = default_registry()
    registry.register(ReadRuleTool())
    registry.register(probe)
    session = Session(cwd="/tmp", model="test/model")
    loop = AgentLoop(
        _Scripted(script),
        registry,
        session,
        rule_registry=rules,
        max_iterations=40,
        settings=Settings(),
    )
    if streamed:
        stop_reason = ""
        async for event in loop.astream_turn("do a multi-step thing"):
            if isinstance(event, AgentDone):
                stop_reason = event.stop_reason
            elif isinstance(event, AgentStatus):
                _STATUSES.append(event.message)
    else:
        stop_reason = (await loop.arun_turn("do a multi-step thing")).stop_reason
    return session, loop, probe, stop_reason


def _rule_results(session: Session) -> list[ToolResultBlock]:
    return [
        block
        for message in session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock) and block.tool_use_id.startswith("rule-")
    ]


def _kinds(results: list[ToolResultBlock]) -> list[str]:
    """Each answer as ``text`` (the rule's text), ``pointer`` or ``refused``."""
    return [
        "refused"
        if (b.data or {}).get("plan_only_bound")
        else "pointer"
        if (b.data or {}).get(RULE_POINTER)
        else "text"
        for b in results
    ]


def _first_reads() -> list[LLMResult]:
    return [_rule_call(n, name) for n, name in enumerate(("a-rule", "b-rule", "c-rule"))]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_model_cycling_through_rules_it_holds_ends_the_turn_stuck(streamed: bool) -> None:
    script = [_plan_call(_steps("a", "b")), _judge_ok(), *_first_reads()]
    script += [_rule_call(n, _RULES[n % 3]) for n in range(3, 40)]
    script += [_done(), _review_ok()]
    session, loop, probe, stop_reason = await _turn(script, streamed)
    assert stop_reason == "stuck"
    # Three first reads, two asks answered by a pointer, five refused, and the fifth refusal is
    # the stop: not 374 and counting.
    assert _kinds(_rule_results(session)) == ["text"] * 3 + ["pointer"] * 2 + ["refused"] * 5
    refused = _rule_results(session)[5]
    assert refused.is_error
    assert "only asks read_rule for rules already in your context" in refused.output
    assert "current step is 1 'a': apply the rules you have to it" in refused.output
    notes = [e for e in loop._trace.events if e.data.get("kind") == "plan_only_bound"]
    assert len(notes) == 5
    assert notes[0].detail == "response 3 in a row only asked read_rule for rules already held"
    stops = [e for e in loop._trace.events if e.data.get("kind") == "stuck"]
    assert stops and stops[-1].data.get("lookup_only") == 7
    # Rung 1's seeded step says what was counted, and a lookup does not finish it.
    seeded = [t for t in session.task_network.leaves() if t.title.startswith("Do: one piece")]
    assert seeded and seeded[0].note.startswith(
        "you have been asking read_rule for rules already in your context"
    )
    assert "A rule lookup is not that work" in seeded[0].note
    assert probe.runs == 0
    if streamed:
        assert "idle bound: the calls were not executed" in _STATUSES


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_new_rule_or_a_refresh_at_the_bound_is_answered_never_refused(
    streamed: bool,
) -> None:
    script = [_plan_call(_steps("a", "b")), _judge_ok(), *_first_reads()]
    script += [_rule_call(3, "a-rule"), _rule_call(4, "b-rule"), _rule_call(5, "d-rule")]
    script += [_rule_call(6, "a-rule"), _rule_call(7, "b-rule")]
    script += [_rule_call(8, "c-rule", refresh=True)]
    script += [_rule_call(9, "a-rule"), _rule_call(10, "b-rule"), _rule_call(11, "c-rule")]
    script += [_probe_call(1), _rule_call(12, "a-rule"), _done(), _review_ok()]
    session, _loop, probe, _stop = await _turn(script, streamed)
    results = _rule_results(session)
    assert _kinds(results) == [
        *("text", "text", "text"),
        *("pointer", "pointer", "text"),  # d-rule is new at the bound: read
        *("pointer", "pointer", "text"),  # refresh=true at the bound: its text again
        *("pointer", "pointer", "refused"),  # the third held ask in a row
        "pointer",  # another tool ran: answered again
    ]
    assert results[5].output.startswith("# d-rule")
    assert results[8].output.startswith("[sent again on refresh] Rule 'c-rule':\n# c-rule")
    assert probe.runs == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_asks_between_real_steps_are_never_refused(streamed: bool) -> None:
    """Positive control: two held asks, then work, again and again, stay under the bound."""
    script = [_plan_call(_steps("a", "b")), _judge_ok(), *_first_reads()]
    for n in range(1, 5):
        script += [_probe_call(n), _rule_call(10 * n, "a-rule"), _rule_call(10 * n + 1, "b-rule")]
    script += [_done(), _review_ok()]
    session, loop, probe, stop_reason = await _turn(script, streamed)
    assert stop_reason == "completed"
    assert _kinds(_rule_results(session)) == ["text"] * 3 + ["pointer"] * 8
    assert probe.runs == 4
    assert not [e for e in loop._trace.events if e.data.get("kind") == "plan_only_bound"]

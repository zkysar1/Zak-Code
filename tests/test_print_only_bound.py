"""The bound on consecutive print-only completions (ADR-0373).

A shell call whose whole command is one ``echo`` of literal words reads nothing and changes
nothing, and each string can differ, so the doom guard never sees a repeat. The third such
completion in a row is answered unexecuted, naming the step to do; any other call ends the run.
Tracker-level tests pin the counting and the predicate, loop-level tests pin the refusal on BOTH
turn paths (a behaviour written at one of the two is silently absent on the other).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_loop_planning import _done, _judge_ok, _plan_call, _review_ok, _Scripted
from tests.test_plan_only_bound import _Probe, _probe_call, _steps
from zakcode.agent.loop import _PLAN_TOOLS, _SHELL_TOOLS, AgentLoop
from zakcode.agent.stuck import SIG_PLAN_ONLY, SIG_PRINT_ONLY, StuckAction, StuckTracker
from zakcode.config import PermissionTier, Settings
from zakcode.events import AgentDone, AgentStatus
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import LLMResult, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.usage import Usage

# ── the tracker ─────────────────────────────────


def _bash(command: str, n: int = 0, name: str = "Bash") -> list[ToolCall]:
    return [ToolCall(id=f"b{n}", name=name, arguments={"command": command})]


def _tracker() -> StuckTracker:
    return StuckTracker(plan_tools=_PLAN_TOOLS, shell_tools=_SHELL_TOOLS)


def _ran(tracker: StuckTracker, calls: list[ToolCall], output: str = "ok") -> None:
    tracker.observe(calls, [ToolResultBlock(tool_use_id=c.id, output=output) for c in calls])


def _print(tracker: StuckTracker, n: int) -> None:
    _ran(tracker, _bash(f'echo "NEXT{n}"', n), f"NEXT{n}")


PRINTS = [
    'echo "NEXT97"',
    "echo NEXT3",
    '  echo "two words"  ',
    "echo 'not $expanded'",
    "echo -n done",
    "echo several bare words",
    'echo "a line\nand another"',
]
NOT_PRINTS = [
    'echo "$HOME"',
    "echo $HOME",
    "echo a\\b",
    "echo (a)",
    "echo $(date)",
    "echo `date`",
    'echo "`date`"',
    'echo "a\\b"',
    "echo a | tee out",
    "echo a > out",
    "echo a; ls",
    "echo a && ls",
    "echo *.py",
    "echo ~",
    "echo {a,b}",
    "echo [ab]",
    "echo a?",
    "echo a # note",
    'echo "esc\\"aped"',
    "echo",
    "echoes a",
    "ls",
    'printf "x"',
]


def test_the_third_print_in_a_row_is_the_one_refused_though_every_string_differs() -> None:
    tracker = _tracker()
    assert not tracker.refuses_print_only(_bash('echo "NEXT1"'))
    _print(tracker, 1)
    assert not tracker.refuses_print_only(_bash('echo "NEXT2"'))
    _print(tracker, 2)
    assert tracker.print_only_run == 2
    assert tracker.refuses_print_only(_bash('echo "NEXT3"'))


@pytest.mark.parametrize("command", PRINTS)
def test_what_counts_as_a_print(command: str) -> None:
    tracker = _tracker()
    _ran(tracker, _bash(command, 1))
    _ran(tracker, _bash(command, 2))
    assert tracker.print_only_run == 2
    assert tracker.refuses_print_only(_bash(command, 3))


@pytest.mark.parametrize("command", NOT_PRINTS)
def test_what_does_not_count_and_ends_the_run(command: str) -> None:
    tracker = _tracker()
    _print(tracker, 1)
    _print(tracker, 2)
    assert not tracker.refuses_print_only(_bash(command))
    _ran(tracker, _bash(command, 3))
    assert tracker.print_only_run == 0
    assert not tracker.refuses_print_only(_bash('echo "NEXT4"'))


@pytest.mark.parametrize("name", sorted(_SHELL_TOOLS))
def test_every_shell_tool_counts_and_no_other_tool_does(name: str) -> None:
    tracker = _tracker()
    _ran(tracker, _bash("echo one", 1, name))
    _ran(tracker, _bash("echo two", 2, name))
    assert tracker.refuses_print_only(_bash("echo three", 3, name))
    other = _tracker()
    _ran(other, _bash("echo one", 1, "probe"))
    _ran(other, _bash("echo two", 2, "probe"))
    assert other.print_only_run == 0


def test_a_print_beside_real_work_in_one_completion_is_no_print_only_completion() -> None:
    tracker = _tracker()
    _print(tracker, 1)
    _print(tracker, 2)
    mixed = [*_bash('echo "NEXT3"', 3), ToolCall(id="r1", name="read_file", arguments={"path": "x"})]
    assert not tracker.refuses_print_only(mixed)
    _ran(tracker, mixed)
    assert tracker.print_only_run == 0


def test_two_prints_in_one_completion_are_print_only() -> None:
    tracker = _tracker()
    _print(tracker, 1)
    _print(tracker, 2)
    assert tracker.refuses_print_only([*_bash("echo a", 3), *_bash("echo b", 4)])


def test_a_run_that_mixes_prints_and_plan_calls_is_refused_by_neither_bound() -> None:
    # Measured in ordinary work (ADR-0373): "lpp", "llp", "plpp". Neither count sees the other's.
    tracker = _tracker()
    plan = [ToolCall(id="p1", name="update_plan", arguments={"tasks": []})]
    for n in range(0, 12, 3):
        _ran(tracker, plan, "Plan updated: 0/1 steps done")
        _print(tracker, n)
        _print(tracker, n + 1)
        assert not tracker.refuses_print_only(plan)
        assert not tracker.refuses_plan_only(plan)
    assert tracker.print_only_run == 2  # the two prints since the last plan call
    assert tracker.plan_only_run == 0


def test_the_bound_is_inert_without_shell_tools_and_cleared_by_a_reset() -> None:
    inert = StuckTracker(plan_tools=_PLAN_TOOLS)
    for n in range(4):
        _print(inert, n)
    assert not inert.refuses_print_only(_bash('echo "NEXT9"'))
    tracker = _tracker()
    _print(tracker, 1)
    _print(tracker, 2)
    tracker.reset()  # a TURN_END veto sent the model round again
    assert not tracker.refuses_print_only(_bash('echo "NEXT3"'))


def test_a_refusal_is_its_own_signal_and_climbs_the_ladder_to_a_stop() -> None:
    tracker = _tracker()
    _print(tracker, 1)
    _print(tracker, 2)
    actions = []
    for n in range(3, 8):
        calls = _bash(f'echo "NEXT{n}"', n)
        refusal = ToolResultBlock(
            tool_use_id=calls[0].id,
            output=tracker.print_only_message("1 '1'"),
            is_error=True,
            data={"plan_only_bound": True},
        )
        tracker.observe(calls, [refusal])
        assert tracker.last_signals == [SIG_PRINT_ONLY]
        assert tracker.evidence() == {"signals": SIG_PRINT_ONLY, "print_only": n}
        assert tracker.last_was_print_only and not tracker.last_was_plan_only
        actions.append(tracker.next_action())
    assert tracker.failing_tools() == []  # never "Bash has failed N times"
    assert tracker.plan_only_run == 0
    assert actions == [
        StuckAction.CONTINUE,
        StuckAction.NUDGE,
        StuckAction.NARROW,
        StuckAction.STEP_BACK,
        StuckAction.STOP,
    ]


def test_a_plan_only_refusal_is_still_told_as_one() -> None:
    tracker = _tracker()
    plan = [ToolCall(id="p1", name="update_plan", arguments={"tasks": []})]
    refusal = ToolResultBlock(
        tool_use_id="p1", output="Not executed", is_error=True, data={"plan_only_bound": True}
    )
    tracker.observe(plan, [refusal])
    assert tracker.last_signals == [SIG_PLAN_ONLY]
    assert tracker.plan_only_run == 1 and tracker.print_only_run == 0
    assert "plan tools" in tracker.nudge_message()


def test_what_the_rails_say_is_true_of_what_was_counted() -> None:
    tracker = _tracker()
    _print(tracker, 1)
    _print(tracker, 2)
    text = tracker.print_only_message("2 'Run the tests'")
    assert text.startswith("Not executed: this is response 3 in a row that only prints a fixed")
    assert "current step is 2 'Run the tests': do that step now" in text
    assert "or answer in text if the work is done" in text
    assert "plan tools" not in text
    assert "Do the next piece of actual work" in tracker.print_only_message(None)
    calls = _bash('echo "NEXT3"', 3)
    refusal = ToolResultBlock(
        tool_use_id=calls[0].id, output=text, is_error=True, data={"plan_only_bound": True}
    )
    tracker.observe(calls, [refusal])
    nudge = tracker.nudge_message()
    assert nudge.startswith("You have now sent 3 responses in a row that only print a fixed")
    assert "plan tools" not in nudge


# ── the loop, both turn paths ────────────────────────────�


class _Shell(Tool):
    """Stands in for the shell, so a print that ran is counted without spawning one."""

    spec = ToolSpec(
        name="powershell",
        description="runs a command",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self) -> None:
        self.runs: list[str] = []

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.runs.append(args["command"])
        return ToolResult.ok(args["command"])


def _print_call(n: int) -> LLMResult:
    call = ToolCall(id=f"print-{n}", name="powershell", arguments={"command": f'echo "NEXT{n}"'})
    return LLMResult(text="", tool_calls=[call], usage=Usage(total_tokens=1))


_STATUSES: list[str] = []


async def _turn(
    script: list[LLMResult], streamed: bool
) -> tuple[Session, AgentLoop, _Shell, _Probe, str]:
    _STATUSES.clear()
    shell, probe = _Shell(), _Probe()
    registry = default_registry()
    registry._tools[shell.spec.name] = shell  # the real shell tool, replaced
    registry.activate(shell.spec.name)
    registry.register(probe)
    session = Session(cwd="/tmp", model="test/model")
    loop = AgentLoop(_Scripted(script), registry, session, max_iterations=40, settings=Settings())
    if streamed:
        stop_reason = ""
        async for event in loop.astream_turn("do a multi-step thing"):
            if isinstance(event, AgentDone):
                stop_reason = event.stop_reason
            elif isinstance(event, AgentStatus):
                _STATUSES.append(event.message)
    else:
        stop_reason = (await loop.arun_turn("do a multi-step thing")).stop_reason
    return session, loop, shell, probe, stop_reason


def _print_results(session: Session) -> list[ToolResultBlock]:
    return [
        block
        for message in session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock) and block.tool_use_id.startswith("print-")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_the_loop_refuses_the_third_print_and_runs_one_after_work(streamed: bool) -> None:
    script = [
        _plan_call(_steps("a", "b")),
        _judge_ok(),
        _print_call(1),
        _print_call(2),
        _print_call(3),  # the third in a row: refused
        _probe_call(1),  # another tool ends the run
        _print_call(4),  # runs again
        _done(),
        _review_ok(),
    ]
    session, loop, shell, probe, _stop = await _turn(script, streamed)
    results = _print_results(session)
    assert [bool((b.data or {}).get("plan_only_bound")) for b in results] == [
        False,
        False,
        True,
        False,
    ]
    refused = results[2]
    assert refused.is_error
    assert "only prints a fixed string" in refused.output
    assert "current step is 1 'a': do that step now" in refused.output
    assert shell.runs == ['echo "NEXT1"', 'echo "NEXT2"', 'echo "NEXT4"']
    assert probe.runs == 1
    notes = [e for e in loop._trace.events if e.data.get("kind") == "plan_only_bound"]
    assert len(notes) == 1
    assert notes[0].detail == "response 3 in a row only printed a fixed string"
    if streamed:
        assert "idle bound: the calls were not executed" in _STATUSES


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_model_that_keeps_printing_ends_the_turn_stuck(streamed: bool) -> None:
    script = [_plan_call(_steps("a", "b")), _judge_ok()]
    script += [_print_call(n) for n in range(1, 30)]
    script += [_done(), _review_ok()]
    session, loop, shell, _unused, stop_reason = await _turn(script, streamed)
    assert stop_reason == "stuck"
    assert len(shell.runs) == 2
    # Two ran, five were refused, and the fifth refusal is the stop: not 149 and counting.
    assert len(_print_results(session)) == 7
    kinds = [e.data.get("kind") for e in loop._trace.events]
    assert kinds.count("plan_only_bound") == 5
    # Rung 1's seeded step says what was counted, and a print does not finish it.
    seeded = [t for t in session.task_network.leaves() if t.title.startswith("Do: one piece")]
    assert seeded and seeded[0].note.startswith("you have been printing fixed strings")
    assert "more than a print" in seeded[0].note


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_print_between_real_steps_is_never_refused(streamed: bool) -> None:
    script = [_plan_call(_steps("a", "b")), _judge_ok()]
    for n in range(1, 6):
        script += [_probe_call(n), _print_call(n), _print_call(n + 10)]
    script += [_done(), _review_ok()]
    session, loop, shell, probe, stop_reason = await _turn(script, streamed)
    assert stop_reason == "completed"
    assert not any((b.data or {}).get("plan_only_bound") for b in _print_results(session))
    assert len(shell.runs) == 10
    assert probe.runs == 5
    assert not [e for e in loop._trace.events if e.data.get("kind") == "plan_only_bound"]

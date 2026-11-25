"""The poll signal (ADR-0406): the same shell check run again and again, answered the same.

A model that waits by checking runs one command repeatedly, and a number in it (a counter, a
time, the file its output goes to) makes every call a new string. The doom guard compares batches
byte for byte, and the outcome signal skips an empty or short answer and reads a small number as
new, so neither sees the repeat. Tracker-level tests pin the counting, the masking and the ladder;
loop-level tests pin the rail, the plan step and the climb to a stop on BOTH turn paths (a
behaviour written at one of the two is silently absent on the other).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.test_stuck_decompose import _c, _drain, _rails, _ScriptByCallProvider, _titles
from zakcode.agent.loop import _PLAN_TOOLS, _SHELL_TOOLS, AgentLoop
from zakcode.agent.stuck import (
    POLL_AT,
    SIG_ALL_ERRORS,
    SIG_POLL,
    SIG_REPEATED_OUTCOME,
    StuckAction,
    StuckTracker,
    poll_answer,
    poll_command,
)
from zakcode.config import PermissionTier, Settings
from zakcode.events import AgentDone, AgentStatus
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import LLMResult, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec

#: What the shell tool answers a successful command that printed nothing.
EMPTY_OK = "[exit code: 0]"
#: An answer that says something, the same each time.
UNCHANGED = "no change since the last look"

# ── the tracker ─────────────────────────────────


def _shell(command: str, n: int, name: str = "Bash") -> list[ToolCall]:
    return [ToolCall(id=f"s{n}", name=name, arguments={"command": command})]


def _tracker(**kw: Any) -> StuckTracker:
    return StuckTracker(plan_tools=_PLAN_TOOLS, shell_tools=_SHELL_TOOLS, **kw)


def _ran(
    tracker: StuckTracker, calls: list[ToolCall], output: str = EMPTY_OK, *, is_error: bool = False
) -> None:
    tracker.observe(
        calls, [ToolResultBlock(tool_use_id=c.id, output=output, is_error=is_error) for c in calls]
    )


def _status_check(n: int) -> str:
    # The measured shape: one query run again, its output sent to a new numbered file each time.
    return f"bash q.sh --job j-7 --field status > /tmp/out_{n}.txt 2>&1"


def _poll(tracker: StuckTracker, n: int) -> None:
    _ran(tracker, _shell(_status_check(n), n))


def _run(
    tracker: StuckTracker,
    commands: list[str],
    answer: Callable[[int], str] = lambda n: EMPTY_OK,
) -> int:
    """Run one shell completion per command and return how many completions fired the signal."""
    fired = 0
    for n, command in enumerate(commands):
        _ran(tracker, _shell(command, n), answer(n))
        fired += tracker.last_was_poll
    return fired


def test_the_measured_shape_is_a_poll_at_the_sixth_completion_though_no_two_calls_match() -> None:
    commands = [_status_check(n) for n in range(1, POLL_AT + 1)]
    assert len(set(commands)) == POLL_AT  # the floor: the doom guard's byte test sees nothing
    tracker = _tracker()
    for n in range(1, POLL_AT):
        _poll(tracker, n)
        assert tracker.last_signals == [] and not tracker.last_was_poll
        assert tracker.poll_run == n
    _poll(tracker, POLL_AT)
    assert tracker.last_signals == [SIG_POLL] and tracker.last_was_poll
    assert tracker.poll_run == POLL_AT == 6
    assert tracker.evidence() == {"signals": SIG_POLL, "poll": 6}
    assert tracker.failing_tools() == []  # nothing failed: no step may say a tool keeps failing


def test_the_ladder_climbs_by_the_length_of_the_run() -> None:
    tracker = _tracker()
    actions = []
    for n in range(1, 11):
        _poll(tracker, n)
        actions.append(tracker.next_action())
    assert actions == [
        *[StuckAction.CONTINUE] * 5,
        StuckAction.NUDGE,
        StuckAction.NARROW,
        StuckAction.STEP_BACK,
        StuckAction.STOP,
        StuckAction.STOP,
    ]
    assert tracker.evidence() == {
        "signals": SIG_POLL,
        "poll": 10,
    }  # the run's length, not the threshold


LETTERS = "abcdefg"
#: What changes between runs of one check, besides where its output goes.
NUMBER_VARIANTS: dict[str, Callable[[int], str]] = {
    "a counter in an argument": lambda n: f"bash q.sh --job j-7 --attempt {n}",
    "a time in the command": lambda n: f"bash q.sh --since 2026-11-25T08:{10 + n}:00",
    "a number in a path": lambda n: f"cat /tmp/session-{n}/state",
}
#: Where its output goes, and how the command is spaced.
TARGET_VARIANTS: dict[str, Callable[[int], str]] = {
    "an output file with a number": lambda n: f"bash q.sh --job j-7 > /tmp/out_{n}.txt",
    "an output file with no number": lambda n: f"bash q.sh --job j-7 > /tmp/o_{LETTERS[n]}.txt",
    "an appended log": lambda n: f"bash q.sh --job j-7 >> /tmp/log-{LETTERS[n]} 2>&1",
    "the error stream sent to a file": lambda n: f"bash q.sh --job j-7 2>/tmp/e{LETTERS[n]}",
    "both streams sent to a file": lambda n: f"bash q.sh --job j-7 &>/tmp/all_{LETTERS[n]}",
    "spacing": lambda n: "bash q.sh" + (" ", "\t", "  ", " \t", "   ", "\t\t")[n] + "--job j-7",
}
VARIANTS = {**NUMBER_VARIANTS, **TARGET_VARIANTS}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_what_varies_between_the_runs_of_one_check(variant: str) -> None:
    commands = [VARIANTS[variant](n) for n in range(POLL_AT)]
    assert len(set(commands)) == POLL_AT  # the floor: every call is a new string
    tracker = _tracker()
    assert _run(tracker, commands[:-1], lambda n: UNCHANGED) == 0
    assert _run(tracker, commands[-1:], lambda n: UNCHANGED) == 1
    assert tracker.poll_run == POLL_AT


@pytest.mark.parametrize("variant", sorted(TARGET_VARIANTS))
def test_a_check_that_printed_nothing_is_the_same_check_when_only_its_output_file_differs(
    variant: str,
) -> None:
    commands = [TARGET_VARIANTS[variant](n) for n in range(POLL_AT)]
    assert len(set(commands)) == POLL_AT
    assert _run(_tracker(), commands) == 1  # EMPTY_OK: nothing printed


@pytest.mark.parametrize("variant", sorted(NUMBER_VARIANTS))
def test_commands_that_printed_nothing_and_differ_in_a_number_are_not_one_check(
    variant: str,
) -> None:
    # `py step1.py`, `py step2.py`, ...: each prints nothing, and each is another job. Nothing came
    # back to say the questions were the same, so the number in the command is the difference.
    commands = [NUMBER_VARIANTS[variant](n) for n in range(2 * POLL_AT)]
    assert len({poll_command(c) for c in commands}) == 1  # the floor: one command up to numbers
    assert _run(_tracker(), commands) == 0
    # the control: the same calls that each said the same thing are a poll
    assert _run(_tracker(), commands, lambda n: UNCHANGED) == POLL_AT + 1


def test_a_result_with_no_text_at_all_is_the_same_silence_as_the_exit_code_line() -> None:
    counters = [NUMBER_VARIANTS["a counter in an argument"](n) for n in range(2 * POLL_AT)]
    assert _run(_tracker(), counters, lambda n: "") == 0  # numbers kept: another job each time
    files = [TARGET_VARIANTS["an output file with a number"](n) for n in range(POLL_AT)]
    assert _run(_tracker(), files, lambda n: "") == 1  # only where the output goes differs


PACKAGES = ("api", "core", "db", "models", "utils", "views", "tests", "cli", "web")
#: What a model that scaffolds by shell writes, a command to a completion: one text into a new
#: file each time. Nothing is read, so nothing is asked of the world.
LITERAL_WRITES: dict[str, Callable[[int], str]] = {
    "an echo into a new package file": lambda n: f"echo '' > src/{PACKAGES[n]}/__init__.py",
    "a printf with an escape": lambda n: f"printf '# header\\n' > src/{PACKAGES[n]}/mod.py",
    "an append": lambda n: f"echo done >> logs/{PACKAGES[n]}.log",
    "numbered files": lambda n: f"echo '' > part{n}.txt",
}


@pytest.mark.parametrize("variant", sorted(LITERAL_WRITES))
def test_one_text_written_into_different_files_is_no_check_however_often(variant: str) -> None:
    commands = [LITERAL_WRITES[variant](n) for n in range(len(PACKAGES))]
    assert len(commands) >= POLL_AT + 3  # the stop, had they read as one check
    tracker = _tracker()
    assert _run(tracker, commands) == 0  # EMPTY_OK: nothing printed
    assert tracker.poll_run == 1
    assert tracker.next_action() is StuckAction.CONTINUE


def test_a_different_check_is_not_the_same_check_though_the_answers_match() -> None:
    fields = ["status", "title", "size", "notes", "label", "age", "kind", "mode"]
    commands = [f"bash q.sh --job j-7 --field {name}" for name in fields]
    assert len({poll_command(c) for c in commands}) == len(fields)
    assert _run(_tracker(), commands, lambda n: UNCHANGED) == 0
    # the control: the same eight completions as one check fire from the sixth on
    same = [f"bash q.sh --job j-7 --field status --try {n}" for n in range(len(fields))]
    assert _run(_tracker(), same, lambda n: UNCHANGED) == len(fields) - POLL_AT + 1


def test_a_call_to_another_tool_ends_the_run() -> None:
    tracker = _tracker()
    for n in range(1, POLL_AT):
        _poll(tracker, n)
    assert tracker.poll_run == POLL_AT - 1
    read = [ToolCall(id="r1", name="read_file", arguments={"path": "x"})]
    _ran(tracker, read, "contents")
    assert tracker.poll_run == 0
    for n in range(10, 10 + POLL_AT - 1):
        _poll(tracker, n)
        assert not tracker.last_was_poll
    assert tracker.poll_run == POLL_AT - 1
    _poll(tracker, 20)  # the control: one more and the second run fires
    assert tracker.last_was_poll


def test_a_completion_with_no_calls_is_no_part_of_a_run() -> None:
    # The loop gives the tracker only completions that called a tool; this is its own rule for a
    # batch with none.
    tracker = _tracker()
    for n in range(1, POLL_AT):
        _poll(tracker, n)
    tracker.observe([], [])
    assert tracker.poll_run == 0 and not tracker.last_was_poll


def test_a_different_shell_command_starts_a_new_run() -> None:
    tracker = _tracker()
    for n in range(1, POLL_AT):
        _poll(tracker, n)
    _ran(tracker, _shell("ls /tmp", 50), "a b c")
    assert tracker.poll_run == 1 and not tracker.last_was_poll
    for n in range(60, 60 + POLL_AT - 1):
        _poll(tracker, n)
    assert tracker.poll_run == POLL_AT - 1 and not tracker.last_was_poll


def test_a_shell_command_run_after_each_edit_is_no_poll() -> None:
    # edit -> test -> edit -> test is progress: the test command is the same, the world is not.
    tracker = _tracker()
    edit = [ToolCall(id="e1", name="edit_file", arguments={"path": "a.py"})]
    shells = 0
    for n in range(2 * POLL_AT):
        _ran(tracker, edit, "Made 1 replacement")
        _ran(tracker, _shell("pytest -q tests/test_a.py", n), "1 passed in 0.12s\n[exit code: 0]")
        shells += 1
        assert not tracker.last_was_poll and tracker.poll_run == 1
    assert shells == 2 * POLL_AT  # the floor: long enough to fire, had no edit stood between
    bare = _tracker()
    fired = _run(bare, ["pytest -q tests/test_a.py"] * (2 * POLL_AT), lambda n: "1 passed in 0.12s")
    assert fired == POLL_AT + 1  # the control: the same shell calls with no edit between them


def test_reading_a_file_in_pages_is_no_poll_when_every_page_answers_in_other_words() -> None:
    commands = [f"sed -n '{50 * n + 1},{50 * n + 50}p' big.txt" for n in range(2 * POLL_AT)]
    assert len({poll_command(c) for c in commands}) == 1  # the floor: one command up to numbers
    assert _run(_tracker(), commands, lambda n: f"line {n} " + "text " * (n + 3)) == 0
    # the control: the same calls answered with one page each time are a poll
    assert _run(_tracker(), commands, lambda n: "(end of file)") == POLL_AT + 1


#: Items asked for one at a time, each answered in words that differ only in their numbers. The
#: masks cannot tell them from one check run again (ADR-0406, Not done), so the ladder reaches its
#: stop at the ninth. The cost is pinned so that a change to it is made on purpose.
ENUMERATIONS: dict[str, tuple[Callable[[int], str], Callable[[int], str]]] = {
    "numbered scripts": (lambda n: f"python ex{n}.py", lambda n: f"Exercise {n} OK"),
    "pages of a log": (
        lambda n: f"sed -n '{40 * n + 1},{40 * n + 40}p' train.log",
        lambda n: " ".join(f"loss {n}.{i}" for i in range(40)),
    ),
    "items by number": (
        lambda n: f"curl -s localhost:8080/items/{n}",
        lambda n: f'{{"id": {n}, "state": "ok"}}',
    ),
}


@pytest.mark.parametrize("name", sorted(ENUMERATIONS))
def test_items_asked_for_by_number_and_answered_alike_read_as_one_check_the_known_cost(
    name: str,
) -> None:
    command, answer = ENUMERATIONS[name]
    tracker = _tracker()
    actions: list[StuckAction] = []
    for n in range(1, 10):
        _ran(tracker, _shell(command(n), n), answer(n))
        actions.append(tracker.next_action())
    assert tracker.poll_run == 9 and actions[-1] is StuckAction.STOP


def test_numbers_in_the_answer_are_not_a_change_and_words_are() -> None:
    commands = [_status_check(n) for n in range(2 * POLL_AT)]
    assert _run(_tracker(), commands, lambda n: f"waiting {n}s of 30s\n{EMPTY_OK}") == POLL_AT + 1
    states = ["pending"] * (POLL_AT - 1) + ["running"] * (POLL_AT - 1)
    assert _run(_tracker(), commands[: len(states)], lambda n: states[n]) == 0
    # a changed answer starts a new run, and that run needs its own POLL_AT
    answers = ["pending"] * POLL_AT + ["done"] * (POLL_AT - 1)
    assert _run(_tracker(), commands[: len(answers)], lambda n: answers[n]) == 1


def test_a_check_that_starts_failing_has_a_new_answer_and_a_failing_poll_is_still_one() -> None:
    commands = [_status_check(n) for n in range(POLL_AT)]
    tracker = _tracker()
    for n in range(POLL_AT - 1):
        _ran(tracker, _shell(commands[n], n), "connection refused")
    _ran(tracker, _shell(commands[-1], 9), "connection refused", is_error=True)
    assert tracker.poll_run == 1 and not tracker.last_was_poll
    erroring = _tracker()
    for n in range(POLL_AT):
        _ran(erroring, _shell(commands[n], n), "connection refused", is_error=True)
    assert erroring.last_was_poll
    assert SIG_ALL_ERRORS in erroring.last_signals and SIG_POLL in erroring.last_signals


def test_a_completion_is_in_a_run_only_when_every_call_in_it_is_a_shell_call() -> None:
    def pair(n: int) -> list[ToolCall]:
        return [*_shell(f"q.sh a {n}", 2 * n), *_shell(f"q.sh b {n}", 2 * n + 1)]

    tracker = _tracker()
    for n in range(POLL_AT):
        _ran(tracker, pair(n), UNCHANGED)
    assert tracker.last_was_poll  # two shell calls in a completion, six completions in a row
    mixed = _tracker()
    for n in range(POLL_AT - 1):
        _ran(mixed, pair(n), UNCHANGED)
    read = [ToolCall(id="r1", name="read_file", arguments={"path": "x"})]
    _ran(mixed, [*pair(9), *read], UNCHANGED)
    assert mixed.poll_run == 0 and not mixed.last_was_poll
    one = _tracker()
    for n in range(POLL_AT - 1):
        _ran(one, pair(n), UNCHANGED)
    _ran(one, _shell("q.sh a 9", 40), UNCHANGED)  # one of the two checks alone is another question
    assert one.poll_run == 1


def test_a_call_with_no_command_or_no_result_ends_the_run_and_is_never_read() -> None:
    # A provider hands back {"_raw": ...} for arguments it could not decode, and a call can have no
    # result: neither reads as a check, and neither may fail the tracker.
    undecoded = [{"_raw": '{"command": "q.sh'}, {}, {"command": None}, {"command": ["q.sh"]}]
    for arguments in undecoded:
        tracker = _tracker()
        for n in range(1, POLL_AT):
            _poll(tracker, n)
        assert tracker.poll_run == POLL_AT - 1  # the floor: a run stood
        _ran(tracker, [ToolCall(id="u1", name="Bash", arguments=arguments)])
        assert tracker.poll_run == 0 and not tracker.last_was_poll
    unanswered = _tracker()
    for n in range(1, POLL_AT):
        _poll(unanswered, n)
    unanswered.observe(_shell(_status_check(90), 90), [])
    assert unanswered.poll_run == 0 and not unanswered.last_was_poll


def test_only_a_completion_the_narrow_step_turned_away_whole_is_transparent() -> None:
    tracker = _tracker()
    for n in range(1, POLL_AT):
        _poll(tracker, n)
    calls = [
        ToolCall(id="m1", name="Bash", arguments={"command": _status_check(80)}),
        ToolCall(id="m2", name="Bash", arguments={"command": _status_check(81)}),
    ]
    turned_away = ToolResultBlock(
        tool_use_id="m1",
        output="Tool 'Bash' is unavailable on this step.",
        is_error=True,
        data={"step_restricted": True, "tool": "Bash"},
    )
    tracker.observe(calls, [turned_away, ToolResultBlock(tool_use_id="m2", output=EMPTY_OK)])
    assert tracker.poll_run == 1  # one of its two calls ran: a completion of its own, a new run


def _run_named(tracker: StuckTracker, name: str) -> int:
    fired = 0
    for n in range(POLL_AT):
        _ran(tracker, _shell(_status_check(n), n, name))
        fired += tracker.last_was_poll
    return fired


@pytest.mark.parametrize("name", sorted(_SHELL_TOOLS))
def test_every_shell_tool_counts_and_no_other_tool_does(name: str) -> None:
    assert _run_named(_tracker(), name) == 1
    other = _tracker()
    assert _run_named(other, "probe") == 0 and other.poll_run == 0


def test_the_signal_is_inert_without_shell_tools_and_its_threshold_is_the_trackers() -> None:
    inert = StuckTracker(plan_tools=_PLAN_TOOLS)
    assert _run(inert, [_status_check(n) for n in range(3 * POLL_AT)]) == 0
    assert inert.poll_run == 0
    eager = _tracker(poll_at=3)
    assert _run(eager, [_status_check(n) for n in range(3)]) == 1


def test_a_turn_end_veto_and_a_refused_completion_each_end_the_run() -> None:
    tracker = _tracker()
    for n in range(1, POLL_AT):
        _poll(tracker, n)
    tracker.reset()  # a TURN_END veto sent the model round again
    assert tracker.poll_run == 0
    _poll(tracker, 30)
    assert tracker.poll_run == 1
    for n in range(2, POLL_AT):
        _poll(tracker, 30 + n)
    calls = _shell('echo "NEXT1"', 90)
    refusal = ToolResultBlock(
        tool_use_id=calls[0].id,
        output="Not executed",
        is_error=True,
        data={"plan_only_bound": True},
    )
    tracker.observe(calls, [refusal])
    assert tracker.poll_run == 0
    for n in range(POLL_AT - 1):
        _poll(tracker, 100 + n)
        assert not tracker.last_was_poll
    _poll(tracker, 200)
    assert tracker.last_was_poll


def test_a_call_the_narrow_step_turned_away_neither_extends_nor_ends_the_run() -> None:
    tracker = _tracker()
    actions: list[StuckAction] = []
    for n in range(1, POLL_AT + 2):
        _poll(tracker, n)
        actions.append(tracker.next_action())
    assert tracker.poll_run == POLL_AT + 1 and actions[-1] is StuckAction.NARROW
    calls = _shell(_status_check(90), 90)
    turned_away = ToolResultBlock(
        tool_use_id=calls[0].id,
        output="Tool 'Bash' is unavailable on this step.",
        is_error=True,
        data={"step_restricted": True, "tool": "Bash"},
    )
    tracker.observe(calls, [turned_away])
    # It did nothing, so the run stands where it was and the signal does not fire for it. The
    # rung it earns is the ordinary one: a completion of nothing but errors and no progress.
    assert tracker.poll_run == POLL_AT + 1 and not tracker.last_was_poll
    actions.append(tracker.next_action())
    for n in (91, 92):
        _poll(tracker, n)  # the model checks again once the step is over: the run goes on
        assert tracker.last_was_poll
        actions.append(tracker.next_action())
    assert tracker.poll_run == POLL_AT + 3
    assert actions == [
        *[StuckAction.CONTINUE] * 5,
        StuckAction.NUDGE,
        StuckAction.NARROW,
        StuckAction.STEP_BACK,
        StuckAction.STOP,
        StuckAction.STOP,
    ]


# ── what is masked ──────────────────────────────


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("q.sh --n 17", "q.sh --n 3"),
        ("q.sh > out_17.txt", "q.sh > out_3.txt"),
        ("q.sh > a.txt", "q.sh > b.txt"),
        ("q.sh >out.txt", "q.sh   >   other.txt"),
        ("q.sh >> a.log 2>&1", "q.sh >> b.log 2>&1"),
        ("q.sh 2> a.err", "q.sh 2> b.err"),
        ("q.sh &> a.all", "q.sh &>b.all"),
        ("q.sh   a\n b", "q.sh a b"),
    ],
)
def test_commands_that_differ_only_in_numbers_or_redirect_targets_are_one(
    one: str, other: str
) -> None:
    assert one != other
    assert poll_command(one) == poll_command(other)


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("q.sh a", "q.sh b"),
        ("git status", "git diff"),
        ("q.sh 2>&1", "q.sh"),
        ("q.sh 2>&1", "q.sh 2>err"),
        ("q.sh > a", "q.sh >> a"),
        ("ls out", "ls in"),
        ("q.sh --n 1", "q.sh --m 1"),
    ],
)
def test_commands_that_differ_in_anything_else_are_two(one: str, other: str) -> None:
    assert poll_command(one) != poll_command(other)


def test_without_numbers_masked_only_where_the_output_goes_and_the_spacing_are() -> None:
    assert poll_command("py step1.py > out1.txt", numbers=False) == poll_command(
        "py   step1.py > out2.txt", numbers=False
    )
    assert poll_command("py step1.py", numbers=False) != poll_command("py step2.py", numbers=False)
    assert poll_command("py step1.py", numbers=False) == "py step1.py"


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("echo '' > src/api/__init__.py", "echo '' > src/core/__init__.py"),
        ("printf '# header\\n' > a/mod.py", "printf '# header\\n' > b/mod.py"),
        ("echo done >> a.log", "echo done >> b.log"),
        ("echo x > part1.txt", "echo x > part2.txt"),
    ],
)
def test_a_literal_write_keeps_the_file_it_writes_into(one: str, other: str) -> None:
    assert poll_command(one, numbers=False) != poll_command(other, numbers=False)


@pytest.mark.parametrize(
    "template",
    [
        "echo $(cat state) > {}",
        "echo `cat state` > {}",
        "echo $HOME > {}",
        "echo x; bash q.sh > {}",
        "echo x && bash q.sh > {}",
        "echo x | bash q.sh > {}",
        "echo x < state > {}",
        "echo /tmp/jobs/*.done > {}",
        "echo x > a.txt; bash q.sh > {}",
        "bash q.sh; echo ok > {}",
        "echo2 q > {}",
        "printfx q > {}",
    ],
)
def test_a_write_that_reads_or_runs_something_is_still_read_by_what_it_asks(template: str) -> None:
    assert poll_command(template.format("a")) == poll_command(template.format("b"))


@pytest.mark.parametrize("char", list("|;&<()`$*?["))
def test_one_pipe_separator_expansion_or_glob_anywhere_makes_a_write_something_else(
    char: str,
) -> None:
    # In the words, or after the arrow where no space marks the end of the file: `echo x > a;q.sh`
    # writes a file and then runs a command.
    for template in ("echo a{c}b > {t}", "printf a{c}b > {t}", "echo x > {t}{c}q.sh"):
        one = poll_command(template.format(c=char, t="x"))
        assert one == poll_command(template.format(c=char, t="y"))


HOSTILE = {
    "a long run of digits": "0" * 100_000,
    "digits and an arrow": "0" * 100_000 + "> f",
    "arrows": ">" * 100_000,
    "arrows and spaces": "> " * 50_000,
    "a long echo": "echo " + "a " * 100_000 + "> f",
    "an unfinished echo": "echo " + "a" * 100_000,
    "arrows ending in a pipe": "echo " + ">" * 40_000 + "|",
}


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_a_hostile_command_is_masked_in_linear_time(name: str) -> None:
    # The command is read on the event loop, once per shell completion. A `\d*` before the arrow
    # was retried from every digit of a run: 40,000 digits took seconds and 100,000 take a minute.
    started = time.perf_counter()
    poll_command(HOSTILE[name])
    poll_command(HOSTILE[name], numbers=False)
    assert time.perf_counter() - started < 1.0


def test_an_answer_is_its_text_without_numbers_and_only_its_head() -> None:
    assert poll_answer("waiting 17s\n[exit code: 0]") == "waiting #s [exit code: #]"
    assert poll_answer("") == ""
    assert len(poll_answer("word " * 5000)) == 4000
    assert poll_answer("a" * 4000 + "x") == poll_answer("a" * 4000 + "y")  # past the head: unread


# ── what the rails say ──────────────────────────


def test_the_nudge_says_what_was_counted_and_what_to_do_instead() -> None:
    tracker = _tracker()
    for n in range(1, POLL_AT + 1):
        _poll(tracker, n)
    nudge = tracker.nudge_message()
    assert nudge.startswith("You have now run the same check 6 times in a row")
    assert "answered the same each time apart from its numbers" in nudge
    assert "(ScheduleWakeup)" in nudge and "end your turn" in nudge
    for wrong in ("failed", "stuck", "same tool result", "plan tools"):
        assert wrong not in nudge  # said only what the harness knows
    for n in (7, 8):  # the number is the run's length, not the threshold
        _poll(tracker, n)
    assert tracker.nudge_message().startswith("You have now run the same check 8 times in a row")


def test_the_poll_nudge_is_the_one_said_when_the_answers_also_repeat_as_outcomes() -> None:
    tracker = _tracker()
    long_answer = "3 jobs pending, none started, the list has not moved since the last look"
    for n in range(1, POLL_AT + 1):
        _ran(tracker, _shell(_status_check(n), n), long_answer)
    assert SIG_REPEATED_OUTCOME in tracker.last_signals and SIG_POLL in tracker.last_signals
    assert tracker.nudge_message().startswith("You have now run the same check 6 times")
    plain = _tracker()
    for n in range(3):
        _ran(plain, _shell(f"bash q{n}.sh", n), long_answer)
    assert plain.nudge_message().startswith("You have now observed the SAME tool result")


# ── the loop, both turn paths ───────────────────


class _Shell(Tool):
    """Stands in for the shell: every command prints nothing and succeeds. Like the real one it
    is no read-only tool, so a stuck NARROW step turns it away."""

    spec = ToolSpec(
        name="Bash",
        description="runs a command",
        required_permission=PermissionTier.WORKSPACE_WRITE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(EMPTY_OK)


def _shell_loop(
    provider: _ScriptByCallProvider, tmp_path: Path, *, max_iterations: int = 15
) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(_Shell())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=max_iterations,
        settings=Settings(permission_mode="allow"),
    )


def _polls_then_answer() -> _ScriptByCallProvider:
    def script(n: int) -> LLMResult:
        if n <= POLL_AT:
            return LLMResult(tool_calls=[_c(f"c{n}", "Bash", command=_status_check(n))])
        return LLMResult(text="Nothing to act on yet: armed a wake-up and ended the turn.")

    return _ScriptByCallProvider(script)


def _polls_forever() -> _ScriptByCallProvider:
    return _ScriptByCallProvider(
        lambda n: LLMResult(tool_calls=[_c(f"c{n}", "Bash", command=_status_check(n))])
    )


DECIDE = "Decide: act on what the check showed, or wait without checking"


def test_the_loop_adds_one_decision_step_and_says_what_was_counted(tmp_path: Path) -> None:
    loop = _shell_loop(_polls_then_answer(), tmp_path)
    result = asyncio.run(loop.arun_turn("wait for the lock to be released"))
    assert result.stop_reason == "completed" and result.iterations == POLL_AT + 1
    # one step, and none of the steps that would say what is not so: nothing failed
    assert _titles(loop) == [DECIDE]
    added = [r for r in _rails(loop) if "I added 1 investigative steps" in r]
    assert len(added) == 1
    assert "run the same check 6 times in a row" in added[0] and "ScheduleWakeup" in added[0]


def test_the_streaming_twin_adds_the_same_step_and_announces_it(tmp_path: Path) -> None:
    loop = _shell_loop(_polls_then_answer(), tmp_path)
    events = _drain(loop, "wait for the lock to be released")
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.stop_reason == "completed"
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("added 1 investigative steps in the plan" in m for m in statuses)
    assert _titles(loop) == [DECIDE]
    assert any("run the same check 6 times in a row" in r for r in _rails(loop))


def _restricted_results(loop: AgentLoop) -> list[ToolResultBlock]:
    return [
        block
        for message in loop.session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock) and (block.data or {}).get("step_restricted")
    ]


@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
def test_a_model_that_keeps_checking_is_nudged_narrowed_stepped_back_and_stopped(
    tmp_path: Path, streamed: bool
) -> None:
    loop = _shell_loop(_polls_forever(), tmp_path)
    if streamed:
        events = _drain(loop, "wait for the lock to be released")
        stop_reason = next(e for e in events if isinstance(e, AgentDone)).stop_reason
    else:
        stop_reason = asyncio.run(loop.arun_turn("wait for the lock to be released")).stop_reason
    assert stop_reason == "stuck"
    rails = "\n".join(_rails(loop))
    assert "run the same check 6 times in a row" in rails  # the nudge
    assert "only read-only tools are available; Bash is refused" in rails  # the narrow step
    assert "take a step back" in rails
    # the narrow step turned ONE check away, and the run went on past it to the stop
    assert len(_restricted_results(loop)) == 1
    assert loop._trace is not None and [
        e.detail for e in loop._trace.events if e.data.get("kind") == "stuck"
    ][-1].startswith("stuck")


def test_a_model_that_works_between_its_checks_is_never_told(tmp_path: Path) -> None:
    def script(n: int) -> LLMResult:
        if n > 3 * POLL_AT:
            return LLMResult(text="done")
        if n % 2:
            return LLMResult(tool_calls=[_c(f"c{n}", "Bash", command=_status_check(n))])
        return LLMResult(tool_calls=[_c(f"c{n}", "Bash", command=f"git log -n {n}")])

    loop = _shell_loop(_ScriptByCallProvider(script), tmp_path, max_iterations=4 * POLL_AT)
    result = asyncio.run(loop.arun_turn("watch the build"))
    assert result.iterations == 3 * POLL_AT + 1  # the floor: the whole script ran
    assert not any("same check" in r for r in _rails(loop))

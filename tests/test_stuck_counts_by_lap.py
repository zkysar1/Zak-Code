"""ADR-0209: the stuck ladder's repeated-outcome signal counts per LAP of a loop the session is
running, and a receipt of a change is not a look at the world.

A served perpetual loop is ONE turn for the whole run: a turn-end hook refuses every stop and
sends the model round its loop skill again. Counting identical outcomes over that whole turn, a
healthy loop's once-a-lap housekeeping climbed the ladder by itself. Measured on gpt-5.6-luna in
four served runs of 35 minutes (``bench/results/served-luna-preregistration.log``, the ladder
re-read): 47 rungs, 8 of them STOPs; all 33 on a tool that looks at the world had a lap boundary
between their repeats, and the other 14 fell on the plan tool's "Plan updated" receipt while the
model was journalling into its plan between real steps.

This change makes rungs STOP APPEARING, and a dead ladder draws no rung either. So every case
here stands beside a CONTROL that must still climb: the same calls with no lap between them, a
lap that showed nothing new (the bound), a receipt with no work in between (plan churn, which
this signal is the only net for), an unflagged result, a pointer, another skill's body. The
loop-level cases go through the real doors: the turn-end hook and the harness's delivery, the
Skill call answered with a body, and the real plan tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.test_refused_stop_silences_answer_now import (
    MODEL,
    PLAIN_REASON,
    SKILL_REASON,
    _compose,
    _Hook,
    _judge_ok,
    _review_ok,
    _run,
    _say,
    _Scripted,
)
from zakcode.agent.loop import _UNOBSERVING_TOOLS, AgentLoop
from zakcode.agent.stuck import (
    SIG_REPEATED_OUTCOME,
    StuckAction,
    StuckTracker,
    outcome_signature,
)
from zakcode.config import PermissionTier
from zakcode.messages import ToolResultBlock
from zakcode.permissions import PermissionMode, PermissionPolicy
from zakcode.providers.base import LLMResult, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import TaskNetwork
from zakcode.tools import default_registry
from zakcode.tools.base import (
    RECEIPT_OF_CHANGE,
    ConcurrencyClass,
    SkillLoad,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.tools.builtins.use_skill import UseSkillTool
from zakcode.usage import Usage

C, N, NA, SB, ST = (
    StuckAction.CONTINUE,
    StuckAction.NUDGE,
    StuckAction.NARROW,
    StuckAction.STEP_BACK,
    StuckAction.STOP,
)
LADDER = [C, C, N, NA, SB, ST]
#: The once-a-lap housekeeping of a healthy loop: the same answer every time it is asked.
STATE = "state: RUNNING | mode: autonomous | queue: open | nothing is waiting on a person"
RECEIPT = "Plan updated: 2/7 steps done · current: 3 Carry out the goal that was selected"


def _tracker() -> StuckTracker:
    """The tracker exactly as both of the loop's paths build it."""
    return StuckTracker(uncounted_outcome_tools=_UNOBSERVING_TOOLS)


class _Feed:
    """Hands a tracker one single-call iteration at a time, with the loop's three readings."""

    def __init__(self, tracker: StuckTracker) -> None:
        self.tracker, self.n = tracker, 0

    def __call__(
        self,
        output: str,
        *,
        tool: str = "Bash",
        lap: int = 0,
        work: int = 0,
        data: dict[str, Any] | None = None,
        is_error: bool = False,
    ) -> StuckAction:
        self.n += 1
        call = ToolCall(id=f"c{self.n}", name=tool, arguments={"n": self.n})
        block = ToolResultBlock(tool_use_id=call.id, output=output, is_error=is_error, data=data)
        self.tracker.observe([call], [block], assistant_text="working", lap=lap, work=work)
        return self.tracker.next_action()


def _work(lap: int, step: int = 0) -> str:
    return f"ok: the result of step {step} of lap {lap}, which no other lap produces"


# ── the lap ──────────────────────────────────────────────────────────────────


def test_a_healthy_loop_of_many_laps_draws_no_rung() -> None:
    feed = _Feed(_tracker())
    for lap in range(12):
        assert feed(STATE, lap=lap) is C
        assert feed(_work(lap), lap=lap) is C
    assert SIG_REPEATED_OUTCOME not in feed.tracker.last_signals


def test_control_the_same_calls_with_no_lap_between_them_climb_the_whole_ladder() -> None:
    """What the healthy loop above drew until ADR-0209, and what a long ordinary turn that
    keeps re-measuring one thing still draws: the absence above is the lap's doing."""
    feed = _Feed(_tracker())
    actions, evidence = [], {}
    for lap in range(6):
        actions.append(feed(STATE))
        evidence = feed.tracker.evidence()  # of the housekeeping call, before the work call
        assert feed(_work(lap)) is C
    assert actions == LADDER
    assert evidence == {"signals": SIG_REPEATED_OUTCOME, "tool": "Bash", "repeats": 6}


def test_circling_inside_one_lap_still_climbs() -> None:
    """The ADR-0038 incident's shape, late in a served run: several healthy laps, then one
    result again and again between other probes inside a single lap."""
    feed = _Feed(_tracker())
    for lap in range(5):
        assert feed(STATE, lap=lap) is C
        assert feed(_work(lap), lap=lap) is C
    actions = []
    for k in range(6):
        actions.append(feed(STATE, lap=5))
        if k < 5:  # a novel probe between the repeats does not hide them (ADR-0038)
            assert feed(_work(5, k), lap=5) is C
    assert actions == LADDER


def test_the_batch_that_arrives_with_a_new_lap_belongs_to_that_lap() -> None:
    """The loop hands over ``lap`` AFTER the batch ran, so the batch in which the loop skill's
    body arrived carries the new number. The boundary is settled before that batch is counted:
    its sighting is the new lap's first, not the old lap's third."""
    feed = _Feed(_tracker())
    assert feed(STATE, lap=0) is C
    assert feed(_work(0), lap=0) is C
    assert feed(STATE, lap=0) is C  # the second sighting of lap 0
    assert feed(STATE, lap=1) is C  # would be the third: a NUDGE, were it counted first
    assert feed.tracker.evidence() == {"signals": ""}


# ── the bound ────────────────────────────────────────────────────────────────


def test_a_lap_that_showed_nothing_new_does_not_start_the_counts_over() -> None:
    """THE BOUND. A model that probes, stops, is sent round again by the hook and probes the
    same thing sees a lap boundary between every two sightings. Only the first lap showed
    anything new, so only the first boundary starts the counts over: the ladder is one lap
    late, and whole."""
    feed = _Feed(_tracker())
    actions = [feed(STATE, lap=lap) for lap in range(7)]
    assert actions == [C, *LADDER]


def test_new_means_new_to_the_turn_not_to_the_lap() -> None:
    """A spin that alternates two probes from lap to lap shows each lap something the lap
    before it did not have. It is not new to the TURN, so the counts carry on and it climbs."""
    feed = _Feed(_tracker())
    other = "queue: 3 goals open | 0 blocked | the selector has nothing above the threshold"
    actions = [feed(STATE if lap % 2 == 0 else other, lap=lap) for lap in range(12)]
    assert actions[:6] == [C] * 6  # each probe: new once, then counted 1 and 2
    assert actions[6:10] == [N, N, NA, NA]  # both climb, a lap apart
    assert StuckAction.STEP_BACK in actions[10:]


def test_one_new_result_in_a_lap_is_what_makes_it_a_lap_of_work() -> None:
    """The same spin with ONE distinct result in every lap is, to anything that reads outputs,
    a healthy loop: housekeeping that repeats and work that does not. It draws no rung, and
    that is the limit of what this signal can know. A loop that goes round doing nothing worth
    doing is the hook owner's to end (the hook is what keeps sending it round)."""
    feed = _Feed(_tracker())
    for lap in range(10):
        assert feed(STATE, lap=lap) is C
        assert feed(_work(lap), lap=lap) is C


# ── the receipt ──────────────────────────────────────────────────────────────


def test_a_receipt_of_change_does_not_repeat_while_work_succeeds_in_between() -> None:
    """The measured shape: the model journals into its plan between real steps. The receipt
    reads the same each time (same done count, same current step) although the plan moved."""
    feed = _Feed(_tracker())
    for k in range(1, 9):
        assert feed(_work(0, k), work=k) is C
        assert feed(RECEIPT, tool="update_plan", work=k, data={RECEIPT_OF_CHANGE: True}) is C
    assert SIG_REPEATED_OUTCOME not in feed.tracker.last_signals


def test_control_the_same_receipt_with_no_work_in_between_is_plan_churn_and_climbs() -> None:
    """A model that rewrites its plan again and again and does nothing else: this signal is
    the only net under it (ADR-0192 names the bill that churn once ran up), so it holds."""
    feed = _Feed(_tracker())
    flagged = {RECEIPT_OF_CHANGE: True}
    actions = [feed(RECEIPT, tool="update_plan", work=4, data=flagged) for _ in range(6)]
    assert actions == LADDER
    assert feed.tracker.evidence()["tool"] == "update_plan"


def test_the_rail_and_the_note_for_plan_churn_say_what_is_true_of_a_receipt() -> None:
    """The ladder's words for a repeat ("observed the SAME tool result ... without changing
    anything in between") are false of a receipt twice over: it is no observation, and the
    model DID change its plan each time. What is true: the tool was called N times, nothing
    else succeeded in between, and the receipt read the same. The trace note says it was a
    receipt, so a reader can tell plan churn from a repeated look at the world."""
    feed = _Feed(_tracker())
    flagged = {RECEIPT_OF_CHANGE: True}
    actions = [feed(RECEIPT, tool="update_plan", work=4, data=flagged) for _ in range(3)]
    assert actions[-1] is N and feed.tracker.last_outcome_was_receipt
    said = feed.tracker.nudge_message()
    assert "called update_plan 3 times with no other work succeeding in between" in said
    assert "observed the SAME tool result" not in said and "without changing" not in said
    assert feed.tracker.evidence() == {
        "signals": SIG_REPEATED_OUTCOME,
        "tool": "update_plan",
        "repeats": 3,
        "receipt": True,
    }


def test_control_a_repeated_look_at_the_world_keeps_its_words_and_its_note() -> None:
    feed = _Feed(_tracker())
    actions = [feed(STATE, work=4) for _ in range(3)]
    assert actions[-1] is N and not feed.tracker.last_outcome_was_receipt
    assert "observed the SAME tool result 3 times" in feed.tracker.nudge_message()
    assert feed.tracker.evidence() == {
        "signals": SIG_REPEATED_OUTCOME,
        "tool": "Bash",
        "repeats": 3,
    }


def test_the_words_follow_the_result_the_batch_counted_highest() -> None:
    """One batch can hold a look and a receipt. The rail speaks of the one the tracker counted
    highest, which is the one ``evidence`` names: the two are read off the same fact."""

    def batch(tracker: StuckTracker, n: int, *, look: str, work: int) -> None:
        calls = [
            ToolCall(id=f"look{n}", name="Bash", arguments={"n": n}),
            ToolCall(id=f"plan{n}", name="update_plan", arguments={"n": n}),
        ]
        blocks = [
            ToolResultBlock(tool_use_id=f"look{n}", output=look),
            ToolResultBlock(tool_use_id=f"plan{n}", output=RECEIPT, data={RECEIPT_OF_CHANGE: True}),
        ]
        tracker.observe(calls, blocks, assistant_text="working", work=work)

    looked = _tracker()  # the look repeats; work moves, so each receipt is a new one
    for n in range(3):
        batch(looked, n, look=STATE, work=n + 1)
    assert (looked.evidence()["tool"], looked.last_outcome_was_receipt) == ("Bash", False)
    assert "observed the SAME tool result 3 times" in looked.nudge_message()

    # The other way round: every look reads differently, and the work count is held still (as
    # the loop holds it when nothing between two receipts SUCCEEDED), so the receipt repeats.
    churned = _tracker()
    for n in range(3):
        batch(churned, n, look=_work(0, n), work=7)
    assert (churned.evidence()["tool"], churned.last_outcome_was_receipt) == ("update_plan", True)
    assert "called update_plan 3 times" in churned.nudge_message()


def test_control_an_unflagged_result_repeats_whatever_work_ran_in_between() -> None:
    """ "Plan unchanged" carries no flag: it answers the same plan sent again, which is the
    churn. So does every look at the world. Work in between hides neither (ADR-0038)."""
    feed = _Feed(_tracker())
    unchanged = "Plan unchanged: 2/7 steps done · current: 3 Carry out the goal. Nothing moved"
    actions = []
    for k in range(1, 7):
        assert feed(_work(0, k), work=k) is C
        actions.append(feed(unchanged, tool="update_plan", work=k, data={"unchanged": True}))
    assert actions == LADDER


def test_an_errored_result_is_never_a_receipt_of_change() -> None:
    feed = _Feed(_tracker())
    flagged = {RECEIPT_OF_CHANGE: True}
    failed = "update_plan failed: the plan could not be saved to the session store (disk full)"
    actions = []
    for k in range(1, 4):
        assert feed(_work(0, k), work=k) is C
        actions.append(feed(failed, tool="update_plan", work=k, data=flagged, is_error=True))
    assert actions[-1] is N  # counted like any result: nothing changed, so nothing is excused


def test_a_receipt_is_never_the_something_new_of_a_lap() -> None:
    """The bound again, with a plan in the spin: probe, update the plan, stop, be sent round.
    The probe succeeded, so the work count moved and the receipt's identity with it. Were that
    "something new", every such lap would start the counts over and the probe never climb."""
    feed = _Feed(_tracker())
    actions = []
    for lap in range(7):
        actions.append(feed(STATE, lap=lap, work=lap + 1))
        flagged = {RECEIPT_OF_CHANGE: True}
        assert feed(RECEIPT, tool="update_plan", lap=lap, work=lap + 1, data=flagged) is C
    assert actions == [C, *LADDER]


def test_the_signature_keys_a_receipt_on_the_work_count_and_nothing_else_on_it() -> None:
    plain = outcome_signature("update_plan", RECEIPT, 2)
    assert plain is not None and plain == outcome_signature("update_plan", RECEIPT, 2, work=None)
    at_4 = outcome_signature("update_plan", RECEIPT, 2, work=4)
    assert at_4 is not None and at_4 != plain
    assert at_4 == outcome_signature("update_plan", RECEIPT, 2, work=4)
    assert at_4 != outcome_signature("update_plan", RECEIPT, 2, work=5)
    assert outcome_signature("update_plan", "ok", 0, work=4) is None  # still too short to count


# ── the real plan tool says which receipt is which ───────────────────────────

PLAN = [
    {"title": "Select the goal", "status": "done", "outcome": "g-1 selected"},
    {"title": "Carry out the goal that was selected", "status": "in_progress", "note": "began"},
    {"title": "Close the iteration"},
]


async def test_the_plan_tool_flags_plan_updated_and_neither_unchanged_receipt() -> None:
    ctx = ToolContext(workspace_root=Path("/tmp"), task_network=TaskNetwork())
    first = await UpdatePlanTool().execute({"tasks": PLAN}, ctx)
    noted = [PLAN[0], dict(PLAN[1], note="began; the script ran clean"), PLAN[2]]
    second = await UpdatePlanTool().execute({"tasks": noted}, ctx)
    for result in (first, second):
        assert result.output.startswith("Plan updated: 1/3 steps done")
        assert result.data is not None and result.data[RECEIPT_OF_CHANGE] is True
    assert first.output == second.output  # the plan moved; the receipt reads the same

    fresh = ToolContext(workspace_root=Path("/tmp"), task_network=TaskNetwork())
    untouched = [{"title": "Look around"}, {"title": "Then act"}]
    await UpdatePlanTool().execute({"tasks": untouched}, fresh)
    resent = await UpdatePlanTool().execute({"tasks": untouched}, fresh)
    assert resent.output.startswith("Plan unchanged")
    assert resent.data is not None and RECEIPT_OF_CHANGE not in resent.data

    # A step that carries its result and is sent back as it stands is advanced FOR the model
    # (ADR-0168 lever N). The model changed nothing, so that receipt is no receipt of change.
    worked = [PLAN[0], dict(PLAN[1], outcome="the script ran clean"), PLAN[2]]
    sent = await UpdatePlanTool().execute({"tasks": worked}, ctx)
    assert sent.data is not None and sent.data[RECEIPT_OF_CHANGE] is True
    advanced = await UpdatePlanTool().execute({"tasks": worked}, ctx)
    assert advanced.data is not None and advanced.data.get("autoadvanced") is True
    assert RECEIPT_OF_CHANGE not in advanced.data


# ── the loop: both doors, on both of its paths ───────────────────────────────


class _Probe(Tool):
    spec = ToolSpec(
        name="probe",
        description="the loop's housekeeping: the same answer every time",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(STATE)


class _Work(Tool):
    spec = ToolSpec(
        name="work",
        description="distinct, successful work",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(f"ok: step {args.get('n')} finished with a result of its own")


class _Resolver:
    """Answers the REAL Skill tool. ``pointers`` names the skills answered with the loader's
    "[already loaded]" pointer; every other known skill gets its body, flagged a re-entry."""

    def __init__(self, pointers: frozenset[str] = frozenset()) -> None:
        self._pointers = pointers

    def names(self) -> list[str]:
        return ["cycle", "report"]

    def body(self, name: str) -> str | None:
        return f"# {name.title()}\n\nOpen the next {name} and carry it out.\n"

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        if name in self._pointers:
            text = f"[already loaded] the instructions for skill {name!r} are in your context."
            return SkillLoad(found=True, name=name, body=text, pointer=True)
        return SkillLoad(found=True, name=name, body=self.body(name), reentry=True)


def _call(tool: str, n: int, **arguments: object) -> LLMResult:
    call = ToolCall(id=f"{tool}-{n}", name=tool, arguments={"n": n, **arguments})
    return LLMResult(text="", tool_calls=[call], usage=Usage(total_tokens=1))


def _served(
    provider: _Scripted, tmp_path: Path, hook: _Hook, pointers: frozenset[str] = frozenset()
) -> AgentLoop:
    """The ADR-0205 tests' loop (a vetoable turn, an in-process hook, a fake composer) with the
    two tools of a lap and the real Skill tool in front of a scripted resolver."""
    registry = default_registry()
    for tool in (_Probe(), _Work()):
        registry.register(tool)
    registry.register(UseSkillTool(), aliases=["use_skill"])
    loop = AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model=MODEL),
        workspace_root=tmp_path,
        max_iterations=60,
        turn_end_vetoable=True,
        compose_skill=_compose,
        skill_resolver=_Resolver(pointers),
        permission_policy=PermissionPolicy(PermissionMode.ALLOW),
    )
    loop.hook_manager.register_turn_end(hook)
    return loop


def _stuck_notes(loop: AgentLoop) -> list[tuple[str, int]]:
    """``(tool, repeats)`` of every ladder note on the turn's trace, in order."""
    notes = [e.data for e in loop._trace.events if e.data.get("kind") == "stuck"]
    return [(str(n.get("tool")), int(n.get("repeats") or 0)) for n in notes]


def _lap_of_work(n: int) -> list[LLMResult]:
    return [_call("probe", n), _call("work", n), _say(f"lap {n} is done")]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_loop_the_hook_sends_round_draws_no_rung(tmp_path: Path, streamed: bool) -> None:
    """Door one. Four laps of housekeeping and work; the hook refuses the first three stops,
    naming the loop skill, and the harness delivers it each time (three, so the ADR-0187 fence
    stays out of it)."""
    script = [r for n in range(4) for r in _lap_of_work(n)]
    provider, hook = _Scripted(script), _Hook([SKILL_REASON] * 3)
    loop = _served(provider, tmp_path, hook)
    assert await _run(loop, "run the loop", streamed=streamed) == "completed"
    assert (hook.consulted, provider.calls, loop._loop_laps) == (4, len(script), 3)
    assert loop.session.loop_skill == "cycle loop"
    assert _stuck_notes(loop) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_control_a_refusal_in_plain_words_is_no_lap(tmp_path: Path, streamed: bool) -> None:
    """The same four laps under a hook that names no skill. Nothing says a loop is running, so
    nothing is a lap, and the third and fourth sighting draw the rungs they always drew."""
    script = [r for n in range(4) for r in _lap_of_work(n)]
    provider, hook = _Scripted(script), _Hook([PLAIN_REASON] * 3)
    loop = _served(provider, tmp_path, hook)
    assert await _run(loop, "run the loop", streamed=streamed) == "completed"
    assert (hook.consulted, loop._loop_laps) == (4, 0)
    assert _stuck_notes(loop) == [("probe", 3), ("probe", 4)]


def _own_loads(skill: str, laps: int) -> list[LLMResult]:
    """Laps the model starts itself: its own Skill call, the housekeeping, distinct work."""
    script: list[LLMResult] = []
    for n in range(laps):
        script += [_call("Skill", n, skill=skill, args="loop"), *_lap_of_work(n)[:2]]
    return [*script, _say("the run is over")]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_the_loop_skills_body_answering_the_models_own_call_is_a_lap(
    tmp_path: Path, streamed: bool
) -> None:
    """Door two. A hook named the loop skill in an earlier turn (the session keeps it); in this
    one the model goes round by loading that skill itself, and gets the BODY each time."""
    provider = _Scripted(_own_loads("cycle", 5))
    loop = _served(provider, tmp_path, _Hook([]))
    loop.session.loop_skill = "cycle loop"
    assert await _run(loop, "run the loop", streamed=streamed) == "completed"
    assert loop._loop_laps == 5
    assert _stuck_notes(loop) == []


@pytest.mark.asyncio
async def test_control_a_pointer_is_not_a_lap(tmp_path: Path) -> None:
    provider = _Scripted(_own_loads("cycle", 5))
    loop = _served(provider, tmp_path, _Hook([]), pointers=frozenset({"cycle"}))
    loop.session.loop_skill = "cycle loop"
    assert (await loop.arun_turn("run the loop")).stop_reason == "completed"
    assert loop._loop_laps == 0
    assert _stuck_notes(loop) == [("probe", 3), ("probe", 4), ("probe", 5)]


@pytest.mark.asyncio
async def test_control_the_body_of_another_skill_is_not_a_lap(tmp_path: Path) -> None:
    """A model that hops between skills while it circles is still circling."""
    provider = _Scripted(_own_loads("report", 5))
    loop = _served(provider, tmp_path, _Hook([]))
    loop.session.loop_skill = "cycle loop"
    assert (await loop.arun_turn("run the loop")).stop_reason == "completed"
    assert loop._loop_laps == 0
    assert _stuck_notes(loop) == [("probe", 3), ("probe", 4), ("probe", 5)]


@pytest.mark.asyncio
async def test_control_no_skill_is_the_loop_until_a_hook_has_named_one(tmp_path: Path) -> None:
    provider = _Scripted(_own_loads("cycle", 5))
    loop = _served(provider, tmp_path, _Hook([]))
    assert loop.session.loop_skill == ""
    assert (await loop.arun_turn("run the loop")).stop_reason == "completed"
    assert loop._loop_laps == 0
    assert _stuck_notes(loop) == [("probe", 3), ("probe", 4), ("probe", 5)]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_model_that_stops_after_every_probe_is_still_caught(
    tmp_path: Path, streamed: bool
) -> None:
    """THE BOUND, through both doors at once, where nothing else would end it: every lap the
    model loads the loop skill itself (so the ADR-0187 fence starts over each time), probes,
    stops in words and is refused. No lap after the first shows anything new, so the counts
    carry on and the ladder ends the turn. The hook is out of reasons by then and lets it."""
    lap = [_call("Skill", 0, skill="cycle", args="loop"), _call("probe", 0), _say("nothing to do")]
    provider, hook = _Scripted(lap * 12), _Hook([SKILL_REASON] * 6)
    loop = _served(provider, tmp_path, hook)
    assert await _run(loop, "run the loop", streamed=streamed) == "stuck"
    # One lap late (the first lap WAS new), and whole: nudge, read-only, step back, stop.
    assert _stuck_notes(loop) == [("probe", 3), ("probe", 4), ("probe", 5), ("probe", 6)]
    assert hook.consulted == 7  # six stops in words refused, then the ladder's own stop let go


# ── the loop: the real plan tool's receipt ───────────────────────────────────


def _plan_noted(k: int) -> LLMResult:
    """The same three steps, the second in progress, its note carrying what was learned so
    far. The done count and the current step never move, so neither does the receipt."""
    tasks = [
        {"title": "Select the goal", "status": "done", "outcome": "g-1 selected"},
        {"title": "Carry out the goal", "status": "in_progress", "note": f"so far: {k} checks"},
        {"title": "Close the iteration"},
    ]
    call = ToolCall(id=f"plan-{k}", name="update_plan", arguments={"tasks": tasks})
    return LLMResult(text="", tool_calls=[call], usage=Usage(total_tokens=1))


def _journal(entries: int, *, work_in_between: bool) -> list[LLMResult]:
    script = [_plan_noted(0), _judge_ok()]  # the judge reads a turn's first plan: a side call
    for k in range(1, entries):
        if work_in_between:
            script.append(_call("work", k))
        script.append(_plan_noted(k))
    return [*script, _say("the goal is carried out: the answer"), _review_ok()]


def _receipts(loop: AgentLoop) -> list[str]:
    """The output of every ``update_plan`` result of the session, in order."""
    return [
        block.output.split("\n", 1)[0]
        for message in loop.session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock) and block.tool_use_id.startswith("plan-")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_model_journalling_into_its_plan_between_steps_draws_no_rung(
    tmp_path: Path, streamed: bool
) -> None:
    provider = _Scripted(_journal(7, work_in_between=True))
    loop = _served(provider, tmp_path, _Hook([]))
    await _run(loop, "carry out the goal", streamed=streamed)
    receipts = _receipts(loop)
    # THE PREMISE, from the real tool: seven edits that each changed the plan, one receipt text.
    assert len(receipts) == 7 and len(set(receipts)) == 1
    assert receipts[0].startswith("Plan updated: 1/3 steps done")
    assert _stuck_notes(loop) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_control_rewriting_the_plan_with_nothing_in_between_still_climbs(
    tmp_path: Path, streamed: bool
) -> None:
    """The same seven edits with no work between them: plan churn. The third identical receipt
    draws the first rung, as it did before ADR-0209."""
    provider = _Scripted(_journal(7, work_in_between=False))
    loop = _served(provider, tmp_path, _Hook([]))
    await _run(loop, "carry out the goal", streamed=streamed)
    assert _stuck_notes(loop)[:1] == [("update_plan", 3)]
    # ...and it is NAMED for what it is, on the trace and on the wire (the rail is a user
    # message of the session): a receipt come back with no work in between, not a result
    # observed again with nothing changed.
    first = next(e.data for e in loop._trace.events if e.data.get("kind") == "stuck")
    assert first.get("receipt") is True
    rails = [m.text for m in loop.session.messages if m.role == "user"]
    assert any("called update_plan 3 times with no other work succeeding" in r for r in rails)
    assert not any("observed the SAME tool result" in r for r in rails)


def test_the_step_seeded_for_plan_churn_is_the_work_and_not_a_re_measurement(
    tmp_path: Path,
) -> None:
    """Rung 1 turns the evidence into plan steps (ADR-0057). For a repeated look the step asks
    what the result already tells you. A receipt tells nothing, so the step is the work itself,
    and one successful work call is also what makes the next receipt a new one. CONTROL: the
    repeated look keeps its step."""
    loop = _served(_Scripted([_say("unused")]), tmp_path, _Hook([]))
    churn, look = _Feed(_tracker()), _Feed(_tracker())
    for _ in range(3):
        churn(RECEIPT, tool="update_plan", work=4, data={RECEIPT_OF_CHANGE: True})
        look(STATE, work=4)
    seeded = [step.title for step in loop._seed_investigation_steps(churn.tracker)]
    assert seeded[0] == "Do: one piece of actual work on the step you were on"
    assert not any("re-measuring" in title for title in seeded)
    seeded = [step.title for step in loop._seed_investigation_steps(look.tracker)]
    assert seeded[0] == "Investigate: what the result you keep re-measuring already tells you"
    assert not any(title.startswith("Do:") for title in seeded)

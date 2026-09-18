"""The stuck ladder counts observations of the world, never the harness's own deliveries
(ADR-0038, amended 2026-09-18).

Measured on a served Mind loop (gpt-5.6-luna, 2026-09-18, 316 calls). The framework's stop hook
orders ``Skill('aspirations') with args='loop'``; the harness has already delivered that skill,
so the loader answers the model's call with the same "[already loaded]" pointer every time. The
3rd, 4th and 5th pointer of one turn drew nudge, narrow and step-back although distinct,
successful work ran between them and the tracker was reset at every veto; the step-back rail
landed on the completion right after a veto, the model answered in text, and the turn ended
``veto_stall``. One turn later the graceful-stop body, asked for again after work six times,
drew the whole ladder and a STOP in the middle of the stop itself.

Every case here has its control beside it: the same tracker still climbs on a repeated
observation of the world, so the absence these tests pin is the exemption's doing and not a
ladder that stopped working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import _UNOBSERVING_TOOLS, AgentLoop
from zakcode.agent.stuck import (
    SIG_REPEATED_OUTCOME,
    StuckAction,
    StuckTracker,
    outcome_signature,
)
from zakcode.config import PermissionTier, load_settings
from zakcode.evals.harness import ScriptedProvider, call_tool, reply
from zakcode.messages import ToolResultBlock
from zakcode.permissions import PermissionMode, PermissionPolicy
from zakcode.providers.base import ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins.schedule_wakeup import ScheduleWakeupTool
from zakcode.wakeup import LOOP_SENTINEL, WakeupSlot

#: The loader's reply to a skill that is already in context, as ``SkillLoader`` writes it.
POINTER = (
    "[arguments: loop]\n\n[already loaded] Nothing new was loaded: the full instructions for "
    "skill 'aspirations' are already in your context THIS turn, unchanged — in the /command "
    "message you were given, or an earlier Skill result — and you have run no tool on them "
    "since they arrived. Loading a skill does not run it. Carry those instructions out now, "
    "starting from their first step: your next action is that step's tool call, not another "
    "Skill call and not a summary."
)
PROBE_OUTPUT = (
    "[runner-claim] acquire: HELD (backend=local) — another machine owns a live claim\n"
    "ACQUIRE_RC=4\n[exit code: 0]"
)
LADDER = [
    StuckAction.CONTINUE,
    StuckAction.CONTINUE,
    StuckAction.NUDGE,
    StuckAction.NARROW,
    StuckAction.STEP_BACK,
    StuckAction.STOP,
]


def _observe(tracker: StuckTracker, n: int, name: str, output: str, **args: object) -> StuckAction:
    call = ToolCall(id=f"c{n}", name=name, arguments=dict(args))
    tracker.observe(
        [call], [ToolResultBlock(tool_use_id=call.id, output=output)], assistant_text="working"
    )
    return tracker.next_action()


def _loop_tracker() -> StuckTracker:
    """The tracker exactly as both of the loop's paths build it."""
    return StuckTracker(uncounted_outcome_tools=_UNOBSERVING_TOOLS)


# ── the tracker ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["Skill", "use_skill"])
def test_the_loaders_pointer_never_climbs_the_ladder(name: str) -> None:
    """The measured shape: distinct work, a veto (the loop resets the tracker), the pointer."""
    tracker = _loop_tracker()
    n = 0
    for door in range(1, 8):
        for step in range(3):
            n += 1
            out = f"ok: a distinct result for door {door} step {step}, long enough to count"
            assert _observe(tracker, n, "Bash", out, command=f"s-{door}-{step}") is (
                StuckAction.CONTINUE
            )
        tracker.reset()
        n += 1
        assert _observe(tracker, n, name, POINTER, skill="aspirations", args="loop") is (
            StuckAction.CONTINUE
        )
        assert SIG_REPEATED_OUTCOME not in tracker.last_signals


def test_control_the_same_pointer_climbs_a_tracker_that_does_not_exempt_it() -> None:
    """Without the exemption the pointer IS a repeated outcome: the defect, pinned."""
    tracker = StuckTracker()
    actions = [
        _observe(tracker, i, "Skill", POINTER, skill="aspirations", args="loop")
        for i in range(1, 7)
    ]
    # Same arguments every time, so the repeated-batch signal joins in from the second call;
    # the ladder is the same one the served run climbed.
    assert actions == LADDER


def test_control_a_repeated_observation_of_the_world_still_climbs_the_loop_tracker() -> None:
    tracker = _loop_tracker()
    actions = [
        _observe(tracker, i, "Bash", PROBE_OUTPUT, command=f"# try {i}\ngit for-each-ref")
        for i in range(1, 7)
    ]
    assert actions == LADDER
    assert tracker.evidence() == {"signals": SIG_REPEATED_OUTCOME, "tool": "Bash", "repeats": 6}


def test_the_exemption_is_per_call_not_per_batch() -> None:
    """A pointer riding in the same batch does not hide the probe beside it."""
    tracker = _loop_tracker()
    actions: list[StuckAction] = []
    for i in range(1, 4):
        skill = ToolCall(id=f"s{i}", name="Skill", arguments={"skill": "aspirations"})
        probe = ToolCall(id=f"p{i}", name="Bash", arguments={"command": f"probe {i}"})
        tracker.observe(
            [skill, probe],
            [
                ToolResultBlock(tool_use_id=skill.id, output=POINTER),
                ToolResultBlock(tool_use_id=probe.id, output=PROBE_OUTPUT),
            ],
            assistant_text="x",
        )
        actions.append(tracker.next_action())
    assert actions == [StuckAction.CONTINUE, StuckAction.CONTINUE, StuckAction.NUDGE]
    assert tracker.evidence() == {"signals": SIG_REPEATED_OUTCOME, "tool": "Bash", "repeats": 3}


def test_every_other_signal_still_sees_a_skill_call() -> None:
    """A skill call that FAILS the same way three times is stuck, exempt tool or not."""
    tracker = _loop_tracker()
    actions: list[StuckAction] = []
    for i in range(1, 4):
        call = ToolCall(id=f"c{i}", name="Skill", arguments={"skill": "no-such-skill"})
        tracker.observe(
            [call],
            [ToolResultBlock(tool_use_id=call.id, output="unknown skill", is_error=True)],
            assistant_text="",
        )
        actions.append(tracker.next_action())
    assert actions[-1] is StuckAction.NUDGE
    assert SIG_REPEATED_OUTCOME not in tracker.last_signals
    assert "all-errors" in tracker.evidence()["signals"]  # type: ignore[operator]


# ── the wake-up's acknowledgement, from the real tool ────────────────────────


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


async def test_a_re_armed_wakeup_acknowledges_identically_and_is_not_counted(
    tmp_path: Path,
) -> None:
    """A Mind's loop re-arms its deadman net before EVERY re-entry, by contract. The tool's
    acknowledgement differs only in a clock time the signature masks, so a healthy loop's
    fourth re-arm in one turn read as a third identical observation."""
    clock = _Clock()
    slot = WakeupSlot(Session(cwd="/w", model="test"), clock=clock)
    ctx = ToolContext(workspace_root=tmp_path, wakeup_slot=slot)
    acks: list[str] = []
    for _ in range(8):
        clock.now += 431.0  # a different due time every time
        res = await ScheduleWakeupTool().execute({"prompt": LOOP_SENTINEL}, ctx)
        assert not res.is_error, res.output
        acks.append(res.output)
    assert len(set(acks[1:])) > 1  # the raw text differs (the due time) ...
    signatures = {outcome_signature("ScheduleWakeup", ack) for ack in acks[1:]}
    assert len(signatures) == 1 and None not in signatures  # ... the detector's view does not

    exempt, plain = _loop_tracker(), StuckTracker()
    exempt_actions, plain_actions = [], []
    for i, ack in enumerate(acks[1:], start=1):
        # Different arguments each time, so only the OUTCOME can repeat.
        exempt_actions.append(_observe(exempt, i, "ScheduleWakeup", ack, reason=f"iteration {i}"))
        plain_actions.append(_observe(plain, i, "ScheduleWakeup", ack, reason=f"iteration {i}"))
    assert set(exempt_actions) == {StuckAction.CONTINUE}
    assert plain_actions[:6] == LADDER  # the control: uncounted is what keeps it quiet


# ── the loop, on both of its paths ───────────────────────────────────────────


class _Pointer(Tool):
    """Stands in for the skill tool at the veto door: the same pointer, every time."""

    spec = ToolSpec(
        name="Skill",
        description="answers with the loader's pointer",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(POINTER)


class _Work(Tool):
    spec = ToolSpec(
        name="work",
        description="distinct, successful work",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(f"ok: step {args.get('n')} finished with a result of its own")


class _Probe(Tool):
    spec = ToolSpec(
        name="probe",
        description="a probe whose answer never changes",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(PROBE_OUTPUT)


def _loop(tmp_path: Path, script: list[Any]) -> AgentLoop:
    registry = ToolRegistry()
    for tool in (_Pointer(), _Work(), _Probe()):
        registry.register(tool)
    return AgentLoop(
        ScriptedProvider(script),
        registry,
        Session(cwd=str(tmp_path), model="test"),
        settings=load_settings(workspace_root=tmp_path),
        workspace_root=tmp_path,
        permission_policy=PermissionPolicy(PermissionMode.ALLOW),
        max_iterations=60,
    )


async def _run(loop: AgentLoop, prompt: str, path: str) -> None:
    if path == "buffered":
        await loop.arun_turn(prompt)
        return
    async for _ in loop.astream_turn(prompt):
        pass


def _stuck_notes(loop: AgentLoop) -> list[dict[str, Any]]:
    return [e.data for e in loop._trace.events if e.data.get("kind") == "stuck"]


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_six_pointers_between_distinct_work_end_the_turn_clean(
    tmp_path: Path, path: str
) -> None:
    script: list[Any] = []
    for door in range(1, 7):
        script.append(call_tool("work", {"n": door}, id=f"w{door}"))
        script.append(call_tool("Skill", {"skill": "aspirations", "args": "loop"}, id=f"s{door}"))
    script.append(reply("done"))
    loop = _loop(tmp_path, script)
    await _run(loop, "run the loop", path)
    assert _stuck_notes(loop) == []
    rails = [m.text for m in loop.session.messages if m.role == "user"]
    assert not any("observed the SAME tool result" in r for r in rails)
    assert loop.session.last_stop_reason == "completed"


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_control_six_identical_probes_still_end_the_turn_stuck_and_say_why(
    tmp_path: Path, path: str
) -> None:
    script = [call_tool("probe", {"q": f"variant {i}"}, id=f"p{i}") for i in range(1, 9)]
    loop = _loop(tmp_path, script)
    await _run(loop, "find the stale claim ref", path)
    assert loop.session.last_stop_reason == "stuck"
    notes = _stuck_notes(loop)
    assert [n["repeats"] for n in notes] == [3, 4, 5, 6]  # nudge, narrow, step back, stop
    assert {n["tool"] for n in notes} == {"probe"}
    assert all(n["signals"] == SIG_REPEATED_OUTCOME for n in notes)

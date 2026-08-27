"""Repeated-outcome ladder (ADR-0038) and its file-edit epoch exemption.

Field incident 2026-08-27 (coach on zc-03): 135 iterations, 103 minutes, 10.5M tokens. The
model re-ran the same probe with a different comment each time, every command wrapped in
``|| echo`` so nothing ever errored, and observed the same 5-line output ~15 times. The doom
guard needs byte-identical consecutive batches; every stuck signal keyed on an error. Nothing
fired. These tests pin the fifth signal: the same tool, the same output, no edit in between.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop
from zakcode.agent.stuck import (
    SIG_REPEATED_OUTCOME,
    StuckAction,
    StuckTracker,
    outcome_signature,
)
from zakcode.config import PermissionTier
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

PROBE_OUTPUT = (
    "---\n---\nNo mind refs in packed-refs\n---\n"
    "[runner-claim] acquire: HELD (backend=local) — another machine owns a live claim\n"
    "ACQUIRE_RC=4\n[exit code: 0]"
)


def _c(i: int, name: str = "run", **args: object) -> ToolCall:
    return ToolCall(id=f"c{i}", name=name, arguments=dict(args))


def _r(i: int, output: str = PROBE_OUTPUT, *, is_error: bool = False) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=f"c{i}", output=output, is_error=is_error)


# ── the signature ────────────────────────────────────────────────────────────


def test_outcome_signature_masks_volatile_fragments_and_keys_on_tool_and_epoch() -> None:
    a = outcome_signature("run", "pid 4242 at 2026-08-27T01:12:03 took 0.31s: still HELD here", 0)
    b = outcome_signature("run", "pid 9876 at 2026-08-27T02:55:41 took 0.87s: still HELD here", 0)
    assert a is not None and a == b  # pids, clock times and durations do not distinguish
    assert outcome_signature("run", "ok", 0) is None  # too short to mean anything
    assert outcome_signature("run", PROBE_OUTPUT, 0) != outcome_signature("run", PROBE_OUTPUT, 1)
    assert outcome_signature("run", PROBE_OUTPUT, 0) != outcome_signature("read", PROBE_OUTPUT, 0)


# ── the ladder ───────────────────────────────────────────────────────────────


def test_same_result_with_varied_commands_climbs_nudge_narrow_step_back_stop() -> None:
    tracker = StuckTracker()
    actions: list[StuckAction] = []
    nudges: list[str] = []
    for i in range(1, 7):
        # A different comment every time — the doom guard never sees a repeat.
        tracker.observe(
            [_c(i, command=f"# probe {i}\ngit for-each-ref")], [_r(i)], assistant_text="checking"
        )
        action = tracker.next_action()
        actions.append(action)
        if action is StuckAction.NUDGE:
            nudges.append(tracker.nudge_message())
    assert actions == [
        StuckAction.CONTINUE,
        StuckAction.CONTINUE,
        StuckAction.NUDGE,
        StuckAction.NARROW,
        StuckAction.STEP_BACK,
        StuckAction.STOP,
    ]
    assert SIG_REPEATED_OUTCOME in tracker.last_signals
    assert nudges and "3 times" in nudges[0] and "Re-measuring" in nudges[0]


def test_interleaved_novel_probes_do_not_hide_the_repeat() -> None:
    """The field loop alternated probes: a consecutive streak never formed."""
    tracker = StuckTracker()
    actions: list[StuckAction] = []
    for i in range(1, 6):
        output = (
            PROBE_OUTPUT if i % 2 else f"a novel result number {i} that is long enough to count"
        )
        tracker.observe([_c(i, command=str(i))], [_r(i, output)], assistant_text="x")
        actions.append(tracker.next_action())
    assert actions[-1] is StuckAction.NUDGE  # the third identical observation, at i=5


def test_a_file_edit_between_identical_results_is_progress_not_a_loop() -> None:
    tracker = StuckTracker()
    for i, epoch in enumerate((0, 1, 2), start=1):  # edit → test → edit → test
        tracker.observe([_c(i, command="pytest -q")], [_r(i)], assistant_text="fixing", epoch=epoch)
        assert tracker.next_action() is StuckAction.CONTINUE
    assert SIG_REPEATED_OUTCOME not in tracker.last_signals


def test_short_acknowledgements_never_count() -> None:
    tracker = StuckTracker()
    for i in range(1, 8):
        tracker.observe(
            [_c(i, "write_file", path=f"f{i}.py")], [_r(i, "File written.")], assistant_text="w"
        )
        assert tracker.next_action() is StuckAction.CONTINUE


def test_identical_error_outputs_count_too() -> None:
    tracker = StuckTracker()
    err = "fatal: could not read Username for 'https://github.com': No such device or address"
    for i in range(1, 4):
        tracker.observe(
            [_c(i, command=f"git push --tags # try {i}")],
            [_r(i, err, is_error=True)],
            assistant_text="t",
        )
    assert SIG_REPEATED_OUTCOME in tracker.last_signals


# ── the loop ends the turn ───────────────────────────────────────────────────


class _Probe(Tool):
    spec = ToolSpec(
        name="probe",
        description="a probe whose answer never changes",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(PROBE_OUTPUT)


class _Script(Provider):
    """Issues a differently-argued probe call every iteration, forever."""

    def __init__(self) -> None:
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return LLMResult(
            tool_calls=[
                ToolCall(
                    id=f"c{self.calls}", name="probe", arguments={"q": f"variant {self.calls}"}
                )
            ],
            finish_reason="tool_calls",
        )

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=200_000)


def test_loop_ends_stuck_after_six_identical_observations(tmp_path: Path) -> None:
    provider = _Script()
    registry = ToolRegistry()
    registry.register(_Probe())
    loop = AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=40,
    )
    result = asyncio.run(loop.arun_turn("find the stale claim ref"))
    assert result.stop_reason == "stuck"
    assert provider.calls == 6  # nudge at 3, read-only at 4, step back at 5, stop at 6
    rails = [m.text for m in loop.session.messages if m.role == "user"]
    assert any("observed the SAME tool result" in r for r in rails)

"""ADR-0094: ``schedule_wakeup`` — Claude Code's ``ScheduleWakeup`` for a Zak Code session.

A Mind's autonomous loop arms a deadman wake-up before every re-entry and a parked worker
Body arms an hourly re-poll; Zak Code answered both with ``unknown tool``. Now one wake-up is
held per session (replace-slot; ``stop`` cancels), the delay is clamped to [60, 3600], the
slot is persisted with the session, and the REPL's idle wait hands the due prompt over as a
``(harness)`` line — never mid-turn. Hermetic: a fake clock, scripted providers, tmp stores.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop
from zakcode.cli import _InputMux
from zakcode.hooks import HookEvent, HookPayload, HookSpec, wire_payload
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.say_inbox import (
    BUSY_STALE_SECONDS,
    busy_elsewhere,
    busy_path,
    say_pending,
    write_say,
)
from zakcode.session.store import Session, SessionStore
from zakcode.tools import default_registry
from zakcode.tools.base import ToolContext, ToolRegistry
from zakcode.tools.builtins.schedule_wakeup import ScheduleWakeupTool
from zakcode.wakeup import (
    DEFAULT_DELAY_SECONDS,
    LOOP_LINE,
    LOOP_SENTINEL,
    MAX_DELAY_SECONDS,
    MIN_DELAY_SECONDS,
    WakeupSlot,
    clamp_delay,
    fired_line,
)


class _Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _slot(clock: _Clock, changes: list[int] | None = None) -> tuple[Session, WakeupSlot]:
    session = Session(cwd="/w", model="test")
    on_change = None if changes is None else (lambda: changes.append(1))
    return session, WakeupSlot(session, on_change=on_change, clock=clock)


# ── the slot ─────────────────────────────────────────────────────────────────


def test_delay_is_clamped_to_claude_codes_window_and_defaults_when_unusable() -> None:
    assert clamp_delay(10) == MIN_DELAY_SECONDS
    assert clamp_delay(99_999) == MAX_DELAY_SECONDS
    assert clamp_delay("120") == 120
    assert clamp_delay(None) == DEFAULT_DELAY_SECONDS
    assert clamp_delay("soon") == DEFAULT_DELAY_SECONDS


def test_arm_holds_one_wakeup_and_take_due_consumes_it_only_once_due() -> None:
    clock = _Clock(1_000.0)
    changes: list[int] = []
    session, slot = _slot(clock, changes)
    assert slot.pending() is None
    assert slot.take_due() is None

    armed = slot.arm("check the PR", 600)
    assert armed.due_at == 1_600.0 and armed.delay_seconds == 600
    assert session.pending_wakeup is armed
    assert len(changes) == 1  # persisted on arm

    clock.now = 1_599.0
    assert slot.take_due() is None  # not yet
    assert session.pending_wakeup is armed

    clock.now = 1_600.0
    assert slot.take_due() == "[harness] scheduled wake-up: check the PR"
    assert session.pending_wakeup is None  # firing consumed the slot
    assert slot.take_due() is None
    assert len(changes) == 2  # persisted on fire


def test_a_new_arm_replaces_the_held_wakeup_and_stop_cancels_it() -> None:
    clock = _Clock(1_000.0)
    _, slot = _slot(clock)
    assert slot.cancel() is False
    slot.arm("first", 600)
    second = slot.arm("second", 3_600)
    assert slot.pending() is second  # replace-slot, never a queue
    assert slot.cancel() is True
    assert slot.pending() is None
    clock.now = 10_000.0
    assert slot.take_due() is None  # the cancelled wake-up never fires


def test_the_loop_sentinel_fires_as_the_re_entry_line() -> None:
    assert fired_line(LOOP_SENTINEL) == LOOP_LINE
    assert fired_line(f"  {LOOP_SENTINEL} ") == LOOP_LINE
    assert "aspirations loop" in LOOP_LINE and "Re-arm" in LOOP_LINE
    assert fired_line("  poll CI  ") == "[harness] scheduled wake-up: poll CI"


def test_the_held_wakeup_survives_a_session_round_trip(tmp_path: Path) -> None:
    clock = _Clock(1_000.0)
    session, slot = _slot(clock)
    slot.arm(LOOP_SENTINEL, 600)
    store = SessionStore(tmp_path / "sessions")
    store.save(session)

    loaded = store.load(session.id)
    assert loaded.pending_wakeup is not None
    assert loaded.pending_wakeup.prompt == LOOP_SENTINEL
    assert loaded.pending_wakeup.due_at == 1_600.0
    # The resumed process's slot fires it: due time is epoch seconds, not process-relative.
    assert WakeupSlot(loaded, clock=_Clock(1_600.0)).take_due() == LOOP_LINE


# ── the tool ─────────────────────────────────────────────────────────────────


def _ctx(tmp_path: Path, slot: WakeupSlot | None) -> ToolContext:
    return ToolContext(workspace_root=tmp_path, wakeup_slot=slot)


async def test_tool_arms_replaces_and_cancels(tmp_path: Path) -> None:
    clock = _Clock(1_000.0)
    session, slot = _slot(clock)
    tool = ScheduleWakeupTool()

    res = await tool.execute({"prompt": LOOP_SENTINEL}, _ctx(tmp_path, slot))
    assert not res.is_error, res.output
    assert res.data == {
        "armed": True,
        "delay_seconds": DEFAULT_DELAY_SECONDS,
        "due_at": 1_600.0,
        "replaced": False,
    }
    assert "600s" in res.output and "one is held at a time" in res.output

    res = await tool.execute(
        {"prompt": "poll again", "delaySeconds": 3_600, "reason": "hourly re-poll"},
        _ctx(tmp_path, slot),
    )
    assert not res.is_error
    assert res.data is not None and res.data["replaced"] is True
    assert res.data["delay_seconds"] == 3_600
    assert session.pending_wakeup is not None and session.pending_wakeup.prompt == "poll again"

    res = await tool.execute({"stop": True}, _ctx(tmp_path, slot))
    assert not res.is_error
    assert res.data == {"armed": False, "cancelled": True}
    assert session.pending_wakeup is None

    res = await tool.execute({"stop": True}, _ctx(tmp_path, slot))
    assert not res.is_error and res.data == {"armed": False, "cancelled": False}


async def test_tool_accepts_the_snake_case_delay_and_clamps_it(tmp_path: Path) -> None:
    _, slot = _slot(_Clock(1_000.0))
    res = await ScheduleWakeupTool().execute(
        {"prompt": "x", "delay_seconds": 5}, _ctx(tmp_path, slot)
    )
    assert not res.is_error
    assert res.data is not None and res.data["delay_seconds"] == MIN_DELAY_SECONDS


async def test_tool_without_a_prompt_or_without_a_session_errors_cleanly(tmp_path: Path) -> None:
    _, slot = _slot(_Clock(1_000.0))
    tool = ScheduleWakeupTool()

    res = await tool.execute({}, _ctx(tmp_path, slot))
    assert res.is_error and "'prompt' is required" in res.output
    assert slot.pending() is None

    res = await tool.execute({"prompt": "x"}, _ctx(tmp_path, None))
    assert res.is_error and "not available" in res.output


def test_the_tool_answers_to_claude_codes_name_and_its_hooks_fire_on_it() -> None:
    registry = default_registry()
    tool = registry.get("schedule_wakeup")
    assert isinstance(tool, ScheduleWakeupTool)
    # A Mind's loop calls it as Claude Code names it; the alias routes silently.
    assert registry.get("ScheduleWakeup") is tool
    assert registry.get("wakeup") is tool
    # The Mind's PreToolUse gate is written as matcher "ScheduleWakeup": it must fire, and
    # read the Claude Code tool name on the wire.
    assert HookSpec(event=HookEvent.PRE_TOOL_USE, command=["x"], matcher="ScheduleWakeup").matches(
        "schedule_wakeup"
    )
    payload = HookPayload(
        event=HookEvent.PRE_TOOL_USE, tool_name="schedule_wakeup", arguments={"prompt": "p"}
    )
    assert b'"tool_name": "ScheduleWakeup"' in wire_payload(payload)


# ── the loop's seam ──────────────────────────────────────────────────────────


class _ArmsThenStops(Provider):
    """Call 1 arms a wake-up (as a Mind's loop would, by Claude Code's name); call 2 ends."""

    def __init__(self) -> None:
        self.calls = 0

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="ScheduleWakeup",
                        arguments={"prompt": LOOP_SENTINEL, "delaySeconds": 600},
                    )
                ]
            )
        return LLMResult(text="armed; carrying on")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=32_768)


def test_a_wakeup_the_model_arms_is_on_disk_before_the_turn_ends(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ScheduleWakeupTool(), aliases=["ScheduleWakeup"])
    store = SessionStore(tmp_path / "sessions")
    session = Session(cwd=str(tmp_path), model="test")
    loop = AgentLoop(
        _ArmsThenStops(), registry, session, workspace_root=tmp_path, store=store, max_iterations=5
    )
    result = loop.run_turn("start the loop")
    assert result.stop_reason == "completed" and result.tool_results[0].is_error is False

    held = session.pending_wakeup
    assert held is not None and held.prompt == LOOP_SENTINEL and held.delay_seconds == 600
    assert loop.wakeup_slot.pending() is held
    on_disk = store.load(session.id).pending_wakeup
    assert on_disk is not None and on_disk.due_at == held.due_at


# ── the REPL door ────────────────────────────────────────────────────────────


def test_idle_wait_delivers_a_due_wakeup_as_a_harness_line(tmp_path: Path) -> None:
    clock = _Clock(1_000.0)
    _, slot = _slot(clock)
    slot.arm("poll the reducer", 60)
    mux = _InputMux(tmp_path / "say", tmp_path / "stop", keyboard=False, wakeup_probe=slot.take_due)

    # Not due: the wait keeps waiting (the asker's stop event ends it).
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    assert mux.next_input(idle=True, stop=stop) == ("cancelled", None)
    assert mux.try_input() is None

    clock.now = 1_060.0
    assert mux.try_input() == ("harness", "[harness] scheduled wake-up: poll the reducer")
    assert slot.pending() is None  # consumed
    assert mux.try_input() is None


def test_a_due_wakeup_fires_from_the_blocking_idle_wait_too(tmp_path: Path) -> None:
    _, slot = _slot(_Clock(2_000.0))
    slot.arm(LOOP_SENTINEL, 60)
    mux = _InputMux(
        tmp_path / "say",
        tmp_path / "stop",
        keyboard=False,
        wakeup_probe=lambda: slot.take_due(now=5_000.0),
    )
    assert mux.next_input(idle=True) == ("harness", LOOP_LINE)


def test_a_wakeup_never_fires_mid_turn(tmp_path: Path) -> None:
    asked: list[int] = []

    def probe() -> str | None:
        asked.append(1)
        return "[harness] scheduled wake-up: x"

    mux = _InputMux(tmp_path / "say", tmp_path / "stop", keyboard=False, wakeup_probe=probe)
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    # A permission prompt is a mid-turn consumer (idle=False): the wake-up must wait.
    assert mux.next_input(idle=False, stop=stop) == ("cancelled", None)
    assert asked == []


def test_typed_and_said_input_win_over_a_due_wakeup(tmp_path: Path) -> None:
    _, slot = _slot(_Clock(1_000.0))
    slot.arm("later", 60)
    mux = _InputMux(
        tmp_path / "say",
        tmp_path / "stop",
        keyboard=False,
        wakeup_probe=lambda: slot.take_due(now=5_000.0),
    )
    mux.queue.put(("line", "a human typed this"))
    assert mux.try_input() == ("line", "a human typed this")
    assert slot.pending() is not None  # still held: the person's line went first
    assert mux.try_input() == ("harness", "[harness] scheduled wake-up: later")


def test_a_due_wakeup_fires_while_another_process_turn_holds_the_workspace(
    tmp_path: Path,
) -> None:
    # ADR-0094 amendment: the busy marker (ADR-0060) guards the say inbox — ONE slot the
    # whole workspace shares — not the session's own wake-up. Eight Bodies on one checkout
    # keep the marker fresh around the clock (zc-03, 2026-08-30: 12/12 samples over 60 s),
    # so a parked Body that stood back behind it never woke.
    _, slot = _slot(_Clock(1_000.0))
    slot.arm(LOOP_SENTINEL, 60)
    inbox = tmp_path / "say"
    mux = _InputMux(
        inbox, tmp_path / "stop", keyboard=False, wakeup_probe=lambda: slot.take_due(now=5_000.0)
    )
    assert write_say(inbox, "for the runner")
    marker = busy_path(tmp_path)
    foreign = {"pid": os.getpid() + 100_000, "session": "other", "since": time.time()}
    marker.write_text(json.dumps(foreign) + "\n", encoding="utf-8")
    assert busy_elsewhere(marker)  # a FRESH marker names another process's turn

    # The say stands back for the turn that owns the inbox (ADR-0060, untouched) — the
    # session's own wake-up does not: both doors hand it over under the marker.
    assert mux.try_input() == ("harness", LOOP_LINE)
    assert say_pending(inbox) and slot.pending() is None
    slot.arm("poll the reducer", 60)
    stop = threading.Event()  # bounds the wait: a regression here otherwise spins until the
    threading.Timer(2.0, stop.set).start()  # marker ages out (120 s) and the say arrives
    woke = mux.next_input(idle=True, stop=stop)
    assert woke == ("harness", "[harness] scheduled wake-up: poll the reducer")
    assert say_pending(inbox)

    # Positive control: the marker is what held the say back — aged out, the say arrives.
    stale = time.time() - BUSY_STALE_SECONDS - 5
    os.utime(marker, (stale, stale))
    assert mux.try_input() == ("say", "for the runner")

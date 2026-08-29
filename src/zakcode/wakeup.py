"""A scheduled wake-up: ONE held line the harness hands the session at an idle prompt
(ADR-0094).

Claude Code's ``ScheduleWakeup`` is the primitive a Mind's autonomous loop is built on: the
reducer arms a "deadman" wake-up before every loop re-entry (it fires only if the re-entry
chain breaks), and a worker Body that parks — its reducer gone — arms an hourly re-poll.
Zak Code had no such tool: the calls came back ``unknown tool``, the reducer ran with no net,
and a parked Body sat at its prompt forever. Measured 2026-08-29 (zc-03): a reducer restart
re-minted the runner token, every worker's liveness poll parked, and all four Bodies were
dead until an operator cycled them — the re-poll they had armed never existed.

The contract, matching Claude Code's: a single replace-slot per session (a new call replaces
the held wake-up; ``stop`` cancels it), a delay clamped to [60, 3600] seconds, and delivery
at the next idle prompt on or after the due time — never mid-turn. Firing CONSUMES the slot.
The slot is persisted on the session so it survives the ADR-0034 restart into a new build.
Pure: no threads, no I/O — the REPL's idle wait asks :meth:`WakeupSlot.take_due`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

#: Claude Code clamps a wake-up's delay to this window; so does Zak Code.
MIN_DELAY_SECONDS = 60
MAX_DELAY_SECONDS = 3600
#: A wake-up armed with ``delaySeconds`` missing or unusable gets this delay.
DEFAULT_DELAY_SECONDS = 600

#: Claude Code resolves this sentinel back to its autonomous-loop instructions at fire time;
#: a Mind's deadman net arms exactly this. Zak Code hands over the line below instead.
LOOP_SENTINEL = "<<autonomous-loop-dynamic>>"
LOOP_LINE = (
    "[harness] the wake-up armed as the autonomous-loop sentinel fired: the loop that armed it "
    "did not re-enter on its own and nobody is at the prompt. Re-arm a wake-up first, then "
    "re-enter the loop — invoke the skill that runs it (the aspirations loop, args 'loop') "
    "and carry on from where the plan stands. Do not stop to wait for instructions."
)


def clamp_delay(value: Any) -> int:
    """``value`` as a delay in seconds within the window; the default when unusable."""
    try:
        delay = int(value)
    except (TypeError, ValueError):
        return DEFAULT_DELAY_SECONDS
    return max(MIN_DELAY_SECONDS, min(MAX_DELAY_SECONDS, delay))


def fired_line(prompt: str) -> str:
    """The line the session receives when a wake-up armed with ``prompt`` fires."""
    if prompt.strip() == LOOP_SENTINEL:
        return LOOP_LINE
    return f"[harness] scheduled wake-up: {prompt.strip()}"


class Wakeup(BaseModel):
    """The one held wake-up: what to say, and when it may be said."""

    prompt: str
    #: Epoch seconds (``time.time()``), so the due time survives a process restart.
    due_at: float
    armed_at: float
    delay_seconds: int

    def is_due(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) >= self.due_at


class WakeupSlot:
    """The tool's handle on the session's wake-up: arm, cancel, and — for the REPL — take it
    once due. ``on_change`` (the loop's persist) runs after every mutation so the held
    wake-up is on disk before the turn that armed it ends."""

    def __init__(
        self,
        session: Any,
        *,
        on_change: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._session = session
        self._on_change = on_change
        self._clock = clock

    def pending(self) -> Wakeup | None:
        return getattr(self._session, "pending_wakeup", None)

    def arm(self, prompt: str, delay_seconds: Any) -> Wakeup:
        """Hold ``prompt`` for delivery ``delay_seconds`` from now, replacing any held one."""
        now = self._clock()
        delay = clamp_delay(delay_seconds)
        wakeup = Wakeup(prompt=prompt, due_at=now + delay, armed_at=now, delay_seconds=delay)
        self._session.pending_wakeup = wakeup
        self._changed()
        return wakeup

    def cancel(self) -> bool:
        """Drop the held wake-up; ``False`` when none was held."""
        if self.pending() is None:
            return False
        self._session.pending_wakeup = None
        self._changed()
        return True

    def take_due(self, now: float | None = None) -> str | None:
        """The line to deliver when the held wake-up is due — the slot is consumed — else
        ``None``. Called from the REPL's idle wait; never mid-turn."""
        wakeup = self.pending()
        if wakeup is None or not wakeup.is_due(self._clock() if now is None else now):
            return None
        self._session.pending_wakeup = None
        self._changed()
        return fired_line(wakeup.prompt)

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

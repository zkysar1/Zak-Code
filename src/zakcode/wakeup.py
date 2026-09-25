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

ONE MORE RULE, AND IT IS THE ONLY THING HERE THAT SAYS NO (ADR-0216). A fired sentinel hands
the session a line that opens "re-arm a wake-up FIRST, then re-enter the loop" — deliberately,
because a net that fires while it is being replaced is a net with a hole. But the model obeys
that instruction BEFORE it can discover whether there is anything to re-enter, so a loop that
cannot run re-arms its own resurrection and the pair repeats until someone notices. Measured
2026-09-22 on a served Mind: six turns, each ~3-5M tokens, each ending with the identical
verdict that the agent was IDLE and the loop would not start. Nothing in the product could see
it, because every repeat detector it has — the doom guard and the stuck ladder both — is scoped
to ONE TURN, and these repeats are one turn apart.

So :meth:`WakeupSlot.note_turn_end` carries the one piece of state that outlives a turn: the
fingerprint of how the last SENTINEL-fired turn ended. A sentinel turn that ends exactly as the
last sentinel turn did has resurrected nothing, and the sentinel it re-armed is cancelled rather
than allowed to fire again. Narrow on purpose, three ways: only a turn a sentinel OPENED is
judged, only the SENTINEL is ever cancelled (a hook's specific "come back and do X" is another
instruction entirely and is left alone), and one repeat is the whole threshold — the second
identical turn is already proof, and waiting for a third only spends more of them.

AND ONE EXCEPTION TO THAT RULE, WHICH IS NOT A LOOPHOLE BUT ITS OWN CASE (ADR-0250). A sentinel
turn the PROVIDER failed — every model call refused, no answer at all — proves nothing about
the loop, because the model never got to act; and a provider outage produces identical endings
by construction, two of them in a row on the second cycle. Measured 2026-09-25 on three worker
Bodies during a 12-hour pod outage: each cycle was a fired sentinel, a compaction whose
summarizer call failed, four 900-second retry budgets, a ``veto_stall`` and a fresh 600-second
net; on the second identical cycle the repeat guard cancelled that net, and all three Bodies sat
at their prompts for four and a half hours after the pod came back, until an operator typed
``/start``. A cancel converts a temporary outage into a permanent park. So a provider-failed
sentinel turn HOLDS its net instead: the sentinel is re-armed at a delay that doubles per
consecutive provider failure up to the clamp (600, 1200, 2400, 3600 s), the record is kept, and
the streak resets the moment a sentinel turn actually runs. The REPL door applies the same hold
BEFORE opening the turn when the provider does not answer a cheap probe, so an outage costs one
probe per cycle rather than a compaction and an hour of retries. An outage may delay the loop; it
must never end it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from hashlib import blake2b
from typing import Any

from pydantic import BaseModel

#: Claude Code clamps a wake-up's delay to this window; so does Zak Code.
MIN_DELAY_SECONDS = 60
MAX_DELAY_SECONDS = 3600
#: A wake-up armed with ``delaySeconds`` missing or unusable gets this delay.
DEFAULT_DELAY_SECONDS = 600

#: Claude Code resolves this sentinel back to its autonomous-loop instructions at fire time;
#: a Mind's deadman net arms exactly this. Zak Code resolves it to the skill a turn-end hook
#: last asked the loop to re-enter with (``Session.loop_skill``, ADR-0187), composed by the
#: harness with :data:`LOOP_WAKE_NOTE` in the frame; when no such skill is known yet it hands
#: over :data:`LOOP_LINE` instead.
LOOP_SENTINEL = "<<autonomous-loop-dynamic>>"
LOOP_WAKE_NOTE = (
    "the wake-up armed as the autonomous-loop sentinel fired: the loop that armed it did not "
    "re-enter on its own and nobody is at the prompt. Re-arm a wake-up with ScheduleWakeup "
    "first, then carry out these instructions from where the plan stands. Do not stop to "
    "wait for instructions."
)
LOOP_LINE = (
    "[harness] the wake-up armed as the autonomous-loop sentinel fired: the loop that armed it "
    "did not re-enter on its own and nobody is at the prompt. Re-arm a wake-up first, then "
    "re-enter the loop — invoke the skill that runs it, with the arguments it was started "
    "with — and carry on from where the plan stands. Do not stop to wait for instructions."
)


def clamp_delay(value: Any) -> int:
    """``value`` as a delay in seconds within the window; the default when unusable."""
    try:
        delay = int(value)
    except (TypeError, ValueError):
        return DEFAULT_DELAY_SECONDS
    return max(MIN_DELAY_SECONDS, min(MAX_DELAY_SECONDS, delay))


def provider_hold_delay(streak: int) -> int:
    """Seconds before the sentinel re-fires after ``streak`` consecutive provider failures
    (ADR-0250): the default delay doubled per failure, never past the clamp — 600, 1200, 2400,
    3600, 3600 … A loop that cannot reach its provider is retried at most hourly, and the
    moment it can the next firing runs it."""
    return clamp_delay(DEFAULT_DELAY_SECONDS * 2 ** max(0, streak - 1))


def turn_fingerprint(stop_reason: str, assistant_text: str) -> str:
    """How a turn ENDED, as one short stable string (ADR-0216).

    Two things and no more: why the turn stopped, and the last thing it said. Whitespace is
    collapsed so a re-wrapped answer is still the same answer. Hashed rather than stored raw
    because this value is PERSISTED on the session, and a session file should not grow a second
    copy of the model's prose to answer a question that only ever needs equality.

    Tool calls are deliberately NOT in it. A sentinel turn re-arms its wake-up as its first act
    on instruction, so every such turn shares that call whatever else it does; and the turns this
    exists to catch differ in nothing at all. What distinguishes a productive re-entry from a
    barren one is what the turn CONCLUDED, which is exactly these two fields.
    """
    canonical = f"{stop_reason}\n{' '.join(assistant_text.split())}"
    return blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


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
    #: The arming turn's own word on whether anything changed since the last wake-up
    #: (Claude Code's ``noop``): ``True`` for a quiet hold, ``False`` when it did something,
    #: ``None`` when the arm did not say. Recorded, never acted on; a wake-up armed before
    #: the field existed loads as ``None``.
    noop: bool | None = None

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

    def arm(self, prompt: str, delay_seconds: Any, *, noop: bool | None = None) -> Wakeup:
        """Hold ``prompt`` for delivery ``delay_seconds`` from now, replacing any held one."""
        now = self._clock()
        delay = clamp_delay(delay_seconds)
        wakeup = Wakeup(
            prompt=prompt, due_at=now + delay, armed_at=now, delay_seconds=delay, noop=noop
        )
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
        prompt = self.take_due_prompt(now)
        return None if prompt is None else fired_line(prompt)

    def take_due_prompt(self, now: float | None = None) -> str | None:
        """The held wake-up's own prompt when it is due — the slot is consumed — else
        ``None``. The REPL door resolves the sentinel itself (ADR-0187); :meth:`take_due`
        renders the plain line."""
        wakeup = self.pending()
        if wakeup is None or not wakeup.is_due(self._clock() if now is None else now):
            return None
        self._session.pending_wakeup = None
        if wakeup.prompt.strip() == LOOP_SENTINEL:
            # The turn this line is about to open is a SENTINEL turn, and note_turn_end judges
            # it when it ends (ADR-0216). Marked here because this is the only moment anything
            # knows the sentinel fired: the slot is consumed on the next line and the turn that
            # follows is otherwise indistinguishable from one a person typed.
            self._session.sentinel_turn_open = True
        self._changed()
        return wakeup.prompt

    def note_turn_end(self, fingerprint: str, *, provider_failed: bool = False) -> bool:
        """Judge a turn that a fired sentinel opened; ``True`` when it merely repeated (ADR-0216).

        Called once at every turn end. It is a no-op for a turn a person opened, and for the
        first sentinel turn after any different one — those only record how they ended.

        When a sentinel turn ends exactly as the previous sentinel turn did, the wake-up re-armed
        during it resurrected nothing, and it is cancelled so the pair cannot run again. ONLY the
        sentinel is ever cancelled: a turn-end hook that armed its own prompt said something
        specific about when to come back, and this has no standing to overrule it — the same
        precedence ``_arm_stall_net`` already keeps, where the framework's net outranks the
        harness's.

        The recorded fingerprint is CLEARED on a cancel, so the next sentinel to fire starts
        clean rather than being judged against a turn that is now two nets old.

        ``provider_failed`` names the one ending that is never a verdict on the loop (ADR-0250):
        the provider refused every call and the model never acted. Such a turn is still judged
        (the return value still says whether it repeated) but its net is HELD, not cancelled —
        :meth:`hold_for_provider` re-arms the sentinel at a backoff and keeps the record — because
        an outage produces identical endings by construction and a cancel would park the loop
        for good. Any sentinel turn that actually ran resets that backoff.
        """
        if not getattr(self._session, "sentinel_turn_open", False):
            return False
        self._session.sentinel_turn_open = False
        previous = getattr(self._session, "last_sentinel_outcome", "")
        repeated = bool(previous) and previous == fingerprint
        if provider_failed:
            self._session.last_sentinel_outcome = fingerprint
            self.hold_for_provider()  # persists
            return repeated
        self._session.sentinel_provider_repeats = 0
        if repeated:
            self._session.last_sentinel_outcome = ""
            held = self.pending()
            if held is not None and held.prompt.strip() == LOOP_SENTINEL:
                self._session.pending_wakeup = None
            self._changed()
            return True
        self._session.last_sentinel_outcome = fingerprint
        self._changed()
        return False

    def hold_for_provider(self) -> Wakeup:
        """Keep the loop's net through a provider failure, backing off (ADR-0250).

        One more consecutive provider failure is counted on the session, and the sentinel is
        re-armed at :data:`DEFAULT_DELAY_SECONDS` doubled per failure, clamped — 600, 1200, 2400,
        3600 s — replacing a held SENTINEL (the 600 s a turn-end hook or the stall fence just
        armed, sized for a loop that can run) or an empty slot. A hook's OWN prompt is left in
        place, the precedence :meth:`note_turn_end` already keeps; the count still advances so the
        next hold after it is sized right. Returns the wake-up now held. Called by
        :meth:`note_turn_end` for a provider-failed sentinel turn and by the REPL door for a
        sentinel the provider probe says would fail, so both count on one streak.
        """
        streak = int(getattr(self._session, "sentinel_provider_repeats", 0) or 0) + 1
        self._session.sentinel_provider_repeats = streak
        held = self.pending()
        if held is not None and held.prompt.strip() != LOOP_SENTINEL:
            self._changed()
            return held
        return self.arm(LOOP_SENTINEL, provider_hold_delay(streak))

    def hold_unfired_sentinel(self) -> Wakeup:
        """The REPL door took the sentinel but will not open its turn — the provider does not
        answer — so put it back at the provider backoff (ADR-0250). :meth:`take_due_prompt`
        marked the turn it was about to open as a sentinel turn; no turn opens, so the mark is
        lifted here, or the next turn a person types would be judged as one."""
        self._session.sentinel_turn_open = False
        return self.hold_for_provider()

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

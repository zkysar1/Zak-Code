"""Multi-signal stuck detection + an escalating recovery ladder (Bet 2).

A weak local model frequently gets *stuck* in ways the byte-identical doom-loop guard
(``loop.py``) misses: it retries the same broken call with slightly tweaked arguments, or
every tool call this iteration errors, or it churns without ever producing a successful
result. The doom guard only catches the *exact same batch* three times in a row.

:class:`StuckTracker` generalizes "no progress" into a **vote across several independent
signals** and, when enough of them fire for enough iterations in a row, drives an
**escalating recovery ladder** instead of immediately giving up:

* **nudge** — inject a corrective hint and let the model try a different approach;
* **narrow** — restrict the next iteration to read-only tools so the model is forced to
  *investigate* (re-read the file, the error, the directory) before mutating again;
* **step back** — one last, once-per-turn reassessment prompt: every attempt failing
  usually means they share a wrong *assumption* (a path that does not exist here, a tool
  that is not installed, an interface that differs), so the model is told to restate the
  goal and verify that assumption from the ground up with read-only probes before acting
  again. Firing it resets the streak, because a model that takes the advice starts with a
  failing discovery probe or two (verified in the field: the first post-prompt probe
  failed, the second found the real path) — without the reset those honest probes would
  trip the stop threshold mid-recovery;
* **stop** — end the turn cleanly with ``stop_reason="stuck"`` rather than burning the
  whole iteration budget flailing.

The ladder fires only on *sustained* trouble (a streak of ≥2-signal iterations), so a
capable model that hits a single transient error never triggers it — and the doom guard
still owns the pure exact-repeat case (it fires *before* execution, so exact repeats end as
``doom_loop`` exactly as before). This is always-on and self-pacing — not a feature flag.

Pure per-turn state, mirroring :class:`~zakcode.agent.recipe.RecipeCursor`: the loop creates
one tracker per turn, feeds it each tool-call iteration's ``(calls, results, assistant_text)``
via :meth:`observe`, then consults :meth:`next_action` to decide the ladder step. No
provider/transport knowledge.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from enum import Enum

from zakcode.messages import ToolResultBlock
from zakcode.providers.base import ToolCall

# ── canonical tool-call identity (shared with the loop's doom guard) ─────────────


def call_signature(call: ToolCall) -> tuple[str, str]:
    """A stable, hashable identity for a tool call (name + canonical arguments).

    Arguments are serialized with sorted keys so two logically-identical calls compare
    equal regardless of dict ordering. Falls back to ``repr`` for the (vanishingly rare)
    non-JSON-serializable argument value.
    """
    try:
        args = json.dumps(call.arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args = repr(sorted(call.arguments.items()))
    return (call.name, args)


def batch_signature(calls: list[ToolCall]) -> tuple[tuple[str, str], ...]:
    """Signature for a whole batch of tool calls requested in one iteration."""
    return tuple(call_signature(c) for c in calls)


# ── signals + ladder ────────────────────────────────────────────────────────────

#: The model asked for the byte-identical batch it asked for last iteration.
SIG_REPEATED_BATCH = "repeated-batch"
#: Every tool result this iteration was an error.
SIG_ALL_ERRORS = "all-errors"
#: Some call has now failed ``repeated_failure_at`` times this turn (not necessarily in a
#: row) — the model keeps retrying the same broken thing with other calls interleaved.
SIG_REPEATED_FAILURE = "repeated-failure"
#: A non-empty batch produced no successful result AND the model emitted no reasoning text:
#: it is neither acting successfully nor thinking out loud.
SIG_NO_PROGRESS = "no-progress"
#: The same tool produced the SAME output it already produced ``outcome_repeat_at`` times this
#: turn with no file edit in between (ADR-0038). Every other signal keys on an ERROR; a model
#: that re-measures the same thing with slightly different commands, each exiting 0, fires
#: none of them — field incident 2026-08-27: 135 iterations, 103 minutes, the same 5-line
#: probe output observed ~15 times, every command wrapped in ``|| echo`` so nothing ever
#: errored. Re-observing a known result is not progress; this signal is STRONG (it counts as
#: stuck on its own) and it drives the ladder by the repeat count, not by a consecutive streak,
#: because the re-measurements were interleaved with other probes.
SIG_REPEATED_OUTCOME = "repeated-outcome"

#: Outputs shorter than this (normalized) never count as a repeated outcome: ``ok`` / ``done``
#: style acknowledgements repeat legitimately.
_OUTCOME_MIN_CHARS = 24
#: Only the head of a long output is hashed — enough to identify it, bounded cost.
_OUTCOME_HEAD_CHARS = 4000
#: Volatile fragments masked before comparing: timestamps, pids, ports, hashes, durations.
_VOLATILE_RE = re.compile(
    r"\d{4,}|\d{1,2}:\d{2}(?::\d{2})?|0x[0-9a-f]{6,}|\b[0-9a-f]{12,}\b|\d+\.\d+s\b",
    re.IGNORECASE,
)


def outcome_signature(name: str, output: str, epoch: int = 0) -> str | None:
    """A stable identity for a tool OUTCOME, or ``None`` when it is too short to mean anything.

    ``epoch`` is the loop's count of successful FILE-EDIT calls so far this turn: the same
    output after an edit is a fresh measurement of a changed world (edit → test → edit → test
    is progress, not a loop) and must not compare equal to the one before the edit.
    """
    normalized = _VOLATILE_RE.sub("#", " ".join((output or "").split()))
    if len(normalized) < _OUTCOME_MIN_CHARS:
        return None
    return f"{name}\x00{epoch}\x00{normalized[:_OUTCOME_HEAD_CHARS]}"


class StuckAction(Enum):
    """The recovery step the loop should take after an :meth:`StuckTracker.observe`."""

    CONTINUE = "continue"  # not stuck (or not yet) — proceed normally
    NUDGE = "nudge"  # inject a corrective hint, keep going
    NARROW = "narrow"  # restrict the next iteration to read-only tools, keep going
    STEP_BACK = "step_back"  # once per turn: reassess assumptions from scratch, streak resets
    STOP = "stop"  # give up gracefully (stop_reason="stuck")


class StuckTracker:
    """Per-turn 'is the model making progress?' detector + recovery ladder.

    Each tool-call iteration is scored by how many independent stuck-signals fire
    (:meth:`observe`); ``vote_threshold`` or more makes the iteration "stuck", and a run of
    consecutive stuck iterations escalates the ladder at ``nudge_at`` → ``narrow_at`` →
    ``step_back_at`` → ``stop_at`` (:meth:`next_action`). Streaks reset the moment the model
    makes progress, so transient trouble never escalates.

    The STEP_BACK rung is once per turn and *consumes the streak*: choosing it resets the
    counter (and the per-call failure counts) so the reassessment gets the same runway a
    fresh approach would. If the model climbs all the way back, the second arrival at
    ``step_back_at`` is a STOP. ``stop_at`` (> ``step_back_at``) is a pure backstop for
    custom threshold layouts where the streak can pass ``step_back_at`` without landing on
    it; with the defaults the reset makes it unreachable.
    """

    def __init__(
        self,
        *,
        vote_threshold: int = 2,
        nudge_at: int = 3,
        narrow_at: int = 4,
        step_back_at: int = 5,
        stop_at: int = 6,
        repeated_failure_at: int = 2,
        outcome_repeat_at: int = 3,
    ) -> None:
        self.vote_threshold = vote_threshold
        self.nudge_at = nudge_at
        self.narrow_at = narrow_at
        self.step_back_at = step_back_at
        self.stop_at = stop_at
        self.repeated_failure_at = repeated_failure_at
        self.outcome_repeat_at = outcome_repeat_at
        self._streak = 0  # consecutive stuck (>= vote_threshold signals) iterations
        self._prev_sig: tuple[tuple[str, str], ...] | None = None
        self._error_counts: Counter[tuple[str, str]] = Counter()  # per-call failures this turn
        self._outcome_counts: Counter[str] = Counter()  # identical outcomes this turn (ADR-0038)
        self._last_outcome_repeats = 0  # the worst repeat count seen on the most recent observe
        self._last_signals: list[str] = []  # signals fired on the most recent observe
        self._actions: list[str] = []  # ladder actions taken this turn (observability)
        self._step_back_used = False  # the reassessment rung is once per turn

    # ── inspection ───────────────────────────────────────────────────────────
    @property
    def streak(self) -> int:
        """How many stuck iterations have occurred in a row (0 when progressing)."""
        return self._streak

    @property
    def last_signals(self) -> list[str]:
        """The signal names that fired on the most recent :meth:`observe`."""
        return list(self._last_signals)

    @property
    def actions(self) -> list[str]:
        """The ladder actions taken this turn, in order (e.g. ``["nudge", "narrow"]``)."""
        return list(self._actions)

    @property
    def took_action(self) -> bool:
        """Whether any recovery step (nudge/narrow/stop) has fired this turn."""
        return bool(self._actions)

    def error_signatures(self) -> list[tuple[str, str]]:
        """Call signatures ``(name, canonical-args)`` that failed at least ``repeated_failure_at``
        times this turn, most-failed first.

        The symptom set a recovered-failure lesson (research R1) is built from — exposed as a
        clean accessor so the writer never reaches into the private failure Counter.
        """
        return sorted(
            (sig for sig, n in self._error_counts.items() if n >= self.repeated_failure_at),
            key=lambda sig: self._error_counts[sig],
            reverse=True,
        )

    # ── core ─────────────────────────────────────────────────────────────────
    def observe(
        self,
        calls: list[ToolCall],
        results: list[ToolResultBlock],
        *,
        assistant_text: str = "",
        epoch: int = 0,
    ) -> None:
        """Score one tool-call iteration, updating the stuck streak.

        Call once per iteration that requested tool calls, after the batch has executed.
        Iterations with no tool calls (a text/empty completion) are not stuck by definition
        and should not be passed here. ``epoch`` is the turn's successful file-edit count
        (see :func:`outcome_signature`).
        """
        sig = batch_signature(calls)
        by_id = {r.tool_use_id: r for r in results}

        errored_now: list[tuple[str, str]] = []
        for call in calls:
            result = by_id.get(call.id)
            if result is not None and result.is_error:
                cs = call_signature(call)
                self._error_counts[cs] += 1
                errored_now.append(cs)

        signals: list[str] = []
        if sig and self._prev_sig is not None and sig == self._prev_sig:
            signals.append(SIG_REPEATED_BATCH)
        if results and all(r.is_error for r in results):
            signals.append(SIG_ALL_ERRORS)
        if any(self._error_counts[cs] >= self.repeated_failure_at for cs in errored_now):
            signals.append(SIG_REPEATED_FAILURE)
        produced_success = any((r := by_id.get(c.id)) is not None and not r.is_error for c in calls)
        if calls and not produced_success and not assistant_text.strip():
            signals.append(SIG_NO_PROGRESS)

        # Repeated outcome (ADR-0038): count identical (tool, epoch, output) observations
        # across the whole turn — NOT consecutively — and read the worst count this batch.
        worst = 0
        for call in calls:
            result = by_id.get(call.id)
            if result is None:
                continue
            osig = outcome_signature(call.name, result.output or "", epoch)
            if osig is None:
                continue
            self._outcome_counts[osig] += 1
            worst = max(worst, self._outcome_counts[osig])
        self._last_outcome_repeats = worst
        if worst >= self.outcome_repeat_at:
            signals.append(SIG_REPEATED_OUTCOME)

        self._last_signals = signals
        if len(signals) >= self.vote_threshold:
            self._streak += 1
        else:
            self._streak = 0
        if worst >= self.outcome_repeat_at:
            # A strong signal: the Nth identical observation lands on rung N of the ladder
            # (3 → nudge, 4 → narrow, 5 → step back, 6 → stop with the defaults) regardless
            # of what the interleaved iterations did — the field loop alternated probes, so a
            # consecutive streak never formed while the same result came back fifteen times.
            self._streak = max(self._streak, self.nudge_at + (worst - self.outcome_repeat_at))
        self._prev_sig = sig

    def next_action(self) -> StuckAction:
        """The ladder step implied by the current streak (call once per :meth:`observe`).

        Returns :attr:`StuckAction.NUDGE` / ``NARROW`` exactly once as the streak crosses
        each threshold; the first arrival at ``step_back_at`` is ``STEP_BACK`` (which
        resets the streak — the reassessment gets fresh runway), the second is ``STOP``;
        ``stop_at`` is a ``>=`` backstop. ``CONTINUE`` otherwise. Records the chosen step
        for :attr:`actions` / :attr:`took_action`.
        """
        if self._streak >= self.stop_at:
            action = StuckAction.STOP
        elif self._streak == self.step_back_at:
            action = StuckAction.STOP if self._step_back_used else StuckAction.STEP_BACK
        elif self._streak == self.narrow_at:
            action = StuckAction.NARROW
        elif self._streak == self.nudge_at:
            action = StuckAction.NUDGE
        else:
            action = StuckAction.CONTINUE
        if action is StuckAction.STEP_BACK:
            # Consume the one-shot rung and give the reassessment the same runway a fresh
            # approach would get. The field trace that motivated this rung recovered with a
            # FAILING first probe (List on the assumed path) before the second probe found
            # the real one — without the reset, that honest probe would trip the stop.
            self._step_back_used = True
            self._streak = 0
            self._prev_sig = None
            self._error_counts.clear()
        if action is not StuckAction.CONTINUE:
            self._actions.append(action.value)
        return action

    def reset(self) -> None:
        """Reset streak and signal state after a TURN_END veto continues the loop.

        Prevents immediate re-triggering of the stuck ladder when the loop
        re-enters with an injected continuation prompt. Deliberately does NOT restore the
        one-shot STEP_BACK charge — a turn gets one reassessment no matter how many veto
        continuations it earns, so the ladder stays bounded.
        """
        self._streak = 0
        self._prev_sig = None
        self._error_counts.clear()

    # ── recovery messages ────────────────────────────────────────────────────
    def nudge_message(self) -> str:
        """The corrective hint injected on a :attr:`StuckAction.NUDGE`."""
        if SIG_REPEATED_OUTCOME in self._last_signals:
            return (
                f"You have now observed the SAME tool result {self._last_outcome_repeats} "
                "times this turn without changing anything in between. Re-measuring a known "
                "result is not progress. Act on what you already know: either change the "
                "question (a different probe that could FALSIFY your current hypothesis), make "
                "the change the evidence supports, or stop and report what you found — and "
                "read the error text you already have before forming a new theory."
            )
        return (
            "You appear to be stuck: the last few steps made no progress (a repeated or "
            "all-failing tool call). Stop and reconsider — re-read the relevant file or the "
            "exact error message, identify the specific cause, and try a DIFFERENT approach "
            "rather than repeating an action that is not working."
        )

    def narrow_message(self) -> str:
        """The corrective hint injected on a :attr:`StuckAction.NARROW`."""
        return (
            "You are still stuck, so for this step only read-only tools are available. Use "
            "them to investigate first — read the file, the error, or the directory — and "
            "take one focused step to find the real cause before attempting another change."
        )

    def step_back_message(self) -> str:
        """The reassessment prompt injected on a :attr:`StuckAction.STEP_BACK`.

        Modeled on the operator intervention that recovered a real stuck-stopped turn in the
        field ("take a step back, and think about what the right path is, and try again"):
        it attacks the shared PREMISE of the failed attempts, not the method, and demands
        cheap read-only verification before any retry.
        """
        return (
            "Stop and take a step back — do not retry anything yet. Several different "
            "attempts have all failed, which usually means they share one wrong assumption: "
            "a path that does not exist here, a command or tool that is not available, an "
            "interface that differs from what you expect. First state in one sentence what "
            "you are trying to accomplish. Then verify the assumption every failed attempt "
            "depended on, from the ground up, with read-only probes: list a directory you "
            "KNOW exists (such as the workspace root) and walk down to find the real path; "
            "run the command with --help to see its real interface; read the file you "
            "believe is there. Rebuild your approach from what the probes actually show, "
            "and only then act. Do not repeat any earlier failing call until a probe has "
            "confirmed the assumption it depends on."
        )

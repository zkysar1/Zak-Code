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


class StuckAction(Enum):
    """The recovery step the loop should take after an :meth:`StuckTracker.observe`."""

    CONTINUE = "continue"  # not stuck (or not yet) — proceed normally
    NUDGE = "nudge"  # inject a corrective hint, keep going
    NARROW = "narrow"  # restrict the next iteration to read-only tools, keep going
    STOP = "stop"  # give up gracefully (stop_reason="stuck")


class StuckTracker:
    """Per-turn 'is the model making progress?' detector + recovery ladder.

    Each tool-call iteration is scored by how many independent stuck-signals fire
    (:meth:`observe`); ``vote_threshold`` or more makes the iteration "stuck", and a run of
    consecutive stuck iterations escalates the ladder at ``nudge_at`` → ``narrow_at`` →
    ``stop_at`` (:meth:`next_action`). Streaks reset the moment the model makes progress, so
    transient trouble never escalates.
    """

    def __init__(
        self,
        *,
        vote_threshold: int = 2,
        nudge_at: int = 3,
        narrow_at: int = 4,
        stop_at: int = 5,
        repeated_failure_at: int = 2,
    ) -> None:
        self.vote_threshold = vote_threshold
        self.nudge_at = nudge_at
        self.narrow_at = narrow_at
        self.stop_at = stop_at
        self.repeated_failure_at = repeated_failure_at
        self._streak = 0  # consecutive stuck (>= vote_threshold signals) iterations
        self._prev_sig: tuple[tuple[str, str], ...] | None = None
        self._error_counts: Counter[tuple[str, str]] = Counter()  # per-call failures this turn
        self._last_signals: list[str] = []  # signals fired on the most recent observe
        self._actions: list[str] = []  # ladder actions taken this turn (observability)

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
        self, calls: list[ToolCall], results: list[ToolResultBlock], *, assistant_text: str = ""
    ) -> None:
        """Score one tool-call iteration, updating the stuck streak.

        Call once per iteration that requested tool calls, after the batch has executed.
        Iterations with no tool calls (a text/empty completion) are not stuck by definition
        and should not be passed here.
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

        self._last_signals = signals
        if len(signals) >= self.vote_threshold:
            self._streak += 1
        else:
            self._streak = 0
        self._prev_sig = sig

    def next_action(self) -> StuckAction:
        """The ladder step implied by the current streak (call once per :meth:`observe`).

        Returns :attr:`StuckAction.NUDGE` / ``NARROW`` exactly once as the streak crosses
        each threshold, and ``STOP`` once it reaches ``stop_at``; ``CONTINUE`` otherwise.
        Records the chosen step for :attr:`actions` / :attr:`took_action`.
        """
        if self._streak >= self.stop_at:
            action = StuckAction.STOP
        elif self._streak == self.narrow_at:
            action = StuckAction.NARROW
        elif self._streak == self.nudge_at:
            action = StuckAction.NUDGE
        else:
            action = StuckAction.CONTINUE
        if action is not StuckAction.CONTINUE:
            self._actions.append(action.value)
        return action

    # ── recovery messages ────────────────────────────────────────────────────
    def nudge_message(self) -> str:
        """The corrective hint injected on a :attr:`StuckAction.NUDGE`."""
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

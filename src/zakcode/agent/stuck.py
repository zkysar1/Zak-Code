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
from collections.abc import Sequence
from enum import Enum
from hashlib import blake2b

from zakcode.messages import ToolResultBlock
from zakcode.providers.base import ToolCall
from zakcode.tools.base import RECEIPT_OF_CHANGE

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
#:
#: It counts OBSERVATIONS of the world. A tool whose result is the harness's own delivery or
#: acknowledgement -- a skill's body, its "[already loaded]" pointer, a wake-up's "armed" line --
#: is identical by construction and measures nothing, so the loop names those tools in
#: ``uncounted_outcome_tools`` and they never feed this signal (amended 2026-09-18). Measured on
#: a served workspace loop (gpt-5.6-luna): the stop hook orders ``Skill('<loop-skill>')``,
#: the loader answers with the same pointer every time, and the 3rd, 4th and 5th pointer of
#: one turn drew
#: nudge, narrow and step-back although distinct, successful work ran between them; the
#: graceful-stop body, asked for again after work, drew the whole ladder and a STOP in the
#: middle of the stop itself. Every OTHER signal still sees those calls.
#:
#: It counts them PER LAP of a loop the session is running, and a RECEIPT is not one of them
#: (ADR-0209). A served perpetual loop is ONE turn that lasts the whole run: a turn-end hook
#: refuses every stop and sends the model round its loop skill again. Counting over that whole
#: turn, a healthy loop's housekeeping (the same state probe, the same listing, once a lap)
#: climbed the ladder by itself, and because the Nth sighting lands on rung N, late in a run
#: ONE sighting of a familiar output was a STOP. Measured on gpt-5.6-luna, four served runs of
#: 35 minutes (bench/results/served-luna-preregistration.log, the ladder re-read): 47 rungs,
#: 8 of them stops; all 33 on a tool that looks at the world had a lap boundary between their
#: repeats, and the other 14 were on the plan tool's own receipt. So:
#:
#: * the counts start over when the loop goes round (``lap``, which the loop counts: the
#:   session's loop skill delivered again, by the harness at a refused stop or as the BODY
#:   answering the model's own Skill call). THE BOUND: only if the lap that just ended showed
#:   at least one outcome never seen before in this turn. A lap that showed nothing new was no
#:   lap of work, so a model that stops after every probe and is sent round again still climbs;
#: * a result whose tool flags it ``RECEIPT_OF_CHANGE`` (the plan tool's "Plan updated") is the
#:   harness's acknowledgement of a write that changed something, not a look at the world. It
#:   reads the same whenever the done count and the current step do, although the plan moved.
#:   It is the same observation as an earlier one only while NO work call has succeeded in
#:   between (``work``, ADR-0196's count): a model journalling into its plan between real
#:   steps is working, and one that rewrites its plan again and again with nothing in between
#:   is the churn this signal is the only net for, and still climbs. A receipt is never the
#:   "something new" of the bound above: it would be new after every work call.
SIG_REPEATED_OUTCOME = "repeated-outcome"

#: The bound on consecutive plan-only completions (ADR-0364), the net ADR-0192 left undone. A
#: completion whose every call is a plan tool, and that neither closed a step nor drew the
#: harness's advance (ADR-0168 lever N), cannot move the plan: a step moves on a status the model
#: sends, an outcome it records, or the work done while it is current, and such a completion
#: supplies none of them. The receipt rails could not see this run when each call CHANGED the
#: plan: the unchanged rail answers only a resend, lever N needs evidence, and the doom guard
#: needs a byte-identical batch. Measured on a served fleet (ten agents, one 35B-class local
#: model, 2026-11-05, 4 h, 2,175 calls): 12.8% of all calls sat in runs of three or more such
#: completions, at least 14% of agent time. The completion that would be the ``plan_only_at``-th
#: in a row is refused unexecuted and told the step to do; any other call ends the run, as does a
#: plan call that moved the plan. A refused completion is no failure of the plan tool and no look
#: at the world, so the error and outcome counts never see it (a ladder that called the refusals
#: "update_plan has failed 3 times" seeded "Investigate: why update_plan keeps failing", which is
#: false and more plan work); it fires ``SIG_PLAN_ONLY`` alone, which drives the ladder one rung
#: per refusal after the first (the first is itself the intervention), so a model that keeps
#: planning is still taken to a stop.
PLAN_ONLY_AT = 3
#: A completion the plan-only bound refused unexecuted (ADR-0364). STRONG, like a repeated outcome.
SIG_PLAN_ONLY = "plan-only"

#: The same bound on completions that only PRINT (ADR-0373). A shell call whose whole command is
#: one ``echo`` of literal words reads nothing and changes nothing, and each string can differ,
#: which the doom guard's byte-identical test cannot see and the outcome signal skips as too
#: short. Measured on a served fleet (ten sessions, one 35B-class local model, 2026-11-07, 24 h,
#: 7,723 completions): one session ran 149 such completions in a row, one every 3 s, until an
#: operator interrupted it, and no other run passed 2. Runs that MIX a print with plan calls
#: occur in ordinary work, so a run here is prints alone, counted apart: ADR-0364's count is
#: untouched and a print ends a plan-only run as any other call does. The completion that would
#: be the ``plan_only_at``-th in a row is refused unexecuted and climbs the ladder as a plan-only
#: refusal does, on ``SIG_PRINT_ONLY``, so what the rails say is true of what was counted.
SIG_PRINT_ONLY = "print-only"
#: A whole shell command that only prints literal words: ``echo``, then words with no expansion,
#: substitution, escape, glob, pipe, redirect, separator or comment in them. The words are
#: joined by spaces and tabs only: a newline ends the ``echo``, and whatever follows it runs.
_PRINT_ONLY_RE = re.compile(
    r"""\s*echo(?:[ \t]+(?:"[^"`$\\]*"|'[^']*'|[^\s"'`$\\|;&<>()#*?\[\]~{}]+))+\s*"""
)

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


def outcome_signature(
    name: str, output: str, epoch: int = 0, *, work: int | None = None
) -> str | None:
    """A stable identity for a tool OUTCOME, or ``None`` when it is too short to mean anything.

    ``epoch`` is the loop's count of successful FILE-EDIT calls so far this turn: the same
    output after an edit is a fresh measurement of a changed world (edit → test → edit → test
    is progress, not a loop) and must not compare equal to the one before the edit.

    ``work`` is given for a RECEIPT only (ADR-0209): the loop's count of successful work calls.
    Two identical receipts compare equal only while that count has not moved between them.
    """
    normalized = _VOLATILE_RE.sub("#", " ".join((output or "").split()))
    if len(normalized) < _OUTCOME_MIN_CHARS:
        return None
    signature = f"{name}\x00{epoch}\x00{normalized[:_OUTCOME_HEAD_CHARS]}"
    return signature if work is None else f"{signature}\x00{work}"


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
        uncounted_outcome_tools: frozenset[str] = frozenset(),
        plan_tools: frozenset[str] = frozenset(),
        plan_only_at: int = PLAN_ONLY_AT,
        shell_tools: frozenset[str] = frozenset(),
    ) -> None:
        self.vote_threshold = vote_threshold
        self.nudge_at = nudge_at
        self.narrow_at = narrow_at
        self.step_back_at = step_back_at
        self.stop_at = stop_at
        self.repeated_failure_at = repeated_failure_at
        self.outcome_repeat_at = outcome_repeat_at
        self.uncounted_outcome_tools = uncounted_outcome_tools
        #: The loop's plan tools (ADR-0364). Empty, the plan-only bound never counts anything.
        self.plan_tools = plan_tools
        self.plan_only_at = plan_only_at
        #: The loop's shell tools (ADR-0373). Empty, the print-only bound never counts anything.
        self.shell_tools = shell_tools
        self._plan_only_run = 0  # consecutive plan-only completions that did not move the plan
        self._print_only_run = 0  # consecutive completions that only printed literal words
        self._plan_finished: int | None = None  # the done count the last plan receipt reported
        self._streak = 0  # consecutive stuck (>= vote_threshold signals) iterations
        self._prev_sig: tuple[tuple[str, str], ...] | None = None
        self._error_counts: Counter[tuple[str, str]] = Counter()  # per-call failures this turn
        #: Identical outcomes since the counts last started over: at the turn's start, or at a
        #: lap boundary that followed a lap with something new in it (ADR-0038, ADR-0209).
        self._outcome_counts: Counter[str] = Counter()
        #: A digest of every OBSERVATION counted this turn. Never cleared: "new" means new to
        #: the turn, not to the lap, or a loop that alternates two probes would always be new.
        self._seen_outcomes: set[bytes] = set()
        self._lap = 0  # the loop's lap count as of the last observe (ADR-0209)
        self._lap_saw_new = False  # the lap under way has shown an outcome new to this turn
        self._last_outcome_repeats = 0  # the worst repeat count seen on the most recent observe
        self._last_outcome_tool = ""  # the tool that worst count belongs to
        self._last_outcome_receipt = False  # ...and whether it was a receipt of change
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
    def last_outcome_was_receipt(self) -> bool:
        """Whether the repeated outcome of the most recent :meth:`observe` was a tool's
        RECEIPT for a change, come back again with no work in between (plan churn), and not
        a look at the world (ADR-0209). ONE predicate: the rail, the trace note and the step
        the loop seeds at rung 1 all read it, so what is said in one cannot differ from another."""
        return SIG_REPEATED_OUTCOME in self._last_signals and self._last_outcome_receipt

    @property
    def last_was_plan_only(self) -> bool:
        """Whether the most recent :meth:`observe` was a completion the plan-only bound refused
        (ADR-0364). Like a receipt, it has no result to interpret: the remedy is the work."""
        return SIG_PLAN_ONLY in self._last_signals

    @property
    def last_was_print_only(self) -> bool:
        """Whether the most recent :meth:`observe` was a completion the print-only bound refused
        (ADR-0373). There is no result to interpret here either: the remedy is the work."""
        return SIG_PRINT_ONLY in self._last_signals

    @property
    def print_only_run(self) -> int:
        """How many completions in a row only printed literal words (ADR-0373); 0 after any
        other call."""
        return self._print_only_run

    @property
    def plan_only_run(self) -> int:
        """How many completions in a row called only the plan tools without moving the plan
        (ADR-0364); 0 after any other call."""
        return self._plan_only_run

    def refuses_plan_only(self, calls: Sequence[ToolCall]) -> bool:
        """Whether ``calls``, not yet executed, are the completion the plan-only bound refuses:
        plan tools only, after ``plan_only_at - 1`` such completions in a row (ADR-0364)."""
        return self._is_plan_only(calls) and self._plan_only_run >= self.plan_only_at - 1

    def refuses_print_only(self, calls: Sequence[ToolCall]) -> bool:
        """Whether ``calls``, not yet executed, are the completion the print-only bound refuses:
        prints alone, after ``plan_only_at - 1`` such completions in a row (ADR-0373)."""
        return self._is_print_only(calls) and self._print_only_run >= self.plan_only_at - 1

    @property
    def actions(self) -> list[str]:
        """The ladder actions taken this turn, in order (e.g. ``["nudge", "narrow"]``)."""
        return list(self._actions)

    @property
    def took_action(self) -> bool:
        """Whether any recovery step (nudge/narrow/stop) has fired this turn."""
        return bool(self._actions)

    def evidence(self) -> dict[str, object]:
        """What the ladder acted on, for the trace note: the signals that fired on the most
        recent :meth:`observe` and, when a repeated outcome is among them, the tool and how
        many times its result has now come back. Names and counts only, never the output.
        ``receipt`` is present, and ``True``, only when what came back was the tool's receipt
        for a change with no work in between (plan churn) and not a look at the world
        (ADR-0209), so a reader of the trace can tell the two apart.

        A served run's trace said ``no progress`` eight times in one turn and nothing could say
        which signal or which tool (2026-09-18); the outputs had been compacted away.
        """
        data: dict[str, object] = {"signals": ",".join(self._last_signals)}
        if self.last_was_plan_only:
            data["plan_only"] = self._plan_only_run
        if self.last_was_print_only:
            data["print_only"] = self._print_only_run
        if SIG_REPEATED_OUTCOME in self._last_signals:
            data["tool"] = self._last_outcome_tool
            data["repeats"] = self._last_outcome_repeats
            if self.last_outcome_was_receipt:
                data["receipt"] = True
        return data

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

    def failing_tools(self) -> list[tuple[str, int]]:
        """Tool names that failed at least ``repeated_failure_at`` times this turn, with the
        count, most-failed first — regardless of arguments.

        :meth:`error_signatures` sees the model that retries the SAME call; this sees the one
        that varies the arguments and fails every time (the wrong-premise shape: the path,
        command, or interface is what is wrong, not the argument). The decompose-on-stuck
        steps (ADR-0057) name the tool either way.
        """
        by_name: Counter[str] = Counter()
        for (name, _args), count in self._error_counts.items():
            by_name[name] += count
        return sorted(
            ((name, n) for name, n in by_name.items() if n >= self.repeated_failure_at),
            key=lambda item: item[1],
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
        lap: int = 0,
        work: int = 0,
    ) -> None:
        """Score one tool-call iteration, updating the stuck streak.

        Call once per iteration that requested tool calls, after the batch has executed.
        Iterations with no tool calls (a text/empty completion) are not stuck by definition
        and should not be passed here. Three readings of the loop's own counters, each taken
        AFTER the batch ran (see ``SIG_REPEATED_OUTCOME``): ``epoch`` is the turn's successful
        file-edit count, ``lap`` how many times the session's loop skill has been delivered
        again, ``work`` the successful work calls so far. Of ``lap`` and ``work`` only a CHANGE
        between two calls means anything, so the loop never resets either. A caller that passes
        none of them gets the plain whole-turn count.
        """
        sig = batch_signature(calls)
        by_id = {r.tool_use_id: r for r in results}
        if calls and all(
            (r := by_id.get(c.id)) is not None and (r.data or {}).get("plan_only_bound")
            for c in calls
        ):
            # ADR-0364: the bound answered this completion unexecuted. Its own signal alone, one
            # rung per refusal after the first, and nothing else counted (see SIG_PLAN_ONLY).
            # ADR-0373 refuses prints the same way, on their own count and signal.
            if self._is_print_only(calls):
                self._print_only_run += 1
                run, signal = self._print_only_run, SIG_PRINT_ONLY
            else:
                self._plan_only_run += 1
                run, signal = self._plan_only_run, SIG_PLAN_ONLY
            refusals = run - (self.plan_only_at - 1)
            self._last_signals = [signal]
            self._streak = max(self._streak, self.nudge_at + refusals - 2)
            self._prev_sig = sig
            return

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

        # A lap boundary (ADR-0209): the loop went round since the last observe, so the counts
        # start over. THE BOUND: only when the lap that just ended showed something new. This
        # batch already belongs to the new lap, so the boundary is settled before it is counted.
        if lap != self._lap:
            self._lap = lap
            if self._lap_saw_new:
                self._outcome_counts.clear()
            self._lap_saw_new = False

        # Repeated outcome (ADR-0038): count identical (tool, epoch, output) observations
        # since the counts last started over — NOT consecutively — and read the worst count
        # this batch.
        worst, worst_tool, worst_receipt = 0, "", False
        # ONE batch is one moment: the model asked for all of it before it saw any answer, so
        # identical results inside it are one observation, never a re-observation of a known
        # result. The epoch is read once, after the batch, so without this a batch of N
        # single-line edits to one file (N x "Made 1 replacement in <path>") counted N sightings
        # and jumped to rung N (ADR-0038, amended 2026-09-27).
        batch_outcomes: set[str] = set()
        for call in calls:
            result = by_id.get(call.id)
            if result is None:
                continue
            if call.name in self.uncounted_outcome_tools:
                continue  # the harness's own delivery: identical by construction, no measurement
            # The tool's own flag, never a search of the text (ADR-0203). An ERROR is never a
            # receipt of change, whatever its data says: nothing changed.
            receipt = not result.is_error and bool((result.data or {}).get(RECEIPT_OF_CHANGE))
            osig = outcome_signature(
                call.name, result.output or "", epoch, work=work if receipt else None
            )
            if osig is None or osig in batch_outcomes:
                continue
            batch_outcomes.add(osig)
            if not receipt:
                # Only a look at the world can be the lap's "something new": a receipt is new
                # after every work call, and would let any spin with a plan in it off the bound.
                digest = blake2b(osig.encode("utf-8", "surrogatepass"), digest_size=8).digest()
                if digest not in self._seen_outcomes:
                    self._seen_outcomes.add(digest)
                    self._lap_saw_new = True
            self._outcome_counts[osig] += 1
            if self._outcome_counts[osig] > worst:
                worst, worst_tool, worst_receipt = self._outcome_counts[osig], call.name, receipt
        self._last_outcome_repeats = worst
        self._last_outcome_tool = worst_tool
        self._last_outcome_receipt = worst_receipt
        if worst >= self.outcome_repeat_at:
            signals.append(SIG_REPEATED_OUTCOME)

        self._last_signals = signals
        if len(signals) >= self.vote_threshold:
            self._streak += 1
        else:
            self._streak = 0
        if worst >= self.outcome_repeat_at:
            # A strong signal: the Nth identical observation of a lap lands on rung N of the ladder
            # (3 → nudge, 4 → narrow, 5 → step back, 6 → stop with the defaults) regardless
            # of what the interleaved iterations did — the field loop alternated probes, so a
            # consecutive streak never formed while the same result came back fifteen times.
            self._streak = max(self._streak, self.nudge_at + (worst - self.outcome_repeat_at))
        self._prev_sig = sig
        self._observe_plan_only(calls, by_id)
        self._print_only_run = self._print_only_run + 1 if self._is_print_only(calls) else 0

    def _is_plan_only(self, calls: Sequence[ToolCall]) -> bool:
        return bool(calls and self.plan_tools) and all(c.name in self.plan_tools for c in calls)

    def _is_print_only(self, calls: Sequence[ToolCall]) -> bool:
        return bool(calls) and all(
            c.name in self.shell_tools
            and isinstance(command := c.arguments.get("command"), str)
            and _PRINT_ONLY_RE.fullmatch(command) is not None
            for c in calls
        )

    def _observe_plan_only(
        self, calls: Sequence[ToolCall], by_id: dict[str, ToolResultBlock]
    ) -> None:
        """Extend or end the plan-only run (ADR-0364). The plan MOVED when a receipt says the
        harness advanced it (lever N) or reports more steps done than the last receipt did: a
        model closing steps on work it already did, or walked by the advance, is not churning."""
        moved = False
        for call in calls:
            result = by_id.get(call.id)
            if result is None or result.is_error or not isinstance(result.data, dict):
                continue
            if result.data.get("autoadvanced"):
                moved = True
            finished = result.data.get("finished")
            if isinstance(finished, int) and not isinstance(finished, bool):
                if self._plan_finished is not None and finished > self._plan_finished:
                    moved = True
                self._plan_finished = finished
        self._plan_only_run = (
            self._plan_only_run + 1 if self._is_plan_only(calls) and not moved else 0
        )

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
        continuations it earns, so the ladder stays bounded. Nor does it touch the outcome
        counts: whether THOSE start over is the lap boundary's call, made in :meth:`observe`
        (ADR-0209), and a refusal that names no loop skill is no lap.
        """
        self._streak = 0
        self._prev_sig = None
        self._error_counts.clear()
        self._plan_only_run = 0
        self._print_only_run = 0

    # ── recovery messages ────────────────────────────────────────────────────
    def plan_only_message(self, step: str | None) -> str:
        """The result every call of a refused plan-only completion gets (ADR-0364). ``step``
        names the plan's current step, or is None when the plan has none to name."""
        then = (
            f"The plan in force is unchanged, and its current step is {step}: do that step now "
            "with a tool call."
            if step
            else "The plan in force is unchanged: act on it now with a tool call, or answer if "
            "the work is done."
        )
        return (
            f"Not executed: this is response {self._plan_only_run + 1} in a row that calls only "
            "the plan tools, with no other tool call between them. A step moves on a status you "
            "send, an outcome you record, or the work done while it is current, and these calls "
            f"supply none of them. {then} The plan tools are accepted again once another tool has "
            "run."
        )

    def print_only_message(self, step: str | None) -> str:
        """The result every call of a refused print-only completion gets (ADR-0373). ``step``
        names the plan's current step, or is None when the plan has none to name."""
        then = (
            f"The plan's current step is {step}: do that step now with a tool call that does the "
            "work, or answer in text if the work is done."
            if step
            else "Do the next piece of actual work with a tool call, or answer in text if the "
            "work is done."
        )
        return (
            f"Not executed: this is response {self._print_only_run + 1} in a row that only prints "
            "a fixed string, with no other tool call between them. A print reads nothing and "
            f"changes nothing. {then} A print runs again once another tool has run."
        )

    def nudge_message(self) -> str:
        """The diagnosis injected on a :attr:`StuckAction.NUDGE` — WHY the model is stuck.

        The remedy no longer rides here as advice: the loop turns the same evidence into
        investigative plan steps (ADR-0057) and says what it added. This stays the symptom,
        in the model's own terms.
        """
        if self.last_was_plan_only:
            # True of what was counted (ADR-0198): the calls were refused, not failed.
            return (
                f"You have now sent {self._plan_only_run} responses in a row that call only the "
                "plan tools, and the harness did not execute the last of them. Planning is not "
                "progress on the task: do the current step with a tool call."
            )
        if self.last_was_print_only:
            return (
                f"You have now sent {self._print_only_run} responses in a row that only print a "
                "fixed string, and the harness did not execute the last of them. Printing is not "
                "progress on the task: do the current step with a tool call, or answer in text "
                "if the work is done."
            )
        if SIG_REPEATED_OUTCOME in self._last_signals:
            if self.last_outcome_was_receipt:
                # Words that are TRUE of a receipt (ADR-0209, and ADR-0198's rule that a rail
                # says only what the harness knows). The model DID change something each
                # time, so "without changing anything" would be false, and a receipt is no
                # observation. What did not happen is any work in between.
                return (
                    f"You have now called {self._last_outcome_tool} "
                    f"{self._last_outcome_repeats} times with no other work succeeding in "
                    "between, and its receipt read the same each time. Recording a change "
                    "is not progress on the task: do the next piece of actual work."
                )
            return (
                f"You have now observed the SAME tool result {self._last_outcome_repeats} "
                "times without changing anything in between. Re-measuring a known "
                "result is not progress."
            )
        failing = self.failing_tools()
        if failing:
            name, count = failing[0]
            return (
                f"You appear to be stuck: {name} has failed {count} times this turn. "
                f"Calling {name} again with the same inputs is unlikely to succeed. Read the "
                f"error {name} returned — it may name a better tool — and switch to a tool "
                "whose description matches what you are working with, rather than repeating "
                "the same call."
            )
        return (
            "You appear to be stuck: the last few steps made no progress (a repeated or "
            "all-failing tool call)."
        )

    def narrow_message(self, refused: Sequence[str] = ()) -> str:
        """The corrective hint injected on a :attr:`StuckAction.NARROW`.

        ``refused`` names the tools the execution seam turns away on the next response. The
        schema and the system prompt still list them, unchanged (ADR-0268), so the rail says
        which ones are refused. Measured on a local model, naming the refused tools held 20
        first responses of 20 to read-only tools; naming the allowed ones held 19.
        """
        available = "only read-only tools are available"
        if refused:
            names = list(refused)
            listed = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
            available += f"; {listed} {'is' if len(names) == 1 else 'are'} refused"
        return (
            f"You are still stuck, so for your NEXT response {available} (the full toolset "
            "returns after that). Use them to investigate: read the file, the error, or the "
            "directory, and find the real cause before attempting another change."
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
            "attempts have all failed, which usually means they share one wrong "
            "assumption: a path that "
            "does not exist here, a command or tool that is not available, an interface "
            "that differs from what you expect. Do this, in order:\n"
            "1. State in one sentence what you are trying to accomplish.\n"
            "2. Name the assumption every failed attempt depended on.\n"
            "3. Check that assumption with read-only probes: list a directory you KNOW "
            "exists (such as the workspace root) and walk down to the real path; run the "
            "command with --help to see its real interface; read the file you believe is "
            "there.\n"
            "4. Rebuild your approach from what the probes actually show, and only then "
            "act.\n"
            "Do not repeat any earlier failing call until a probe has confirmed the "
            "assumption it depends on."
        )

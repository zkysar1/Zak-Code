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
#: a served Mind loop (gpt-5.6-luna): the stop hook orders ``Skill('aspirations')``, the loader
#: answers with the same pointer every time, and the 3rd, 4th and 5th pointer of one turn drew
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
    ) -> None:
        self.vote_threshold = vote_threshold
        self.nudge_at = nudge_at
        self.narrow_at = narrow_at
        self.step_back_at = step_back_at
        self.stop_at = stop_at
        self.repeated_failure_at = repeated_failure_at
        self.outcome_repeat_at = outcome_repeat_at
        self.uncounted_outcome_tools = uncounted_outcome_tools
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
            if osig is None:
                continue
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

    # ── recovery messages ────────────────────────────────────────────────────
    def nudge_message(self) -> str:
        """The diagnosis injected on a :attr:`StuckAction.NUDGE` — WHY the model is stuck.

        The remedy no longer rides here as advice: the loop turns the same evidence into
        investigative plan steps (ADR-0057) and says what it added. This stays the symptom,
        in the model's own terms.
        """
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
        return (
            "You appear to be stuck: the last few steps made no progress (a repeated or "
            "all-failing tool call)."
        )

    def narrow_message(self) -> str:
        """The corrective hint injected on a :attr:`StuckAction.NARROW`."""
        return (
            "You are still stuck, so for your NEXT response only read-only tools are "
            "available (the full toolset returns after that). Use them to investigate: "
            "read the file, the error, or the directory, and find the real cause before "
            "attempting another change."
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

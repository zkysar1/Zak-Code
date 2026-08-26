"""Detect a completion degenerating into repetition — the tail-loop attractor.

Field incident (2026-08-26, ADR-0018): gemini-2.5-flash-lite, driven at the harness's
then-default temperature 0.0, fell into the documented Gemini 2.5 repetition attractor and
streamed "I will now provide the information you requested." once a second toward a
~65k-token output cap. Nothing harness-side bounded or recognized it; the operator's Ctrl-C
was the only thing that ended the turn. Google documents the trigger (temperature below 1.0
on Gemini 2.5+) and the shape (loops until max_output_tokens); small local models fail the
same way.

Design rules (mirroring :mod:`zakcode.agent.stuck`):

* **Pure.** No provider, no I/O — one function over a string, shared by both turn paths.
* **Bounded work.** Only a fixed-size tail is examined, so periodic streaming checks stay
  cheap no matter how long the completion grows.
* **Two branches, each convicting only pathological shapes.**

  - *Line branch*: the last :data:`_TAIL_LINES` non-empty lines are dominated
    (>= :data:`_LINE_REPEATS`) by ONE short normalized line. A super-majority — not
    unanimity — convicts, because real loops mutate occasionally at the token level
    ("the information *I* requested").
  - *Period branch*: the raw tail is exactly periodic with a short period repeated many
    times — catches no-newline loops ("abc abc abc"), control-character floods, and
    single-character runs that have no line structure at all.

* **False positives are survivable by construction.** The loop's first strike discards the
  completion and retries once with an explicit "do not repeat yourself" rail; a
  legitimately repetitive answer (a fixture of identical lines) comes back essentially the
  same ONCE more and only then ends the turn — labeled degraded, never crashed. The
  detector therefore tunes for catching real loops within seconds, not for courtroom
  certainty; the thresholds still sit above any plausible legitimate run (12 identical
  trailing lines / an exactly-periodic 400-char tail).

A repeating unit longer than :data:`_MAX_UNIT_CHARS` is deliberately NOT convicted — a
model restating a whole paragraph is a quality problem, not the runaway-stream failure this
guard exists to stop (the per-completion output cap bounds that one instead).
"""

from __future__ import annotations

from collections import Counter

#: Window of trailing non-empty lines the line branch examines.
_TAIL_LINES = 15
#: How many of those must be ONE normalized line for the line branch to convict.
_LINE_REPEATS = 12
#: A repeating unit longer than this is restated prose, not a loop — never convicted.
_MAX_UNIT_CHARS = 200
#: No verdict before this much text exists (early completions are too short to judge).
_MIN_TAIL_CHARS = 400
#: Raw-character window for the exact-period branch.
_PERIOD_WINDOW = 400
#: Full repetitions of the period inside that window required to convict.
_PERIOD_REPEATS = 8


def repeated_tail(text: str) -> str | None:
    """The unit the tail of ``text`` is stuck repeating, or ``None`` when healthy."""
    if len(text) < _MIN_TAIL_CHARS:
        return None
    # Line branch: mutation-tolerant, matches the field incident's shape (one short
    # sentence repeated as its own paragraph, with occasional token-level variants).
    lines = [" ".join(line.split()).lower() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) >= _TAIL_LINES:
        unit, count = Counter(lines[-_TAIL_LINES:]).most_common(1)[0]
        if count >= _LINE_REPEATS and len(unit) <= _MAX_UNIT_CHARS:
            return unit
    # Period branch: the ENTIRE raw tail is periodic with a short period — detected by
    # self-overlap (``t`` has period ``p`` iff ``t[p:] == t[:-p]``; partial final unit
    # included, and unlike the string-doubling rotation trick this holds when ``p`` does
    # not divide the window length). Ordinary prose is never exactly periodic over 400
    # chars, so this convicts only floods; each comparison is one C-speed slice equality,
    # bounded by the window, so the scan stays cheap at streaming cadence.
    tail = text[-_PERIOD_WINDOW:]
    max_period = min(_MAX_UNIT_CHARS, len(tail) // _PERIOD_REPEATS)
    for period in range(1, max_period + 1):
        if tail[period:] == tail[:-period]:
            return tail[:period]
    return None

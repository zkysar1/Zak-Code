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


#: Burst-repetition (tool-argument) thresholds. Unlike :func:`repeated_tail` this scans
#: for a pathological run ANYWHERE in the text, because degeneration inside tool-call
#: arguments is buried mid-payload (field incident 2026-08-26: a python -c command carried
#: ``import json; `` ×28 and ``YOUR_`` ×38 mid-string and executed unjudged — the
#: completion-text detector explicitly skips tool-call batches). The thresholds convict
#: only runs no legitimate command contains: ≥150 chars of ONE short unit repeated ≥12×
#: consecutively. Single-character runs (divider lines, padding, newline floods) and
#: whitespace units are never convicted — those are everyday formatting.
_BURST_MIN_RUN_CHARS = 150
_BURST_MIN_REPEATS = 12
_BURST_MAX_UNIT_CHARS = 64
_BURST_SCAN_CAP = 65_536
#: Probe window / stride: a fully-periodic 128-char window every 32 chars guarantees
#: detection of any run ≥160 chars (window + stride), at C-speed slice-compare cost.
_BURST_PROBE_CHARS = 128
_BURST_STRIDE = 32


def burst_repetition(text: str) -> tuple[str, int] | None:
    """The ``(unit, repeats)`` of a consecutive repetition burst anywhere in ``text``.

    Returns ``None`` when healthy. Scans at most :data:`_BURST_SCAN_CAP` chars.
    """
    s = text[:_BURST_SCAN_CAP]
    n = len(s)
    if n < _BURST_MIN_RUN_CHARS:
        return None
    for i in range(0, n - _BURST_PROBE_CHARS + 1, _BURST_STRIDE):
        probe = s[i : i + _BURST_PROBE_CHARS]
        for period in range(2, _BURST_MAX_UNIT_CHARS + 1):
            if probe[period:] != probe[:-period]:
                continue
            unit = probe[:period]
            if len(set(unit)) < 2 or not unit.strip():
                break  # single-char/whitespace unit: formatting, never convicted
            # Measure the true run around the probe: walk whole units left, then right.
            start = i
            while start - period >= 0 and s[start - period : start] == unit:
                start -= period
            end = start + period
            while end + period <= n and s[end : end + period] == unit:
                end += period
            repeats = (end - start) // period
            if repeats >= _BURST_MIN_REPEATS and repeats * period >= _BURST_MIN_RUN_CHARS:
                return unit, repeats
            break  # smallest period found but the run is short — this probe is done
    return None

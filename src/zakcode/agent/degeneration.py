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
* **Three branches, each convicting only pathological shapes.**

  - *Line branch*: the last :data:`_TAIL_LINES` non-empty lines are dominated
    (>= :data:`_LINE_REPEATS`) by ONE short normalized line. A super-majority — not
    unanimity — convicts, because real loops mutate occasionally at the token level
    ("the information *I* requested").
  - *Near-duplicate branch* (ADR-0033): the same window is dominated by lines that share
    most of their WORDS with one short line without being identical — the apology spiral
    ("Let's try this again. I will try to add the skill correctly." / "Let's try again. I
    will try to create the skill correctly." / "I will try to create the skill correctly."),
    which mutates a word or two per line so the exact branch never reaches its bar.
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

import re
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

#: Near-duplicate branch (ADR-0033) — measured on the 2026-08-26 serene transcript, where
#: the exact branch topped out at 3 identical lines of 15 while 10–11 of those 15 shared
#: >= 60% of their words with ONE short sentence. Two healthy look-alikes score just as
#: high on word overlap and are acquitted by the two extra predicates:
#:
#: * a template LISTING ("Line 5: a distinct observation about file number 5." ×15) —
#:   every line introduces a token seen nowhere else in the window, whereas a spiral
#:   recycles a closed vocabulary (``_NEAR_DUP_MAX_NOVEL_FRAC``);
#: * an IDENTICAL-line fixture below the exact branch's bar — no mutation at all, which is
#:   that branch's jurisdiction, and a fuzzier branch must not undercut its 12-of-15 verdict
#:   (``_NEAR_DUP_MIN_VARIANTS``).
_NEAR_DUP_LINES = 8
_NEAR_DUP_SIMILARITY = 0.6
_NEAR_DUP_MIN_VARIANTS = 3
_NEAR_DUP_MIN_TOKENS = 4
_NEAR_DUP_MAX_NOVEL_FRAC = 0.5
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _near_duplicate_unit(lines: list[str]) -> str | None:
    """The short line most of the tail ``lines`` are near-duplicates of, or ``None``.

    ``lines`` are normalized (lowercased, whitespace-collapsed, non-empty); only the last
    :data:`_TAIL_LINES` are judged. Similarity is word-set Jaccard, so token-level mutation
    ("add" → "create", "this again" → "again") does not acquit the way it does for the exact
    branch.
    """
    window = lines[-_TAIL_LINES:]
    if len(window) < _NEAR_DUP_LINES:
        return None
    tokens = [set(_TOKEN_RE.findall(line)) for line in window]
    # How many window lines each token occurs in: a line whose every token recurs
    # elsewhere adds no vocabulary — the signature of a spiral, never of a listing.
    line_counts: Counter[str] = Counter(tok for toks in tokens for tok in toks)
    best: tuple[int, str] | None = None
    for unit, unit_tokens in zip(window, tokens, strict=True):
        if len(unit) > _MAX_UNIT_CHARS or len(unit_tokens) < _NEAR_DUP_MIN_TOKENS:
            continue
        similar = [
            j
            for j, other in enumerate(tokens)
            if len(unit_tokens & other) / len(unit_tokens | other) >= _NEAR_DUP_SIMILARITY
        ]
        if len(similar) < _NEAR_DUP_LINES:
            continue
        if len({window[j] for j in similar}) < _NEAR_DUP_MIN_VARIANTS:
            continue  # exact repeats only: the exact branch's jurisdiction, and its bar
        novel = sum(1 for j in similar if any(line_counts[t] == 1 for t in tokens[j]))
        if novel >= len(similar) * _NEAR_DUP_MAX_NOVEL_FRAC:
            continue  # the similar lines keep bringing new words: a listing, not a spiral
        if best is None or len(similar) > best[0]:
            best = (len(similar), unit)
    return best[1] if best is not None else None


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
    # Near-duplicate branch: the same window, judged on shared words instead of equality.
    near = _near_duplicate_unit(lines)
    if near is not None:
        return near
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

"""Front-matter ``description:`` reader for the bench catalogue scripts (stdlib only).

The one-line regex ``^description:\\s*(.+)$`` reads a YAML block scalar -- ``description: >-``
followed by indented lines -- as the two-character indicator. Measured 2026-09-12: 11 of 145 Mind
skills carry one; every catalogue rendered them as ``- name: >-``, they scored FULL 0.364 against
0.623 for the other 129 and 0.091 in every BM25 shortlist arm (a name-only entry is unretrievable),
and they filled the bottom of the lowest-fidelity tertile by construction (ADR-0158 second
addendum). ``coach_choosability.py`` carries a verbatim copy of ``read_description`` because it
runs as a single stdlib file on the coach box; ``tests/test_bench_frontmatter.py`` pins the two
copies to the same AST.
"""

from __future__ import annotations

import re

_KEY = re.compile(r"^description:[ \t]*(.*)$", re.M)
YAML_INDICATORS = frozenset({">", ">-", ">+", "|", "|-", "|+"})


def read_description(text: str) -> str | None:
    """The ``description:`` value of a SKILL.md front matter, folded to ONE line.

    Handles plain scalars, quoted scalars, and block scalars (``>``/``|`` with a chomping
    indicator), whose continuation lines are the indented lines that follow. A multi-line plain
    scalar's indented continuation lines are folded in the same way. Returns ``None`` when there
    is no description or it is empty -- never the indicator itself.
    """
    m = _KEY.search(text)
    if not m:
        return None
    first = m.group(1).strip()
    rest = text[m.end() :].splitlines()[1:]  # lines after the ``description:`` line
    continuation: list[str] = []
    for line in rest:
        if line.strip() == "":
            continue  # a blank line inside a block folds to nothing; it does not end the block
        if not line.startswith((" ", "\t")):
            break
        continuation.append(line.strip())
    if first in YAML_INDICATORS:
        return " ".join(continuation).strip() or None
    if len(first) >= 2 and first[0] == first[-1] and first[0] in "\"'":
        first = first[1:-1].strip()
    folded = " ".join([first, *continuation]).strip()
    return folded or None


def assert_sane(entries: list[tuple], min_chars: int = 12) -> None:
    """Refuse a catalogue whose descriptions include a YAML indicator or (near-)empty text.

    An instrument that renders ``- name: >-`` does not error; it measures something else and
    reports it with full confidence. ``entries`` are ``(name, description, ...)`` tuples.
    """
    bad = [e[0] for e in entries if not e[1] or e[1] in YAML_INDICATORS or len(e[1]) < min_chars]
    if bad:
        raise RuntimeError(
            f"catalogue instrument failure: {len(bad)} skill(s) with a missing, indicator-only or "
            f"<{min_chars}-char description: {bad[:8]}{' ...' if len(bad) > 8 else ''}"
        )

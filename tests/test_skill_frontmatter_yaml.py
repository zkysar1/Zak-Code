"""The skill frontmatter parser reads indentation as YAML does.

Measured 2026-09-23 against PyYAML: of a live Mind's 146 skills, 36 reached the model's
catalogue with the wrong description. 11 were YAML block scalars (``description: >-`` then
indented lines) listed as the indicator ``>-``, and a folded line with a colon in it became a
key of its own. 25 had an ``arguments:`` list whose items carry a ``description:`` of their own,
and the parser, which stripped every line's indentation, let the LAST ``description:`` win, so
the skill was listed with one of its arguments' descriptions. Coach's deployment: 20 of 62. The
bench's catalogue reader had been fixed for block scalars on 2026-09-12 (ADR-0158 second
addendum, ``tests/test_bench_frontmatter.py``); the product parser that builds the real
catalogue had not.

Every expected string below is what YAML reads. Where PyYAML is installed (it arrives with
litellm), the differential test holds the parser to it directly.
"""

from __future__ import annotations

import pytest

from zakcode.skills import parse_frontmatter


def _fm(frontmatter: str) -> tuple[str, dict[str, str | list[str]]]:
    """The description and extras of a SKILL.md with this frontmatter (after `name`)."""
    fm, body = parse_frontmatter(f"---\nname: s\n{frontmatter}\n---\nbody\n")
    assert body == "body"
    return fm.description, fm.extras


FOLDED_WITH_A_COLON = """\
description: >-
  Summarise the notes in a folder, one paragraph per file.
  Reads each file once (step 2: the index), then writes
  the summary.
user-invocable: false"""

ARGUMENTS_WITH_THEIR_OWN_DESCRIPTIONS = """\
description: Count the words in a text file.
arguments:
  - name: path
    description: The file to count
    required: true
  - name: mode
    description: Words or lines
triggers:
  - /word-count"""

#: Each case: frontmatter after ``name:`` and the description YAML reads from it.
CASES = {
    "folded, stripped": (
        "description: >-\n  one line\n  and the next\n",
        "one line and the next",
    ),
    "folded, with a colon in a line": (
        FOLDED_WITH_A_COLON,
        "Summarise the notes in a folder, one paragraph per file. Reads each file once "
        "(step 2: the index), then writes the summary.",
    ),
    "folded, a blank line is a break": (
        "description: >-\n  first paragraph\n  continues\n\n  second paragraph\n",
        "first paragraph continues\nsecond paragraph",
    ),
    "folded, a more-indented line keeps its breaks": (
        "description: >-\n  intro\n    - kept as written\n  outro\n",
        "intro\n  - kept as written\noutro",
    ),
    "literal keeps each break": (
        "description: |\n  line one\n  line two\n",
        "line one\nline two",
    ),
    "plain value continued on indented lines": (
        "description: starts here\n  and goes on\n  to the end\n",
        "starts here and goes on to the end",
    ),
    "double-quoted with escapes": (
        'description: "asks \\"what have you done\\", then a tab\\tand a backslash \\\\"\n',
        'asks "what have you done", then a tab\tand a backslash \\',
    ),
    "single-quoted with a doubled quote": (
        "description: 'it''s quoted'\n",
        "it's quoted",
    ),
    "an argument's description does not replace the skill's": (
        ARGUMENTS_WITH_THEIR_OWN_DESCRIPTIONS,
        "Count the words in a text file.",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_description_is_what_yaml_reads(case: str) -> None:
    frontmatter, expected = CASES[case]
    description, _ = _fm(frontmatter)
    assert description == expected


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_description_matches_pyyaml(case: str) -> None:
    yaml = pytest.importorskip("yaml")
    frontmatter, _ = CASES[case]
    reference = yaml.safe_load(f"name: s\n{frontmatter}")
    description, _ = _fm(frontmatter)
    assert description == str(reference["description"]).strip()


def test_an_indented_line_is_never_a_key_of_its_own() -> None:
    _, extras = _fm(FOLDED_WITH_A_COLON)
    assert set(extras) == {"user_invocable"}
    assert extras["user_invocable"] == "false"  # the key after the block still parses

    _, extras = _fm(ARGUMENTS_WITH_THEIR_OWN_DESCRIPTIONS)
    assert set(extras) == {"arguments", "triggers"}
    # A list of maps keeps each item's first line, as before; the item's other keys stay in
    # the item instead of leaking to the top level.
    assert extras["arguments"] == ["name: path", "name: mode"]
    assert extras["triggers"] == ["/word-count"]


def test_a_nested_mapping_stays_under_its_key() -> None:
    _, extras = _fm("stats:\n  runs: 3\n  last_run: never\nmode: x")
    assert extras == {"stats": "", "mode": "x"}


def test_a_sequence_level_with_its_key_is_still_a_list() -> None:
    _, extras = _fm('triggers:\n- "/one"\n- /two\nmode: any')
    assert extras == {"triggers": ["/one", "/two"], "mode": "any"}


def test_a_nested_sequence_inside_an_item_is_not_an_item() -> None:
    _, extras = _fm("steps:\n  - name: a\n    subs:\n      - inner\n  - name: b")
    assert extras["steps"] == ["name: a", "name: b"]


@pytest.mark.parametrize(
    ("header", "expected"),
    [("|-", "a\nb"), ("|", "a\nb\n"), ("|+", "a\nb\n\n\n"), (">", "a b\n"), (">+", "a b\n\n\n")],
)
def test_an_extra_keeps_the_ending_its_chomping_asks_for(header: str, expected: str) -> None:
    frontmatter = f"notes: {header}\n  a\n  b\n\n\nnext: x"
    _, extras = _fm(frontmatter)
    assert extras["notes"] == expected
    assert extras["next"] == "x"
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load(f"name: s\n{frontmatter}")["notes"] == expected


def test_an_unbalanced_quote_keeps_the_old_reading() -> None:
    description, extras = _fm("description: \"half open\nmode: 'x")
    assert description == "half open"
    assert extras["mode"] == "x"

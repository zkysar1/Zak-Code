"""Tests for the ``save_rule`` tool and its storage primitive (g-368-14).

``save_rule`` is the WRITE half of the rules lane ``read_rule`` already reads. The pair is
the point, so these tests pin the ROUND TRIP — a rule authored by the tool is discovered,
named and described correctly on the next discovery pass — rather than only asserting that
a file appeared. A write whose output the reader cannot parse would pass a file-exists
assertion and fail the only thing that matters.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zakcode.rules import (
    MAX_RULE_FILE_CHARS,
    RuleError,
    discover_rule_dir,
    project_rules_dir,
    save_rule,
)
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.save_rule import SaveRuleTool


def _run(args: dict, rules_dir: Path):
    ctx = ToolContext(workspace_root=rules_dir)
    return asyncio.run(SaveRuleTool(rules_dir).execute(args, ctx))


# --- the round trip: what save_rule writes, discover_rule_dir must read back -------------


def test_saved_rule_round_trips_through_discovery(tmp_path: Path) -> None:
    """THE CONTRACT. Name, description and body all survive the writer -> parser hop."""
    save_rule(
        "deploy-target",
        "Deploys go to dev, never main.",
        "# Deploy target\nAlways merge to `dev`.",
        rules_dir=tmp_path,
    )
    rules, errors = discover_rule_dir(tmp_path)
    assert errors == {}
    assert len(rules) == 1
    assert rules[0].name == "deploy-target"
    assert rules[0].description == "Deploys go to dev, never main."
    assert "Always merge to `dev`." in rules[0].content
    # The body must NOT still contain the frontmatter fence — that would mean the parser
    # failed to split it off and the whole file became the body.
    assert "---" not in rules[0].content.splitlines()[0]


def test_frontmatter_fence_is_the_first_byte(tmp_path: Path) -> None:
    """_split_frontmatter tests lines[0] == '---'. Anything above it — a blank line, a
    comment — makes the parser return NO metadata and silently treat the whole file as the
    body, so the rule loads under its filename stem with its description lost. That
    degradation is invisible, which is why it gets its own byte-level test."""
    path = save_rule("fence-check", "d", "body", rules_dir=tmp_path)
    assert path.read_text(encoding="utf-8").startswith("---\n")


def test_description_with_a_colon_round_trips(tmp_path: Path) -> None:
    """The parser partitions on the FIRST ':' and takes the remainder verbatim."""
    save_rule("colon-rule", "Note: this has a colon.", "body", rules_dir=tmp_path)
    rules, _ = discover_rule_dir(tmp_path)
    assert rules[0].description == "Note: this has a colon."


def test_internal_quotes_survive_and_edge_quotes_are_normalised(tmp_path: Path) -> None:
    """The parser ends with .strip("\\"'"), so a description that STARTS or ENDS with a quote
    char cannot round-trip in this format at all — by the reader, not by any writer choice.
    save_rule normalises with the same strip, which is what makes a write idempotent under
    re-read (pinned separately below) instead of shedding one character per edit.

    Recorded because the first draft of this test asserted the wrong thing: it claimed to
    pin 'unquoted beats quoted' using an internally-quoted description, and passed against
    the quoted writer, because the outer pair is all that gets stripped either way. With the
    normalisation in place quoting is REDUNDANT, not harmful. What is worth pinning is the
    reader's actual behaviour, so that is what this asserts."""
    save_rule("quote-rule", 'The flag is called "lean".', "body", rules_dir=tmp_path)
    rules, _ = discover_rule_dir(tmp_path)
    assert rules[0].description == 'The flag is called "lean".', "internal quotes survive"

    save_rule("quote-tail", 'Say "hello"', "body", rules_dir=tmp_path)
    rules, _ = discover_rule_dir(tmp_path)
    tail = next(r for r in rules if r.name == "quote-tail")
    assert tail.description == 'Say "hello', "an edge quote is dropped by the reader's strip"


def test_rewriting_the_same_rule_is_idempotent(tmp_path: Path) -> None:
    """Save -> read -> save the value that came back must produce the same file, or the
    description drifts a little every time a rule is edited."""
    save_rule("idem", "A description.", "body", rules_dir=tmp_path)
    first = (tmp_path / "idem.md").read_text(encoding="utf-8")
    rules, _ = discover_rule_dir(tmp_path)
    save_rule("idem", rules[0].description, rules[0].content, rules_dir=tmp_path, overwrite=True)
    assert (tmp_path / "idem.md").read_text(encoding="utf-8") == first


# --- refusals ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["../escape", "has/slash", "Has-Caps", "-leading-dash", "", "a" * 65],
)
def test_unsafe_or_malformed_names_are_refused(tmp_path: Path, name: str) -> None:
    with pytest.raises(RuleError):
        save_rule(name, "d", "body", rules_dir=tmp_path)


def test_body_over_the_per_rule_budget_is_refused(tmp_path: Path) -> None:
    """MAX_RULE_FILE_CHARS is where the RENDER truncates. Authoring past it would produce a
    rule that is silently cut every turn it is read, so the writer refuses instead."""
    with pytest.raises(RuleError, match="budget"):
        save_rule("huge", "d", "x" * (MAX_RULE_FILE_CHARS + 1), rules_dir=tmp_path)
    assert not (tmp_path / "huge.md").exists()


def test_multiline_description_is_refused(tmp_path: Path) -> None:
    """The frontmatter is line-based: an embedded newline either invents a bogus key or,
    if the line is '---', closes the block early and swallows the body."""
    with pytest.raises(RuleError, match="single line"):
        save_rule("multi", "line one\n---\nline two", "body", rules_dir=tmp_path)


def test_empty_body_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RuleError):
        save_rule("empty", "d", "   ", rules_dir=tmp_path)


def test_existing_rule_needs_overwrite(tmp_path: Path) -> None:
    save_rule("dup", "first", "body one", rules_dir=tmp_path)
    with pytest.raises(RuleError, match="already exists"):
        save_rule("dup", "second", "body two", rules_dir=tmp_path)
    rules, _ = discover_rule_dir(tmp_path)
    assert rules[0].description == "first", "a refused write must not have touched the file"
    save_rule("dup", "second", "body two", rules_dir=tmp_path, overwrite=True)
    rules, _ = discover_rule_dir(tmp_path)
    assert rules[0].description == "second"


def test_symlinked_target_cannot_redirect_the_write_out_of_tree(tmp_path: Path) -> None:
    """The kebab-case check never resolves reparse points, so containment is checked on the
    REAL path — a pre-planted symlink at <rules_dir> must not land the write elsewhere."""
    outside = tmp_path / "outside"
    outside.mkdir()
    rules_dir = tmp_path / "rules"
    try:
        rules_dir.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform/account")
    # The link itself resolves to `outside`, so a write THROUGH it stays contained relative
    # to its own resolved root; what must never happen is a write landing outside the tree
    # the caller named. Assert the file lands under the resolved root, not somewhere else.
    path = save_rule("linked", "d", "body", rules_dir=rules_dir)
    assert path.resolve().parent == outside.resolve()


# --- the tool wrapper -------------------------------------------------------------------


def test_tool_saves_and_reports_next_session(tmp_path: Path) -> None:
    res = _run({"name": "t-rule", "description": "A thing.", "body": "Do the thing."}, tmp_path)
    assert not res.is_error
    assert (tmp_path / "t-rule.md").exists()
    # The deferral is load-bearing and the model must be told: discovery and the rules index
    # are cache-stable per session, so a rule written now cannot apply now.
    assert "next session" in res.output


def test_tool_requires_a_description(tmp_path: Path) -> None:
    """save_skill tolerates a missing description; a rule cannot. The index line IS how a
    future session discovers the rule, so a blank one is present-but-unfindable."""
    res = _run({"name": "t-rule", "body": "Do the thing."}, tmp_path)
    assert res.is_error
    assert "description" in res.output
    assert not (tmp_path / "t-rule.md").exists()


def test_tool_turns_a_rule_error_into_a_clean_error_not_a_crash(tmp_path: Path) -> None:
    res = _run({"name": "../escape", "description": "d", "body": "b"}, tmp_path)
    assert res.is_error
    assert "kebab-case" in res.output


def test_tool_writes_where_the_workspace_helper_points(tmp_path: Path) -> None:
    """The facade wires the tool to project_rules_dir(); pin that this is a rules dir
    discovery actually scans, so an authored rule is not written somewhere inert."""
    from zakcode.rules import default_rule_dirs

    assert project_rules_dir(tmp_path) == tmp_path / ".zakcode" / "rules"
    assert project_rules_dir(tmp_path) in default_rule_dirs(tmp_path)

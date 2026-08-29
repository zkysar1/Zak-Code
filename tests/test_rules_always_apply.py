"""ADR-0105 — ``alwaysApply: true`` keeps a rule's full body in the prompt under the lean index."""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.rules import (
    MAX_RULE_FILE_CHARS,
    MAX_RULES_TOTAL_CHARS,
    Rule,
    RuleRegistry,
    _split_frontmatter,
    discover_rule_dir,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _registry(*rules: Rule) -> RuleRegistry:
    registry = RuleRegistry()
    for rule in rules:
        registry.add(rule)
    return registry


class TestFrontmatter:
    @pytest.mark.parametrize("value", ["true", "True", "yes", "1", '"true"'])
    def test_truthy_spellings_pin(self, tmp_path: Path, value: str) -> None:
        _write(tmp_path / "a.md", f"---\ndescription: d\nalwaysApply: {value}\n---\nBODY")
        rules, errors = discover_rule_dir(tmp_path)
        assert errors == {} and rules[0].always_apply is True
        assert rules[0].content == "BODY" and rules[0].description == "d"

    @pytest.mark.parametrize("value", ["false", "no", "0", "maybe", ""])
    def test_anything_else_does_not_pin(self, tmp_path: Path, value: str) -> None:
        _write(tmp_path / "a.md", f"---\nalwaysApply: {value}\n---\nBODY")
        rules, _ = discover_rule_dir(tmp_path)
        assert rules[0].always_apply is False

    def test_absent_flag_is_off_and_other_keys_still_parse(self) -> None:
        meta, body = _split_frontmatter("---\nname: n\ndescription: d\npaths:\n  - x\n---\nB")
        assert meta == {"name": "n", "description": "d"} and body == "B"


class TestLeanIndex:
    def test_pinned_body_rides_in_full_and_the_rest_stay_one_line(self, tmp_path: Path) -> None:
        pinned = Rule(
            "return-protocol",
            "END EVERY TURN WITH A TOOL CALL.\nSecond line.",
            tmp_path / "r.md",
            description="End every turn with a tool call",
            always_apply=True,
        )
        other = Rule(
            "archive-before-delete",
            "ARCHIVE FIRST — the body.",
            tmp_path / "a.md",
            description="Archive before delete",
        )
        out = _registry(other, pinned).render_index()
        assert out.startswith("Pinned rules (full text — always apply these):")
        assert "## return-protocol\nEND EVERY TURN WITH A TOOL CALL.\nSecond line." in out
        assert "- archive-before-delete: Archive before delete [" in out
        assert "ARCHIVE FIRST" not in out  # the unpinned body is NOT in the prompt
        assert "- return-protocol:" not in out  # a pinned rule is not ALSO indexed
        assert out.index("Pinned rules") < out.index("Project rules (operator-authored")

    def test_without_pins_the_index_is_byte_identical_to_before(self, tmp_path: Path) -> None:
        a = Rule("a", "A body", tmp_path / "a.md", description="alpha")
        b = Rule("b", "B body", tmp_path / "b.md", description="beta")
        out = _registry(a, b).render_index()
        assert "Pinned rules" not in out
        assert out.startswith("Project rules (operator-authored standing guidance). Each rule")
        assert f"- a: alpha [{tmp_path / 'a.md'}]\n- b: beta [{tmp_path / 'b.md'}]" in out

    def test_pinned_body_is_capped_per_file_and_the_total_is_bounded(self, tmp_path: Path) -> None:
        big = Rule("big", "x" * (MAX_RULE_FILE_CHARS + 500), tmp_path / "big.md", always_apply=True)
        out = _registry(big).render_index()
        assert out.split("## big\n", 1)[1].count("x") == MAX_RULE_FILE_CHARS
        rules = [
            Rule(f"p{i:02d}", "y" * MAX_RULE_FILE_CHARS, tmp_path / f"p{i}.md", always_apply=True)
            for i in range(8)
        ]
        out = _registry(*rules).render_index()
        assert len(out) <= MAX_RULES_TOTAL_CHARS
        assert "further rule(s) omitted" in out
        # Pinned bodies take the budget first; whatever pinned rules did not fit are the
        # ones counted as dropped, and no unpinned line was ever in the running.
        kept = out.count("## p")
        assert kept >= 1 and f"[{8 - kept} further rule(s) omitted" in out

    def test_pinned_bodies_leave_room_for_the_index_lines_that_fit(self, tmp_path: Path) -> None:
        pinned = Rule("p", "z" * 100, tmp_path / "p.md", always_apply=True)
        others = [
            Rule(f"o{i}", f"body {i}", tmp_path / f"o{i}.md", description=f"o{i} summary")
            for i in range(3)
        ]
        out = _registry(pinned, *others).render_index()
        assert "## p\n" + "z" * 100 in out
        for i in range(3):
            assert f"- o{i}: o{i} summary [" in out


class TestFullRender:
    def test_pinned_rules_are_folded_first_so_the_budget_drops_them_last(
        self, tmp_path: Path
    ) -> None:
        # 5 rules of the per-file cap each: only 3 fit. Without pins the first three by
        # discovery order survive; with the LAST one pinned it jumps the queue.
        rules = [Rule(f"r{i}", "w" * MAX_RULE_FILE_CHARS, tmp_path / f"r{i}.md") for i in range(5)]
        plain = _registry(*rules).render()
        assert "## r0" in plain and "## r2" in plain and "## r4" not in plain
        rules[4].always_apply = True
        pinned = _registry(*rules).render()
        assert "## r4" in pinned and pinned.index("## r4") < pinned.index("## r0")
        assert "## r2" not in pinned  # one fewer unpinned rule fits, as it must
        assert "[2 further rule(s) omitted" in pinned

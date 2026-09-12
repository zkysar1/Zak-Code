"""The bench catalogue reader must read a YAML block-scalar description, never its indicator.

Measured 2026-09-12: ``^description:\\s*(.+)$`` read ``description: >-`` as the string ``>-`` for
11 of 145 skills, which entered every catalogue as ``- name: >-`` (ADR-0158 second addendum).
The coach box runs a single stdlib file, so ``coach_choosability.py`` carries a verbatim copy of
the reader; the parity test pins the two copies to one AST.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"


def _load():
    spec = importlib.util.spec_from_file_location(
        "bench_frontmatter_under_test", BENCH / "_frontmatter.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


FOLDED = """---
name: x
description: >-
  Audit the alignment between an account's rows and the
  directories it serves.

  Second paragraph folds too.
tags: [a]
---
# body
"""
LITERAL = "---\ndescription: |\n  line one\n  line two\n---\n"
PLAIN = "---\ndescription: Verify the SHA landed.\ntags: [a]\n---\n"
QUOTED = '---\ndescription: "Quoted text: with a colon."\n---\n'
MULTILINE_PLAIN = "---\ndescription: starts here\n  and continues here\nname: y\n---\n"


@pytest.mark.parametrize(
    ("text", "want"),
    [
        (
            FOLDED,
            "Audit the alignment between an account's rows and the directories it serves. "
            "Second paragraph folds too.",
        ),
        (LITERAL, "line one line two"),
        (PLAIN, "Verify the SHA landed."),
        (QUOTED, "Quoted text: with a colon."),
        (MULTILINE_PLAIN, "starts here and continues here"),
        ("---\nname: z\n---\n", None),
        ("---\ndescription: >-\nname: z\n---\n", None),  # indicator with no body: None, never ">-"
    ],
)
def test_read_description(text, want):
    assert _load().read_description(text) == want


def test_assert_sane_refuses_indicators_and_stubs():
    mod = _load()
    with pytest.raises(RuntimeError, match="instrument failure"):
        mod.assert_sane([("ok-skill", "A real description of a skill."), ("bad-skill", ">-")])
    with pytest.raises(RuntimeError, match="bad-skill"):
        mod.assert_sane([("bad-skill", "short")])
    mod.assert_sane([("ok-skill", "A real description of a skill.")])  # positive control: passes


def test_coach_copy_matches_the_shared_reader():
    """coach_choosability.py inlines read_description/assert_sane; both copies must be one logic."""

    def fn(path: Path, name: str) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        node.body = [
            b
            for b in node.body
            if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))
        ]
        return ast.dump(node)

    for name in ("read_description", "assert_sane"):
        assert fn(BENCH / "_frontmatter.py", name) == fn(BENCH / "coach_choosability.py", name)

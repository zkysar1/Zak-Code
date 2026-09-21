"""ADR-0190, decision 7, kept true: prose follows the names.

The tool list a model is offered says ``Read``, ``Write``, ``Edit``, ``LS``, ``Glob``, ``Grep``,
``Bash``, ``WebFetch``, ``WebSearch``, ``Skill``. The names those tools carried before ADR-0190
still RESOLVE (they are registry aliases), but ADR-0190's own finding was that "aliases change
what resolves; a small model calls what it can SEE". A refusal that says "use write_file", a
rail that says "run it now with use_skill" or a prompt section headed "Skills (use_skill)" names
a tool the model cannot see in its list. Three dozen such sentences outlived the rename
(found 2026-09-21: one of them read "call Write with the first part ... then edit_file").

This test reads every string literal in the package, docstrings aside, and fails on a SENTENCE
that names a tool the old way. A literal that is nothing but the name is an identifier (an alias
row, a role set that reads resumed history, a renderer's table) and is what decision 4 asks for.
Anything else that must keep an old spelling is listed in ``ALLOWED`` with the reason.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from zakcode.tool_names import PRE_0190_TOOL_NAMES

SRC = Path(__file__).resolve().parents[1] / "src" / "zakcode"

#: Old names that are also ordinary English or shell words. For these only a TOOL-shaped
#: mention counts: call syntax, or the word "tool" beside the quoted name.
AMBIGUOUS = {"bash", "glob", "grep"}
_PLAIN = sorted(set(PRE_0190_TOOL_NAMES) - AMBIGUOUS)
_OLD_NAME = re.compile(r"(?<![\w./-])(" + "|".join(map(re.escape, _PLAIN)) + r")(?![\w-])")
_OLD_TOOL_SHAPED = re.compile(
    r"(?<![\w./-])(?:bash|glob|grep)\("  # grep(pattern=...)
    r"|[`'](?:bash|glob|grep)[`'] tool"  # the `bash` tool
    r"|\bthe (?:bash|glob|grep) tool\b"
    r"|\buse (?:glob|grep) with\b"
)

#: (path under src/zakcode, a substring of the literal) -> why the old spelling stays.
ALLOWED: dict[tuple[str, str], str] = {
    ("agent/loop.py", r"\b(?:Skill|use_skill)\("): (
        "the re-entry pattern reads hook text written for either spelling (ADR-0187)"
    ),
    ("__init__.py", "invoked via use_skill"): "a LOG line, and bench/served_stops.py reads it",
    ("__init__.py", "use_skill deduped"): "a LOG line, and bench/served_stops.py reads it",
    ("__init__.py", "use_skill answered"): "a LOG line, and bench/served_stops.py reads it",
}


def stale_mentions(source: str, rel: str) -> list[str]:
    """``file:line: text`` for every non-docstring string literal in ``source`` that names a
    tool by a pre-ADR-0190 spelling inside a sentence."""
    tree = ast.parse(source)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        text = node.value
        if id(node) in docstrings or text.strip() in PRE_0190_TOOL_NAMES:
            continue  # a docstring, or an identifier: an alias row, a role set, a table key
        if not (_OLD_NAME.search(text) or _OLD_TOOL_SHAPED.search(text)):
            continue
        if any(path == rel and part in text for path, part in ALLOWED):
            continue
        found.append(f"{rel}:{node.lineno}: {' '.join(text.split())[:140]}")
    return found


def test_no_sentence_names_a_tool_the_old_way() -> None:
    found: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        found += stale_mentions(path.read_text(encoding="utf-8"), path.relative_to(SRC).as_posix())
    assert not found, (
        "these sentences name a tool by its pre-ADR-0190 spelling; say it as the tool list "
        "does (or add the literal to ALLOWED with the reason):\n  " + "\n  ".join(found)
    )


def test_the_scan_can_fire_and_knows_an_identifier_from_a_sentence() -> None:
    # The positive control: a check whose passing state is "nothing found" proves nothing until
    # it has been seen to find something.
    planted = (
        'FIX = "create it with write_file first"\n'
        'HINT = f"run {n} now with use_skill"\n'
        "SHAPED = 'Run grep(pattern=\"x\") from the root'\n"
        'WINDOWS = "the `bash` tool runs under cmd.exe"\n'
    )
    assert len(stale_mentions(planted, "planted.py")) == 4
    clean = (
        '"""A docstring may say use_skill: it is documentation, not something a model reads."""\n'
        'ALIASES = {"use_skill": "Skill", "read_file": "Read"}\n'
        'ROLE = frozenset({"Write", "write_file"})\n'
        'SHELL = "run it as `bash script.sh`, or pipe through grep/head"\n'
        'FIX = "create it with Write first, or check the path with LS/Glob"\n'
    )
    assert stale_mentions(clean, "clean.py") == []


def test_every_allowance_is_still_needed() -> None:
    # An allowance nobody needs any more is a hole waiting for the next stale sentence.
    sources = {
        p.relative_to(SRC).as_posix(): p.read_text(encoding="utf-8") for p in SRC.rglob("*.py")
    }
    for (rel, part), why in ALLOWED.items():
        assert part in sources[rel], (
            f"ALLOWED entry no longer matches anything: {rel} / {part} ({why})"
        )

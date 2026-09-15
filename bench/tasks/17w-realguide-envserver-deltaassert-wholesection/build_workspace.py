#!/usr/bin/env python3
"""Build the 17w-realguide-envserver-deltaassert-wholesection workspace: thrust 27's decomposition
of the 27B's abridged-rule miss (thrust 26 stage 2: the ADR-0176 abridged block was read 11/12 by
the 35B and 7/12 by the 27B).

Cell 17's repo, task and verify, and cell 17's private guide with ONE change: the
`### Test-authoring gotchas` sub-section (7,988 chars fence-aware, eleven items) is reduced to its
rule sentence alone — the heading, then "Assert on before/after DELTAS for shared static counters,
never absolutes." — and nothing else of the guide moves. Under ADR-0176 the fold then admits the
rule WHOLE at the same priority position (heading tier: "gotchas") with no marker and no sibling
mandates; the budget the abridged block took goes back to the plain sections it displaced. The
derived guide is produced in memory from cell 17's guide asset at build time (never written beside
this script) and its md5 is pinned, so the measurement stays reproducible for anyone holding
cell 17's file.

Usage: build_workspace.py [<target>] [<guide_file>]
"""
import hashlib
import importlib.util
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BASE_DIR = _HERE.parent / "17-realguide-envserver-deltaassert"
_spec = importlib.util.spec_from_file_location("cell17_builder", _BASE_DIR / "build_workspace.py")
_cell17 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cell17)

DERIVED_MD5 = "4832dea75891"  # first 12 hex of the derived guide's md5 (2026-09-15 snapshot)
_ORIGINAL_ASSET = _cell17.GUIDE_ASSET  # cell 17's private guide file
_ORIGINAL_MD5 = _cell17.GUIDE_MD5
_HEADING = re.compile(r"^#{1,6}\s+\S")


def transform(text: str) -> str:
    """Cell 17's guide with the rule sub-section's body replaced by the rule sentence alone."""
    lines = text.split("\n")
    fence = False
    heads = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if not fence and _HEADING.match(line):
            heads.append(i)
    start = next(i for i in heads if lines[i].startswith(_cell17.RULE_HEADER))
    end = next((i for i in heads if i > start), len(lines))
    return "\n".join(lines[: start + 1] + ["", _cell17.RULE_TEXT, ""] + lines[end:])


class _DerivedAsset:
    """Stands in for cell 17's GUIDE_ASSET: the same file, transformed on read."""

    def __str__(self) -> str:
        return f"{_ORIGINAL_ASSET} (derived: rule sub-section reduced to its rule)"

    def exists(self) -> bool:
        return _ORIGINAL_ASSET.exists()

    def read_text(self, encoding: str = "utf-8") -> str:
        raw = _ORIGINAL_ASSET.read_text(encoding=encoding)
        md5 = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        assert md5 == _ORIGINAL_MD5, f"cell 17's guide md5 {md5} != pinned {_ORIGINAL_MD5}"
        return transform(raw)


_cell17.GUIDE_ASSET = _DerivedAsset()
_cell17.GUIDE_MD5 = DERIVED_MD5

if __name__ == "__main__":
    args = sys.argv[1:]
    _cell17.main(*args) if args else _cell17.main(_HERE / "workspace")
    print(f"17w: rule sub-section reduced to its rule (whole, no marker); md5 {DERIVED_MD5}")

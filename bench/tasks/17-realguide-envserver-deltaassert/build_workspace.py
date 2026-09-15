#!/usr/bin/env python3
"""Build the 17-realguide-envserver-deltaassert workspace: the OVERSIZED-SECTION residual (thrust 26).

ADR-0175's fleet census left one over-cap guide whose rules still fold out: the Ayoai-Environment-Server CLAUDE.md
(25,222 chars, 30 sections) keeps its test-authoring rules under `### Test-authoring gotchas (suite-order hazards, …)`,
a 7,087-char sub-section with no sub-headings (one 5.5K bullet list and four short paragraphs). The shipped fold keeps
sections whole or not at all, so that section — promoted by its heading — never fits behind the sections before it and
is the first name in the omitted list. This cell installs the guide verbatim and tests ONE rule stated only inside
that section: "Assert on before/after DELTAS for shared static counters, never absolutes." The workspace is a small
Python package whose `Metrics` class carries process-wide class-level counters and whose existing tests assert
ABSOLUTES (the imitation default). The task adds a test for `Metrics.record_hit()`; it passes iff the new test asserts
a before/after delta on `Metrics.hits` and never an absolute value.

The guide file is the author's private repository documentation and is NOT committed with this cell (`.gitignore`);
place a copy at `guide-CLAUDE.md` beside this script (on the author's boxes:
/opt/GitHub/Ayoai/Ayoai-Environment-Server/CLAUDE.md). The selfcheck pins its md5 so the measurement is reproducible
by anyone holding the file.

  MODE="guide"   — the guide is installed as CLAUDE.md.
  MODE="noguide" — the same repo with NO guide file: the DEFAULT-RATE control (17n-…-noguide).

Usage: build_workspace.py [<target>] [<guide_file>]
"""
import hashlib
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_TARGET = _HERE / "workspace"
GUIDE_ASSET = _HERE / "guide-CLAUDE.md"
GUIDE_MD5 = "c190292441f6"  # first 12 hex of the file's md5 (2026-09-15 snapshot)
DEFAULT_GUIDE = "CLAUDE.md"
MODE = "guide"  # "guide" | "noguide"
FOLD_CAP = 8_192  # == zakcode prompt.py MAX_CONTEXT_FILE_CHARS
RULE_HEADER = "### Test-authoring gotchas"
RULE_TEXT = "Assert on before/after DELTAS for shared static counters, never absolutes."
RULE_MARK = "DELTAS"  # must appear ONLY in the guide

COUNTERS = '''"""Shared process-wide counters (class-level state; every test in the run sees the same values)."""


class Metrics:
    """Counters shared across the whole process. Tests share these too."""

    hits = 0
    misses = 0
    evictions = 0

    @classmethod
    def record_hit(cls):
        cls.hits += 1

    @classmethod
    def record_miss(cls):
        cls.misses += 1

    @classmethod
    def record_eviction(cls):
        cls.evictions += 1
'''

EXISTING_TESTS = '''from metrics.counters import Metrics


def test_record_miss_increments():
    Metrics.record_miss()
    assert Metrics.misses == 1


def test_record_eviction_increments():
    Metrics.record_eviction()
    assert Metrics.evictions == 1
'''


def _mod(name: str, n: int) -> str:
    return f'"""metrics.{name} — application module."""\n\n\ndef {name}_value(x):\n    return x * {n}\n'


def _files(guide_file: str) -> dict:
    f = {}
    f["metrics/__init__.py"] = '"""metrics package."""\n'
    f["metrics/counters.py"] = COUNTERS
    for i, m in enumerate(["cache", "report", "config", "export", "window"], start=2):
        f[f"metrics/{m}.py"] = _mod(m, i)
    if MODE == "guide":
        if not GUIDE_ASSET.exists():
            raise SystemExit(f"guide asset missing: {GUIDE_ASSET} (see the module docstring for its source)")
        f[guide_file] = GUIDE_ASSET.read_text(encoding="utf-8")
    f["README.md"] = "# metrics\n\nA small metrics library.\n"
    f["docs/architecture.md"] = "# Architecture\n\nModules under metrics/.\n"
    f["pyproject.toml"] = '[project]\nname = "metrics"\nversion = "0.1.0"\n'
    f["Makefile"] = "test:\n\tpytest -q\n"
    f[".gitignore"] = "__pycache__/\n*.pyc\n"
    f["tests/__init__.py"] = ""
    f["tests/test_counters.py"] = EXISTING_TESTS
    f["tests/test_metrics.py"] = '"""Tests for metrics.counters (hits)."""\n\nfrom metrics.counters import Metrics  # noqa: F401\n'
    return f


def _selfcheck(root: Path, guide_file: str) -> None:
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    n = len(files)
    assert n <= 40, f"{n} files exceeds the small-repo budget (<=40)"
    carriers = sorted(
        str(p.relative_to(root)) for p in files if RULE_MARK in p.read_text(encoding="utf-8", errors="ignore")
    )
    if MODE == "guide":
        assert carriers == [guide_file], f"the rule must be in exactly {guide_file}, found in {carriers}"
        raw = (root / guide_file).read_text(encoding="utf-8")
        md5 = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        assert md5 == GUIDE_MD5, f"guide md5 {md5} != pinned {GUIDE_MD5}: not the measured snapshot"
        content = raw.strip()
        assert content.count(RULE_TEXT) == 1, "the rule sentence must appear exactly once"
        head = content.find(RULE_HEADER)
        off = content.find(RULE_TEXT)
        assert 0 < head < off, "the rule must sit inside its sub-section"
        nxt = content.find("\n#", head + 1)
        assert nxt == -1 or off < nxt, "the rule must sit before the next heading"
        section_len = (nxt if nxt != -1 else len(content)) - head
        note = (
            f"guide {len(content)} chars md5 {md5}; rule at char {off} ({off / len(content):.0%}) inside "
            f"'{RULE_HEADER}' ({section_len} chars, no sub-headings) — larger than the fold keeps whole"
        )
    else:
        assert carriers == [], f"noguide: the rule leaked into {carriers}"
        assert not (root / guide_file).exists(), "noguide: no guide file"
        note = "no guide file (default-rate control)"
    print(f"selfcheck OK: {n} files (<=40); mode={MODE}; {note}")


def main(target=_DEFAULT_TARGET, guide_file=DEFAULT_GUIDE) -> None:
    root = Path(target)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for rel, content in _files(guide_file).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    _selfcheck(root, guide_file)
    n = sum(1 for f in root.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    print(f"built {n} files under {root} (guide={guide_file if MODE == 'guide' else 'NONE'}, mode={MODE})")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args) if args else main()

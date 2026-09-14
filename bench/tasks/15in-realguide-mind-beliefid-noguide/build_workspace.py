#!/usr/bin/env python3
"""Build the 15in-realguide-mind-beliefid-noguide workspace: the 15i repo with NO guide — the default-rate control (thrust 18).

Thrust 17 (15-realguide-mind-timestamp) measured the ADR-0171 fold on a real guide where the rule sits in a section
the heading tier KEEPS ("### Naming Rules"). This cell is the residual on the same real guide: the rule sits in
"### ID Formats" — a plain topical heading, no emphasized mandate in its body — which the ADR-0171 fold of the
47,511-char guide OMITS (canonical probe: 17 of 39 sections kept; "ID Formats" is named in the note's omitted list).
Its parent heading is "## Universal Conventions": the author's outline says these ARE conventions, but the fold
scores every section on its own heading and body alone. The convention under test: belief-record ids are
`bel-NNN`. The task — implement `next_belief_id(existing_ids)` — has NO default route to "bel-": with no ids yet
assigned the prefix must come from knowledge, so the empty-list case is the discriminator (with ids present a
prefix-copying implementation passes on its own, which is why the verify runs the empty case too).

  MODE="guide"   — the guide is installed as CLAUDE.md (the measured arms: zakcode's 8K priority fold vs Claude
                   Code's native fold of the whole file).
  MODE="noguide" — the same repo with NO guide file: the DEFAULT-RATE control (15in-realguide-mind-beliefid-noguide).

_selfcheck asserts: the guide is >= 3x the cap with the rule header past the cap and the rule text absent from a
head cut; "bel-" appears ONLY in the guide; the stub raises; the verify regex accepts padded and unpadded
`bel-N` and rejects the plausible defaults (belief-1, b-1, bare 1, bel_1). <= 40 files.

Usage: build_workspace.py [<target>] [<guide_file>]
"""
import re
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_TARGET = _HERE / "workspace"
GUIDE_ASSET = _HERE.parent / "15-realguide-mind-timestamp" / "guide-CLAUDE.md"  # the thrust-17 snapshot, shared
DEFAULT_GUIDE = "CLAUDE.md"  # the filename BOTH loops fold
MODE = "noguide"  # "guide" | "noguide"
FOLD_CAP = 8_192  # == zakcode prompt.py MAX_CONTEXT_FILE_CHARS
RULE_HEADER = "### ID Formats"
RULE_MARK = "bel-"  # the convention's prefix; must appear ONLY in the guide
ID_RE = re.compile(r"^bel-\d+$")


def _mod(name: str, n: int) -> str:
    return f'"""app.{name} — application module."""\n\n\ndef {name}_value(x):\n    return x * {n}\n'


def _files(guide_file: str) -> dict:
    f = {}
    f["app/__init__.py"] = '"""records application package."""\n'
    f["app/ids.py"] = (
        '"""Record id helpers."""\n\n\n'
        "def next_belief_id(existing_ids):\n"
        '    """Return the id to assign to the next belief record, given the ids already assigned to belief\n'
        "    records (possibly empty). Ids are allocated sequentially and never reused.\n"
        '    """\n'
        '    raise NotImplementedError("allocate the next belief id")\n'
    )
    if MODE == "guide":
        f[guide_file] = GUIDE_ASSET.read_text(encoding="utf-8")
    for i, m in enumerate(["models", "store", "cli", "settings", "errors", "render", "queue", "tasks"], start=2):
        f[f"app/{m}.py"] = _mod(m, i)
    f["README.md"] = "# records-app\n\nA small record-keeping application.\n"
    f["docs/architecture.md"] = "# Architecture\n\nModules under app/.\n"
    f["pyproject.toml"] = '[project]\nname = "records-app"\nversion = "0.1.0"\n'
    f["Makefile"] = "test:\n\tpytest -q\n"
    f[".gitignore"] = "__pycache__/\n*.pyc\n"
    f["tests/__init__.py"] = ""
    f["tests/test_app.py"] = "def test_app_smoke():\n    assert True\n"
    f["tests/test_ids.py"] = (
        "# NB: intentionally no assertion on the id FORMAT here — that is a project convention.\n"
        "def test_ids_import():\n"
        "    import app.ids  # noqa: F401\n"
    )
    return f


def _selfcheck(root: Path, guide_file: str) -> None:
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    n = len(files)
    assert n <= 40, f"{n} files exceeds the small-repo budget (<=40)"
    stub = (root / "app/ids.py").read_text(encoding="utf-8")
    assert "def next_belief_id(" in stub and "NotImplementedError" in stub, "stub missing/short"
    carriers = sorted(str(p.relative_to(root)) for p in files if RULE_MARK in p.read_text(encoding="utf-8", errors="ignore"))
    if MODE == "guide":
        assert carriers == [guide_file], f"prefix must be in exactly {guide_file}, found in {carriers}"
        content = (root / guide_file).read_text(encoding="utf-8").strip()
        off = content.find(RULE_HEADER)
        assert len(content) >= 3 * FOLD_CAP, f"guide {len(content)} chars is not >= 3x the cap"
        assert off > FOLD_CAP, f"rule header at {off} is not past the {FOLD_CAP}-char cap"
        assert "`bel-NNN`" in content[off:off + 800], "rule not in its section"
        assert RULE_MARK not in content[:FOLD_CAP], "a head cut must drop the rule"
        assert content.count(RULE_MARK) == 1, "the prefix must be stated exactly once in the guide"
        note = f"guide {len(content)} chars; rule header at char {off} ({off / len(content):.0%}); a head cut at {FOLD_CAP} drops it"
    else:
        assert carriers == [], f"noguide: prefix leaked into {carriers}"
        assert not (root / guide_file).exists(), "noguide: no guide file"
        note = "no guide file (default-rate control)"
    # the verify regex: accepts the convention padded or not, rejects the plausible defaults
    for ok in ("bel-001", "bel-1", "bel-042", "bel-1043"):
        assert ID_RE.match(ok), f"{ok} must PASS"
    for bad in ("belief-1", "belief-001", "b-1", "1", "001", "bel_1", "BEL-1", "bel-", "belief_001"):
        assert not ID_RE.match(bad), f"{bad} must FAIL"
    print(f"selfcheck OK: {n} files (<=40); mode={MODE}; {note}; verify accepts bel-N padded or not, rejects belief-1 / b-1 / 1")


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

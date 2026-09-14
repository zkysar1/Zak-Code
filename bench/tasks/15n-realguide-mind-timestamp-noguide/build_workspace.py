#!/usr/bin/env python3
"""Build the 15n-realguide-mind-timestamp-noguide workspace: a REAL 47K guide, a real convention, a contrary default (campaign thrust 17).

Thrusts 11-16 built the fold's priority policy (ADR-0170 heading tier, ADR-0171 emphasized-body tier) on
synthetic guides. This cell asks the practical question with a REAL guide: `guide-CLAUDE.md` is a verbatim
snapshot (2026-09-14) of the Ayoai-Mind repository's CLAUDE.md — 47,511 chars, 39 sections, the working guide
of a live fleet — installed as the workspace's CLAUDE.md, the filename BOTH loops fold (Claude Code natively,
zakcode via discover_context). Its "### Naming Rules" section (starting at char 26,940, 57% in) states a real,
unprompted convention: timestamps are naive ISO-8601, `%Y-%m-%dT%H:%M:%S`, no zone suffix. A head cut at the
8,192-char cap drops it; the section-priority fold keeps it (heading tier: "Rules"). The task — implement
`journal_stamp()` returning "the current time as a timestamp string" — has a strong contrary default
(`datetime.now().isoformat()` carries microseconds; a space separator or a zone suffix also fail), so a pass
means the convention was applied.

  MODE="guide"   — the guide is installed as CLAUDE.md (the measured arms: zakcode's 8K priority fold of it vs
                   Claude Code's native fold of the whole file).
  MODE="noguide" — the same repo with NO guide file: the DEFAULT-RATE control (sibling dir 15n-realguide-mind-timestamp-noguide).

_selfcheck asserts: the guide is >= 3x the cap with the rule header past the cap and the rule text absent from a
head cut; the format string appears ONLY in the guide; the stub raises; the verify regex rejects the isoformat()
default and accepts the convention (and notes the one lucky default, isoformat(timespec="seconds")). <= 40 files.

Usage: build_workspace.py [<target>] [<guide_file>]
"""
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_TARGET = _HERE / "workspace"
GUIDE_ASSET = _HERE / "guide-CLAUDE.md"  # verbatim snapshot of Ayoai-Mind CLAUDE.md, 2026-09-14
DEFAULT_GUIDE = "CLAUDE.md"  # the filename BOTH loops fold
MODE = "noguide"  # "guide" | "noguide"
FOLD_CAP = 8_192  # == zakcode prompt.py MAX_CONTEXT_FILE_CHARS
RULE_HEADER = "### Naming Rules"
RULE_MARK = "%Y-%m-%dT%H:%M:%S"  # the convention's format string; must appear ONLY in the guide
RULE_PHRASE = "no zone suffix"
STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


def _mod(name: str, n: int) -> str:
    return f'"""app.{name} — application module."""\n\n\ndef {name}_value(x):\n    return x * {n}\n'


def _files(guide_file: str) -> dict:
    f = {}
    f["app/__init__.py"] = '"""journal application package."""\n'
    f["app/journal.py"] = (
        '"""Journal helpers."""\n\n\n'
        "def journal_stamp():\n"
        '    """Return the current time as a timestamp string for a journal entry."""\n'
        '    raise NotImplementedError("produce the timestamp string")\n'
    )
    if MODE == "guide":
        f[guide_file] = GUIDE_ASSET.read_text(encoding="utf-8")
    for i, m in enumerate(["models", "store", "cli", "settings", "errors", "render", "queue", "tasks"], start=2):
        f[f"app/{m}.py"] = _mod(m, i)
    f["README.md"] = "# journal-app\n\nA small journaling application.\n"
    f["docs/architecture.md"] = "# Architecture\n\nModules under app/.\n"
    f["pyproject.toml"] = '[project]\nname = "journal-app"\nversion = "0.1.0"\n'
    f["Makefile"] = "test:\n\tpytest -q\n"
    f[".gitignore"] = "__pycache__/\n*.pyc\n"
    f["tests/__init__.py"] = ""
    f["tests/test_app.py"] = "def test_app_smoke():\n    assert True\n"
    f["tests/test_journal.py"] = (
        "# NB: intentionally no assertion on the stamp FORMAT here — that is a project convention.\n"
        "def test_journal_import():\n"
        "    import app.journal  # noqa: F401\n"
    )
    return f


def _selfcheck(root: Path, guide_file: str) -> None:
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    n = len(files)
    assert n <= 40, f"{n} files exceeds the small-repo budget (<=40)"
    stub = (root / "app/journal.py").read_text(encoding="utf-8")
    assert "def journal_stamp(" in stub and "NotImplementedError" in stub, "stub missing/short"
    carriers = sorted(str(p.relative_to(root)) for p in files if RULE_MARK in p.read_text(encoding="utf-8", errors="ignore"))
    if MODE == "guide":
        assert carriers == [guide_file], f"format string must be in exactly {guide_file}, found in {carriers}"
        content = (root / guide_file).read_text(encoding="utf-8").strip()
        off = content.find(RULE_HEADER)
        assert len(content) >= 3 * FOLD_CAP, f"guide {len(content)} chars is not >= 3x the cap"
        assert off > FOLD_CAP, f"rule header at {off} is not past the {FOLD_CAP}-char cap"
        assert RULE_PHRASE in content[off:off + 800] and RULE_MARK in content[off:off + 800], "rule not in its section"
        assert RULE_PHRASE not in content[:FOLD_CAP] and RULE_MARK not in content[:FOLD_CAP], "a head cut must drop the rule"
        note = f"guide {len(content)} chars; rule header at char {off} ({off / len(content):.0%}); a head cut at {FOLD_CAP} drops it"
    else:
        assert carriers == [], f"noguide: format string leaked into {carriers}"
        assert not (root / guide_file).exists(), "noguide: no guide file"
        note = "no guide file (default-rate control)"
    # the verify regex: rejects the contrary defaults, accepts the convention, notes the lucky default
    now = datetime.now()
    assert not STAMP_RE.match(now.isoformat()), "isoformat() default must FAIL (microseconds)"
    assert not STAMP_RE.match(now.strftime("%Y-%m-%d %H:%M:%S")), "space separator must FAIL"
    assert not STAMP_RE.match(now.strftime("%Y-%m-%dT%H:%M:%SZ")), "zone suffix must FAIL"
    assert STAMP_RE.match(now.strftime(RULE_MARK)), "the convention must PASS"
    assert STAMP_RE.match(now.isoformat(timespec="seconds")), "isoformat(timespec=seconds) is the one lucky default"
    print(f"selfcheck OK: {n} files (<=40); mode={MODE}; {note}; verify rejects isoformat()/space/zone, accepts the convention")


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

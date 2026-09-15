#!/usr/bin/env python3
"""Build the 16p-realguide-envproc-aspirationkeys workspace: a SECOND real guide, a rule under `## Constraints` (thrust 25).

The fold's tiers were built on synthetic guides and one real one (the Mind's CLAUDE.md). A census of the author's
other guides (39 files) found the house style keeps its hard rules under `## Constraints` (28 of 39) — a heading
that names none of the fold's mandate words — and that of the five guides over the cap the shipped fold drops the
rules heading in four. This cell installs one of them verbatim: the Ayoai-Environment-Processor CLAUDE.md
(14,010 chars, 16 sections, md5 f3003a699a7c). Its `## Constraints` section states, for aspiration records,
`ayoTaskKey` for task references and `targetAyoKey` for object targets — names no model can guess, stated ONLY
there (the section starts past the cap; the shipped fold names it in the omitted list). The task — implement
`make_aspiration(task_key, target_key)` returning the record — passes iff both names are used.

The guide file is the author's private repository documentation and is NOT committed with this cell
(`.gitignore`); place a copy at `guide-CLAUDE.md` beside this script (on the author's boxes:
/opt/GitHub/Ayoai/Ayoai-Environment-Processor/CLAUDE.md). The selfcheck pins its md5 so the measurement is
reproducible by anyone holding the file.

  MODE="guide"   — the guide is installed as CLAUDE.md.
  MODE="noguide" — the same repo with NO guide file: the DEFAULT-RATE control (16pn-…-noguide).

Usage: build_workspace.py [<target>] [<guide_file>]
"""
import hashlib
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_TARGET = _HERE / "workspace"
GUIDE_ASSET = _HERE / "guide-CLAUDE.md"
GUIDE_MD5 = "f3003a699a7c"  # first 12 hex of the file's md5 (2026-09-15 snapshot)
DEFAULT_GUIDE = "CLAUDE.md"
MODE = "guide"  # "guide" | "noguide"
FOLD_CAP = 8_192  # == zakcode prompt.py MAX_CONTEXT_FILE_CHARS
RULE_HEADER = "## Constraints"
RULE_MARKS = ("ayoTaskKey", "targetAyoKey")  # the convention's names; must appear ONLY in the guide


def _mod(name: str, n: int) -> str:
    return f'"""app.{name} — application module."""\n\n\ndef {name}_value(x):\n    return x * {n}\n'


def _files(guide_file: str) -> dict:
    f = {}
    f["app/__init__.py"] = '"""environment-processor application package."""\n'
    f["app/aspirations.py"] = (
        '"""Aspiration records."""\n\n\n'
        "def make_aspiration(task_key, target_key):\n"
        '    """Build the aspiration record for a task reference and an object target; return a dict."""\n'
        '    raise NotImplementedError("build the aspiration record")\n'
    )
    if MODE == "guide":
        if not GUIDE_ASSET.exists():
            raise SystemExit(f"guide asset missing: {GUIDE_ASSET} (see the module docstring for its source)")
        f[guide_file] = GUIDE_ASSET.read_text(encoding="utf-8")
    for i, m in enumerate(["models", "store", "cli", "settings", "errors", "render", "queue", "tasks"], start=2):
        f[f"app/{m}.py"] = _mod(m, i)
    f["README.md"] = "# environment-processor\n\nA small processing application.\n"
    f["docs/architecture.md"] = "# Architecture\n\nModules under app/.\n"
    f["pyproject.toml"] = '[project]\nname = "environment-processor"\nversion = "0.1.0"\n'
    f["Makefile"] = "test:\n\tpytest -q\n"
    f[".gitignore"] = "__pycache__/\n*.pyc\n"
    f["tests/__init__.py"] = ""
    f["tests/test_app.py"] = "def test_app_smoke():\n    assert True\n"
    f["tests/test_aspirations.py"] = (
        "# NB: intentionally no assertion on the record's KEY NAMES here — that is a project convention.\n"
        "def test_aspirations_import():\n"
        "    import app.aspirations  # noqa: F401\n"
    )
    return f


def _selfcheck(root: Path, guide_file: str) -> None:
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    n = len(files)
    assert n <= 40, f"{n} files exceeds the small-repo budget (<=40)"
    stub = (root / "app/aspirations.py").read_text(encoding="utf-8")
    assert "def make_aspiration(" in stub and "NotImplementedError" in stub, "stub missing/short"
    carriers = sorted(
        str(p.relative_to(root))
        for p in files
        if any(m in p.read_text(encoding="utf-8", errors="ignore") for m in RULE_MARKS)
    )
    if MODE == "guide":
        assert carriers == [guide_file], f"the key names must be in exactly {guide_file}, found in {carriers}"
        raw = (root / guide_file).read_text(encoding="utf-8")
        md5 = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        assert md5 == GUIDE_MD5, f"guide md5 {md5} != pinned {GUIDE_MD5}: not the measured snapshot"
        content = raw.strip()
        off = content.find(RULE_HEADER)
        assert off > FOLD_CAP, f"rule header at {off} is not past the {FOLD_CAP}-char cap"
        section = content[off : off + 1200]
        assert all(m in section for m in RULE_MARKS), "rule not in its section"
        assert not any(m in content[:FOLD_CAP] for m in RULE_MARKS), "a head cut must drop the rule"
        note = f"guide {len(content)} chars md5 {md5}; rule header at char {off} ({off / len(content):.0%}); a head cut at {FOLD_CAP} drops it"
    else:
        assert carriers == [], f"noguide: key names leaked into {carriers}"
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

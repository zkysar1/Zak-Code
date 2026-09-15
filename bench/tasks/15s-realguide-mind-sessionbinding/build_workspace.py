#!/usr/bin/env python3
"""Build the 15s-realguide-mind-sessionbinding workspace: the REAL 47K guide, a convention with NO lexical signal (thrust 20).

Thrust 18 (15i) measured the ADR-0171 fold's residual on the real guide — a rule under a plain heading whose PARENT names
the mandate — and ADR-0172 (the heading tier reads the outline path) closed it. This cell is the next residual on the
same guide, and it carries NO lexical signal at all: "## Session Binding (Phase 2.6)" is a plain heading with no mandate
word in the heading, in any ancestor, or emphasized in its body, and it is the ONLY place the guide states where the
agent-session binding file lives (`agents/<name>/sessions/<SID>/binding.yaml`). It also RETIRES the legacy
`.active-agent-<SID>` form that two OTHER sections of the same guide still describe. Under ADR-0172 the section is
omitted (the tier-0 set alone fills the 8,192-char cap) and named in the note's omitted list. The task — implement
`binding_path(root, agent, sid)` — has no default route to the layout: the path must come from knowledge.

  MODE="guide"   — the guide is installed as CLAUDE.md (the measured arms: zakcode's 8K priority fold vs Claude
                   Code's native fold of the whole file).
  MODE="noguide" — the same repo with NO guide file: the DEFAULT-RATE control (15sn-realguide-mind-sessionbinding-noguide).

_selfcheck asserts: the guide is >= 3x the cap with the rule header past the cap and the file name absent from a head
cut; "binding.yaml" and ".active-agent" appear ONLY in the guide; the stub raises; the verify's layout check accepts the
documented path as a Path, a str, or a root-relative path and rejects the plausible wrong layouts. <= 40 files.

Usage: build_workspace.py [<target>] [<guide_file>]
"""
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_TARGET = _HERE / "workspace"
GUIDE_ASSET = _HERE.parent / "15-realguide-mind-timestamp" / "guide-CLAUDE.md"  # the thrust-17 snapshot, shared
DEFAULT_GUIDE = "CLAUDE.md"  # the filename BOTH loops fold
MODE = "guide"  # "guide" | "noguide"
FOLD_CAP = 8_192  # == zakcode prompt.py MAX_CONTEXT_FILE_CHARS
RULE_HEADER = "## Session Binding (Phase 2.6)"
RULE_MARK = "binding.yaml"  # the convention's file name; must appear ONLY in the guide
LEGACY_MARK = ".active-agent"  # the retired form the guide still describes elsewhere; also ONLY in the guide


def expected_path(root: Path, agent: str, sid: str) -> Path:
    """The layout the guide documents (mirrors verify.py)."""
    return root / "agents" / agent / "sessions" / sid / "binding.yaml"


def layout_matches(got, root: Path, agent: str, sid: str) -> bool:
    """Accept the documented layout as a Path, a str, or a root-relative path (mirrors verify.py)."""
    if not isinstance(got, (str, Path)):
        return False
    p = Path(got)
    if not p.is_absolute():
        p = root / p
    return p.resolve() == expected_path(root, agent, sid).resolve()


def _mod(name: str, n: int) -> str:
    return f'"""app.{name} — application module."""\n\n\ndef {name}_value(x):\n    return x * {n}\n'


def _files(guide_file: str) -> dict:
    f = {}
    f["app/__init__.py"] = '"""records application package."""\n'
    f["app/binding.py"] = (
        '"""Agent-session binding helpers."""\n\n\n'
        "def binding_path(root, agent, sid):\n"
        '    """Return the path of the file where this repository stores the agent-session binding for agent\n'
        "    `agent` and session id `sid`, under the repository root `root`.\n"
        '    """\n'
        '    raise NotImplementedError("locate the agent-session binding file")\n'
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
    f["tests/test_binding.py"] = (
        "# NB: intentionally no assertion on the binding LAYOUT here — that is a project convention.\n"
        "def test_binding_import():\n"
        "    import app.binding  # noqa: F401\n"
    )
    return f


def _selfcheck(root: Path, guide_file: str) -> None:
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    n = len(files)
    assert n <= 40, f"{n} files exceeds the small-repo budget (<=40)"
    stub = (root / "app/binding.py").read_text(encoding="utf-8")
    assert "def binding_path(" in stub and "NotImplementedError" in stub, "stub missing/short"
    texts = {str(p.relative_to(root)): p.read_text(encoding="utf-8", errors="ignore") for p in files}
    carriers = sorted(k for k, t in texts.items() if RULE_MARK in t or LEGACY_MARK in t)
    if MODE == "guide":
        assert carriers == [guide_file], f"the layout must be in exactly {guide_file}, found in {carriers}"
        content = texts[guide_file].strip()
        off = content.find(RULE_HEADER)
        assert len(content) >= 3 * FOLD_CAP, f"guide {len(content)} chars is not >= 3x the cap"
        assert off > FOLD_CAP, f"rule header at {off} is not past the {FOLD_CAP}-char cap"
        assert "`agents/<name>/sessions/<SID>/binding.yaml`" in content[off:off + 400], "rule not in its section"
        assert RULE_MARK not in content[:FOLD_CAP], "a head cut must drop the rule"
        legacy_n = content.count(LEGACY_MARK)
        assert legacy_n >= 3, f"the guide should still describe the legacy form elsewhere (found {legacy_n})"
        note = (
            f"guide {len(content)} chars; rule header at char {off} ({off / len(content):.0%}); "
            f"a head cut at {FOLD_CAP} drops it; legacy form mentioned {legacy_n}x"
        )
    else:
        assert carriers == [], f"noguide: layout leaked into {carriers}"
        assert not (root / guide_file).exists(), "noguide: no guide file"
        note = "no guide file (default-rate control)"
    # the verify's layout check: accepts the documented layout in three spellings, rejects the plausible wrong ones
    probe_root = root / "_probe_root"
    probe_root.mkdir()
    try:
        a, s = "alpha", "8913fdff-ddee-4554-b289-c3714f63c0de"
        good = [
            expected_path(probe_root, a, s),
            str(expected_path(probe_root, a, s)),
            Path("agents") / a / "sessions" / s / "binding.yaml",
        ]
        for g in good:
            assert layout_matches(g, probe_root, a, s), f"{g} must PASS"
        bad = [
            probe_root / f".active-agent-{s}",
            probe_root / "agents" / a / "session" / "binding.yaml",
            probe_root / "agents" / a / "sessions" / s / "session-summary.yaml",
            probe_root / "sessions" / s / "binding.yaml",
            probe_root / "agents" / a / f"{s}.yaml",
            probe_root / "agents" / a / "sessions" / s,
            probe_root / "agents" / "sessions" / s / "binding.yaml",
            None,
            42,
        ]
        for b in bad:
            assert not layout_matches(b, probe_root, a, s), f"{b} must FAIL"
    finally:
        shutil.rmtree(probe_root)
    print(
        f"selfcheck OK: {n} files (<=40); mode={MODE}; {note}; verify accepts the documented layout as Path/str/relative, "
        "rejects .active-agent-<SID> / session (singular) / sessions/<SID> without the file / session-summary.yaml"
    )


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

#!/usr/bin/env python3
"""Build the 13-agents-md-nameorder workspace: UNPROMPTED-CONVENTION ADHERENCE, robustness cell.

Campaign thrust 10. Thrust 9 (12-agents-md-convention) found the first clean beyond-parity result:
CC 0/12 vs zakcode 12/12 on an unprompted AGENTS.md convention (version format), because zakcode
FOLDS AGENTS.md into the system prompt while CC (headless `claude -p`) globs-but-does-not-read it.
This cell tests ROBUSTNESS: a DIFFERENT KIND of convention (name ORDERING, not string formatting) in
AGENTS.md, to confirm the effect is not specific to version-format. Same structure, same discriminator.

Convention (AGENTS.md only): full_name(first, last) returns "LAST, FIRST" (last name, comma+space,
first name) — overriding the model's strong "First Last" default. Prompt never mentions it; solvable-
but-wrong without consulting AGENTS.md; trivial to apply once known → a fail is an ADHERENCE miss.

CONTROLLED PROPERTIES (asserted by _selfcheck):
  * full_name stub present, raises NotImplementedError.
  * The convention (the "LAST, FIRST" rule) appears in the guide file ONLY.
  * verify triples' expected outputs are NOT pre-written (AGENTS.md uses a DIFFERENT example name).
  * small realistic repo (<=40 files), guide at root (visible — this is adherence, not discovery).

Usage: build_workspace.py [<target>] [<guide_file>]   # guide_file default AGENTS.md
"""
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_TARGET = _HERE / "workspace"
DEFAULT_GUIDE = "AGENTS.md"
RULE_MARK = "LAST, FIRST"  # convention-unique phrase; must appear in the guide file ONLY


def _mod(pkg: str, name: str, n: int) -> str:
    return (
        f'"""{pkg}.{name} — application module."""\n\n'
        f"def {name}_value(x):\n"
        f"    return x * {n}\n"
    )


def _guide_body() -> str:
    return (
        "# Project Guide\n\n"
        "Contributor and agent guidance for this repository.\n\n"
        "## Setup\n\n"
        "Install with `pip install -e .`. Run tests with `pytest`.\n\n"
        "## Naming (MANDATORY)\n\n"
        f"Whenever this project renders a person's name as a single string, it MUST be formatted as\n"
        f"`{RULE_MARK}` — the LAST name, then a comma and a single space, then the FIRST name. It is\n"
        "NEVER rendered as `First Last`. For example, first=\"Grace\", last=\"Hopper\" renders as\n"
        "`Hopper, Grace`. Every function that produces a full-name string MUST follow this format.\n\n"
        "## Testing\n\n"
        "Tests live under `tests/`. All new code needs a test.\n"
    )


def _files(guide_file: str) -> dict:
    f = {}
    f["app/__init__.py"] = '"""demo application package."""\n'
    f["app/people.py"] = (
        '"""People helpers."""\n\n\n'
        "def full_name(first, last):\n"
        '    """Return a person\'s full name as a single string, from their first and last names."""\n'
        '    raise NotImplementedError("render the full name from the components")\n'
    )
    f[guide_file] = _guide_body()
    app_mods = [
        "models", "views", "handlers", "router", "settings", "auth", "cache", "session",
        "errors", "middleware", "cli", "tasks",
    ]
    for i, m in enumerate(app_mods, start=2):
        f[f"app/{m}.py"] = _mod("app", m, i)
    f["README.md"] = "# demo-app\n\nA small demo application. See the project guide for conventions.\n"
    f["docs/architecture.md"] = "# Architecture\n\nLayers: app modules under app/.\n"
    f["docs/api.md"] = "# API\n\nEndpoints are defined in app/router.py.\n"
    f["docs/faq.md"] = "# FAQ\n\nCommon questions and answers.\n"
    f["pyproject.toml"] = '[project]\nname = "demo-app"\nversion = "0.1.0"\n'
    f["setup.cfg"] = "[metadata]\nname = demo-app\n"
    f["Makefile"] = "test:\n\tpytest -q\n"
    f[".gitignore"] = "__pycache__/\n*.pyc\n"
    f["tests/__init__.py"] = ""
    f["tests/test_app.py"] = "def test_app_smoke():\n    assert True\n"
    f["tests/test_people.py"] = (
        "# NB: intentionally no assertion on the name FORMAT here — that lives in the project guide.\n"
        "def test_people_import():\n"
        "    import app.people  # noqa: F401\n"
    )
    return f


def _selfcheck(root: Path, guide_file: str) -> None:
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    n = len(files)
    assert n <= 40, f"{n} files exceeds the small-repo budget (<=40)"
    stub = (root / "app/people.py").read_text(encoding="utf-8")
    assert "def full_name(" in stub and "NotImplementedError" in stub, "stub missing/short"
    carriers = [p for p in files if RULE_MARK in p.read_text(encoding="utf-8", errors="ignore")]
    rel = sorted(str(p.relative_to(root)) for p in carriers)
    assert rel == [guide_file], f"rule {RULE_MARK!r} must be in exactly {guide_file}, found in {rel}"
    guide_text = (root / guide_file).read_text(encoding="utf-8")
    assert "NEVER rendered as `First Last`" in guide_text, "guide must state the override"
    # verify answers must not be pre-written (AGENTS.md example is Hopper, Grace; verify uses others)
    for ans in ("Lovelace, Ada", "Turing, Alan", "Curie, Marie"):
        leaks = [str(p.relative_to(root)) for p in files if ans in p.read_text(encoding="utf-8", errors="ignore")]
        assert not leaks, f"verify answer {ans!r} leaked into {leaks}"
    print(f"selfcheck OK: {n} files (<=40); rule in exactly {guide_file}; verify answers not pre-written")


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
    print(f"built {n} files under {root} (guide={guide_file})")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args) if args else main()

#!/usr/bin/env python3
"""Deterministically build the 12-agents-md-convention workspace: UNPROMPTED-CONVENTION ADHERENCE.

Campaign thrust 9. Thrust 8 falsified the discovery-determinism edge: CC's glob('*') returns the
complete file inventory ≈ zakcode's survey listing, so both DISCOVER a REQUIRED value 12/12. The
remaining candidate edge (06 family) is zakcode's deterministic zero-cost CONTENT-FOLD of guide
files (AGENTS.md / CLAUDE.md / ZAK.md) into the system prompt, which should raise ADHERENCE to an
UNPROMPTED convention — one the task never mentions and that the agent has no functional reason to
hunt for.

This task isolates that mechanism cleanly:
  * The task (implement version_string) is SOLVABLE with a plausible, self-consistent answer WITHOUT
    reading any guide file — so an agent is tempted to just implement it and move on.
  * A convention in AGENTS.md OVERRIDES the model's strong default: version strings are `v<MAJOR>.<MINOR>`
    and OMIT the patch component. The model's default for version_string(major,minor,patch) is the full
    semver "MAJOR.MINOR.PATCH" — so an agent that does NOT consult AGENTS.md emits a CONFIDENT WRONG
    answer (e.g. "2.5.3"), while an agent with the convention in context emits "v2.5".
  * Application is TRIVIAL (a one-line format) — no serializer, no stdlib-import temptation — so a fail
    is an ADHERENCE miss (did the agent consult the unprompted convention?), not a coding-skill miss.

DISCRIMINATOR: zakcode FOLDS AGENTS.md CONTENT into the system prompt every run (deterministic,
zero-cost) → the convention is unconditionally in context → adheres. Claude Code (whose glob SEES
AGENTS.md, per thrust 8) must proactively READ + APPLY it for a task that does not require it — the
open empirical question is whether CC does so unprompted. Prediction fork:
  * CC adheres (proactively reads guide files) -> PARITY -> the fold is not an edge because CC is
    diligent about AGENTS.md.
  * CC does NOT adhere (implements the trivial task without consulting AGENTS.md) -> ZAK>CC -> the
    content-fold is the real beyond-parity lever for unprompted conventions.
Either outcome is a sharp, honest result and (with the CLAUDE.md control cell, run next if a gap
appears) pins the mechanism.

CONTROLLED PROPERTIES (asserted by _selfcheck at build time):
  * version_string stub present and raises NotImplementedError.
  * The convention (the `v<MAJOR>.<MINOR>` rule / patch-omission) appears in AGENTS.md ONLY — the sole
    channel to adherence.
  * The verify triples' expected outputs are NOT pre-written anywhere the agent could copy (AGENTS.md
    uses a DIFFERENT example triple, 2.5.3 -> v2.5, forcing a GENERAL implementation, not a copy).
  * Small realistic repo (<=40 files) with AGENTS.md at the root where guide files live — this is NOT
    a discovery test; AGENTS.md is plainly visible. The question is CONSULTATION, not location.

Output is byte-identical for a given target on any box (no timestamps / randomness).

Usage:
    build_workspace.py                 # build ./workspace
    build_workspace.py <target>
    build_workspace.py <target> <guide_file>   # guide_file in {AGENTS.md, CLAUDE.md, ZAK.md}
                                                # (control-cell variants; default AGENTS.md)
"""
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_TARGET = _HERE / "workspace"

# The unprompted convention, carried in a guide file (default AGENTS.md; CLAUDE.md / ZAK.md are the
# control cells). The rule OVERRIDES the model's full-semver default on TWO axes (v-prefix + drop
# patch), so an accidental adherent output without reading the guide is near-impossible.
DEFAULT_GUIDE = "CLAUDE.md"
# convention-unique phrase (must appear in the guide file ONLY)
RULE_MARK = "v<MAJOR>.<MINOR>"


def _mod(pkg: str, name: str, n: int) -> str:
    return (
        f'"""{pkg}.{name} — application module."""\n\n'
        f"def {name}_value(x):\n"
        f'    """Return a derived value."""\n'
        f"    return x * {n}\n"
    )


def _guide_body() -> str:
    return (
        "# Project Guide\n\n"
        "Contributor and agent guidance for this repository.\n\n"
        "## Setup\n\n"
        "Install with `pip install -e .`. Run the test suite with `pytest`.\n\n"
        "## Code style\n\n"
        "Follow PEP 8. Keep functions small and pure; prefer explicit returns.\n\n"
        "## Versioning (MANDATORY)\n\n"
        f"Version strings in this project are ALWAYS formatted as `{RULE_MARK}` — the literal\n"
        "letter `v`, the major number, a dot, then the minor number. They NEVER include the patch\n"
        "component. For example, for major=2, minor=5, patch=3 the version string is `v2.5` (the\n"
        "patch `3` is intentionally omitted). Every function in this project that produces a version\n"
        "string MUST follow this exact format.\n\n"
        "## Testing\n\n"
        "Tests live under `tests/`. All new code needs a test.\n"
    )


def _files(guide_file: str) -> dict:
    f = {}
    # --- the real task target: a stub the agent implements ---
    f["app/__init__.py"] = '"""demo application package."""\n'
    f["app/versioning.py"] = (
        '"""Version helpers."""\n\n\n'
        "def version_string(major, minor, patch):\n"
        '    """Return a version string built from the given major, minor and patch numbers."""\n'
        '    raise NotImplementedError("build the version string from the components")\n'
    )
    # --- the guide file carrying the unprompted convention (AGENTS.md by default) ---
    f[guide_file] = _guide_body()
    # --- clutter: application modules (small, realistic repo) ---
    app_mods = [
        "models", "views", "handlers", "router", "settings", "auth", "cache", "session",
        "errors", "middleware", "cli", "tasks",
    ]
    for i, m in enumerate(app_mods, start=2):
        f[f"app/{m}.py"] = _mod("app", m, i)
    # --- clutter: docs / config / tests (NONE carry the versioning rule) ---
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
    f["tests/test_versioning.py"] = (
        "# NB: intentionally no assertion on the version FORMAT here — that lives in the\n"
        "# project guide, not in the tests.\n"
        "def test_versioning_import():\n"
        "    import app.versioning  # noqa: F401\n"
    )
    return f


def _selfcheck(root: Path, guide_file: str) -> None:
    """Fail loudly if a controlled property is violated (so a broken build never ships)."""
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    n = len(files)
    assert n <= 40, f"{n} files exceeds the small-repo budget (<=40)"
    # stub present + raises NotImplementedError
    stub = (root / "app/versioning.py").read_text(encoding="utf-8")
    assert "def version_string(" in stub and "NotImplementedError" in stub, "stub missing/short"
    # the convention rule appears in the guide file ONLY
    carriers = [p for p in files if RULE_MARK in p.read_text(encoding="utf-8", errors="ignore")]
    rel = sorted(str(p.relative_to(root)) for p in carriers)
    assert rel == [guide_file], f"rule {RULE_MARK!r} must be in exactly {guide_file}, found in {rel}"
    # the patch-omission instruction is present (so once read, adherence is unambiguous)
    guide_text = (root / guide_file).read_text(encoding="utf-8")
    assert "NEVER include the patch" in guide_text, "guide must state the patch-omission rule"
    # the verify triples' answers are NOT pre-written anywhere (agent must GENERALIZE, not copy):
    # AGENTS.md uses example (2,5,3)->v2.5; verify uses OTHER triples. Assert those answers absent.
    for ans in ("v1.0", "v10.11", "v3.4", "v0.9"):
        leaks = [str(p.relative_to(root)) for p in files if ans in p.read_text(encoding="utf-8", errors="ignore")]
        assert not leaks, f"verify answer {ans!r} leaked into {leaks} — agent could copy it"
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

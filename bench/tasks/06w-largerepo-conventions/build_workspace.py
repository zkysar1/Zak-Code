#!/usr/bin/env python3
"""Deterministically (re)build 06w's workspace/ from the base-06 workspace.

06w exists to test whether zakcode's deterministic-context-assembly advantage over Claude Code
COMPOUNDS as a repo grows (ADR: campaign thrust 5). It differs from base-06 in EXACTLY TWO
controlled ways:
  (1) CONTRIBUTING.md (root) -> docs/CONVENTIONS.md  [byte-identical rule text; the 06v relocation]
  (2) ~40 non-doc decoy files added (src/, lib/, data/, scripts/, extra tests, root config)
The real task files (plugins/, tests/test_plugins.py) are COPIED VERBATIM from base-06 so verify.py's
correctness bar is byte-identical. NO competing docs and NO extra plugins are added, so the
sibling-renderer signal and the read-selection stay clean -- only DISCOVERY COST changes.

Two measured properties this construction guarantees (see the preregistration in bench/results):
  * zakcode's workspace survey (SURVEY_MAX_ENTRIES=150, depth<=3, sorted walk) LISTS
    docs/CONVENTIONS.md deterministically -- 46 files is far under the 150 cap, so the convention
    file is surfaced regardless of clutter (assembly is clutter-immune).
  * Claude Code has no deterministic equivalent: it must DISCOVER the file agentically among the
    46-file tree, and base-06 showed that discovery succeeds only 2/6 even at the conventional root.

Output is byte-identical for a given (target, base) on any box (no timestamps/randomness), so a
run tree on any bench box matches any other box's survey pre-check exactly.

Usage:
    build_workspace.py                 # build ./workspace from ../06-plugin-conventions/workspace
    build_workspace.py <target> <base>
"""
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_TARGET = _HERE / "workspace"
_DEFAULT_BASE = _HERE.parent / "06-plugin-conventions" / "workspace"


def _mod(pkg, name, n):
    return (f'"""{pkg}.{name} — application module (decoy)."""\n\n'
            f'def {name}_entry(x):\n'
            f'    """Return a derived value."""\n'
            f'    return x * {n}\n\n\n'
            f'class {name.capitalize()}Service:\n'
            f'    def handle(self, payload):\n'
            f'        return {{"module": "{name}", "n": {n}}}\n')


def _decoys():
    files = {}
    files["README.md"] = "# demo-app\n\nA small demo application. See `docs/` for details.\n"
    files["pyproject.toml"] = '[project]\nname = "demo-app"\nversion = "0.1.0"\n'
    files["setup.cfg"] = "[metadata]\nname = demo-app\n"
    files["tox.ini"] = "[tox]\nenvlist = py311\n"
    files["Makefile"] = "test:\n\tpytest -q\n"
    files[".gitignore"] = "__pycache__/\n*.pyc\n"
    app_mods = ["models", "views", "handlers", "router", "settings", "auth",
                "cache", "session", "logging_conf", "errors", "middleware", "cli"]
    for i, m in enumerate(app_mods, start=2):
        files[f"src/app/{m}.py"] = _mod("app", m, i)
    files["src/app/__init__.py"] = '"""demo app package."""\n'
    svc_mods = ["users", "orders", "billing", "notifications", "search", "reports"]
    for i, m in enumerate(svc_mods, start=2):
        files[f"src/app/services/{m}.py"] = _mod("services", m, i)
    files["src/app/services/__init__.py"] = '"""services package."""\n'
    lib_mods = ["dates", "strings", "numbers", "collections_ext", "io_utils"]
    for i, m in enumerate(lib_mods, start=2):
        files[f"lib/{m}.py"] = _mod("lib", m, i)
    for i in range(1, 6):
        files[f"data/fixture_{i}.json"] = '{"id": %d, "ok": true}\n' % i
    files["data/sample.csv"] = "a,b\n1,2\n3,4\n"
    files["scripts/seed.py"] = '"""seed script."""\nif __name__ == "__main__":\n    print("seed")\n'
    files["scripts/migrate.py"] = '"""migrate script."""\nif __name__ == "__main__":\n    print("migrate")\n'
    files["tests/test_app.py"] = "def test_app_smoke():\n    assert True\n"
    files["tests/test_services.py"] = "def test_services_smoke():\n    assert True\n"
    return files


def main(target=_DEFAULT_TARGET, base=_DEFAULT_BASE):
    root, base = Path(target), Path(base)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for sub in ("plugins", "tests"):  # real task files, verbatim
        shutil.copytree(base / sub, root / sub,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (root / "docs").mkdir()  # relocate rules: CONTRIBUTING.md -> docs/CONVENTIONS.md
    shutil.copyfile(base / "CONTRIBUTING.md", root / "docs" / "CONVENTIONS.md")
    for rel, content in _decoys().items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    nfiles = sum(1 for f in root.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    print(f"built {nfiles} files under {root}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args) if args else main()

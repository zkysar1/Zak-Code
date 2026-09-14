#!/usr/bin/env python3
"""Deterministically build the 11-discovery-token workspace: a PURE-DISCOVERY task.

Campaign thrust 8. The 06-family measures convention adherence, where the correctness signal is
SWAMPED at N=6 by shared model residuals (stdlib-only adherence + YAML-emitter bugs). This task
removes those residuals so that pass ⟺ discovery: the ONLY hard part is FINDING an un-guessable
magic token that lives in exactly ONE non-obvious file among ~110 clutter files; compliance is
trivial once found (return the string). There is no serializer to get wrong and no import
temptation, so a fail is a discovery miss and a pass is a discovery hit -- the clean measurement
of the discovery-DETERMINISM gap (zakcode's deterministic survey LISTING surfaces the file's name
every run; Claude Code must discover it by agentic exploration).

CONTROLLED PROPERTIES (asserted by _selfcheck at build time):
  * The token string appears in EXACTLY ONE file (ops/release-registry.md) -- found ⟺ that file read.
  * Total files <= SURVEY_MAX_ENTRIES (150) so zakcode's survey LISTS every file (no truncation),
    and the token file sits at depth 2 (<= SURVEY_MAX_DEPTH=3) so it is surveyed.
  * The token is un-guessable (a random-looking tag) and appears in NO other file, so the task
    cannot be passed without reading the registry file.
  * The word "release" appears in several decoy files (CHANGELOG, a deploy script, a process doc)
    as grep-noise, but the TOKEN VALUE is only in the registry -- grepping "release" is not enough.

Output is byte-identical for a given target on any box (no timestamps / randomness).

Usage:
    build_workspace.py                 # build ./workspace
    build_workspace.py <target>
"""
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_TARGET = _HERE / "workspace"

# The un-guessable magic token. Lives in ops/release-registry.md ONLY.
TOKEN = "ZK-9F3A-2231-RELEASE"
TOKEN_FILE = "ops/release-registry.md"


def _mod(pkg: str, name: str, n: int) -> str:
    return (
        f'"""{pkg}.{name} — application module."""\n\n'
        f"def {name}_value(x):\n"
        f'    """Return a derived value."""\n'
        f"    return x * {n}\n\n\n"
        f"class {name.capitalize()}Handler:\n"
        f"    def handle(self, payload):\n"
        f'        return {{"module": "{name}", "n": {n}}}\n'
    )


def _files() -> dict:
    f = {}
    # --- the real task target: a stub the agent implements ---
    f["app/__init__.py"] = '"""demo application package."""\n'
    f["app/meta.py"] = (
        '"""Project metadata accessors."""\n\n\n'
        "def release_tag() -> str:\n"
        '    """Return the project\'s official release tag.\n\n'
        "    The exact value is defined in the project's documentation/configuration —\n"
        "    find it there and return it verbatim.\n"
        '    """\n'
        '    raise NotImplementedError("return the official release tag defined in the project docs")\n'
    )
    # --- the ONE file carrying the token (non-obvious location: ops/, not docs/) ---
    f[TOKEN_FILE] = (
        "# Release Registry\n\n"
        "This registry is the single source of truth for build and release identifiers.\n\n"
        "## Official release tag\n\n"
        f"The current official release tag for this project is `{TOKEN}`.\n"
        "All published artifacts MUST carry this exact tag. Do not invent a tag; use this value.\n\n"
        "## Rotation\n\n"
        "The tag is rotated only by the release engineering team via this file.\n"
    )
    # --- clutter: application modules (src) ---
    app_mods = [
        "models", "views", "handlers", "router", "settings", "auth", "cache", "session",
        "logging_conf", "errors", "middleware", "cli", "tasks", "events", "metrics",
        "serializers", "validators", "permissions", "pagination", "throttling",
    ]
    for i, m in enumerate(app_mods, start=2):
        f[f"app/{m}.py"] = _mod("app", m, i)
    svc_mods = ["users", "orders", "billing", "notifications", "search", "reports", "inventory", "shipping"]
    for i, m in enumerate(svc_mods, start=2):
        f[f"app/services/{m}.py"] = _mod("services", m, i)
    f["app/services/__init__.py"] = '"""services package."""\n'
    lib_mods = ["dates", "strings", "numbers", "collections_ext", "io_utils", "text", "math_ext", "trees"]
    for i, m in enumerate(lib_mods, start=2):
        f[f"lib/{m}.py"] = _mod("lib", m, i)
    f["lib/__init__.py"] = '"""lib package."""\n'
    # --- clutter: docs (several plausible docs, NONE carry the token) ---
    f["README.md"] = "# demo-app\n\nA small demo application.\n"
    f["docs/architecture.md"] = "# Architecture\n\nLayers: app, services, lib. See modules for detail.\n"
    f["docs/api.md"] = "# API\n\nEndpoints are defined in app/router.py.\n"
    f["docs/testing.md"] = "# Testing\n\nRun `pytest`. Tests live under tests/.\n"
    f["docs/style.md"] = "# Style\n\nFollow PEP 8. Keep functions small.\n"
    f["docs/security.md"] = "# Security\n\nReport issues privately. Rotate secrets quarterly.\n"
    f["docs/onboarding.md"] = "# Onboarding\n\nClone, install, run tests. Ask in #dev.\n"
    f["docs/process.md"] = "# Process\n\nWe follow a two-week release cadence.\n"  # decoy 'release'
    f["docs/faq.md"] = "# FAQ\n\nCommon questions and answers.\n"
    f["docs/glossary.md"] = "# Glossary\n\nDomain terms and definitions.\n"
    # --- clutter: ops / config / scripts (token file lives in ops/) ---
    f["ops/deploy-notes.md"] = "# Deploy notes\n\nBlue/green deploys via the pipeline.\n"
    f["ops/runbook.md"] = "# Runbook\n\nOn-call procedures and escalation.\n"
    f["ops/monitoring.md"] = "# Monitoring\n\nDashboards and alert thresholds.\n"
    f["pyproject.toml"] = '[project]\nname = "demo-app"\nversion = "0.1.0"\n'
    f["setup.cfg"] = "[metadata]\nname = demo-app\n"
    f["tox.ini"] = "[tox]\nenvlist = py311\n"
    f["Makefile"] = "test:\n\tpytest -q\n"
    f["CHANGELOG.md"] = "# Changelog\n\n## Release history\n\n- 0.1.0 initial\n"  # decoy 'release'
    f[".gitignore"] = "__pycache__/\n*.pyc\n"
    f["scripts/seed.py"] = '"""seed script."""\nif __name__ == "__main__":\n    print("seed")\n'
    f["scripts/migrate.py"] = '"""migrate script."""\nif __name__ == "__main__":\n    print("migrate")\n'
    f["scripts/deploy.py"] = '"""deploy script."""\n# release step: tag and publish artifacts\nif __name__ == "__main__":\n    print("deploy")\n'  # decoy 'release'
    for i in range(1, 9):
        f[f"data/fixture_{i}.json"] = '{"id": %d, "ok": true}\n' % i
    f["data/sample.csv"] = "a,b\n1,2\n3,4\n"
    # --- clutter: tests ---
    f["tests/__init__.py"] = ""
    f["tests/test_app.py"] = "def test_app_smoke():\n    assert True\n"
    f["tests/test_services.py"] = "def test_services_smoke():\n    assert True\n"
    f["tests/test_lib.py"] = "def test_lib_smoke():\n    assert True\n"
    return f


def _selfcheck(root: Path) -> None:
    """Fail loudly if a controlled property is violated (so a broken build never ships)."""
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    n = len(files)
    assert n <= 150, f"{n} files exceeds SURVEY_MAX_ENTRIES=150 — zakcode's survey would truncate"
    # token appears in exactly one file
    carriers = [p for p in files if TOKEN in p.read_text(encoding="utf-8", errors="ignore")]
    rel = sorted(str(p.relative_to(root)) for p in carriers)
    assert rel == [TOKEN_FILE], f"token must be in exactly {TOKEN_FILE}, found in {rel}"
    # token file depth <= 3
    depth = len(Path(TOKEN_FILE).parts)
    assert depth <= 3, f"token file depth {depth} > SURVEY_MAX_DEPTH=3"
    print(f"selfcheck OK: {n} files (<=150), token in exactly {TOKEN_FILE} (depth {depth})")


def main(target=_DEFAULT_TARGET) -> None:
    root = Path(target)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for rel, content in _files().items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    _selfcheck(root)
    n = sum(1 for f in root.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    print(f"built {n} files under {root}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args) if args else main()

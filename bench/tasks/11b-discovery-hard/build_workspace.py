#!/usr/bin/env python3
"""Deterministically build the 11b-discovery-hard workspace: HARD pure-discovery.

Campaign thrust 8, iteration 2. The 11-discovery-token pilot saturated (CC unaided 6/6,
zakcode 6/6) because the token was discoverable by TWO trivial channels, both measured in the
pilot's CC run JSON:
  (A) glob('*') surfaced the OBVIOUS filename `ops/release-registry.md` -> read directly (5/6 runs).
  (B) grep 'release.tag' matched the registry prose line "...official release tag ... is `TOKEN`",
      so grep RETURNED THE VALUE on the matching line -- no read needed (run 5, 0 registry reads, PASS).

11b removes BOTH channels on a single clean axis (discovery difficulty), holding everything else
(trivial compliance, no serializer, no import temptation) constant vs 11:
  * NON-OBVIOUS carrier filename at depth 3: ops/internal/appendix-c.md (no release/registry/tag in
    the name), so glob/listing gives no signpost -- channel A dies.
  * The keyword "release tag" appears in the file but NOT on the value's line: the value sits in a
    table under the oblique label `stamp`, and a separate sentence states the stamp IS the release
    tag. The token contains no "release"/"tag" substring, so a keyword grep ("release", "tag",
    "release tag") finds the FILE but returns the SENTENCE, never the value -- channel B dies; the
    agent must READ the file and map stamp->release tag to extract the value.
  * More clutter (~115 files, still <= SURVEY_MAX_ENTRIES=150 so zakcode's survey lists every file;
    carrier at depth 3 <= SURVEY_MAX_DEPTH=3 so it IS surveyed).

So pass ⟺ the agent discovered ops/internal/appendix-c.md AND read it (there is no grep shortcut to
the value and no filename signpost). This isolates the discovery-DETERMINISM hypothesis: zakcode's
survey deterministically LISTS appendix-c.md every run; Claude Code must reach it by agentic search
without the two shortcuts the pilot exposed. Prediction: CC's rate drops below saturation while
zakcode holds (deterministic survey coverage) -> the first clean pure-discovery gap; OR both hold
(discovery is easy whenever the file is grep-findable at all) -> the edge is bounded to surfacing,
not discovery. Either outcome is a sharp, honest result.

CONTROLLED PROPERTIES (asserted by _selfcheck at build time):
  * The token appears in EXACTLY ONE file (ops/internal/appendix-c.md).
  * The token appears on NO line containing (case-insensitive) "release" or "tag" -- so a keyword
    grep for the prompt's own terms cannot return the value (channel B defeated, verified).
  * The carrier filename contains none of release/registry/tag/stamp/version (channel A defeated).
  * Total files <= 150; carrier depth == 3 (<= SURVEY_MAX_DEPTH).
  * Decoy "release" noise in CHANGELOG / deploy.py / process.md (grep 'release' returns noise).

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

# The un-guessable magic token. Contains NO "release"/"tag" substring by design (so a keyword grep
# for the prompt's terms cannot land on the value's line). Lives in ops/internal/appendix-c.md ONLY.
TOKEN = "ZK-7Q2M-8814-K3D9"
TOKEN_FILE = "ops/internal/appendix-c.md"


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
    # --- the ONE file carrying the token: non-obvious name, keyword decoupled from value ---
    # "release tag" appears ONLY in the prose sentence, never on the value line; the value sits in a
    # table under the label `stamp`. The agent must READ the file to map stamp->release tag.
    f[TOKEN_FILE] = (
        "# Appendix C — Build Identifiers\n\n"
        "This appendix records the canonical build identifiers used by the release pipeline.\n\n"
        "| parameter | value |\n"
        "|-----------|-------|\n"
        "| channel   | stable |\n"
        f"| stamp     | {TOKEN} |\n"
        "| retention | 90 |\n\n"
        "The `stamp` value in the table above is this project's canonical release tag. All published\n"
        "artifacts MUST carry it verbatim. Do not invent a value; use the stamp above.\n"
    )
    # --- clutter: application modules (src) ---
    app_mods = [
        "models", "views", "handlers", "router", "settings", "auth", "cache", "session",
        "logging_conf", "errors", "middleware", "cli", "tasks", "events", "metrics",
        "serializers", "validators", "permissions", "pagination", "throttling",
        "backends", "signals", "forms", "widgets", "context", "hooks", "adapters", "registry_mod",
    ]
    for i, m in enumerate(app_mods, start=2):
        f[f"app/{m}.py"] = _mod("app", m, i)
    svc_mods = [
        "users", "orders", "billing", "notifications", "search", "reports", "inventory", "shipping",
        "accounts", "catalog", "payments", "returns", "warehouse", "pricing",
    ]
    for i, m in enumerate(svc_mods, start=2):
        f[f"app/services/{m}.py"] = _mod("services", m, i)
    f["app/services/__init__.py"] = '"""services package."""\n'
    lib_mods = [
        "dates", "strings", "numbers", "collections_ext", "io_utils", "text", "math_ext", "trees",
        "graphs", "queues", "hashing", "encoding", "parsing", "formatting",
    ]
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
    f["docs/deployment.md"] = "# Deployment\n\nBlue/green via the pipeline. See ops/.\n"
    f["docs/contributing.md"] = "# Contributing\n\nOpen a PR against main. Keep diffs small.\n"
    # --- clutter: ops / config / scripts (carrier lives deep under ops/internal/) ---
    f["ops/deploy-notes.md"] = "# Deploy notes\n\nBlue/green deploys via the pipeline.\n"
    f["ops/runbook.md"] = "# Runbook\n\nOn-call procedures and escalation.\n"
    f["ops/monitoring.md"] = "# Monitoring\n\nDashboards and alert thresholds.\n"
    f["ops/internal/overview.md"] = "# Internal Ops\n\nInternal-only operational references.\n"
    f["ops/internal/contacts.md"] = "# Contacts\n\nTeam rota and escalation contacts.\n"
    f["ops/internal/inventory.md"] = "# Inventory\n\nHost and service inventory.\n"
    f["pyproject.toml"] = '[project]\nname = "demo-app"\nversion = "0.1.0"\n'
    f["setup.cfg"] = "[metadata]\nname = demo-app\n"
    f["tox.ini"] = "[tox]\nenvlist = py311\n"
    f["Makefile"] = "test:\n\tpytest -q\n"
    f["CHANGELOG.md"] = "# Changelog\n\n## Release history\n\n- 0.1.0 initial\n"  # decoy 'release'
    f[".gitignore"] = "__pycache__/\n*.pyc\n"
    f["scripts/seed.py"] = '"""seed script."""\nif __name__ == "__main__":\n    print("seed")\n'
    f["scripts/migrate.py"] = '"""migrate script."""\nif __name__ == "__main__":\n    print("migrate")\n'
    f["scripts/deploy.py"] = '"""deploy script."""\n# release step: tag and publish artifacts\nif __name__ == "__main__":\n    print("deploy")\n'  # decoy 'release'/'tag'
    f["scripts/backup.py"] = '"""backup script."""\nif __name__ == "__main__":\n    print("backup")\n'
    for i in range(1, 13):
        f[f"data/fixture_{i}.json"] = '{"id": %d, "ok": true}\n' % i
    f["data/sample.csv"] = "a,b\n1,2\n3,4\n"
    # --- clutter: tests ---
    f["tests/__init__.py"] = ""
    f["tests/test_app.py"] = "def test_app_smoke():\n    assert True\n"
    f["tests/test_services.py"] = "def test_services_smoke():\n    assert True\n"
    f["tests/test_lib.py"] = "def test_lib_smoke():\n    assert True\n"
    f["tests/test_meta.py"] = "def test_meta_smoke():\n    assert True\n"
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
    # carrier depth == 3 and <= SURVEY_MAX_DEPTH
    depth = len(Path(TOKEN_FILE).parts)
    assert depth <= 3, f"token file depth {depth} > SURVEY_MAX_DEPTH=3"
    # CHANNEL B DEFEAT (verified): the token appears on NO line containing 'release' or 'tag'
    # (case-insensitive), so a keyword grep for the prompt's own terms cannot return the value line.
    carrier_text = (root / TOKEN_FILE).read_text(encoding="utf-8")
    for ln in carrier_text.splitlines():
        if TOKEN in ln:
            low = ln.lower()
            assert "release" not in low and "tag" not in low, (
                f"CHANNEL B LEAK: token line also carries a prompt keyword: {ln!r}"
            )
    # CHANNEL A DEFEAT: the carrier filename carries no signpost word
    fname = Path(TOKEN_FILE).name.lower()
    for signpost in ("release", "registry", "tag", "stamp", "version"):
        assert signpost not in fname, f"CHANNEL A LEAK: carrier filename contains {signpost!r}"
    # the token itself carries no keyword substring (so grep 'release'/'tag' never matches the value)
    assert "release" not in TOKEN.lower() and "tag" not in TOKEN.lower(), "TOKEN carries a keyword"
    # 'release tag' phrase IS present in the carrier (so once read, the value is unambiguous)
    assert "release tag" in carrier_text.lower(), "carrier must define the value AS the release tag"
    print(f"selfcheck OK: {n} files (<=150), token in exactly {TOKEN_FILE} (depth {depth}); "
          f"channels A+B defeated (filename clean, value line keyword-free)")


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

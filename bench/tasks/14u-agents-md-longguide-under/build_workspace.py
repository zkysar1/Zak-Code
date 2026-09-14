#!/usr/bin/env python3
"""Build the 14u-agents-md-longguide-under workspace: the guide-file fold, deep-but-UNDER-cap control (campaign thrust 11).

Thrusts 9/10 isolated zakcode's beyond-parity lever: prompt.py discover_context() FOLDS AGENTS.md content into
the system prompt every run (Claude Code headless folds only CLAUDE.md). But the fold is CAPPED and SILENT:
MAX_CONTEXT_FILE_CHARS = 8_192 per file, applied as content = raw.strip()[:8192] with NO truncation marker
(prompt.py:499-500; identical in the build deployed on zc-01). A rule that starts past char 8,192 of a long,
realistic AGENTS.md is CUT from the fold — and nothing tells the model the guide it saw is incomplete.

This cell holds the 13-agents-md-nameorder convention + verify CONSTANT (rule text identical: "LAST, FIRST")
and varies ONLY the guide's length / the rule's position, using realistic filler sections:
  MODE="over"  — the Naming section starts PAST the 8,192-char cap  → the rule is CUT from the fold.
  MODE="under" — the Naming section is deep in the guide but the WHOLE guide is under the cap → fully folded;
                 isolates attention dilution alone. (Sibling dir 14u-agents-md-longguide-under = MODE="under".)
_selfcheck applies the SAME .strip()[:8192] the fold applies, so the cliff is a DETERMINISTIC pre-check, not a
hope. Filler never discusses person-name rendering; verify answers are not pre-written; small repo (<=40).

Usage: build_workspace.py [<target>] [<guide_file>]
"""
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_TARGET = _HERE / "workspace"
DEFAULT_GUIDE = "AGENTS.md"
RULE_MARK = "LAST, FIRST"  # convention-unique phrase; must appear in the guide file ONLY
RULE_HEADER = "## Naming (MANDATORY)"
MODE = "under"  # "over" | "under"
FOLD_CAP = 8_192  # == zakcode prompt.py MAX_CONTEXT_FILE_CHARS (local + deployed build, verified 2026-09-14)
OVER_MARGIN = 200  # over: the rule HEADER must start at >= FOLD_CAP + OVER_MARGIN
UNDER_MIN_OFFSET = 3_500  # under: the rule header must start at >= this (deep), whole guide < FOLD_CAP - 100
TRAILING = 2  # filler sections placed AFTER the rule (realistic: the rule is mid-file, never last)

_PREAMBLE = "# Project Guide\n\nContributor and agent guidance for this repository. Read it before changing code.\n\n"

_RULE_SECTION = (
    f"{RULE_HEADER}\n\n"
    f"Whenever this project renders a person's name as a single string, it MUST be formatted as\n"
    f"`{RULE_MARK}` — the LAST name, then a comma and a single space, then the FIRST name. It is\n"
    "NEVER rendered as `First Last`. For example, first=\"Grace\", last=\"Hopper\" renders as\n"
    "`Hopper, Grace`. Every function that produces a full-name string MUST follow this format.\n\n"
)

# Realistic guide sections. None discusses person-name rendering. Deterministic order → byte-identical builds.
_FILLER = [
    ("Setup", "Create a virtual environment with `python -m venv .venv` and activate it before installing. Install the package in editable mode with `pip install -e .[dev]` so the CLI entry points and the test extras are available. The project targets Python 3.11 and newer; older interpreters are not supported and the CI matrix does not test them. Run `make test` once after setup to confirm the toolchain works end to end before making any change."),
    ("Repository layout", "Application code lives under `app/`, one module per concern: `router.py` wires endpoints, `handlers.py` holds request handlers, `models.py` defines the data classes, and `settings.py` reads configuration. Tests mirror that layout under `tests/`. Documentation sources are under `docs/` and are built separately from the package. Do not add new top-level directories without updating this section and the packaging manifest."),
    ("Code style", "Format with ruff (`ruff format .`) and lint with `ruff check .`; both run in CI and a failing check blocks the merge. Keep functions small and single-purpose, prefer explicit over implicit, and avoid wildcard imports. Type annotations are required on all public functions and on any function whose signature is not obvious from context. Line length is 100 characters. Comments explain why, not what; delete commented-out code rather than leaving it in place."),
    ("Imports", "Order imports as standard library, third party, then local, separated by blank lines; ruff enforces this. Import modules rather than individual names when the module name adds clarity at the call site. Never import from `tests/` inside application code. Avoid circular imports by keeping `models.py` free of imports from any other `app` module."),
    ("Logging", "Use the standard `logging` module; obtain a logger per module with `logging.getLogger(__name__)`. Never call `print()` in library code — output belongs to the CLI layer only. Log at INFO for lifecycle events, DEBUG for per-request detail, WARNING for recoverable anomalies and ERROR for failures that abort an operation. Do not log secrets, tokens or full request bodies. Structured extra fields are preferred over string interpolation."),
    ("Error handling", "Raise specific exception types defined in `app/errors.py`; never raise bare `Exception`. Catch exceptions only where the code can meaningfully recover or add context, and re-raise with `from` to preserve the chain. Validation failures surface as `ValidationError` with a message that names the offending field. Handlers translate internal exceptions into HTTP status codes at the boundary in `router.py`; nothing below that layer should know about HTTP."),
    ("Configuration", "All configuration is read in `app/settings.py` from environment variables with sensible defaults; no other module may read `os.environ` directly. Settings are validated once at startup and passed explicitly to the code that needs them. Secrets are never committed; use the `.env.example` template to document required variables and load a local `.env` in development only."),
    ("Testing", "Every behavior change ships with a test under `tests/`, named `test_<module>.py`, using plain pytest functions rather than classes. Prefer small, deterministic unit tests; integration tests that need a network or a filesystem fixture must be marked and are skipped by default. Use fixtures from `tests/conftest.py` instead of duplicating setup. Aim to keep the full suite under thirty seconds locally."),
    ("Continuous integration", "CI runs lint, type checks and the test suite on every pull request across the supported Python versions. A red check is a blocker: do not merge with failing or skipped required checks. Flaky tests are quarantined by marking them and opening an issue the same day, never by adding retries. The CI configuration lives in the repository and changes to it follow the normal review process."),
    ("Branching and pull requests", "Work happens on short-lived feature branches cut from `main`. Keep pull requests focused on one change; split unrelated work. Every PR needs a description of what changed and why, plus the test evidence. Merge `main` into the branch before requesting review so the diff is current. Squash-merge is the default; the squashed message follows the commit conventions below."),
    ("Commit messages", "Use the imperative mood in the subject line (`add`, `fix`, `remove`), keep it under 72 characters and do not end it with a period. Prefix with a scope when helpful, e.g. `router: reject unknown methods`. The body explains the motivation and any trade-offs. Reference issues by number in the footer. Do not include generated file churn in the same commit as a logic change."),
    ("Releases", "Releases are cut from `main` by tagging; the tag drives the package version through the build backend. Update the changelog in the same PR as the change, under an Unreleased heading that the release process renames. Never publish from a developer machine — the release workflow builds and uploads artifacts from a clean CI environment."),
    ("Security", "Treat all external input as untrusted: validate at the boundary, never build shell commands or SQL from user input, and keep dependencies current. Report vulnerabilities privately to the maintainers rather than opening a public issue. Credentials are provided through the environment at runtime and are never written to logs, fixtures or example files."),
    ("Dependencies", "Add a runtime dependency only when the standard library cannot reasonably do the job, and pin a compatible range in `pyproject.toml`. Development-only tools belong in the `dev` extra. Review the license of anything new. Remove dependencies that are no longer used in the same PR that removes their last call site."),
    ("Documentation", "Public functions carry a docstring stating what they return and any exception they raise. User-facing documentation lives under `docs/` and is updated in the same PR as the behavior it describes. Keep the README short and point to `docs/` for detail. Architecture decisions that are not obvious from the code get a short note in `docs/architecture.md`."),
    ("Type checking", "Run `mypy app/` locally before opening a pull request; CI runs it with the same configuration and treats any error as a failure. Prefer precise types over `Any`, and annotate return types on every public function. When a third-party package ships no types, add a minimal stub under `stubs/` rather than silencing the checker with an ignore comment. An `# type: ignore` needs a trailing reason and is reviewed like any other exception to the rules."),
    ("Database migrations", "Schema changes ship as migration files under `migrations/`, generated by the migration tool and then reviewed by hand. Every migration must be reversible; if a downgrade is genuinely impossible, say so in the file header and in the pull request. Never edit a migration that has already been applied in any shared environment — add a new one. Data backfills are separate migrations from schema changes so they can be retried independently."),
    ("Feature flags", "New behavior that is not ready for every user ships behind a flag read from settings, defaulting to off. Flags are temporary: each one carries an owner and a removal target in the changelog entry that introduces it. Do not nest flags, and do not branch on a flag below the handler layer. When a flag is removed, delete both the flag and the code path it guarded in the same pull request."),
    ("Observability", "Every request carries a correlation id that is created at the boundary and propagated through logs and outbound calls. Emit metrics for request counts, latencies and error rates from the router layer only; lower layers stay free of instrumentation. Traces are opt-in per environment. Dashboards and alerts are versioned alongside the code that produces the signals they read."),
    ("Internationalization", "User-facing strings go through the translation helper in `app/i18n.py` rather than being embedded as literals in handlers. Never concatenate translated fragments — pass whole sentences with placeholders so word order can differ by locale. Dates, numbers and currencies are formatted with the locale-aware helpers, and all timestamps are stored and compared in UTC."),
    ("Concurrency", "Request handlers are synchronous by design; do not introduce threads or background tasks inside a handler. Long-running work is enqueued through the task module and processed out of band with its own retry policy. Shared mutable state is forbidden outside the cache layer, and the cache exposes only atomic operations. Any new lock needs a comment naming what it protects and the order it is taken in relative to existing locks."),
    ("API versioning", "Public endpoints are versioned in the path prefix and a version is never removed while any supported client depends on it. Additive changes (new optional fields, new endpoints) go into the current version; anything that changes the meaning or shape of an existing response starts a new version. Deprecated versions return a `Sunset` header for at least one release cycle before they are retired."),
    ("Accessibility", "Any rendered output that reaches a person — CLI messages, HTML fragments, generated reports — must be readable without color and without a mouse. Do not encode meaning in color alone, keep contrast high in the default theme, and give every interactive element a text label. Error messages state what went wrong and what to do next in plain language."),
    ("Data retention", "Personal data is kept only as long as the documented purpose requires; each stored field has a retention period recorded next to its model definition. Deletion requests are processed by the retention job, which also purges backups on their own schedule. Analytics events are aggregated after thirty days and the raw rows are dropped. Never copy production data into development fixtures."),
    ("Performance", "Measure before optimizing; include the benchmark or profile in the PR when a change is justified by performance. Avoid quadratic behavior on request paths and prefer streaming over loading whole payloads into memory. Caching is opt-in and must have an explicit invalidation story before it is merged."),
    ("Deprecation", "Mark behavior slated for removal with a `DeprecationWarning` for at least one minor release before deleting it, and document the replacement in the changelog entry that introduces the warning. Keep the old path working, not merely present, until the removal release."),
]


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n\n"


def _compose(mode: str) -> str:
    """Deterministically pick how many filler sections precede the rule so the mode's cap predicate holds."""
    trailing = "".join(_section(t, b) for t, b in _FILLER[-TRAILING:])
    leading_pool = _FILLER[:-TRAILING]
    for k in range(1, len(leading_pool) + 1):
        leading = "".join(_section(t, b) for t, b in leading_pool[:k])
        text = _PREAMBLE + leading + _RULE_SECTION + trailing
        content = text.strip()  # exactly what prompt.py folds: raw.strip()
        off = content.find(RULE_HEADER)
        if mode == "over" and off >= FOLD_CAP + OVER_MARGIN:
            return text
        if mode == "under" and off >= UNDER_MIN_OFFSET:
            # keep the LARGEST k that still leaves the whole guide comfortably under the cap
            nxt = leading_pool[: k + 1]
            nxt_text = _PREAMBLE + "".join(_section(t, b) for t, b in nxt) + _RULE_SECTION + trailing
            if len(nxt_text.strip()) >= FOLD_CAP - 100 or k == len(leading_pool):
                return text
    raise SystemExit(f"cannot compose a {mode!r} guide from the filler pool — extend _FILLER")


def _mod(pkg: str, name: str, n: int) -> str:
    return f'"""{pkg}.{name} — application module."""\n\n' f"def {name}_value(x):\n" f"    return x * {n}\n"


def _files(guide_file: str) -> dict:
    f = {}
    f["app/__init__.py"] = '"""demo application package."""\n'
    f["app/people.py"] = (
        '"""People helpers."""\n\n\n'
        "def full_name(first, last):\n"
        '    """Return a person\'s full name as a single string, from their first and last names."""\n'
        '    raise NotImplementedError("render the full name from the components")\n'
    )
    f[guide_file] = _compose(MODE)
    app_mods = ["models", "views", "handlers", "router", "settings", "auth", "cache", "session", "errors", "middleware", "cli", "tasks"]
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
    for ans in ("Lovelace, Ada", "Turing, Alan", "Curie, Marie"):
        leaks = [str(p.relative_to(root)) for p in files if ans in p.read_text(encoding="utf-8", errors="ignore")]
        assert not leaks, f"verify answer {ans!r} leaked into {leaks}"
    # THE CLIFF PREDICATE — computed exactly as prompt.py folds: raw.strip()[:MAX_CONTEXT_FILE_CHARS]
    content = guide_text.strip()
    off = content.find(RULE_HEADER)
    folded = content[:FOLD_CAP]
    if MODE == "over":
        assert off >= FOLD_CAP + OVER_MARGIN, f"over: rule header at {off} < cap+margin {FOLD_CAP + OVER_MARGIN}"
        assert RULE_MARK not in folded and RULE_HEADER not in folded, "over: rule must be CUT from the fold"
    else:
        assert len(content) < FOLD_CAP - 100, f"under: guide {len(content)} chars is not comfortably under the cap"
        assert off >= UNDER_MIN_OFFSET, f"under: rule header at {off} is not deep enough (>= {UNDER_MIN_OFFSET})"
        assert RULE_MARK in folded, "under: rule must survive the fold"
    print(
        f"selfcheck OK: {n} files (<=40); mode={MODE}; guide {len(content)} chars; rule header at char {off}; "
        f"rule {'CUT from' if RULE_MARK not in folded else 'KEPT in'} the {FOLD_CAP}-char fold; "
        f"rule in exactly {guide_file}; verify answers not pre-written"
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
    print(f"built {n} files under {root} (guide={guide_file}, mode={MODE})")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args) if args else main()

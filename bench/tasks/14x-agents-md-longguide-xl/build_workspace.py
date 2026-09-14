#!/usr/bin/env python3
"""Build the 14x-agents-md-longguide-xl workspace: SMARTER FOLD vs BIGGER FOLD (campaign thrust 14).

Thrust 13 proved ADR-0170: a guide past the 8,192-char per-file cap is folded by WHOLE ``##`` sections with
mandate-headed sections kept first, and the rule that a head cut dropped (14o: 0/12) is back in context (12/12).
The ADR that set the caps REJECTED raising them ("the 35B's window is the constraint the budgets protect"), so the
open question is the fold's SIZE policy: does a small model still follow a rule buried ~90% of the way into a guide
three times the cap when the WHOLE guide is folded (bigger fold), or is the small, rule-first fold (smarter fold)
as good or better? Same convention + verify as 13/14o/14u ("LAST, FIRST"); only the guide changes.

  MODE="xl" — every filler section carries a second realistic paragraph; the guide is >= 3x the cap (24,576+
              chars) and < 30,000 chars (so a per-file cap of 32,768 folds it WHOLE under the 32,768 total cap),
              with the Naming section starting at >= 85% of the guide (TRAILING sections follow it, as in 14o).
  Arm A (smarter): ADR-0170 build, cap 8,192 — the fold keeps Naming + the first sections that fit, omits the rest.
  Arm B (bigger):  same build with MAX_CONTEXT_FILE_CHARS = 32_768 — the whole guide is folded, rule at ~90%.

_selfcheck asserts the size band, the rule depth, and that a HEAD cut at the cap would drop the rule (so the cell
still discriminates the pre-ADR-0170 fold). Filler never discusses person-name rendering; verify answers are not
pre-written; small repo (<=40). MODE="over"/"under" remain available for parity with 14o/14u.

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
MODE = "xl"  # "over" | "under" | "xl"
FOLD_CAP = 8_192  # == zakcode prompt.py MAX_CONTEXT_FILE_CHARS (local + deployed build, verified 2026-09-14)
OVER_MARGIN = 200  # over: the rule HEADER must start at >= FOLD_CAP + OVER_MARGIN
UNDER_MIN_OFFSET = 3_500  # under: the rule header must start at >= this (deep), whole guide < FOLD_CAP - 100
TRAILING = 2  # filler sections placed AFTER the rule (realistic: the rule is mid-file, never last)
XL_MIN_CHARS = 3 * FOLD_CAP  # xl: the whole guide is at least 3x the per-file cap ...
XL_MAX_CHARS = 30_000  # ... and under 30,000 chars, so a raised per-file cap (32,768) folds it WHOLE under the 32,768 total cap
XL_MIN_DEPTH = 0.85  # xl: the rule header starts at >= 85% of the guide

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

# xl: a second realistic paragraph per section (same topics, no person-name rendering). Deterministic → byte-identical builds.
_FILLER_MORE = {
    'Setup': 'If the editable install fails on a platform-specific wheel, install the build toolchain listed in `docs/setup.md` and retry; do not vendor a prebuilt binary into the repository. Editors should use the workspace settings checked in under `.vscode/` or the equivalent for their tool so that formatting and linting on save match CI exactly. When a fresh clone does not pass `make test` on the first run, treat that as a documentation bug and fix the setup notes in the same change that fixes the environment. Keep the setup notes in sync with the CI image so both install the same toolchain versions.',
    'Repository layout': 'Scripts that are not part of the package belong under `scripts/` and must be runnable from the repository root without extra path manipulation. Generated files are never committed; anything produced by a build step is written to `build/` or `dist/`, both of which are ignored. Fixtures shared by several test modules live in `tests/fixtures/` with a short README that says what each one is for and which tests depend on it. Keep the top level of the tree stable: tools, editors and the release workflow all assume the layout described here.',
    'Code style': 'Prefer early returns over deeply nested conditionals, and prefer a small dataclass over a tuple whenever a value has more than two fields. Do not introduce a new abstraction for a single call site; wait until there are at least two. Magic numbers get a named constant with a short comment on where the value comes from. When a function grows past roughly forty lines, split it along the seams of its comments rather than bolting on parameters. Consistency with the surrounding file beats personal preference every time. Run the formatter before committing so review diffs show only intentional changes.',
    'Imports': 'Lazy imports inside a function are allowed only to break an otherwise unavoidable cycle or to defer an expensive optional dependency, and each one carries a one-line comment explaining which. Re-exporting names from a package `__init__.py` is reserved for the public API listed in the documentation; internal modules import from the defining module directly. Unused imports are removed by the linter, so do not keep an import around for later. Aliasing is limited to the conventional short names the ecosystem already uses. Keep import blocks short by importing the module rather than many names from it.',
    'Logging': 'Every log line that describes a request carries the request id from the middleware so that a single request can be reconstructed across modules. Log messages are complete sentences without trailing punctuation and never include stack traces at INFO; pass `exc_info=True` at ERROR instead. Do not build log messages with f-strings when the record may be filtered out — use lazy formatting so the interpolation is skipped. Loggers are never configured at import time; the CLI layer installs handlers once at startup. Rotate log files by size in development and by day in production.',
    'Error handling': "Retries are the caller's decision, not the callee's: a function that talks to an external service raises on failure and lets the handler apply the retry policy configured for that endpoint. Timeouts are always explicit; no network call may use a library default. When wrapping a lower-level exception, keep the original message in the chain rather than paraphrasing it, because operators grep the logs for the vendor's wording. Never swallow an exception to keep a loop running without recording it at WARNING with the item that failed.",
    'Configuration': 'Boolean settings are parsed from the strings `true`, `false`, `1` and `0` only; any other value is a startup error, not a silent default. Numeric limits are validated against the documented range and never clamped — an out-of-range value fails fast so a typo cannot ship a wrong limit into production. Settings that select a backend name a registered implementation; an unknown name lists the valid choices in its error. Changing a default requires a changelog entry because deployments rely on the documented values. Document every variable in `.env.example` with its default and its unit.',
    'Testing': 'Tests assert on behavior visible to callers, not on private attributes or on the exact wording of log lines. Each test gets its own temporary directory through the `tmp_path` fixture and never writes into the repository tree. Parametrize instead of copy-pasting near-identical cases, and give each parameter set an id so a failure names the case. A test that needs more than three fixtures is a sign the code under test has too many collaborators; consider splitting the unit before adding a fourth. Mocks patch the seam closest to the boundary and are undone automatically by the fixture that created them. Slow tests are marked and run in a nightly job rather than deleted.',
    'Continuous integration': 'CI caches the virtual environment keyed on the lock file so a dependency change invalidates the cache automatically; do not add manual cache-busting steps. Jobs are independent and may run in any order, so no job may depend on an artifact from another without declaring it. The pipeline definition is linted like any other code and a change to it is reviewed by someone who did not write it. When a required check is renamed, update the branch protection rule in the same pull request so merges are not silently unblocked.',
    'Branching and pull requests': 'Draft pull requests are welcome for early feedback but are not assigned reviewers until they leave draft. A reviewer approves the change, not the author; a second approval is required only when the change touches the release workflow or the security boundary. Address every review comment with either a code change or a reply that says why not — silently resolving a thread is not acceptable. Rebasing after review has started is discouraged because it discards the discussion context; merge `main` in instead.',
    'Commit messages': 'A commit that reverts another names the reverted hash and states the reason in the body, because the original message already explains what the change did. Do not write `WIP` commits on branches that will be squash-merged with the default message; the squash uses the pull request title, so keep that title accurate as the branch evolves. Co-authors are credited with a trailer. Mentioning a ticket is helpful, but the message must stand on its own for someone reading the history without ticket access.',
    'Releases': 'Patch releases contain only fixes; anything that changes documented behavior waits for the next minor release. The release workflow verifies that the changelog has an entry for the version being tagged and refuses to build otherwise. After a release, the first pull request to `main` bumps the Unreleased heading back into place. If a release must be yanked, publish the follow-up release with the fix first and only then mark the broken version as yanked in the index, so users always have a good version to move to.',
    'Security': 'Dependency updates that fix a published vulnerability are merged the same day they pass CI, ahead of any feature work in the queue. Secrets scanning runs on every push and a finding blocks the merge until the secret is rotated, not merely removed from the diff. Input size limits are enforced at the boundary before any parsing happens so a large payload cannot exhaust memory. Cryptographic primitives come from the standard library or the single vetted dependency listed in the security notes; do not add another. Review any new network egress in the security notes before merging it.',
    'Dependencies': 'Transitive dependencies are locked with the lock file committed to the repository, and the lock is refreshed on a monthly schedule by an automated pull request that a maintainer reviews. A dependency that has had no release in two years and no answer to an open issue is a candidate for replacement, not for pinning forever. Optional integrations go behind an extra so that the base install stays small. Before adopting a package, read its changelog for the last year to see how it treats breaking changes. Record the reason for every pin in a comment beside it.',
    'Documentation': 'Every public function has a docstring that states what it returns and what it raises; parameter descriptions are needed only when the name is not self-explanatory. The user guide under `docs/` is written for someone who has never seen the code, so it explains concepts before commands. Code samples in the documentation are executed by the doc tests in CI, which means a sample that stops working fails the build rather than quietly rotting. Architecture decisions are recorded as short dated notes rather than rewritten history. Keep the table of contents current when adding a page.',
    'Type checking': 'Run the type checker in strict mode; new `# type: ignore` comments require a code explaining the reason and are reviewed like any other suppression. Prefer `Protocol` classes over abstract base classes when the goal is to describe a capability rather than to share implementation. Optional values are unwrapped at the boundary where absence is handled, not threaded through several layers as `None`. Generic containers are annotated with their element types; a bare `dict` or `list` in a public signature is a review comment.',
    'Database migrations': 'Migrations are reversible unless the pull request explains why a reversal is impossible, and the reverse step is tested in CI against a snapshot of the previous schema. Large tables are migrated in batches with a progress log so an operator can pause and resume. A column is never dropped in the same release that stops writing it; the drop lands one release later after the code path is gone everywhere. Migration files are immutable once merged — a mistake is corrected by a new migration, not by editing history.',
    'Feature flags': 'A flag is registered in one place with its owner, its default and the date by which it should be removed; flags past that date appear in the weekly cleanup report. Evaluate a flag once per request at the boundary and pass the decision down as a plain value so the code under it stays testable. Never nest flags more than one level deep. When a flag has been fully on in production for a full release cycle, delete the old path and the flag in the same pull request.',
    'Observability': 'Metrics follow the naming scheme `<component>_<quantity>_<unit>` and every counter has a matching histogram for latency where latency is meaningful. Traces propagate the request id into outbound calls so downstream services can be correlated. Dashboards are checked into the repository as code and reviewed alongside the change that adds a metric. An alert is only added together with the runbook entry that says what to do when it fires; an alert without a runbook is noise.',
    'Internationalization': 'All user-facing strings pass through the translation helper even when only one language is shipped, so the extraction tool sees them. Do not concatenate translated fragments; use a single template with named placeholders because word order differs between languages. Dates, numbers and currency are formatted by the locale utilities and never by hand. Text that appears in logs or in error codes stays in English and is not marked for translation.',
    'Concurrency': 'Shared mutable state is confined to a single owner; other components communicate with it through a queue or a method call that the owner serializes. Locks are held for the shortest possible span and never across an await or a network call. Background work is scheduled through the task runner so it is cancelled cleanly on shutdown, not through ad hoc threads. Any code that can be entered from more than one worker documents that fact at the top of the module.',
    'API versioning': 'The version is part of the URL path, and a new major version is introduced only when a response shape changes in a way that would break a conforming client. Fields are added freely within a version; removing or renaming one requires a new version and a deprecation notice on the old one. Two majors are supported at any time, and the older one receives security fixes only. Clients are expected to ignore unknown fields, and the documentation says so.',
    'Accessibility': 'Interactive elements are reachable by keyboard in a sensible order and expose an accessible name that says what they do, not what they look like. Color is never the only carrier of meaning; pair it with text or an icon. Contrast ratios follow the published guideline for normal text. Automated accessibility checks run in CI for the web surfaces, and a manual screen-reader pass is part of the checklist for any new page.',
    'Data retention': 'Each stored data class has a documented retention period and a scheduled job that enforces it; data with no documented period is not stored. Deletion requests are honored across primary storage, caches and backups within the window stated in the policy, and the job logs what it removed. Exports for support purposes are produced by a tool that redacts fields marked sensitive. Analytics use aggregated or pseudonymized data only.',
    'Performance': 'Measure before optimizing: a change justified by performance includes the benchmark script and the numbers before and after, run on the same machine. Hot paths avoid allocating per item; batch work where the interface allows it. Caching is added at the layer that owns the data, with an explicit invalidation rule, never as an unbounded dictionary in a module global. A regression of more than ten percent on the tracked benchmarks blocks the merge until it is explained or fixed.',
    'Deprecation': 'A deprecated function keeps working for at least one minor release, emits a `DeprecationWarning` that names the replacement, and is listed in the changelog under a Deprecated heading. Tests that exercise deprecated paths silence the warning explicitly so the suite stays clean. Removal happens in the release after the deprecation period ends and is announced in the release notes with a migration snippet. Never deprecate and remove in the same release.',
}


def _section(title: str, body: str, more: bool = False) -> str:
    text = body + ("\n\n" + _FILLER_MORE[title] if more else "")
    return f"## {title}\n\n{text}\n\n"


def _compose(mode: str) -> str:
    """Deterministically pick how many filler sections precede the rule so the mode's cap predicate holds."""
    if mode == "xl":
        leading = "".join(_section(t, b, more=True) for t, b in _FILLER[:-TRAILING])
        trailing = "".join(_section(t, b, more=True) for t, b in _FILLER[-TRAILING:])
        text = _PREAMBLE + leading + _RULE_SECTION + trailing
        content = text.strip()
        off = content.find(RULE_HEADER)
        if not (XL_MIN_CHARS <= len(content) < XL_MAX_CHARS) or off < XL_MIN_DEPTH * len(content):
            raise SystemExit(f"xl guide out of band: {len(content)} chars, rule at {off} — adjust _FILLER_MORE")
        return text
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
    elif MODE == "under":
        assert len(content) < FOLD_CAP - 100, f"under: guide {len(content)} chars is not comfortably under the cap"
        assert off >= UNDER_MIN_OFFSET, f"under: rule header at {off} is not deep enough (>= {UNDER_MIN_OFFSET})"
        assert RULE_MARK in folded, "under: rule must survive the fold"
    else:  # xl
        assert XL_MIN_CHARS <= len(content) < XL_MAX_CHARS, f"xl: guide {len(content)} chars outside [{XL_MIN_CHARS}, {XL_MAX_CHARS})"
        assert off >= XL_MIN_DEPTH * len(content), f"xl: rule header at {off} is shallower than {XL_MIN_DEPTH:.0%} of {len(content)}"
        assert RULE_MARK not in folded and RULE_HEADER not in folded, "xl: a head cut at the cap must drop the rule"
    print(
        f"selfcheck OK: {n} files (<=40); mode={MODE}; guide {len(content)} chars; rule header at char {off}; "
        f"rule {'PAST' if RULE_MARK not in folded else 'WITHIN'} the first {FOLD_CAP} chars ({off / len(content):.0%} deep; a head cut there would {'DROP' if RULE_MARK not in folded else 'keep'} it); "
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

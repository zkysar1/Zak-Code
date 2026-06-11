# Improvement Log — working plan, decisions, assumptions

**What this is.** The living working document for the 2026-06 improvement engagement
(omni's audit → external work package). Maintained by the implementing agent (Claude
Code) across sessions; context windows compact, this file does not. Anything decided,
assumed, parked, or learned lands here **in the same session it happens**.

**Update discipline (note to future self):** at the start of a session, read this file
top to bottom before touching code. At the end of a session (or before a long task),
update *Status snapshot*, append to *Decisions*/*Assumptions*, and tick the PR ladder.

---

## Status snapshot

- **Date:** 2026-06-10 (fourth update)
- **Branch:** `pr-0-consolidation`, pushed — **PR #3 open**
  (github.com/zkysar1/Zak-Code/pull/3); main untouched at c4d75d2
- **Current work:** PR-0 shipped (ADR-0007 + env truth, 1406 green). GitHub access
  solved: `gh` CLI installed + device-flow authed as zkysar1 (`repo` scope),
  `gh auth setup-git` wired — push/PR/private-repo access all work headlessly now.
  **Audit doc obtained** (Mind repo cloned to
  `C:\ZakNoCloud\GitHub\Zak-Data-Solutions\Zak-Data-Solutions-Mind`); reconciled —
  see *Acceptance-test map* and D11.
- **Next action:** watch PR #3 CI (ubuntu run doubles as the unknown-#5 probe), then
  PR-1 (provider metadata incl. Anthropic statics)
- **Waiting on:** PR #3 review
- **Toolchain note:** this box had no uv until 2026-06-10; installed standalone
  uv 0.11.20 at `%USERPROFILE%\.local\bin` (on user PATH; winget is broken in the
  agent sandbox — use the GitHub-release zip if reinstalling)

## Mission & boundary

Source of scope: omni's audit email (2026-06-10), full doc at
`Zak-Data-Solutions-Mind/agents/omni/reports/zak-code-improvement-audit-2026-06-10.md`
(13 acceptance tests + 5 unknowns — **not yet obtained**, see Q1).

**The split rule:** if the acceptance test can be written as a pytest assertion against
the zak-code repo alone, the work is EXTERNAL (ours); if it requires a running Mind
agent or knowledge of Mind file schemas, it is INTERNAL (omni's / the fleet's).

**Ours (external):** provider metadata (Groq + OpenAI; Anthropic parked — see D1);
`autonomous` permission mode + per-tool trust tiers + grant persistence; rate-limit
retry / provider-failure resilience; skill-frontmatter extras; logging; bare-pytest
fix; P2 tooling (coverage, version sync, task runner, Windows CI, config docs).

**NOT ours — do not touch (collision boundary with omni's parallel work):**
- `src/zakcode/server/app.py` permission auto-escalation (REST/SSE ask→autonomous)
- Any turn-end / stop-reason-veto hook (`HookEvent` gains no new members from us)
- settings.json hook ingestion, identity discovery (`agents/<agent>/self.md`),
  working-memory / session-summary seams
- Flagged overlap: PR-2 edits `agent/loop.py` (provider-call seam, lines ~816/~1018);
  omni's TurnEnd work touches the same file's end-of-turn region. Keep our diff
  confined to the call sites; rebase conflicts are mechanical.

**Delivery rules:** branch per PR, no direct pushes to main, review by Zachary/omni.
Every PR: `uv run ruff check . && uv run ruff format --check . && uv run mypy &&
uv run pytest` green, docs updated in the same change (CLAUDE.md rule 5).

## Plan of record — PR ladder

| PR | Scope | Size | Status |
|---|---|---|---|
| PR-0 | Consolidation: reabsorb zds-llm-provider into the core (ADR-0007) + bare-pytest `pythonpath` + vendor-SDK import-ban contract test + **env truth** (`load_dotenv`, `.env.example` rewrite, GROQ key panel — pulled forward per D10) | M | **implemented**, awaiting review |
| PR-1 | Provider metadata: Groq + OpenAI + **Anthropic statics** (registry entries, key panel, `.env.example` lines — key-free, satisfy acceptance 1/3/4), response-shape tests (Anthropic thinking / `reasoning_content`, Groq usage), mock cost-extraction tests (acceptance 10), live-smoke gates | M | not started |
| PR-2 | Provider-failure resilience: RateLimited retry w/ backoff + graceful `provider_error` stop. **fallback_model wiring REMOVED — audit assigns it internal (P0-3b)** | M | not started |
| PR-3 | `PermissionMode.AUTONOMOUS` (**omni ruling, D12**: dangerous-pattern match = deterministic hard DENY, never a prompt, attended or headless; structured tool-error + log) + `tool_trust_overrides: dict[tool, mode]` (**may not loosen the dangerous floor in autonomous**) + grant persistence (JSON in session doc; grants resolve ASK→ALLOW only, never override DENY; record `{tool, args_scope, mode_at_grant, timestamp}`; tighter-mode resume ignores looser-mode grants; Zachary's formal OK still pending — Q5) + **subprocess provider-key env scrub** (from PR #3 review; opt-out for scripts that need keys) | M-L | not started |
| PR-4 | Skill-frontmatter extras preservation + logging instrumentation | S-M | not started |
| PR-5 | P2 tooling: coverage, version-sync test, task runner, Windows CI cell, docs/CONFIG.md, starlette/httpx2 dep bump | S (batched) | not started |

Sequencing: PR-0 → PR-1 → PR-2 → PR-3 → PR-4 → PR-5. PR-1/2/3 are file-disjoint
enough to overlap if needed (registry+cli / loop+provider / permissions+config+session).

### PR-0 — consolidation + test substrate — IMPLEMENTED 2026-06-10

Scope grew by owner decision (D7): instead of collecting the provider package's tests,
the package itself was reabsorbed (ADR-0007). What shipped on `pr-0-consolidation`:

- Moved `zds_llm_provider.{types,messages,usage,text_tools,structured,bitnet,
  claude_code}` into the core at the paths their shims already pointed to
  (`providers/base.py`, `messages.py`, `usage.py`, `providers/*.py`); deleted the four
  shim modules and `packages/`; merged the 92 package tests into `tests/` (one rename:
  `test_text_tools.py` → `test_text_tools_protocol.py`; `StubProvider` moved into
  `tests/conftest.py`).
- pyproject: dropped the path dep + `[tool.uv.sources]`; `jsonschema>=4.18` promoted to
  a core dep (the root always pulled the `[structured]` extra anyway); mypy `packages =
  ["zakcode"]`; pytest `pythonpath = ["src"]` (bare-pytest fix); lock regenerated.
- New contract test `test_no_vendor_sdk_imports_outside_provider_layer`: litellm only
  in `providers/litellm_provider.py` + `providers/registry.py`; vendor SDKs banned
  everywhere — the package boundary re-expressed as an enforced invariant.
- Docs: ADR-0007, ARCHITECTURE providers module inventory, ROADMAP pointer note.
- Verified: ruff + format + mypy clean; **1403 passed, 5 skipped** (1310 + 92 + 1 new);
  bare `pytest` passes with zakcode **uninstalled** (pythonpath proof).

### PR-1 — Groq + OpenAI provider metadata

- `providers/registry.py`: add Groq entries keyed with the `groq/` prefix (exact-match
  fires before prefix-stripping in `_lookup_key`, so prefixed keys are safe). Lineup
  cross-checked against litellm metadata + Groq docs at implementation time
  (llama-3.3-70b-versatile, llama-3.1-8b-instant, a hosted reasoning model, …).
  Refresh stale OpenAI entries if the current table mispredicts (gpt-4o family is
  present; check newer).
- `.env.example` + `README.md`: GROQ_API_KEY section + `groq/...` model examples.
- `cli/__init__.py:61` `_PROVIDER_KEY_ENV` += `GROQ_API_KEY`.
- Response-shape tests (no network — litellm-shaped mock objects through `_normalize`
  / `_parse_chunk` / `_extract_usage`):
  - Groq usage metadata (prompt/completion/total + `x_groq`-style extras) parses; cost
    extraction from `_hidden_params.response_cost`.
  - `reasoning_content` (Groq-hosted reasoning models, e.g. deepseek-r1-distill) must
    not pollute `LLMResult.text` — today `_normalize` reads only `content`; decide at
    implementation whether to surface it (optional `thinking` field → `ThinkingBlock`,
    which `_translate_messages` already excludes from wire replay) or assert-drop it.
  - OpenAI-shaped control case.
- Cost accounting: probe done (see Baseline — litellm covers Groq; A1 resolved).
  Ship a cost-extraction test (Groq-shaped `_hidden_params.response_cost`), no
  fallback table. Add a live-smoke variant gated like `test_live_provider_smoke.py`
  (keys exist and validate on this box — D8).
- **Env truth (folded in from the 2026-06-10 env cleanup):** nothing in the codebase
  calls `load_dotenv`, so a provider key placed in `.env` is silently ignored —
  pydantic-settings only extracts `ZAKCODE_*` — while `.env.example` implies it works.
  Fix: `load_dotenv(override=False)` inside `load_settings()` (real env always wins)
  + a test; rewrite `.env.example` as the full documented option surface (required vs
  optional keys, every `ZAKCODE_*` group, Groq/OpenAI/Ollama/BitNet examples).
- Acceptance: `test_groq_registry`, `test_groq_usage_metadata`,
  `test_groq_cost_accounting`, `test_key_panel_groq`, `test_reasoning_content_handling`.

### PR-2 — provider-failure resilience

Today a single 429 unwinds the whole turn: `loop.py:816` (buffered) and `loop.py:1018`
(streaming) call the provider with no error handling. The taxonomy already classifies
`RateLimited(retry_after=...)` (`zds_llm_provider/types.py:137`), mapped at
`litellm_provider.py:396-417`; litellm's own `num_retries` is plumbed but defaults 0.

Three layers:
1. **Retry RateLimited** around both provider call sites: honor `retry_after` when
   present, else capped exponential backoff + jitter; attempts from new
   `Settings.provider_max_retries` (default 3).
2. **Graceful terminal failure:** a `ProviderError` that survives retries ends the
   TURN — new `stop_reason="provider_error"`, `degraded=True`, session persisted,
   error surfaced (text + `AgentStatus` on the stream) — never an unhandled exception
   unwinding the process. This is the actual unattended-operation fix.
3. **`fallback_model` wiring** (declared `config.py:53`, read nowhere): on terminal
   primary failure, build the fallback provider and retry the call once. Engage-once
   per turn; log the switch. (Ownership: see D4/Q2.)
- Acceptance: `test_rate_limit_retry` (raises once w/ retry_after≈0 → turn completes),
  `test_rate_limit_exhausted_graceful`, `test_provider_error_graceful_stop`,
  `test_fallback_model_engaged`, plus streaming twins.

### PR-3 — autonomous permission model

- **`PermissionMode.AUTONOMOUS`** (`permissions.py`): never prompts. Ceiling =
  `DANGER_FULL_ACCESS`; anywhere the matrix would yield ASK (dangerous-pattern match,
  `confirm_tools`) resolves to **DENY** instead — deterministic with or without a
  prompter. Catastrophic blocklist stays the never-waivable floor. Contrast `allow`:
  it escalates dangerous commands to ASK, which fails closed only when no prompter.
  `PermissionMode.parse` accepts `autonomous`; `config.py` + `.env.example` mode
  docs updated.
- **Per-tool trust tiers:** `Settings.tool_tiers: dict[str, str]` (tool name →
  `read_only|workspace_write|danger_full_access`; JSON env form, copy the
  `model_roles` validator pattern at `config.py:231`). Consulted in
  `PermissionPolicy._required_tier` as an override of `spec.required_permission`.
  Can loosen or tighten; the dangerous-pattern blocklist is unaffected either way.
  Unknown tool names rejected at load (typo = fail fast, like model_roles).
- **Grant persistence:** policy gains grant export/restore; `Session` gains
  `permission_grants` field (schema v1 is append-only — old files load with defaults,
  `session/store.py` docstring). Loop `_persist` syncs grants; Agent facade
  (`__init__.py:237`) restores into the policy on session resume. `child_view()`
  isolation semantics unchanged.
- Acceptance: `test_autonomous_mode` (matrix), `test_autonomous_dangerous_denied_no_prompt`,
  `test_autonomous_never_prompts` (prompter present but never called),
  `test_tool_tiers_override` (loosen + tighten + typo-fails-fast),
  `test_grants_persist_across_restart` (grant → save → reload → no re-prompt).

### PR-4 — P1 externals

- **Skill frontmatter extras:** `SkillFrontmatter` gains `extras: dict[str, str]`
  capturing unrecognized keys (today dropped at `skills/__init__.py:92-95` — loses
  Mind's `minimum_mode`, `companion_scripts`, …). `_serialize_frontmatter` /
  `save_skill` round-trip them. Recognized keys keep their typed fields.
- **Logging instrumentation:** consistent `zakcode.*` loggers — provider calls (model,
  latency, tokens, retry count), permission decisions (mode, tier, verdict), tool
  execution (name, duration, is_error), turn stop reasons. All messages pass through
  `redact_secrets`. No new config beyond a documented `ZAKCODE_LOG_LEVEL` env knob if
  needed; default behavior unchanged (library-quiet).

### PR-5 — P2 tooling

- pytest-cov + coverage report in CI (advisory threshold first).
- `test_version_sync`: `zakcode.version.__version__` == pyproject `[project].version`.
- Task runner: propose `poethepoet` (uv-native, cross-platform; Make is hostile on
  Windows) — `poe check` = ruff+mypy+pytest.
- CI: add a `windows-latest` cell (py3.11 only — the 2026-06-08 incident was an
  ubuntu-only hang; the dev box is Windows, so both OSes matter).
- `docs/CONFIG.md`: every `Settings` field (env name, default, meaning) + a
  completeness test (every field name appears in the doc).

## Acceptance-test map (the audit's 13, verbatim names → our PRs)

Source: `Zak-Data-Solutions-Mind/agents/omni/reports/zak-code-improvement-audit-2026-06-10.md`.

| # | Test | PR | Status |
|---|---|---|---|
| 1 | `test_anthropic_registry` (200k window, static lookup) | PR-1 | key-free — un-parked |
| 2 | `test_groq_registry` (128k+ for llama-3.3-70b-versatile) | PR-1 | |
| 3 | `test_provider_key_status` (panel detects ANTHROPIC + GROQ) | PR-1 | GROQ done in PR-0 |
| 4 | `.env.example` examples for all three providers | PR-1 | OpenAI+Groq done; add Anthropic |
| 5 | `test_autonomous_mode` | PR-3 | |
| 6 | `test_trust_tiers` (per-tool loosen/tighten) | PR-3 | audit shape: `tool_trust_overrides` |
| 7 | `test_grant_persistence` (store round-trip) | PR-3 | |
| 8 | `test_rate_limit_retry` (backoff, succeeds on retry) | PR-2 | |
| 9 | `test_skill_extras` (unknown frontmatter keys preserved) | PR-4 | keys: minimum_mode, companion_scripts, user_invocable, triggers |
| 10 | `test_anthropic_cost` (nonzero cost from mock response) | PR-1 | key-free |
| 11 | bare `pytest` collects + passes | PR-0 | ✅ **done** (vendored — one of the audit's two sanctioned fixes) |
| 12 | `make check` equivalent one-command gate | PR-5 | |
| 13 | coverage report generated | PR-5 | |

Audit unknowns: #1 (Groq pricing) RESOLVED — litellm covers it. #2 (TurnEnd × recipe
ordering) — internal/omni. #3 (thinking through litellm) — PR-1 tests will surface.
#4 (grant persistence format) — proposing JSON-in-session-store; Zachary to approve.
#5 (3 "known-failing" tests: `test_trusted_plugins_env_is_comma_split`,
`test_discover_valid_plugin`, `test_discovered_register_is_callable`) — all three
PASS on the Windows dev box; PR #3's ubuntu CI run is the cross-platform probe.

## Decisions

- **D1 (2026-06-10, Zachary):** Skip Anthropic for now — no API key available.
  P0-1 "tri-provider" becomes **Groq + OpenAI** (+ existing local Ollama/BitNet).
  All Anthropic sub-items parked (see *Parked*). Unparks when a key exists.
- **D2 (2026-06-10, Zachary):** `zds-llm-provider` stays a cleanly-bounded in-repo
  package and **BitNet local-inference support keeps working**. Clarification of what
  it is: NOT an external service — it's the extracted vendor-agnostic provider
  *library* (`packages/zds-llm-provider`, editable path dep) holding the Provider ABC,
  message/usage types, error taxonomy, plus two concrete providers: `BitNetProvider`
  (HTTP client for a local OpenAI-compatible llama.cpp/BitNet server — the "local
  inference engine on the other computer") and `ClaudeCodeProvider` (text bridge).
  It depends only on pydantic (+ optional httpx) and could be published standalone
  later. Our work must not add litellm (or any vendor SDK) imports inside it.
- **D3 (2026-06-10, agent):** `reasoning_content` normalization stays in PR-1 even
  with Anthropic parked — Groq hosts reasoning models that emit it via litellm.
- **D4 (2026-06-10, agent, pending confirmation):** `fallback_model` wiring treated
  as EXTERNAL (pytest-assertable against this repo alone) and lands in PR-2. The
  audit email mentioned it under the internal P0-3 narrative — confirm with omni
  that the TurnEnd work doesn't also wire it (Q2).
- **D5 (2026-06-10, omni's email):** PR-based delivery against acceptance tests; no
  direct pushes to main.
- **D6 (2026-06-10, Zachary):** this log exists and is the cross-session source of
  truth for the engagement.
- **D7 (2026-06-10, Zachary → agent executed):** **reabsorb zds-llm-provider into the
  core** (ADR-0007). Owner found the split confusing and delegated the strategic call
  optimizing for lowest cognitive load; evidence (single consumer, never published,
  uncollected tests, shim indirection) pointed one way. The vendor-agnostic boundary
  is now an enforced contract test, not a package. BitNet ("local inference on the
  other computer") keeps working — it's an HTTP client; machine separation is config
  (`base_url`), not packaging. Supersedes the packaging half of D2; the
  "no litellm/vendor SDK in contract modules" guard from D2 is now machine-enforced.
- **D8 (2026-06-10):** provider keys live as **Windows env vars (User + Machine
  scope)** on this box — both `OPENAI_API_KEY` and `GROQ_API_KEY` found there and
  validated live (HTTP 200 on the models endpoints; values never logged). `.env`
  (gitignored) written with `ZAKCODE_*` settings only: `groq/llama-3.3-70b-versatile`
  primary, `openai/gpt-4o-mini` fallback, mode `ask`. Keys-in-env is the canonical
  setup; `.env` keys become *possible* once PR-1 lands `load_dotenv`.
- **D9 (2026-06-10, agent — superseded by D10):** PR-0 was to stay a pure
  zero-behavior-change move with env truth riding PR-1. Owner overrode same day.
- **D11 (2026-06-10, agent — audit reconciliation, supersedes D4 and refines D1/D3):**
  with the full audit doc in hand: (a) `fallback_model` wiring is **internal/omni**
  (audit P0-3b: "interacts with provider selection strategy that may evolve") —
  removed from PR-2; D4's external call was wrong. (b) AUTONOMOUS mode follows the
  audit's written semantics — auto-allow everything; catastrophic patterns escalate
  to ASK (a present prompter may approve; headless fails closed to deny) — near-`allow`
  but it is the documented contract omni's REST/SSE escalation meets; flag the overlap
  in the PR for review. (c) Trust tiers take the audit's shape:
  `tool_trust_overrides: dict[tool_name, PermissionMode]` — per-tool MODE override,
  both directions. (d) Anthropic STATIC metadata (registry entries, panel detection,
  mock-response cost/thinking tests) is key-free and returns to PR-1 satisfying
  acceptance 1/3/4/10; only live-Anthropic items stay parked under D1.
- **D12 (2026-06-10, omni — PR #3 review rulings, all recorded verbatim-in-substance):**
  (a) **AUTONOMOUS semantics: the sharper version wins** — a `DANGEROUS_PATTERNS`
  match in autonomous mode is a deterministic hard DENY, never a prompt, with or
  without a prompter; returned as a structured tool-error the model can adapt to,
  and logged. The distinction vs `allow`: `allow`+prompter can interactively approve
  a catastrophic command; `autonomous` never can. Two invariants: per-tool trust
  overrides cannot loosen the dangerous floor in autonomous mode; persisted grants
  resolve ASK→ALLOW only and never override a DENY. omni corrected the audit doc to
  match; supersedes D11(b). (b) **Grant persistence**: JSON-in-session-doc blessed
  technically (pydantic-default degradation fail-safe — older builds drop grants →
  re-ask, never looser; document it); record shape
  `{tool, args_scope, mode_at_grant, timestamp}`; a session resumed under a tighter
  mode does not honor looser-mode grants. Zachary's formal OK still pending (Q5).
  (c) **Sequencing**: our PR-2 lands first; omni starts internal TurnEnd work after
  it merges. (d) **Unknown #5 fully closed**: `_fails.txt` was a local gitignored
  scratch dump, never repo-tracked; omni corrected the audit, deleted the file; all
  three tests pass everywhere.
- **D13 (2026-06-10, agent — PR #3 scope question resolved by revert):**
  `.env.example`'s uncommented default model reverted to `ollama_chat/llama3.1`
  (matches the code default; fresh-install posture unchanged). Flipping the
  fresh-install default to a cloud model is a product call for Zachary (Q6) — one
  line + a D-entry whenever he wants it. This box's own `.env` keeps Groq primary.
- **D10 (2026-06-10, Zachary):** **the project `.env` is the canonical key store —
  the program must not rely on OS-level environment variables.** Executed on the
  PR-0 branch as its own commit: real key values written into the gitignored `.env`
  (never logged/committed), `load_settings()` now runs `load_dotenv(".env",
  override=False)` (real env still wins when present; missing file is a no-op),
  `.env.example` rewritten as the full documented option surface, `zakcode info`
  panel learns `GROQ_API_KEY`. Proven end-to-end: with OS-level keys stripped from
  the child environment, `zakcode info` reports both keys present and a live
  one-token Groq completion succeeds from `.env` alone. Supersedes D8's
  keys-in-OS-env framing (still true that env vars win if set; they're just no
  longer required).

## Assumptions

- **A1 — RESOLVED 2026-06-10:** litellm's pricing DB fully covers current Groq
  models (14 entries; see Baseline probe). No fallback pricing table; test-only.
- **A2:** The audit's 13 acceptance tests are unavailable; ours are reconstructed
  from the email's named examples (`test_*_registry`, `test_autonomous_mode`,
  `test_rate_limit_retry`). Reconcile names/semantics when the doc arrives (Q1).
- **A3 — CONFIRMED 2026-06-10:** No `_fails.txt` anywhere in the repo, no xfail
  markers, and the full suite is green (see Baseline). Nothing to triage.
- **A4:** omni's parallel internal work = `server/app.py` + loop turn-end region +
  hook ingestion/identity/memory seams. Our changes stay out of those.
- **A5:** GROQ_API_KEY and OPENAI_API_KEY arrive 2026-06-10 (Zachary fetching). All
  PR-1 tests are mock-shaped and key-free regardless; keys only matter for optional
  live smoke (`tests/test_live_provider_smoke.py` pattern: skip-if-unavailable).

## Baseline (PR-0 evidence) — recorded 2026-06-10, Windows 10, py3.11 via uv 0.11.20

ALL GREEN. `uv sync --extra server` then:

| Check | Result |
|---|---|
| `ruff check .` | passed |
| `ruff format --check .` | 195 files already formatted |
| `mypy` | no issues in 96 source files |
| `pytest -q` (main suite) | **1310 passed, 5 skipped**, 38.66s |
| `pytest packages/zds-llm-provider/tests -q` (run manually — uncollected by config) | **92 passed**, 0.57s |

The 5 skips are structural, not failures: 2× Windows symlink-privilege
(`test_builtins_edge.py:80,94`), 3× live-provider opt-ins
(`test_live_provider_smoke.py` — need `LIVE_TESTS=1` + a live model). There are
**zero failing tests** — the audit's "3 known-failing in `_fails.txt`" does not
reproduce here (A3 confirmed; plausibly a misread of the 3 live-provider skips, or
ubuntu-specific). One deprecation warning (starlette testclient/httpx2) — harmless,
worth a dep bump in PR-5.

### litellm Groq pricing probe (audit unknown #1) — RESOLVED 2026-06-10

`litellm.model_cost` contains **14 `groq/` entries** covering the current lineup:
llama-3.1-8b-instant, llama-3.3-70b-versatile, llama-4 scout + maverick, llama-guard-4,
kimi-k2-instruct-0905, gpt-oss-120b/20b/safeguard-20b, qwen3-32b, whisper-large-v3(+turbo),
playai-tts, gemma-7b-it. Spot check `groq/llama-3.3-70b-versatile`: input 5.9e-07,
output 7.9e-07 $/token, 128k in / 32,768 out, `supports_function_calling=True`.
**Conclusion: no fallback pricing table needed** — `_extract_usage`'s
`_hidden_params.response_cost` path should price Groq natively. PR-1 ships a
cost-accounting test instead of a fallback table.

## Code map (verified file:line pointers — compaction survival kit)

| What | Where |
|---|---|
| Capability registry table + lookup order | `src/zakcode/providers/registry.py:17-141` (exact → case-insens. → prefix-stripped → `:tag` base; `ollama/`→`ollama_chat/` aliasing) |
| CLI key panel | `src/zakcode/cli/__init__.py:61` (`_PROVIDER_KEY_ENV`) |
| Permission core | `src/zakcode/permissions.py` — `decide` :346, `authorize` :391 (ask+no-prompter fails closed :417-419), `_MODE_CEILING` :97, blocklist :118, session grants (allow keyed by tool name, deny by tool+args) :276 |
| Settings | `src/zakcode/config.py` — `fallback_model` :53, `permission_mode` :124, list-from-env validator pattern :198, `model_roles` validator to copy :231 |
| Loop provider call sites (no try/except) | `src/zakcode/agent/loop.py:816` (buffered), `:1018` (streaming); stop reasons: completed / max_iterations / doom_loop / stuck / recipe_stalled |
| Error taxonomy | `src/zakcode/providers/base.py` (`RateLimited.retry_after`; was the package's `types.py` pre-ADR-0007) |
| litellm provider | `src/zakcode/providers/litellm_provider.py` — `_map_error` :396, `_extract_retry_after` :356, `num_retries` plumbed-but-0 :161/:440, `_extract_usage` reads `_hidden_params.response_cost` :302-305, `_normalize` drops `reasoning_content` :315-351 |
| Session persistence | `src/zakcode/session/store.py` (schema v1, append-only, atomic save) |
| Agent facade builds policy | `src/zakcode/__init__.py:237` |
| Skills frontmatter parse / serialize | `src/zakcode/skills/__init__.py:65-100` / `:253-267` (unknown keys dropped) |
| Hook events (no turn-end — internal) | `src/zakcode/hooks/__init__.py:40-62` |
| Pytest config (no pythonpath; package tests uncollected) | `pyproject.toml [tool.pytest.ini_options]` |
| CI (ubuntu-only ×3.11-3.13 + SES failure alert) | `.github/workflows/ci.yml` |

## Parked (do when unblocked)

- **Anthropic enablement (blocked on API key, D1):** registry entries
  (claude-opus-4-8 / sonnet-4-6 / haiku-4-5: tools+vision+caching, 200k window),
  `ANTHROPIC_API_KEY` in env/docs/CLI panel, Anthropic thinking-block response tests,
  Anthropic cost-accounting verification. Mostly copies the PR-1 Groq shape.
- **Audit-doc reconciliation (blocked on Q1):** map our acceptance tests onto the 13
  verbatim ones; triage the audit's 5 unknowns list.

## Open questions

- **Q1 — CLOSED 2026-06-10:** audit doc obtained (Mind repo cloned via gh after
  device-flow auth). Reconciled into the Acceptance-test map + D11.
- **Q2 — CLOSED 2026-06-10:** audit answers it — `fallback_model` is internal (P0-3b).
  Removed from PR-2 (D11a).
- **Q3 — CLOSED 2026-06-10:** gh CLI installed + authed as zkysar1 (`repo` scope);
  `gh auth setup-git` wired the git credential helper. Push/PR/private-clone all
  work headlessly now. PR #3 opened.
- **Q4 (Zachary):** when an Anthropic key exists, say so → unpark the LIVE Anthropic
  items (statics already return in PR-1 per D11d).
- **Q5 (Zachary):** approve the grant-persistence format — JSON object inside the
  existing session-store document (audit unknown #4 says principal approves first;
  omni has blessed it technically with the D12(b) constraints).
- **Q6 (Zachary):** should the FRESH-INSTALL default model flip from local
  (`ollama_chat/llama3.1`) to a cloud model (e.g. `groq/llama-3.3-70b-versatile`)?
  Reverted to local for now (D13) — say the word and it's a one-line change.

## Session journal

- **2026-06-10:** Audit email received; every load-bearing claim verified against the
  code (see Code map). Plan drafted, polished with Zachary's corrections (Anthropic
  parked, Groq+OpenAI first, BitNet preserved). This doc created.
  Discovered: provider-package tests uncollected (PR-0 scope); `num_retries` already
  plumbed but defaulting 0 (shrinks PR-2); `_fails.txt` phantom (A3).
  Toolchain: installed uv 0.11.20 standalone (no uv/winget on this box; user PATH
  updated). Baseline recorded: **everything green** — 1310+92 tests pass, ruff/mypy
  clean. Groq pricing probe: litellm DB covers Groq (A1 resolved, no fallback table).
  Mind repo confirmed unreachable with this box's git credentials (Q1 open).
- **2026-06-10 (later):** Env cleanup: Zak-Code had NO `.env` at all (the "divergence"
  was a misremember); keys found as Windows env vars and live-validated (D8); clean
  `.env` written; `load_dotenv` gap discovered → PR-1 scope. zds-llm-provider resolved
  (D7/ADR-0007): consumer scan came back empty, owner delegated the call, **package
  reabsorbed** on branch `pr-0-consolidation` — moves via git mv, imports rewritten,
  tests merged (1403 green), vendor-SDK import-ban contract test added, docs updated
  (ADR-0007 / ARCHITECTURE inventory / ROADMAP note). Bare-pytest proof: suite passes
  with zakcode uninstalled.
- **2026-06-10 (latest):** Owner: no reliance on Windows env vars (D10). Real keys
  moved into `.env`; `load_dotenv` landed in `load_settings()` + 3 tests; `.env.example`
  rewritten as the option reference; GROQ key in the info panel. Live proof with
  OS keys stripped: info panel ✓, one-token Groq completion ✓. Suite: **1406 green**.
  Note for next session: first `git push` attempt hung (no upstream was set) —
  confirm the branch actually reached origin before opening the PR.
- **2026-06-10 (PR #3 review):** omni's verdict: **LGTM pending two amendments** —
  every premise independently reproduced (ADR-0007 evidence, move correctness, test
  arithmetic, Groq pricing to the digit). Amendments landed in this commit:
  RISKS.md row for `.env` keys reaching agent-spawned subprocesses (+ scrub as a
  PR-3 ladder item), ADR-0005→0007 comment fix in the contract test (+ scope note:
  tests/ deliberately unscanned). Scope question resolved by reverting
  `.env.example`'s default model to local (D13/Q6). Rulings recorded as D12 —
  headline: AUTONOMOUS = deterministic hard-deny on dangerous patterns (the sharper
  design wins), PR-2 lands before omni's TurnEnd work, unknown #5 fully closed
  (`_fails.txt` was local gitignored scratch).
- **2026-06-10 (evening):** GitHub access solved: this box had no git-CLI credential
  (the earlier sign-in was GitHub Desktop's own slot). Installed gh 2.94.0 from the
  release zip, device-flow auth as zkysar1, `gh auth setup-git`. **Pushed
  `pr-0-consolidation`, opened PR #3.** Cloned the private Mind repo; read the full
  audit. Reconciled (D11): fallback_model OUT of PR-2 (internal); autonomous-mode
  semantics per audit text; trust tiers as `tool_trust_overrides`; Anthropic statics
  back into PR-1 (key-free). Added the Acceptance-test map. Unknown #5's three
  named tests pass on Windows — ubuntu CI on PR #3 is the cross-platform probe.

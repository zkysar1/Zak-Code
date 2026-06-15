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

- **Date:** 2026-06-10 (final update — ladder complete)
- **State:** the FULL external work package is implemented, fresh-eyes-reviewed per
  phase, and open as a stacked PR chain: **#3** (consolidation + env truth, omni
  LGTM'd + amendments landed) → **#4** (tri-provider metadata) → **#5** (provider
  resilience) → **#6** (autonomous permissions) → **#7** (skill extras + logging) →
  **#8** (tooling). Merge in order; GitHub auto-retargets each as its base merges.
  Suite: **1453 passed**; `uv run poe check` green end to end.
- **All 13 audit acceptance criteria implemented** (see map below — every named
  `-k` selector passes by name; 4 and 11-13 hold by construction/tests).
- **One pending owner action:** pushing the P2-4 CI change (windows cell +
  coverage artifact, local commit on `pr-5-tooling`) needs the `workflow` OAuth
  scope — agent correctly cannot self-grant; Zachary runs
  `gh auth refresh -h github.com -s workflow` (one device code), then `git push`.
- **Waiting on:** PR reviews (omni/Zachary); the workflow-scope grant
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
| PR-1 | Provider metadata: Groq + OpenAI + **Anthropic statics** (registry entries, key panel, `.env.example` lines — key-free, satisfy acceptance 1/3/4), response-shape tests (Anthropic thinking / `reasoning_content`, Groq usage), mock cost-extraction tests (acceptance 10) | M | **implemented** (branch `pr-1-provider-metadata`, stacked on PR-0) |
| PR-2 | Provider-failure resilience: RateLimited retry w/ backoff + graceful `provider_error` stop. **fallback_model wiring REMOVED — audit assigns it internal (P0-3b)** | M | **implemented + fresh-eyes reviewed** (PR #5; review fixes in `8f8c245`: streaming refund symmetry, `error` on AgentDone/ChatResponse, per-attempt accumulators, retry-layering docs) | 
| PR-3 | `PermissionMode.AUTONOMOUS` (D12 hard-deny semantics) + `tool_trust_overrides` + grant persistence (Q5 approved) + subprocess provider-key env scrub w/ opt-out | M-L | **implemented** (branch `pr-3-autonomous-permissions`; 13 new tests incl. acceptance names `test_autonomous_mode` / `test_trust_tiers` / `test_grant_persistence`; 1442 green). Implementation notes: effective-mode = per-tool override else session mode; autonomous (session OR per-tool) → dangerous = hard DENY and confirm_tools fail closed; grants re-decide so they can never override a static DENY; restore filters by `_MODE_LOOSENESS` rank (deny grants always kept); scrub list = `secrets.provider_key_env_names` (exact names + `*_API_KEY` suffix), applied LAST in `_proc.run_capturing` via `ToolContext.scrub_env`; RISKS row → Mitigating |
| PR-4 | Skill-frontmatter extras preservation + logging instrumentation | S-M | **implemented** (branch `pr-4-skills-logging`; acceptance 9 `test_skill_extras` passes by name; extras round-trip through `save_skill`; logging = targeted not exhaustive (D16): registry.execute traceback (the biggest silent swallow), permission denials w/ mode, loop iteration/turn-end lines, provider call latency+tokens+cost at debug — never message contents; 1450 green) |
| PR-6 | **PKG-AUTO** (omni's spec, 2026-06-10; Q6 resolution): `default_model: "auto"` sentinel — startup detection (cheap read-only probes: `/api/tags`, `/v1/models`; never a chat call) + cached, re-probed on failure; resolution = local if viable, else first viable external per configurable preference list (default groq → openai → anthropic), nothing viable → loud startup failure with key-panel diagnosis; resolution logged + in info panel with reason; runtime re-resolution (once per turn) on non-rate-limit ProviderError before the provider_error stop; **fallback_model RELEASED to external** (supersedes D11a) as the explicit-config override of the auto chain; resolver architected as a pluggable interface `(task category, capabilities) → model` with v1 = availability only — Zachary's "zakpick" vision (deep-think/quick-classify/embeddings/planner/coder/writer routing) must land later without API breakage; mocked-detection acceptance matrix + mid-session-fallback + explicit-bypass tests; consider cost-ceiling config; rider: quiet litellm botocore import warnings deliberately | M-L | **implemented** (branch `feat-pkg-auto`, PR #17 — see the 2026-06-11 D21 entry below for deltas vs spec) |
| PR-5 | P2 tooling: coverage, version-sync test, task runner, Windows CI cell, docs/CONFIG.md | S (batched) | **implemented** (branch `pr-5-tooling`): `poe check` one-command gate (acceptance 12), `poe cov` + CI coverage artifact (acceptance 13), `test_version_sync` (P2-2), windows-latest 3.11 CI cell with job-level timeout (GNU `timeout` absent there — split pytest steps), `docs/CONFIG.md` + BOTH-direction completeness tests (fields↔doc), CLAUDE.md gains the one-command gate. **Deferred:** the starlette/httpx2 testclient deprecation warning (test-only, harmless; blind dep churn in the last phase loses) — D17. 1453 green |

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

## Parity backlog (PKG-PARITY) — added 2026-06-11

A second, broader engagement: a full fresh-eyes parity review of Zak-Code against
**claw-code (Claude Code), Hermes, and goose** — not just the Claude-Code surface in
PARITY.md. Method: a 104-agent / ~5.7M-token workflow (8 lanes × 4 codebases → gap
synthesis → adversarial verification against the real code → one plan). **Full verdict,
themes, the 30-item verified backlog, and the "already at-or-ahead, do NOT rebuild" list
live in [`PARITY-GAP-ANALYSIS.md`](PARITY-GAP-ANALYSIS.md)** (D25). Headline: core
engineering is at par-or-ahead of all three; the gaps are *delivered breadth*
(provider-resilience wiring + operability surface), not architecture.

**P1 quick-reference (11; full detail in the parity doc):** (1) failover +
`ContextWindowExceeded` compact-then-retry — *failover half likely already done by
PKG-AUTO #17; re-verify, only the compact-retry sub-item should remain*; (2) emit
`cache_control` + cache-token accounting; (3) headless `zakcode run` with `--json`; (4)
cost/token budget stop; (5) `finish_reason` truncation continuation; (6) normalize commands
before blocklist match; (7) strip invisible Unicode at untrusted boundaries; (8)
overflow-to-file + per-turn output budget; (9) context-overflow progressive prune; (10)
wire CLI session resume + `/fork`; (11) `settings.json` merge + profiles `[CLEAN-ROOM]`
— *note: per-user config home `~/.zakcode` already merged (#16), which may cover part of
this; re-verify*.

> **Freshness:** the review snapshotted the `pr-5-tooling` tree; PKG-AUTO (#17), TurnEnd
> (#12/#18), provider-retry fixes (#13/#15), and the UX track (#9-#11) merged afterward.
> Items #1 (failover) and #30 (outer-loop continuation) are likely partly addressed —
> re-verify against `main` before building. Everything else stands. Coordinate with omni.

## Resilience cluster engagement (PKG-PARITY P1, 2026-06-11)

After PR #19 (parity analysis), Zachary commissioned the **provider-resilience cluster**
(the highest-ROI P1 dead-wiring). **Re-verified against current `main` first** (a
6-agent workflow) — main had moved far ahead, so the scope shrank:

- **#1 failover — DROPPED, already shipped by PKG-AUTO #17** (`Agent._model_failover`:
  `fallback_model` override + auto re-resolution). Left untouched; my work *guards* it.
- **Built (branch `resilience-cluster`, PR pending):** #1b ContextWindowExceeded
  compact-then-retry (caught above the failover branch so an overflow never mis-routes
  into model-switching); #2 `cache_control` emission (Anthropic-only) + cache-token
  accounting in `Usage`; #4 cost/token budget stop (shared-budget ceilings →
  `budget_exhausted`, non-vetoable); #5 `finish_reason` length continuation (bounded,
  flags `degraded`, before TURN_END, no veto-budget draw).
- **37 new tests; 1596 green** (clean env). Fresh-eyes review: 33 candidates → **0
  blocking/major**; actioned the CONFIG.md drift + a streaming-budget test gap + a
  recovery diagnostic-log; deferred a strict context non-progress guard (D23) and the
  buffered/streaming DRY refactor (consistent with the file's established twin structure).
- **Baseline note (flag for omni):** `tests/test_model_auto.py` (PKG-AUTO #17) fails on a
  dev box with a local `.env` because it doesn't isolate `ZAKCODE_FALLBACK_MODEL`; it is
  CI-green. Not touched (omni's file); validated my work in a clean env. A 2-line
  `monkeypatch.delenv` per test would harden it.

- **D23 (2026-06-11, agent):** deferred the strict "only retry if the compacted prompt is
  strictly smaller" guard on ContextWindowExceeded recovery (review #1, rated
  robustness-only). The `_MAX_CONTEXT_RECOVERY=2` bound + graceful `provider_error`
  degradation already guarantee termination; a strict guard tangles with the cap-test
  semantics for marginal gain. Took the cheap half — a before/after message-count
  diagnostic log — so an overflowing-summary edge case is still observable.

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
- **D14 (2026-06-10, Zachary):** blanket approval of all standing recommendations
  ("fully approved") — closes **Q5**: grant persistence as a JSON object in the
  session document is formally approved (with omni's D12(b) constraints). Also a
  standing goal: complete the whole ladder autonomously, fresh-eyes review between
  phases, agent makes long-term strategic decisions.
- **D15 (2026-06-10, agent — PR-2 design):** retry ONLY `RateLimited` (waiting is the
  documented remedy for a 429; retrying auth/context/generic failures wastes spend and
  masks bugs); backoff = `retry_after` when given else `1s·2^(attempt-1)`, clamped to
  `[0, 30s]` so a hostile Retry-After can't stall a turn; **mid-stream failures are
  never retried** (deltas already reached the client — a retry would duplicate them);
  partial streamed text of a failed turn is NOT persisted (session stays at the last
  message boundary); `TurnResult.error` carries the redacted detail; streaming
  surfaces failure as `AgentStatus` (AgentDone schema unchanged — client contract).
- **D18 (2026-06-10, omni stack review + restack):** verdict clean — zero
  blocking/major across #4–#8; #6 security core independently attacked and held.
  Mechanical restack onto main executed (PR #3's squash made the old base
  unreachable for clean merges); all five branches rebased + force-pushed. All
  nine minors folded into their home PRs: (1) `auto_allows` now mirrors
  authorize's grant re-decide; (2) `provider_key_env_names` exported; (3) AWS
  creds decided **keep-narrow** — the scrub targets model-provider keys whose env
  presence is zakcode's own doing; AWS creds are operator-managed workflow
  credentials, agent-run `aws` CLI is first-class here, and the opt-out is global
  (rationale in the docstring); (4) opus-4-8 `max_output` pinned to the no-header
  64k; (5) qwen3-32b `max_output` pinned to Groq-documented 40,960 (litellm DB
  conflates it with the window — re-probed, still 131k/131k); (6) streaming
  reasoning-drop documented as deliberate (no StreamThinkingDelta yet); (7) refund
  comment states the real invariant (partial output discarded); (8) mid-stream
  refund test added; (9) save_skill boolean named. Suite: **1455 green** at tip.
- **D19 (2026-06-10, omni — Q6 resolved → PKG-AUTO):** Zachary's Q6 answer became
  the next external package (see PR-6 ladder row for the full spec). Two contract
  points to honor: **fallback_model is now external** (omni released P0-3b — wire
  it as the explicit override of the auto chain; supersedes D11a), and the
  resolver must be a pluggable interface so "zakpick" (task-category model
  routing) lands later without API breakage. Sequencing: restack → omni merges
  the stack → PKG-AUTO starts → omni starts the internal TurnEnd seam post-#5.
- **D25 (2026-06-11, Zachary → agent, ultracode):** commissioned a full parity review
  of Zak-Code vs **claw-code + Hermes + goose** ("get to par with these three
  harnesses"). Ran as a 104-agent / ~5.7M-token workflow; output is
  `docs/PARITY-GAP-ANALYSIS.md` + the *Parity backlog* section above. Verdict:
  at-par-or-ahead on core engineering, behind on provider-resilience wiring and
  operability breadth; 30 verified items (11 P1), 18 already-ahead. Originally
  numbered D22 to clear omni's commit-referenced D20 (per-user config home, #16) and
  D21 (PKG-AUTO, #17); renumbered D22→D25 in the 06-12 collision cleanup — three
  entries claimed D22, and #21's claude-polish keeps it because its squash-commit
  subject (5fbfb21) cites D22 immutably; omni's TurnEnd entry became D24.
  Clean-room rule enforced on every claw-code reader; `[CLEAN-ROOM]` items must be
  re-expressed, never copied; study material extracted to
  `C:\ZakNoCloud\_zakcode_research\` (read-only, gitignored, never in-repo). The
  review snapshotted the pre-merge `pr-5-tooling` tree, so parity-#1 (failover) and
  parity-#30 (outer-loop continuation) are likely partly addressed by the
  since-merged PKG-AUTO (#17) and TurnEnd (#18) — flagged in the doc; re-verify
  before building; coordinate with omni so the `fallback_model` seam isn't wired twice.
- **D16 (2026-06-10, agent — PR-4 logging scope):** the audit's P1-5 names "67 bare
  except handlers"; instrumenting all 67 mechanically would add noise without value.
  Delivered the TARGETED set instead: `registry.execute`'s wrapped tool exceptions
  (the largest silent swallow — operator now gets the traceback while the model
  still sees the recoverable error), permission denials (warning, with mode +
  reason — also satisfies D12's "log the autonomous hard-deny"), loop iteration /
  turn-end lines (debug/info), and provider call accounting (model, latency, token
  counts, cost — message contents never logged). The remaining handlers are
  best-effort-by-design paths (hooks, compaction, lessons) that already log.
- **D17 (2026-06-10, agent — PR-5 scope):** task runner = **poethepoet** (Make is
  hostile on Windows — the primary dev box; `just` adds a non-Python install; poe
  rides the existing uv dev group). The starlette/httpx2 testclient deprecation
  warning is DEFERRED: it is test-only and harmless, and a blind dependency bump in
  the final phase risks more than it fixes — revisit when the server deps are next
  touched. CI coverage uploads from one cell (linux/3.11) — an artifact per cell
  adds noise, not signal.
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
- **Q5 — CLOSED 2026-06-10 (D14):** Zachary approved the grant-persistence format
  (JSON in the session document, with omni's D12(b) constraints).
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
- **2026-06-10 (ladder complete):** PR-3 implemented + reviewed (4-angle incl.
  security bypass hunter: zero bypasses; fixes: restore dedup, doc truth, drift
  guards — `1445` green). PR-4 implemented + reviewed (fixes: save_skill tool
  extras passthrough, library NullHandler so unconfigured CLIs see no stderr spam —
  `1450` green). PR-5 implemented (`poe check`/`poe cov`, version-sync test,
  CONFIG.md with two-direction completeness tests — `1453` green); its P2-4 CI
  commit is LOCAL-ONLY pending the `workflow` OAuth scope (the auto-mode
  classifier rightly blocked agent self-escalation; owner grants it). Final state:
  six stacked PRs #3-#8 = the complete external package; all 13 acceptance
  criteria implemented.
- **2026-06-10 (PR-2):** Implemented on `pr-2-provider-resilience` (stacked on PR-1):
  `Settings.provider_max_retries` (default 3); `AgentLoop._call_provider` retries
  RateLimited with `retry_after`-aware capped backoff at the buffered site; streaming
  path retries only BEFORE the first event (mid-stream = terminal, no duplicate
  deltas); any surviving `ProviderError` → `stop_reason="provider_error"`,
  `TurnResult.error`, `degraded=True`, `AgentStatus` on the stream, session left at
  the last message boundary. 9 new tests (acceptance #8 `test_rate_limit_retry` by
  name); design rationale in D15. Suite: **1425 green**. loop.py diff confined to
  the two call sites + helpers, per the omni sequencing agreement.
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
- **2026-06-10 (PR-1):** Implemented on `pr-1-provider-metadata` (stacked on PR-0):
  4 Anthropic + 4 Groq registry entries (Claude windows pinned at the standard 200k,
  NOT litellm's 1M beta — no beta header is sent, and the acceptance test agrees);
  `ANTHROPIC_API_KEY` in the key panel; Anthropic lines in `.env.example`;
  `LLMResult.thinking` captures litellm's `reasoning_content` (kept out of `text`;
  streaming reasoning deltas yield no text events); 10 new tests incl. the audit's
  verbatim names — `pytest -k "test_anthropic_registry or test_groq_registry or
  test_provider_key_status or test_anthropic_cost"` → 4 passed. Suite: **1416 green**.
  Acceptance 1/2/3/4/10 done; audit unknown #3 partially answered: litellm DOES
  normalize thinking to `reasoning_content`, now captured (loop persistence of
  ThinkingBlock deliberately deferred — flag for omni/owner if wanted).
- **2026-06-10 (evening):** GitHub access solved: this box had no git-CLI credential
  (the earlier sign-in was GitHub Desktop's own slot). Installed gh 2.94.0 from the
  release zip, device-flow auth as zkysar1, `gh auth setup-git`. **Pushed
  `pr-0-consolidation`, opened PR #3.** Cloned the private Mind repo; read the full
  audit. Reconciled (D11): fallback_model OUT of PR-2 (internal); autonomous-mode
  semantics per audit text; trust tiers as `tool_trust_overrides`; Anthropic statics
  back into PR-1 (key-free). Added the Acceptance-test map. Unknown #5's three
  named tests pass on Windows — ubuntu CI on PR #3 is the cross-platform probe.
- **2026-06-11 (omni):** Post-#9 spec-consistency fix from the PR #9 review record
  (review posted after the merge — found nothing blocking): the terminal tool-wait
  spinner showed the bare verb (`read...`) while UX.md and the web client both
  specify `running read...`. render.py now passes `"running " + verb`;
  `test_spinner_label_matches_ux_spec` pins all three surfaces in agreement.
  Suite: **1466 green**, ruff + mypy clean.
- **2026-06-11 (omni):** Internal TurnEnd package, loop.py-free rungs (PR-T1/T5/T6/T7
  of the ratified TurnEnd Seam Design v1.0): `HookEvent.TURN_END` + veto-capable
  dispatch speaking Claude Code's Stop-hook wire protocol (decision-block JSON on
  exit 0, native exit-2, fail-open on timeout/crash/non-zero, process-tree kill,
  provider-key scrub); settings.json hook ingestion (`Stop`->`TurnEnd` mapping,
  DANGEROUS_PATTERNS hard-deny in autonomous mode, key scrub on ALL workspace hooks,
  appends to a passed hook_manager); `agent_identity_dir` identity discovery for
  `agents/<agent>/self.md`; SESSION_END/PRE_COMPACT payloads enriched with
  session_summary. Omni review amendment: settings.json ingestion is OPT-IN
  (`ZAKCODE_SETTINGS_HOOKS`, default false) instead of CLI-unconditional -- a
  workspace configured for Claude Code would otherwise have its hooks half-fire
  here with a different stdin schema before the T2/T3 loop integration lands.
  Veto dispatch is inert until T2/T3 (deferred behind PKG-AUTO to avoid loop.py
  collisions). Suite: **1504 green**, ruff clean, mypy at main parity.
- **2026-06-11 (omni, D20):** Per-user config home `~/.zakcode` (issue #14, PR #16 —
  spec omni, implementation dev). One obvious place per the `~/.claude` precedent;
  v1 = a single user-level `.env`; precedence defaults -> user .env -> workspace
  .env -> process env (workspace loaded first, then user, both override=False);
  `ZAKCODE_HOME` overrides the directory; the config home is never a workspace
  root (test-pinned). Key panel now names each key's SOURCE (env / workspace .env /
  user .env) — values never shown. RISKS.md notes the wider user-level blast
  radius; the #6 subprocess scrub covers user keys unchanged. Rider landed for
  real: `LITELLM_LOG` default moved to `providers/_env.py`, imported first by the
  providers package `__init__` — the one place guaranteed to run before litellm
  (the load_settings placement we first agreed on provably fired too late).
  Suite 1518 green.
- **2026-06-11 (dev, D21 — PKG-AUTO implemented; rulings from omni's relay):**
  `default_model: "auto"` lands per the PR-6 row (PR #17, entry drafted by
  implementer per the new log convention). Shape: `providers/resolve.py` —
  `AvailabilityResolver` behind a pluggable `ModelResolver` protocol
  (`resolve(task, require_tools=...)`; v1 ignores `task` — zakpick's seam);
  read-only probes (`/api/tags`, `/v1/models`) with a process-lifetime cache,
  probed fresh on the failover path; local wins, then `auto_model_preference`
  (new Settings field, default groq->openai->anthropic); nothing viable raises
  `ModelResolutionError` whose message IS the per-source diagnosis incl. key
  provenance from D20's `env_source`. **Omni ruling folded in:** tool reliability
  is capability metadata — `Capabilities.tools_unreliable`, set on
  `groq/llama-3.3-70b-versatile` (#13 root cause); the resolver skips
  tools-unreliable models whenever tools are required, which lands gpt-oss-120b
  first within groq without a hardcoded sort. **failed_generation salvage:
  parked** (omni ruling — fragile coupling for marginal gain). Runtime: loop ctor
  gains `model_failover` (the ONE loop.py seam, flagged for T2/T3 planning: both
  paths' `except ProviderError` sites ask the callback once per turn, streaming
  only before any event reached the client); `fallback_model` is the explicit
  override of the chain, else auto re-resolution excluding the failed model.
  Deltas vs spec: out-of-the-box default stays `ollama_chat/llama3.1` ("auto" is
  opt-in — flipping the default is a one-line decision left open deliberately);
  cost-ceiling config deferred ("consider" in spec; no consumer yet). Rider
  (litellm warnings) had already landed in #16.
- **2026-06-11 (omni, review fixes on #17):** streaming failover now resets the
  RateLimited retry budget for the replacement provider (buffered-path parity —
  `_call_provider` resets its attempt counter per call; the streaming twin's
  counter is outer-scoped and survived the failover `continue`); `.env.example`
  llama-3.3-70b annotation corrected to tools-UNRELIABLE (it contradicted this
  PR's own registry marking) with gpt-oss-120b added as the reliable groq
  example; stray UTF-8 BOM stripped from this file. Everything else verified
  clean: spec 10/10 clauses, loop seam 7/7 safety points, hermeticity (incl.
  the construction-time probe binding), key handling. 1541 green.
- **2026-06-11 (omni, D24 — TurnEnd T2/T3/T4: loop break-site veto gates;
  logged as D22 at merge, renumbered in the 06-12 collision cleanup):**
  the Stop-hook seam goes live in the loop. `AgentLoop` ctor gains
  `turn_end_veto_budget: int = 0`; at the three VETOABLE break sites
  (`completed` / `doom_loop` / `stuck`, both paths) a new `_fire_turn_end`
  runs TURN_END hooks (T1's runner) and, on veto, injects the hook's
  continuation prompt as a control-rail user message and re-enters the loop —
  at most budget times per turn. `max_iterations` / `provider_error` /
  `recipe_stalled` are never vetoable (hard bounds / the recipe gate's own
  bounded give-up). Veto resets the stall trackers (doom signature+count,
  `StuckTracker.reset()` from T1) so pre-veto repetition can't instantly
  re-trip; `RecipeCursor` is deliberately NOT reset. One wrinkle the design
  ladder missed: at the doom-loop site the repeated batch's tool_use blocks
  are already in the session UNEXECUTED — re-entering would send a dangling
  tool_use to strict providers, so the gate first answers each with a
  synthetic error tool_result (`data.doom_loop_intervention`) before the
  continuation prompt. Streaming twin yields
  `AgentStatus("turn_end hook vetoed stop; continuing")`. Fail-open
  everywhere: budget 0 (the default) short-circuits before payload build —
  byte-identical pre-T2 behavior — and a crashing hook run allows the stop.
  T4: `Settings.turn_end_veto_budget` (`ZAKCODE_TURN_END_VETO_BUDGET`,
  default 0) threaded Agent→loop; CONFIG.md row. Plus
  `HookManager.has_turn_end_hooks()` (matcher-agnostic cheap pre-check).
  15 new tests (budget-zero parity, allow/veto-once/budget-exhausted,
  non-vetoable trio, doom-veto pairing+reset double-threshold proof, payload
  contents, streaming status + budget-zero, fail-open crash, env/Agent
  plumbing). Suite 1556 green.
- **2026-06-11 (dev, D22 — frontend visual overhaul, "claude-polish"; keeps D22
  in the 06-12 collision cleanup — commit 5fbfb21's subject cites it immutably):** Zachary
  directed a ground-up restyle of both clients to reach visual parity with the
  reference harnesses (Claude Code public look, goose, hermes-agent — studied
  clean-room: goose/hermes from their public repos, Claude Code from its
  publicly observable rendering only; a local "from leaked.zip" claw-code copy
  was explicitly NOT used, per the charter). Design produced by a 13-agent
  fan-out (6 recon reviewers -> 3 competing specs -> 3 judges -> synthesis;
  winner "claude-polish" 160 vs 136.5 vs 131), implemented by 3 builders from
  the pinned spec. Shape: one azure spark (color(38)) as the entire brand;
  `●` block / `└` receipt grammar with hanging indents and a `│` rail binding
  result bodies (red under failure); injectable-clock durations on every
  receipt and a state-colored turn receipt replacing the footer rule; painted
  diff bands; head+tail run truncation; welcome + permission as the only two
  boxes (numbered options, humanized tiers); grouped /help; REPL-owned gerund
  wait line (conhost/off-tty/ZAKCODE_NO_SPINNER disabled); web client rebuilt
  on the identical grammar (44rem column, tool cards with pending/orphan/
  abandoned lifecycle, stream-status overlay, scoped y/a/n approvals with
  consent receipts, dark+light). docs/UX.md rewritten as the binding
  cross-client contract incl. the shared-grammar mapping table. Integration
  fixes: spec's invalid light-mode hex (#ecease5 -> #eceae5), ARCHITECTURE
  footer wording, orphaned suspend_live deleted. Suite 1592 green.

- **2026-06-12 (dev, D26 — self-remediation Step 1: declared-vs-undeclared dependency
  gate):** built the first concrete step of the SELF-REMEDIATION roadmap (D-doc PR #25),
  on which this PR stacks. New pure module `src/zakcode/deps_gate.py` — `installed_specs`
  parses a shell command into the package identities it would EXPLICITLY install, and
  `read_declared_packages` reads the project's declared set from `pyproject.toml`
  (project + optional + PEP 735 groups + poetry), `uv.lock` (full resolved set),
  `requirements*.txt`, and `package.json`. Wired into `PermissionPolicy.decide` as a
  **tighten-only** check placed AFTER the dangerous-pattern floor: an install of a package
  no manifest declares escalates ALLOW→ASK, and is a deterministic hard DENY in
  `autonomous` (session-wide or per-tool) — there is no execution sandbox yet, so an
  undeclared install can't auto-run headless. Declared installs, `uv sync`/`npm ci`, and
  editable/local installs (`pip install .`/`-e .`/`-r req.txt`) pass through untouched;
  URL/VCS installs are always treated as undeclared. **Key design call:** the parser is
  *launcher-aware* — it sees through `python -m pip install`, `uv pip install`,
  `uv run pip install`, a leading PowerShell `&`, and a full-interpreter-path invocation —
  precisely the shapes the project's OWN `pip_install_hint` (D-fix-install-hints / PR #22)
  emits, so the self-fix path it was built to enable cannot dodge its own gate by spelling
  the install differently. Gated on an injected `declared_packages` callable (default
  `None` → OFF, so the pure decision matrix is unchanged for every existing caller); the
  Agent facade wires it lazily from `workspace_root` when the new `dependency_gate` setting
  (default **on**) is true. The manifest read happens only when a command actually names an
  install, and a read failure fails toward ASK, never a crash. The suite caught two real
  parser bugs during the build (npm scoped names `@scope/pkg` were dropped by the local-path
  guard; a marker fragment leaked a stray quote into a name), both fixed. Docs travel with it:
  CONFIG.md row, `.env.example` knob, RISKS supply-chain row upgraded, SELF-REMEDIATION Step 1
  marked ✅ SHIPPED.
  **Fresh-eyes adversarial review round (per the standing per-phase-review rule) surfaced and
  fixed 7 more:** (F1) modern Poetry 1.2+ `[tool.poetry.group.*.dependencies]` weren't read;
  (F2) the `requirements*.txt` glob was root-/prefix-only — broadened to `*requirements*.txt`
  + `requirements/*.txt`; (F3) `-r`/`--requirement` includes are now followed (cycle-guarded);
  (F4) a non-string `name` in `uv.lock` raised `AttributeError`, breaking the "never raises"
  contract — now type-guarded; (F5, the one over-PERMISSIVE hole) the flat pip+npm declared
  set let an npm-declared name vouch for a PyPI install of the same string — fixed by
  **ecosystem-tagging** every identity (`pypi:`/`npm:`) end-to-end so a name only vouches
  within its own ecosystem; and (the security-relevant one) a blanket **session ALLOW grant**
  on `bash` re-checked only the dangerous floor + static DENY, not the gate — so an undeclared
  install could ride a prior grant unprompted in interactive modes; the gate is now
  un-waivable by a grant (re-decides in `authorize()`/`auto_allows()`, mirroring the dangerous
  floor). Deferred F6 (per-command manifest re-parse has no cache — minor perf; the re-read is
  intentional for freshness, so a naive cache would risk staleness).
  **Round-2 review (after the session-limit reset) re-ran the 3 cut-short angles and found 8
  more, all fixed:** (A) `uv tool install`/`uvx`/`uv tool run`, (C) versioned pip `pip3.11`,
  (E) interpreter flags before `-m` (`python -E -m pip install`), (D) `env`/`FOO=bar` env-prefix,
  (B) lone `&` background + `(…)`/`{…}` subshell separators — five **blocking** parser bypasses
  where an undeclared install ran UNPROMPTED; (F, blocking) the PreToolUse hook-rewrite seam
  re-checked only the dangerous floor, not the gate — added `undeclared_install_reason()` + a
  loop re-check mirroring audit3 #5; plus two **major** false-positives (G unknown value-bearing
  flags, H trailing `#` comments) that over-blocked legitimate declared installs. **Architectural
  call:** reliably parsing arbitrary shell is unsound, so the gate is documented as **best-effort
  defense-in-depth** — it closes the common/natural spellings (high value, no model), and
  deliberate obfuscation (nested `bash -c`, `eval`, base64) is by design contained by the Step 3
  sandbox, not chased in the parser.
  **Round-3 (convergence) found the position-locked parser still leaked on COMMON spellings, so
  it was restructured, not patched:** global/pip-level flags before the verb (`pip -i <index>
  install`, `uv -q add`, `python -m pip --no-cache-dir install`), Windows backslash venv paths
  (`.venv\Scripts\pip install` — posix `shlex` ate the backslashes), and a `--save-exact`/
  `--prefer-dist` regression (boolean npm flags I'd wrongly put in the value-flag list).
  Fix = **verb-keyword scanning** (find the `install`/`add` verb by scanning, not a fixed slot —
  so a flag can't push it out of position; unknown value-flags now degrade to a *safe
  false-positive*, never a miss) + **`posix=False` tokenizing** (matches `agent/recipe.py`, so
  Windows paths survive) with quote-stripping in `_basename`. This is the principled structural
  fix, not more special cases. **Round-4 (final pre-merge, Zachary-requested) found two last
  common-spelling bypasses, both fixed:** `uv run --with <pkg>` (fetches+runs an ephemeral PyPI
  dep — the `uv run` branch only handled `uv run pip install`, while sibling `uvx --with`/`uv
  tool run --with` were caught — an inconsistent gap, now gated like uvx); and npm `-p`/`-d`
  (lowercase shorts that pip/uv use as value flags but npm treats as BOOLEAN, so `npm install
  -p <pkg>` swallowed the package — fixed by splitting value-flags into ecosystem-specific sets
  `_VALUE_FLAGS`/`_NPM_VALUE_FLAGS`). 1781 suite green (clean env), ruff+mypy clean. Four review
  rounds total hardened the parser; the residual is the documented best-effort boundary (Step 3
  sandbox). **MERGED to main** (PR #25 doc then PR #26 gate).
  Next on the roadmap: Step 2 (autonomy breadth-downgrade +
  protected-path floor), then Step 3 (the real Executor sandbox — the precondition for
  trustworthy autonomy).

- **2026-06-12 (dev, D27 — self-remediation Step 2: un-waivable protected-path floor):** built
  Step 2 on Zachary's "move on to step two". New tighten-only floor in `PermissionPolicy`
  (`PROTECTED_PATH_PATTERNS` + `_protected_path_reason`, a `decide()` block after the dependency
  gate): a WRITE whose path arg matches a sensitive location — `.git/`, `.env` (not
  `.env.example`/`.gitignore`/`.github/`), the venv (`.venv`/`venv`/`site-packages`), or the
  agent's own `.claude/` config — never auto-allows. Escalates to a (session-grantable) prompt
  under `allow`/`acceptEdits`; **hard DENY in `autonomous`**; un-waivable by a session grant OR a
  per-tool trust override (the `authorize()`/`auto_allows()` grant fast-paths re-decide; loose
  modes still hit the floor in `decide()`) AND re-applied to a PreToolUse-hook-rewritten call
  (`protected_path_reason()`, mirroring audit3 #5). This realises BOTH halves of the Step 2 spec:
  the protected-path floor, and the "breadth-downgrade" (a blanket grant/override can no longer
  carry sensitive-write breadth past the floor — closing self-permission-escalation via
  `.claude/`, secret-rewrite via `.env`, dependency-tamper via the venv, repo-corruption via
  `.git/`). `child_view` propagates the combined pattern list; operator extras via the new
  `protected_paths` setting (`compile_protected_paths`, wired in the facade). **KEY SCOPING
  DECISION (the Step-1 lesson, reapplied):** the floor scans the write/edit tools' PATH arg, NOT
  shell commands — a path inside an arbitrary command is usually a read/execute (`.venv/bin/python
  -m pytest`, `cat .git/config`), so scanning shell over-blocks (it broke `test_agent_e2e` on the
  first cut); shell-driven writes to protected paths are best-effort residual the Step 3 sandbox
  contains. 38 new tests (`tests/test_protected_paths.py`); 1819 suite green (clean env),
  ruff+mypy clean. Docs: SELF-REMEDIATION Step 2 ✅ SHIPPED, CONFIG.md + `.env.example`
  (`protected_paths`). Next: fresh-eyes review → PR. Then Step 3 (the Executor sandbox — the big
  one, the precondition for trustworthy autonomy that makes every best-effort residual above
  survivable).

- **2026-06-14 (dev, branch `claude/sdk-task-decomposition-024ug1` — first-layer task
  decomposition / HTN planning):** built the missing proactive-planning layer on the owner's
  ask to bolster near-term task decomposition (keep the domain-agnostic core + skills
  abstraction; this is the "what's next" layer, NOT ayoai-mind's long-horizon planning). Full
  rationale + owner-chosen forks in **ADR-0008**. Summary for a future agent:

  - **`src/zakcode/tasks.py` (NEW, pure, top-level next to `messages.py`/`usage.py`):**
    `TaskNetwork` + `Task`. Compound (decomposed; status DERIVED from children) vs primitive
    (actionable). `normalize()` (called by the tool every edit) assigns dotted ids, **enforces
    single-focus BEFORE deriving parent status** (ordering matters — see the fresh-eyes bug
    below), and derives compound status bottom-up. `current()`/`leaves()`/`actionable_remaining()`/
    `progress()`/`is_complete()`/`render()`. Placed top-level (not in `agent/`) to avoid an
    `agent → session` import cycle, since `Session` holds it.
  - **`src/zakcode/tools/builtins/update_plan.py` (NEW, aliases `plan`/`todo`):** the model's
    TodoWrite-style full-replace handle. `kind` is INFERRED from presence of `subtasks` (depth
    capped at 3 via explicit nested schema — no recursive `$ref`). READ_ONLY (planning never
    gated) + NeverParallel. Mutates `ctx.task_network` in place (the loop persists it). Registered
    in `default_registry.py`.
  - **`src/zakcode/tools/base.py`:** `ToolContext.task_network` (the injection seam, like `spawner`).
  - **`src/zakcode/session/store.py`:** `Session.task_network` — append-only on schema v1 (no
    version bump; same pattern as `permission_grants`; older builds drop it / re-derive).
  - **`src/zakcode/agent/loop.py`:** (1) `_messages_for_call` re-injects the live plan as an
    ephemeral, non-persisted tail message (LAST = highest salience) — fulfils the previously
    aspirational re-injection claim; (2) `_plan_gate_nudge` + `_MAX_PLAN_NUDGES=2`: a bounded,
    self-arming completion gate (sibling to the recipe gate) — won't finish with open steps,
    then completes `degraded` rather than deadlock; (3) `_reset_completed_plan` at turn start
    drops a FINISHED plan so it doesn't bleed into the next turn (an unfinished plan persists);
    (4) `_task_update_event` emitted on the streaming path when the plan render changes. Wired
    on BOTH `_run_turn` and `astream_turn`.
  - **`src/zakcode/events.py`:** `AgentTaskUpdate` added to the `AgentEvent` union (`plan` text +
    structured `tasks` + `finished`/`total`/`complete`).
  - **`src/zakcode/agent/prompt.py`:** domain-agnostic `_PLANNING` guidance (decompose multi-step
    work; one `in_progress`; just-in-time decomposition OK; skip for trivial single actions).
  - **Clients (deprioritized per owner, but completed coherently):** web client (`index.html`)
    handles `task_update` (textContent-only) and maps `update_plan`→`Todo`; CLI maps
    `update_plan`→`Todo` so the tool result renders via the pre-existing `_todo_row` checklist
    grammar. The CLI gracefully IGNORES the `task_update` event (no double-render; the tool
    result already shows the plan in the transcript) — a rich client should use `task_update`
    for a persistent panel instead.
  - **Tests:** `tests/test_tasks.py` (network), `tests/test_loop_planning.py` (re-injection,
    gate nudge→degraded, gate-inert-when-complete, no-plan, cross-turn reset/persist, malformed
    input, streaming event), plus registry/render/webclient-contract updates. **1770 green,
    ruff + mypy clean (`uv run poe check`).**

  - **Fresh-eyes review findings (all fixed before this entry):**
    1. **`normalize()` ordering bug** — deriving compound status BEFORE enforcing single-focus
       left a parent stale `in_progress` after its child was demoted (two compounds each with an
       in_progress child). Fixed: enforce focus first, then derive. Regression test added.
    2. **`update_plan` with all-malformed `tasks`** would silently wipe an existing plan and
       return an empty success. Fixed: reject without mutating; guard + test.
    3. **Completed plan bleeding across turns** — a finished checklist was re-injected/re-emitted
       on the next unrelated turn. Fixed: `_reset_completed_plan` at turn start; tests for both
       reset (complete) and persist (incomplete).

  - **Merge checklist / handoff (no PR opened yet — owner to decide):**
    - [ ] `uv run poe check` green (currently: 1770 passed, 11 skipped — skips are fastapi extra
      + powershell + live-provider, all environmental).
    - [ ] Open PR from `claude/sdk-task-decomposition-024ug1` → `main` if/when the owner asks.
    - [ ] Optional polish (out of scope here): rich-client persistent task panel using
      `task_update`; suppress/collapse the `update_plan` tool result in the web transcript to
      avoid double-render; consider model-settable `kind` to activate the under-decomposition
      gate; route a `planner` role model for decomposition.
    - Risk surface is contained: new code is additive, the gate is bounded (cannot deadlock),
      the substrate is pure + well-tested, and the session field is append-only/back-compatible.

- **2026-06-14 (research, branch `claude/sdk-task-decomposition-024ug1` — best-in-class task-planning
  survey):** ran a multi-agent deep-research sweep (~25 systems, ~40 papers) benchmarking our HTN
  planning layer against the field. Full cited report: `docs/research/task-planning-decomposition.md`.
  Verdict: our fundamentals (model-driven `update_plan`, single-`in_progress`, cache-safe re-injection,
  self-arming bounded gate, completed-plan reset, full-replace) are on the best-in-class path and match
  what Claude Code/Codex/Amp converged on. Prioritized next steps (NOT yet implemented):
  **P0** — (R1) generalize the completion gate from "run the script" to the project's real verifier
  (tests/lint/typecheck) when discoverable, kept domain-agnostic (skill-provided/detected); strongest
  evidence in the literature is external-sound-verification ≫ self-critique. (R2) add optional task
  **dependencies** (`blocked_by`) — the frontier (Claude Code's new `TaskUpdate addBlocks/addBlockedBy`,
  LlamaIndex `SubTask.dependencies`, LLMCompiler/Devin DAGs) moved to dependency-aware plans.
  **P1** — (R3) capability-triggered decomposition: wire `StuckTracker` to nudge "decompose this step"
  on a stuck primitive (ADaPT), plus a complexity floor (~3+ steps, no single-step plans, anti-over-plan).
  (R4) route decomposition through the `planner` role model. **P2** — optional plan-review gate; richer
  delegation summaries; (watch) a facts/assumptions ledger. Caveat recorded in the report: "simple beats
  agentic" on SWE-bench, so the planning layer's justification is weak-model support + multi-turn
  coherence + UX, not benchmark-chasing — keep it sharp.

- **2026-06-14 (dev, branch `claude/sdk-task-decomposition-024ug1` — implement the two research
  P0s):** acted on `docs/research/task-planning-decomposition.md`.
  - **R1 — project-verifier gate (the report's strongest lever: external verification ≫ self-critique).**
    New pure `agent/verify.py` `VerificationGate` (mirrors `RecipeCursor`/`StuckTracker`): arms when a
    turn changes code (successful `write_file`/`edit_file`), satisfied when a run tool executes the
    configured command successfully (`_commands_match` is token-contiguous, tolerant of wrappers like
    `cd x && <cmd>`). New `Settings.verify_command` (`ZAKCODE_VERIFY_COMMAND`). Wired into BOTH loop
    paths between the recipe gate and the plan gate via `_try_project_verify` (mirrors
    `_try_harness_verify`: harness runs it only when it would auto-allow, else nudges; bounded by
    `attempt_cap` → `verification_failed` degraded). New `verification_failed` in
    `_DEGRADED_STOP_REASONS`. **Domain-agnostic** — the engine never guesses the command; inert when
    unset (recipe gate unchanged). Docs: CONFIG.md, .env.example, ARCHITECTURE.md.
  - **R2 — task dependencies.** `Task.blocked_by: list[str]`. `normalize()` now sanitizes edges into a
    DAG (`_sanitize_dependencies` drops self/unknown; `_break_dependency_cycles` drops back-edges) —
    fail-open so a malformed graph never freezes the agent. `current()` returns the first actionable
    leaf whose deps are all terminal, with a fail-open fallback. `update_plan` exposes `blocked_by`
    (referenced by the position ids shown in the plan); `render()` annotates "(after …)".
  - Tests: `tests/test_verify.py` (gate unit + hermetic loop integration with fake write/bash tools —
    no real subprocess) and dependency cases in `tests/test_tasks.py`. **1781 green, ruff + mypy clean.**
  - Remaining (not done): P1 capability-triggered decomposition (wire StuckTracker → "decompose this
    step") + planner-role model; P2 plan-review gate, richer delegation summaries, facts ledger.

- **2026-06-14 (dev, branch `claude/sdk-task-decomposition-024ug1` — research P1/P2 follow-ups):**
  implemented the remaining recommendations from `docs/research/task-planning-decomposition.md`.
  - **R3 capability-triggered decomposition:** `AgentLoop._decompose_hint()` appends a "break step N
    into sub-steps with update_plan" suggestion to the stuck-ladder NUDGE when the live plan's
    `current()` is a primitive (ADaPT's decompose-on-failure, reusing the existing StuckTracker). The
    prompt's `_PLANNING` guidance now sets a ~3-step complexity floor and warns against
    over-decomposition (Claude Code's 3+ heuristic; Codex "no single-step plans"; overthinking).
  - **R4 planner-model decomposition:** the planner-role routing already existed; made concrete by
    adding `update_plan` to `PLAN.allowed_tools` (READ_ONLY, so Plan Mode stays read-only on the
    workspace) + a planner instruction to structure the plan. No planner/executor split (kept the
    single-threaded design per the report's caveat).
  - **R5 plan-first gate (opt-in):** `Settings.require_plan` (`ZAKCODE_REQUIRE_PLAN`, default false).
    New `_is_mutating`/`_plan_first_blocks`; both loop paths withhold a mutating batch (pairing its
    tool_use blocks with recoverable errors + a nudge) until a plan exists, bounded by
    `_MAX_PLAN_FIRST_NUDGES` then fail-open. Read-only investigation is never gated.
  - **R6 structured handoff:** `subagent._HANDOFF` appended to every sub-agent prompt via
    `prompt_builder_for` (the child's final message is the only thing returned, so require a
    self-contained summary — addresses the lossy-boundary failure mode).
  - **R7:** deferred per its own recommendation (facts ledger belongs to the mind).
  - Updated 6 pre-existing tests that asserted the old PLAN toolset / prompt-suffix (intended
    changes). New tests in `test_loop_planning.py` (R3 + R5) and `test_subagent_planning.py` (R4 + R6).
    **1790 green, ruff + mypy clean.** Docs: CONFIG.md, .env.example, ARCHITECTURE.md, ADR-0008, report.

- **2026-06-14 (dev, D28 — self-remediation Step 4: per-task tool-exposure filter; + Step 5
  decision):** Zachary said "get step 4 done, think about step 5" (Step 3, the sandbox, deferred
  as too big a refactor for now). **Step 4 = an operator-set, glob-based tool-exposure filter**
  on `ToolRegistry` (`set_exposure_filter(allow, deny)` + `exposure_allows` + `exposed_names`),
  wired from new settings `tool_exposure_allow` / `tool_exposure_deny` (glob over canonical names;
  deny wins). Narrows the model-facing toolset to a task-appropriate subset — AgentDojo's single
  most effective injection defense (a tool never offered can't be hijacked). **Enforced at BOTH
  seams:** omitted from `definitions()` + the system-prompt list (`_tool_specs` now uses
  `exposed_names()`), AND the loop's execution seam rejects a model call to a filtered-out tool
  (closing the "model names a hidden tool from prior knowledge / injected content" hole). Composes
  over the active set (sticky vs `tool_search` re-activation), propagates into sub-agent
  `subset()`s, exposure-only (never loosens the permission gate; trusted internal `execute()`
  callers unaffected). **Deliberately operator-controlled, NOT model-decided** (a model-chosen
  filter is defeated by the injection it defends against — a wrapping orchestrator like Mind
  declares each task's scope). 15 new tests (`tests/test_tool_exposure.py`), 1841 suite green
  (clean env), ruff+mypy clean. Docs: SELF-REMEDIATION Step 4 ✅ SHIPPED, CONFIG.md, `.env.example`,
  RISKS prompt-injection row.
  **Step 5 decision: NOT building it (likely never as specced).** A gray-zone classifier is the
  weakest layer (~74% F1, worse on a local model), a convenience-to-reduce-prompts at best, never
  a boundary. With Steps 1/2/4 + the deny-first floor shipped, the deterministic layer covers the
  concrete vectors; the honest remaining gap is **containment (Step 3, the sandbox)** — a
  model-quality-independent guarantee — not a smarter judge. Recommendation logged in
  SELF-REMEDIATION §4: close the roadmap at 1/2/4 + floor; revisit Step 5 only if a specific
  high-frequency approval-fatigue pattern emerges after a sandbox lands, and even then as
  propose-policy-enforced-deterministically. **Self-remediation roadmap is now effectively
  complete except Step 3 (the sandbox), which is parked pending an explicit greenlight.**

- **2026-06-14 (Zachary → dev, D29 — Step 3 sandbox FORMALLY DEFERRED):** after a walkthrough of
  what Step 3 is, whether other harnesses ship it, and how important it is, Zachary decided to
  defer it deliberately. Rationale captured in SELF-REMEDIATION §4 (Step 3 decision block):
  the sandbox is the one model-quality-independent guarantee, but its importance is conditional —
  **defense-in-depth for attended/interactive use (the current mode, where the shipped
  deterministic layers + human backstop already protect), the precondition only for
  unattended/headless autonomy.** Cost/ROI is currently poor: L–XL, platform-specific, and the
  primary runtime is **Windows** (weakest/hardest isolation story — worst ROI quadrant). Industry
  practice splits the same way (cloud agents sandbox by default because they're already
  containerized; local-first harnesses run on the host behind a permission gate, where Zak Code
  is). **Revisit when the use case shifts to unattended operation OR a Linux/container
  deployment.** With this, the self-remediation engagement is wrapped: Steps 1/2/4 shipped to
  main, Step 5 declined, Step 3 deferred with a clear revisit trigger.

- **2026-06-14 (dev, branch `fix/test-endpoint-env-isolation` — test hermeticity, post-#28
  fresh-eyes follow-up):** the fresh-eyes review after landing #28 flagged that `uv run poe check`
  was RED locally (2 failures in `tests/test_endpoint_config.py`) while CI was GREEN — a
  non-hermetic test, not a code defect. Root cause: a developer's `.env` reaches `Settings()` two
  ways — pydantic-settings reads the `.env` file directly, AND `load_settings`' `load_dotenv` step
  exports it into `os.environ` so litellm can see provider keys (see `test_dotenv.py`). The two
  tests asserting the UNCONFIGURED defaults (`api_base`/`api_key` is None) therefore failed against
  a dev `ZAKCODE_API_BASE`; CI has no `.env`, so the drift was invisible there. Fix (surgical,
  matches the existing `test_dotenv.py` `monkeypatch.delenv(..., raising=False)` pattern): the two
  default-asserting tests now clear `ZAKCODE_API_BASE`/`ZAKCODE_API_KEY` from the env AND pass
  `Settings(_env_file=None)`, neutralizing both the env-var and the file source so they assert true
  defaults regardless of a local `.env`. Exactly 2 tests affected (full suite confirmed); zero blast
  radius on the other 33 env-touching tests. **Local `poe check` now green (ruff+format+mypy clean,
  1884 passed, 5 skipped).** Deferred (scope): a conftest autouse fixture clearing `ZAKCODE_*` for
  whole-suite hermeticity — revisit only if a third default-asserting test trips the same wire.

- **2026-06-15 (dev, branch `feat/plan-idle-autoclear` — issue #32, the "haunting plan" fix):**
  implemented the bounded auto-clear designed in the post-#28 fresh-eyes review. Problem:
  `_reset_completed_plan` only dropped a plan at turn start when `is_complete()`, so an *abandoned*
  (unfinished, untouched) plan re-injected + re-nudged + degraded every subsequent turn until the
  model called `update_plan([])` — the recurring tax weak local models pay. Fix: generalized it to
  `_reset_stale_or_completed_plan`, which ALSO drops an unfinished plan that has sat byte-identical
  across `_MAX_PLAN_IDLE_TURNS` (3) consecutive turn-starts. Determinism via a new pure
  `TaskNetwork.progress_signature()` (`repr` of each task's `(id, status, title)`); two append-only
  `Session` fields (`plan_idle_turns`, `plan_signature`, no schema bump); the plan-gate completion
  nudge now names the "clear the whole plan" escape hatch. Conservative by design — any edit
  (advance, decompose, retitle) resets the counter, so an active plan is never auto-cleared; only a
  genuinely static one is, and it is freely re-creatable. Wired identically on `_run_turn` and
  `astream_turn`. Tests: `test_tasks.py` (signature equal-when-untouched / changes-on-any-edit) +
  `test_loop_planning.py` (unit threshold + counter, active-never-dropped, nudge escape hatch, and
  both-path integration that an abandoned plan stops haunting). **`uv run poe check` green:
  ruff+format+mypy clean, 1890 passed, 5 skipped; `--extra server --extra dev` full suite 1890
  passed.** Design tracked in GitHub issue #32 (resolved by this change).
- **2026-06-15 (dev, branch `fix/harness-verify-shell-aware` — issue #33, powershell-host harness
  auto-run):** the post-#28 fresh-eyes review flagged that `_try_harness_verify` and
  `_try_project_verify` hardcoded the synthetic verify run as a `bash` ToolCall, so on a
  powershell-preferred host (where the operator granted `powershell`, not `bash`) `auto_allows`
  was false and the harness silently fell back to nudging the model instead of auto-running the
  check. Fix: a shared `AgentLoop._harness_shell_call(command, call_id)` helper that prefers `bash`
  (its cmd.exe / POSIX-sh quoting matches how `resolve_run_command` / `verify_command` build the
  command — so zero change to every existing case) and falls back to `powershell` only when bash
  would prompt or is unregistered. The powershell form is prefixed with the call operator `&` so a
  *quoted* exe path (`resolve_run_command`'s `sys.executable` fallback) executes rather than being
  echoed as a string literal; `&` is a no-op before a bare command and survives both run-matchers
  (`_executed_targets` token split, `_commands_match` token-prefix — both already list `powershell`
  in `_RUN_TOOLS`). Both verify methods now route through the one helper (symmetric, per the issue).
  Tests: 3 unit tests in `test_recipe.py` (bash-preferred-verbatim, powershell-fallback-with-`&`,
  none-when-nothing-auto-allows) — cross-platform, no real subprocess. **`uv run poe check` green:
  ruff+format+mypy clean, 1893 passed, 5 skipped.** Resolves GitHub issue #33.
- **2026-06-15 (dev+research, branch `feat/primitiveness-criteria` — HTN cross-system survey →
  decomposition guidance):** the user asked whether Zak Code's task decomposition could learn from
  the HTN implemented in `Ayoai/Ayoai-Environment-Processor` (and elsewhere). Ran three parallel
  read-only survey agents over: the Processor (dual **HTN + A\*** over a STRIPS world-model), the
  higher mind **`Ayoai-Mind`** (aspirations→goals, 22-criterion goal-selector, scope classes), and
  the omni continual-learning framework's `/decompose` skill. **Finding:** the lean
  full-replace + nesting-inferred-`kind` + ephemeral-plan + issue-#32 design already structurally
  handles most borrowable patterns (idempotency back-refs, hierarchical-cycle detection,
  stale-blocker TTL) — a validation of "simple beats agentic," not a gap. The single additive,
  philosophy-fitting idea — convergent across `/decompose` AND ayoai-mind — was implemented:
  sharpen the primitiveness **stopping-rule** with the two criteria a single-action floor omits — a
  **clear done-condition** and **no hidden 'figure out how'** — in `_PLANNING` (prompt.py) + the
  `update_plan` tool description. Model-facing guidance only: no schema/infra, cache-stable. Pinned
  by `test_planning_guidance_names_primitiveness_criteria` (asserts both criteria in the built
  prompt AND the tool spec, so it can't silently regress). **R7** (facts/assumptions ledger) stays
  deferred — *reinforced*: every sibling keeps belief/world state at the long-horizon layer, never
  the near-term core. Survey conclusions recorded as an ADR-0008 update in `docs/DECISIONS.md`.
  **`uv run poe check` green: ruff+format+mypy clean, 1894 passed, 5 skipped.**
- **2026-06-15 (dev, branch `feat/note-done-conditions` — verification-as-schema via the existing
  `note` field):** the agreed follow-up was an "optional per-task `done_when`" (the schema half of
  verification-as-schema). On reading `tasks.py` the field **already exists**: `Task.note` is
  documented as the "acceptance criterion" slot, parsed by `update_plan`, and rendered into the
  re-injected plan. So a `done_when` field would have just duplicated it — flagged and pivoted to the
  lean path (single-source-of-truth; the handoff's "push back on over-building"). Delivered the same
  value through the existing field: sharpened `note`'s role to *the step's checkable done-condition*
  (the `Task` docstring + the `update_plan` schema field description) and wired `_PLANNING` to record
  each primitive step's done-condition in `note` — so the #36 "steps must be checkable" rule now names
  WHERE the check lives, and it rides in the plan the model re-reads every turn. No data-model change,
  no migration. Pinned by `test_step_note_is_the_checkable_done_condition` (guidance phrase in the
  built prompt + `done-condition` in the update_plan schema). **`uv run poe check` green: 1895 passed,
  5 skipped.**
- **2026-06-15 (dev, branch `feat/plan-decomposition-probe` — measure the decomposition behavior):**
  turned the #36/#37 prompt+schema changes into a *measured behavior*, not just pinned text. The M9
  behavioral-probe suite (`src/zakcode/evals/probes.py`) had end-to-end probes for completion,
  safety, plan-mode, doom-loop, recovery, and compaction — but **none for the planning lifecycle**, a
  real gap given ADR-0008. Added an 8th probe, `plan-decomposition`: it drives the *real* loop with a
  scripted full-replace plan — a COMPOUND step with two primitive children, each carrying a `note`
  done-condition — advances the single frontier step by step, and asserts the turn ends `completed`,
  the plan is fully resolved, the compound's status DERIVES 'done' from its children, every primitive
  carries a checkable `note`, and the done-condition rides in the re-injected plan render. Honest
  about scope: with a `ScriptedProvider` it measures that the *harness* runs a well-formed checkable
  plan to clean completion (the lifecycle the core must never regress) — proving a real model
  produces good steps is the live-eval's job (a noted future follow-up). Suite wiring updated (count
  7→8, name set, CLI `8/8 passed`, standalone `test_probe_plan_decomposition`). **`uv run poe check`
  green: 1896 passed, 5 skipped** (the full-suite gate caught a stale `7/7` CLI assertion the
  targeted run missed — exactly its job).
- **2026-06-15 (dev, branch `feat/live-decomposition-eval` — the honest completion of "measure
  it"):** the #38 probe proves the HARNESS runs a checkable plan; this proves the model half — that
  the sharpened guidance *actually makes a real model* write checkable, decomposed plans. New
  `LIVE_TESTS`-gated `tests/test_live_decomposition.py`: gives a real model a NEUTRAL multi-step task
  (it says nothing about notes/checkability — so any done-conditions come from the system-prompt
  guidance, not the task), then scores the authored plan's `note` coverage (skip if the model didn't
  decompose; assert ≥50% of primitive steps carry a done-condition; the failure message dumps the
  plan). Same cooperation-bias + provider-error-→-skip pattern as the live smoke tests. **Validated
  locally for FREE** against the running llama.cpp `llama-server` (Qwen3.5-9B-Q6 on `:8080`, routed
  via litellm `OPENAI_API_BASE` + a throwaway key → zero OpenAI spend): the 9B authored a clean
  3-step plan (`/health` endpoint → unit test → README) with a `note` done-condition on **3/3 steps
  (100% coverage)** in 18.5s — empirical proof the guidance lands. Docstring documents the local
  llama.cpp invocation (the free path). Skips in the normal gate; **`uv run poe check` green.**
- **2026-06-14 (dev, D30 — zakpick: task-category model routing):** built the realized form of the
  routing seam reserved since PKG-AUTO (D19/D21 — `ModelResolver.resolve(task=...)`) and the general
  form of `model_roles`. Full rationale + the rejected alternative in **ADR-0009**. Summary for a
  future agent:
  - **Shape.** A third `default_model` sentinel `"zakpick"` (beside `"auto"` + a concrete model).
    Under it the engine routes each internal prompt to the model the user assigned to that prompt's
    **task category**, not one model for everything. **The interface is the categories, not a dial:**
    `quick_code` / `deep_code` (main turn) · `summarize` (compaction) · `plan` (plan sub-agent) ·
    `delegate` (general sub-agent / `task` tool) · `classify` (reserved seam, no live caller). The user
    parks a `(model, source)` pair per category via new `Settings.zakpick_models`
    (`ZAKCODE_ZAKPICK_MODELS`, JSON; `source` defaults `"groq"`). `source` is **separate from `model`**
    on purpose (a model name alone is ambiguous — `qwen3-32b` runs at Groq *or* locally);
    `source="local"`→`ollama_chat`, anything else is the litellm prefix
    (`providers.routing.ZakpickModel.litellm_string`).
  - **Defaults out of the box** (`DEFAULT_CATEGORY_MODELS`), drawn from Groq's open-source-only lineup
    (so they double as a "download this to run the category locally" guide), graduated by cost:
    classify→`llama-3.1-8b-instant`; summarize & quick_code→`gpt-oss-20b`; plan→`qwen3-32b`;
    deep_code & delegate→`gpt-oss-120b` (tools-**reliable**). `llama-3.3-70b-versatile` avoided
    (registry `tools_unreliable`). All `source=groq`.
  - **The one automatic decision** is the quick-vs-deep coder split: `classify_main_turn` — a
    deterministic, offline pure function of request length + context fraction + a latched struggle
    signal (no model call, no iteration count) — picks which of the user's **two** coder models drives
    each turn; a one-way **soft latch** flips to `deep_code` on a real struggle signal (stuck ladder /
    doom-loop / verify-gate failure). It only ever switches between the two coder models the user
    already configured — never a model Zak Code chose.
  - **The pivot (recorded as the headline decision).** An earlier design was a **dial**
    (local/cloud/save/max) over a curated **4-tier ladder** with runtime source-masking + degrade-down
    logic. Rejected because it made Zak Code *own a local-vs-cloud tradeoff and tier curation that
    belong to the user*. The category→model map hands every model-choice back to the user; **they own
    the consequences** — a slow local model is slow (their choice), a failing cloud model is handled by
    the existing `fallback_model` seam like any other provider error. Under zakpick `model_failover`
    uses only an explicit `fallback_model` (else `provider_error`); `model_resolution` stays `None`.
  - **Ship/seam split.** Shipped: `providers/routing.py` (vendor-SDK-free — contract test green;
    `ZakpickModel`, `DEFAULT_CATEGORY_MODELS`, `classify_main_turn`, the friendly per-category
    describe), `Settings.zakpick_models` + validator (`config.py`), `ZAKPICK_SENTINEL` +
    `describe_resolution` branch (`resolve.py`), `Agent._resolve_task_provider`/`_main_provider_for`
    (reusing the existing `_provider_for` cache), `main_provider_for` + soft-latch + classifier wiring
    in `agent/loop.py` (buffered + streaming), `category` + `provider_for_task` on `agent/subagent.py`,
    CLI info-panel/`/model`/banner showing `model (source)` per category (never raw slugs; only routed
    categories, so `classify` is never advertised). Deferred seams (in ADR-0009 + ROADMAP, with
    triggers): a cheap difficulty-classifier *model* for `classify` (heuristic-only today; category +
    output shape pre-wired); cost/price metadata on `Capabilities` (defaults encode cost by curation,
    not a price field); an `embeddings` category (no embedding call site yet). **Beyond-parity, Zak-Code
    original** — no reference harness (Claude Code / Hermes / goose) ships task-category routing, so
    provenance is first-principles, not a mirrored design. **1910 tests pass, ruff + mypy clean.**
    CONFIG.md + `.env.example` already updated.
- **2026-06-15 (dev, D31 — zakpick transparency follow-ups, prompted by *Intelligence per Watt*):**
  three small phases off the zakpick base, each motivated by arXiv 2511.07885
  ([`references/intelligence-per-watt.md`](references/intelligence-per-watt.md) — local/small models
  handle most prompts; an 80%-accurate router captures ~80% of the savings; `gpt-oss-120b` is the best
  local model, validating our `deep_code` default). The paper's *auto local-vs-cloud router* stays
  rejected (D30); the value taken is *evidence + transparency*, never a tool-owned routing decision.
  - **Phase 1 — per-model `/cost` attribution.** `Usage.model` tag (+ `Provider.model_id()`,
    forwarded by the text wrapper) → `Session.usage_by_model()` → a `/cost` breakdown shown only when
    a session spanned ≥2 models. Makes the savings the user's config already yields *visible* so they
    can tune it. A "vs all-deep" savings line is stubbed pending the `Capabilities` cost-metadata seam.
  - **Phase 2 — "deep coder wasn't needed" advisory.** `TurnResult`/`AgentDone` gain
    `routed_category` + `routed_escalated`; the REPL counts clean un-escalated `deep_code` turns and,
    once per session after 3, prints one `tip` suggesting a cheaper `deep_code` model may keep up. An
    observation + an option, never an auto-switch.
  - **Phase 3 — evidence-back the deferred classifier-model seam.** Added the paper digest to
    `docs/references`, cited it in ADR-0009 as the router-accuracy→savings yardstick for *whether* to
    upgrade `classify_main_turn`, and recorded this entry. **1918 tests pass, ruff + mypy clean.**

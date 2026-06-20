# Parity gap analysis — Zak-Code vs claw-code, Hermes, and goose

**Date:** 2026-06-11 · **Method:** a 104-agent fresh-eyes review workflow (8 capability
lanes × 4 codebases = 32 structured readers → per-lane gap synthesis → adversarial
verification of every "Zak-Code is behind" claim against the actual code → one
prioritized plan). ~5.7M tokens. The three references:

- **claw-code** — Claude Code, reverse-engineered (Python + a Rust port). **Clean-room
  only**: studied for capabilities/architecture, never copied. Any item sourced here is
  tagged `[CLEAN-ROOM]` and must be re-expressed in our own design.
- **Hermes** — Nous Research Hermes agent (OSS, Python).
- **goose** — Block goose (Apache-2.0, Rust).

> This complements `PARITY.md` (which tracks Claude-Code feature parity). This doc is
> the first comparison against the broader OSS harness field, and it is **evidence-backed
> and verified**: every gap below was checked against Zak-Code's real code, and claims
> that turned out to already exist were dropped (see *Already at or ahead*).

> **⚠ Freshness caveat (read before building).** The review was snapshotted against the
> Zak-Code tree at the `pr-5-tooling` stack tip. Between the snapshot and this doc landing,
> a large amount of work merged to `main`: the whole external improvement stack, omni's
> **PKG-AUTO** auto-model resolver (#17), the **TurnEnd** internal package + veto gates
> (#12, #18), provider retry fixes (#13, #15), and a frontend/UX track (#9–#11). Two items
> below are therefore likely **already addressed** and must be re-verified against `main`
> before any work: **#1's failover half** (PKG-AUTO #17 wired `fallback_model` as the
> explicit override and added runtime re-resolution on non-rate-limit `ProviderError` — so
> only the `ContextWindowExceeded` compact-then-retry sub-item likely remains), and
> **#30's outer-loop self-continuation** (TurnEnd #18 added loop break-site veto gates —
> the Stop-hook seam). Everything else (cache_control, headless `run`, cost budget, command
> normalization, invisible-Unicode stripping, overflow-to-file, session search/fork,
> telemetry, ACP, distribution, …) was untouched by the merged work and stands as written.

---

## Headline verdict

Zak-Code is a genuinely well-architected coding-agent harness whose **core engineering is
at par with — and in several places ahead of — all three references**. Its agent loop
(multi-signal stuck detection with an escalating nudge → narrow → stop ladder, a
delegation-tree-wide shared `IterationBudget`, a unified buffered+streaming tool seam,
atomic message-boundary persistence), its permission core (deny-first, enforced outside
the model's reach, schema-filtered sub-agents, autonomous hard-deny), its SSRF
defense-in-depth, its clean-room no-SDK MCP client, its docs-as-law compliance tests, and
its deterministic behavioral eval harness are all strong and defensible.

Where it falls behind is **not design quality but delivered breadth**, on two axes:

1. **Recovery/robustness depth against real, flaky providers** — failover, cost budgets,
   truncation/context-overflow recovery, and prompt-cache wiring are all either
   dead-wired or absent, leaving real cost and reliability on the table.
2. **Operability / automation surface** — no headless one-shot CLI, no scheduling, no
   telemetry, no IDE/ACP integration, no release/distribution channel, and a JSON-file
   session store with no search/fork.

**Net:** a clean, correct foundation that is feature-narrow versus mature harnesses. Most
gaps are *additive surface work on a sound core*, not corrections to a flawed design.

### Lane scorecard

| Lane | Verdict | Confirmed gaps |
| --- | --- | --- |
| Agent loop & control flow | **at par** | 7 |
| Providers & model routing | **at par** | 5 |
| Tools, execution & sandboxing | **at par** | 7 |
| Permissions & safety | **at par** | 8 |
| Context & memory | **at par** | 8 |
| Extensibility & orchestration | **behind** | 8 |
| Interfaces & platform | **behind** | 8 |
| Ops & developer experience | **behind** | 9 |

Five lanes at par, three behind the best OSS reference (usually goose or Hermes). The
three "behind" lanes are all about *delivered breadth*, not architecture.

---

## Cross-cutting themes

1. **Dead-wired provider resilience** (spans Agent-loop + Providers + Context). A cluster
   of declared-but-unconsumed capabilities: `Settings.fallback_model` is read only by
   `zakcode info` (no failover); `ContextWindowExceeded` is documented to "signal the loop
   to compact" but is referenced nowhere in the loop (the turn just dies as
   `provider_error`); `finish_reason` is captured then discarded as "advisory" (no
   truncation recovery); `supports_caching` is populated but never consulted, so
   `cache_control` is emitted nowhere and the prompt cache is never engaged; `Usage`
   carries no cache-token fields so cache savings are invisible. **Highest-ROI fixes —
   the seams already exist; it's wiring, not new infrastructure.**

2. **No budget/ceiling beyond iteration count.** A turn is bounded only by iteration
   count; `cost_usd`/`total_tokens` are accumulated and logged but never gated. Real spend
   and context-pressure risk on delegation trees and large-context models.

3. **Operability/automation surface is thin** (the dominant reason three lanes land
   behind): no headless one-shot CLI with machine-readable output, no scheduler, no saved
   routine format, no telemetry/OpenTelemetry, no IDE/ACP integration, no release pipeline
   or distribution channel, no log rotation/redaction filter, no doctor diagnostics. These
   separate a well-engineered library from an operable, distributable product — and
   goose/Hermes ship all of them.

4. **Session store is durable but inert.** Atomic, versioned, crash-safe JSON-per-session
   — but no search, no indexed query, no fork/branch/checkpoint, and the CLI `--session`
   resume flag is an explicit stub while the server resume works.

5. **Untrusted-content hardening has a sharp hole.** The permission core is strong, but
   the blocklist matches **raw** command strings (no NFKC/backslash-escape/empty-quote
   normalization — trivially bypassable in allow/autonomous), and the untrusted-content
   defang neutralizes only **visible** protocol frames (invisible Unicode Tag-Block
   instruction smuggling passes through). Both are cheap, conservative, single-chokepoint
   fixes.

6. **Ahead of all three on safety-design and control-flow sophistication** — see *Already
   at or ahead*. Do **not** rebuild these.

---

## Prioritized backlog (the next engagement)

30 verified items. Effort = S/M/L/XL. `[CLEAN-ROOM]` = idea sourced from claw-code; must be
re-expressed in our own design, never copied.

### P1 — material parity gaps (11)

**1. Wire provider-error failover + `ContextWindowExceeded` compact-then-retry** · M ·
*loop/providers*
Hermes drives a failover taxonomy + fallback chain; goose auto-compacts on context-length
errors (recursion-guarded). Verified: `Settings.fallback_model` (config.py:58) is read only
by `zakcode info`; every non-429 `ProviderError` terminates the turn (loop.py:907-916,
1177-1194); `ContextWindowExceeded` is documented to compact (base.py:135-139) but appears
nowhere in `agent/`. **Rec:** a tiny pure `classify(exc) -> {retry, compact_then_retry,
failover, terminal}` over the existing taxonomy: `AuthError`→terminal;
`ContextWindowExceeded`→`compact_now()` (exists, loop.py:413) once then retry;
`RequestFailed`/overloaded→failover once to a provider built from `fallback_model` via the
existing `_provider_for` seam; `RateLimited`→keep today's retry. Emit a degraded event on
each hop. Our own classifier, not a port.

**2. Emit `cache_control` on the stable prompt prefix + track cache tokens in `Usage`** ·
M · *providers/context*
Hermes treats caching as "sacred" and cuts multi-turn input tokens ~75%. Verified: the
`DYNAMIC_BOUNDARY` marker (prompt.py:39) and `supports_caching` (registry) exist, but
`cache_control` appears nowhere, `supports_caching` is never read in the request path, and
`Usage` has no cache fields. **Rec:** when `supports_caching`, split the system string at
`DYNAMIC_BOUNDARY` into content blocks and stamp `cache_control={type:ephemeral}` on the
last stable block (litellm passes it through for Anthropic, drops elsewhere); extend `Usage`
with `cache_read_tokens`/`cache_creation_tokens` from litellm's normalized usage; surface
the hit ratio in `/cost`. Implement the emission half first (accounting reads 0 until then).

**3. Headless one-shot CLI (`zakcode run`) with machine-readable output** · M ·
*interfaces*
claw-code has `prompt` (text/json/ndjson); goose's `run` supports `--output-format`,
stdin, instruction files; Hermes has a batch runner. Verified: the core is headless-capable
but the CLI registers only version/info/eval/chat/serve — no `run`, no `--json`.
ARCHITECTURE.md:14 even claims a one-shot mode the code lacks (a docs-travel-with-code
violation). **This is the gate for CI use, scripting, scheduling, and ACP.** **Rec:**
`zakcode run` reads a task from arg/`--file`/stdin, runs through the core, prints a
transcript or (`--json`) a structured result (text + per-message usage + cost + stop_reason);
defaults to autonomous mode (fails closed); non-zero exit on failure. Fix ARCHITECTURE.md.

**4. Cost/token budget stop condition alongside `IterationBudget`** · M · *loop*
Verified: the only budget stop is iteration count (loop.py:444-448); `cost_usd`/
`total_tokens` are accumulated (loop.py:927) but never compared to a ceiling. Real spend
risk on delegation trees and big-context models. **Rec:** extend the shared
`IterationBudget` (or a parallel `TokenBudget` on the same pool) with an optional per-turn
and tree-wide cost/token ceiling; new `stop_reason="budget_exhausted"`; add
`max_cost_usd`/`max_tokens` config.

**5. Truncation-aware turn completion: use `finish_reason` for length continuation** · M ·
*loop*
Hermes detects `finish_reason="length"` and issues bounded continuation. Verified:
`finish_reason` is captured (base.py:51,115) but the loop discards it as "advisory"
(loop.py:1153-1157) and sets `completed` unconditionally — a length-truncated answer is
mis-reported as complete (a real correctness gap). **Rec:** on length truncation with a
partial message, issue a bounded continuation instead of `completed`; add a `truncated`
stop_reason/degraded flag. Mirror the existing truncated-tool-call recovery in `text_tools`.

**6. Normalize commands before dangerous-pattern matching (obfuscation resistance)** · M ·
*permissions*
Hermes strips ANSI/null, applies NFKC, removes backslash-escapes (`r\m`→`rm`), collapses
empty-string literals (`'r''m'`→`rm`) at detection time. Verified: `DANGEROUS_PATTERNS`
match the **raw** arg string (permissions.py:370-377), so `'r''m -rf /'`, `s\udo`, and
fullwidth variants sail past — and the blocklist is the **last** line of defense in
allow/autonomous mode (where autonomous = hard-deny with no human). **Rec:** a pure
`normalize_for_detection(command)` run before `_dangerous_reason`; match on the normalized
form, echo the original to the operator; unit-test the known vectors.

**7. Strip invisible Unicode Tag-Block / bidi chars at every untrusted boundary** · S ·
*permissions*
goose's `sanitize_unicode_tags()` removes U+E0000–E007F Tag-Block chars that smuggle
instructions invisible to operator and model. Verified: `defang_untrusted`
(text_tools.py:257-273) neutralizes only visible frames; nothing strips Tag-Block,
zero-width, or bidi — at file read-backs, web_fetch, skill bodies, and injected context.
Backs a GUARDRAILS claim that is currently false. **Rec:** extend the single
`defang_untrusted` chokepoint (all four boundaries route through it) to strip/escape
U+E0000–U+E007F, zero-width, and bidi controls; unit-test a Tag-Block payload.

**8. Overflow-to-file output persistence + per-turn aggregate output budget** · M · *tools*
Hermes spills oversized results to a file (returns preview + path; model re-reads via
offset/limit) and enforces a ~200KB per-**turn** budget; goose saves shell overflow to a
file. Verified: every truncating tool slices to an inline marker and **discards** the
remainder (irretrievable for expensive/non-idempotent commands), and there's no per-turn
sum, so an 8-way parallel read batch can stack many near-cap results with no ceiling.
**Rec:** on cap-exceed, write the full output to `<workspace>/.zakcode/results/<id>.txt`
(path-guarded), marker carries the path + "read_file … offset=N for the rest"; add a
`result_file` field; at the batch seam, sum model-facing sizes and spill the largest when
the turn total exceeds a generous ceiling. Implement once, covers every tool.

**9. Context-overflow emergency recovery: progressive prune when compaction is
insufficient** · M · *context*
goose progressively removes tool responses (0/10/20/50/100%) on context-length errors.
Verified: Zak-Code's compaction is purely pre-turn/threshold; if it underestimates or one
huge tool result lands mid-turn, `ContextWindowExceeded` propagates and the turn dies.
**Rec:** on `ContextWindowExceeded`, force `compact_now()` once; if already compacted,
progressively drop the largest/oldest tool-result payloads (keeping the tool-pair-safe
boundary the Compactor respects), then retry once. Shares the branch with item #1.

**10. Wire CLI session resume + `zakcode sessions list` + a `/fork` checkpoint** · M ·
*interfaces*
Verified: `SessionStore` fully supports load/resume/list/delete and **server** resume
works, but the CLI `--session` flag is an explicit stub (cli/__init__.py:731) — the CLI
always starts fresh, the web client always POSTs a new session. **Rec:** load the session
at chat startup and pass it into the agent builder (cumulative usage already reconstructs);
add `zakcode sessions list`; add a lightweight `/fork` that branches by truncating history
(cheap given atomic message-boundary saves).

**11. Project/user JSON config merge with provenance + named profiles** · M · *ops* ·
`[CLEAN-ROOM]`
claw-code does hierarchical user/project/local discovery with deep-merge + per-setting
origin tracking; Hermes has named profiles. Verified: config resolves only
explicit→`ZAKCODE_*`→`.env`→defaults; no `settings.json` discovery, no provenance. The
project's own docs track this at P0/M1, and the `.zakcode/` convention already exists for
mcp/plugins/skills/rules/identity. **Rec:** `settings_customise_sources` to deep-merge
`./.zakcode/settings.json` over user-level over defaults, under the env layer; track
provenance so `zakcode info` shows each value's source; optional `ZAKCODE_PROFILE` for an
isolated state dir. Re-express in pydantic-settings-native design — do not port the loader.

### P2 — needed soon / high-value breadth (17)

**12. MCP streamable-HTTP/SSE transport (+ static-bearer auth)** · M · *extensibility* —
the `Transport` Protocol seam is clean and stdio-only today; HTTP unlocks the hosted-MCP
ecosystem and is the prerequisite for MCP auth.
**13. Scheduler + saved parameterized routine format** · L · *extensibility* — name it
**Routines** (avoid the Recipe Cursor collision): YAML with params/model-override/
allowed-tools, discovered like skills; plus a `JobStore` + file-locked `tick()` driven by
an **external** trigger (OS cron / `zakcode schedule run`), not a baked-in process
scheduler. Builds on item #3.
**14. Optional OS-level execution isolation behind a pluggable `Executor`** · XL ·
*tools/permissions* · `[CLEAN-ROOM]` — Docker/podman + Linux namespace (unshare/bubblewrap)
backends behind the `run_capturing` chokepoint; today's proxy+path-guard is the fallback.
Already tracked as deferred in ROADMAP/RISKS.
**15. OpenTelemetry export (vendor-neutral, off by default)** · L · *ops* — an optional
`zakcode[otel]` extra bridging the existing `AgentEvent` stream into OTel spans at the
`astream_turn` boundary; driven by standard `OTEL_*` env; no vendor SaaS bundled.
**16. Operator logging setup: rotation, redacting filter, session-id correlation** · M ·
*ops* — an opt-in `configure_logging()` (CLI/server only) with a `RotatingFileHandler` and,
crucially, a redacting `logging.Filter` reusing `secrets.py` so no key reaches a file
regardless of level.
**17. Distribution channel: PyPI publish + release pipeline + slim Dockerfile** · M ·
*ops* — tag-triggered Trusted-Publishing (OIDC) job + provenance; a slim server/CLI image;
install matrix docs. The version-sync footgun is already closed.
**18. Scheduled OSV/pip-audit CVE scan + `--frozen` CI + dependency-drift** · M · *ops* —
a daily OSV scan of `uv.lock`, `--frozen`/`--locked` on CI sync so drift fails fast,
unused-deps check. Supply-chain hygiene is documented policy but unenforced.
**19. Context-overflow probe + min-context floor warning for unknown models** · M ·
*providers* — when both the static table and litellm miss, Zak-Code falls silently to a
flat 8192; warn on an implausibly small resolved window at session start.
**20. Move built-in slash commands into the core `CommandRegistry` + expose over server** ·
M · *interfaces* — the built-ins are a hard-coded if-chain in the CLI REPL; move handlers
into the registry (allow async `CommandResult`) and add `POST /command`/a WS command
message so web/IDE clients gain command access. Reinforces the core/interface invariant.
**21. ACP / IDE integration adapter as a new thin client** · XL · *interfaces* — the single
largest Interfaces gap; build over the same `AgentEvent` stream the WS channel emits;
stdio wire first; map the schema from the **public** ACP spec only.
**22. Empty/no-progress completion nudge for weak local models** · M · *loop* — an empty
completion ends the turn cleanly today; for weak local models (a stated target) inject one
"continue / give your final answer" nudge before accepting `completed` (cap 1-2, refund the
budget unit). Reuses existing nudge machinery.
**23. Skills/plugin content threat-scan + OSV install screening** · L · *permissions* — two
error-isolated, fail-open review aids: a content scanner for non-bundled skills/plugins
(exfil/persistence/injection signatures + trust tiers + audit log), and a PreToolUse
inspector recognizing `pip/uv/npm install` and querying OSV (MAL-* advisories), escalating
to ASK (hard-DENY in autonomous) on a hit.
**24. HARDLINE catastrophic floor un-waivable even in allow mode (+ doctor + SECURITY.md)** ·
M · *permissions/ops* — carve a small HARDLINE subset (raw-device writes, mkfs, fork bomb,
`rm -rf` of root, **+ missing power-off/reboot**) that hard-DENYs in **every** mode; write
`docs/SECURITY.md` drawing the real trust boundary; extend `info` into `zakcode doctor`
(env/extras/external-tool/key diagnostics, non-zero exit on hard failure).
**25. Structured compaction summary template + head-anchored protection** · S · *context* —
replace the freeform summary instruction with a structured template (Goal/Decisions/Files/
Open work/Key facts); add `protect_first_n` so the opening task is never summarized away.
Single-string/config change.
**26. Streaming watchdog + partial-stream salvage** · M · *loop* — wrap `provider.astream`
in a stale-stream/read-timeout watchdog (a hung stream surfaces as `RequestFailed`); on
mid-stream failure after deltas streamed, persist the partial as the assistant message and
end degraded instead of discarding it.
**27. Memory & retrieval depth: embedding RAG, transcript search, selective elision** · L ·
*claude-mind* — an embedding-backed recall store (a Mind's, fall back to FTS5); a
`session_search` builtin over indexed transcripts (no LLM in the path); a visibility/elision
concept marking large stale tool results collapsed before full compaction.
**28. Orchestration polish: configurable delegation depth, dynamic tool schemas, plugin
provider registries, live-session orchestration** · L · *extensibility* — make nesting depth
a bounded setting (default 1); add a `dynamic_schema_overrides()` hook so `task`/`tool_search`
advertise live budgets; generalize the `SearchBackend` into a plugin
`register_provider` registry; defer the XL live-session orchestrator until a supervisor
use-case appears.

### P3 — deliberate scope choices (record as decisions) (2)

**29. Model alias shortcuts + broader secret-redaction shapes + nightly perf benchmarks** ·
M · *providers/permissions/ops* · `[CLEAN-ROOM]` — a thin user-extendable alias dict
(`opus`→full string); extend `redact_secrets` with JWTs, DB connection strings, more vendor
prefixes, a mask helper; an opt-out nightly benchmark timing hot paths.
**30. Outer-loop self-continuation, notebook editing, in-process code-exec sandbox, i18n** ·
L · *loop/tools/ops* — genuinely absent but **deliberate** scope choices. **Rec: defer all
four and record each in DECISIONS.md** so they read as choices, not omissions. Build only on
demand (goal/grind continuation if autonomous multi-turn becomes a goal; notebook/exec
sandbox/i18n if target-user scope shifts).

---

## Already at or ahead — do NOT rebuild

The verifier dropped these as already-present or already-best-in-class:

- **Agent loop:** multi-signal stuck detection with a nudge → narrow → stop recovery ladder
  (more sophisticated than goose's tool-level repetition inspector or claw-code's bare
  iteration cap); a **delegation-tree-wide shared `IterationBudget`** (a novel cost-safety
  property none of the three match); unified buffered+streaming paths on one tool seam with
  identical stop semantics, atomic message-boundary persistence, clean cancellation
  (best-in-class); the Recipe Cursor / write-grounding verify-before-finish gates (ahead of
  all three); bounded RateLimited retry + graceful degraded `provider_error` termination
  (shipped in PR #5).
- **Providers:** vendor-agnostic `Provider` ABC enforced by a contract test (only
  `providers/` may import litellm); 100+ backends by config; real tokenizer-based token
  counting (not `len/4`); per-role routing; the `TextToolCallingProvider` wrapper making
  weak local models reliably tool-call, with auto Ollama→text-protocol detection and
  num_ctx-lift-with-window-clamp (a standout the references lack); the static-override →
  litellm-catalog → safe-default capability registry is already the recommended design.
- **Permissions:** deny-first ordered-mode model with a single enforcement seam the model
  cannot reach; autonomous mode = deterministic hard-deny; mode-aware grant persistence;
  child-isolated sub-agent policies; blocklist re-check after PreToolUse hook rewrites;
  read-vs-write auto-approval is the default; per-tool trust overrides already loosen where
  tier would ASK while the catastrophic blocklist stays un-waivable in autonomous. **Ahead
  of all three on the pure-policy layer.**
- **Tools/sandboxing:** SSRF defense-in-depth in `web_fetch` (obfuscated-IPv4 validation,
  connection pinning, per-hop redirect re-pinning, decompression-bomb defense, output
  defang) — **the most thorough of any reference**; uniform centralized process-tree
  teardown on both timeout and cancellation (shell/hooks/MCP) — ahead of claw-code's partial
  teardown; correct concurrency gating (bash NEVER_PARALLEL; read-only-safe tools parallel,
  gated by both tier and `ConcurrencyClass`).
- **Context/memory:** preemptive auto-compaction at a real-token threshold; genuinely
  idempotent compaction (folds a prior summary instead of stacking — cleaner than
  goose/claw-code on re-compaction); stable/dynamic system-prompt split; just-in-time
  agent-guide discovery (`AGENTS.md` / `CLAUDE.md` / `ZAK.md` ancestor-chain + workspace README)
  with caps + content-hash dedup; a generic `PreLLMCall` recall seam a Mind injects
  fenced-untrusted context through (the harness ships no memory of its own);
  per-role summarizer routing.
- **Extensibility:** clean-room no-SDK MCP stdio client with qualified tool naming into one
  registry; a broad lifecycle-hook set with shell + in-process handlers and allow/block/warn
  semantics; three-tier progressive-disclosure skills with runtime authoring; trust-gated
  error-isolated plugins; schema-filtered sub-agents (a planner structurally cannot edit)
  with concurrent fan-out; lazy tool discovery with budgets.
- **Interfaces:** a hard three-layer thin-client split with zero agent logic outside the
  core; a single-event-loop FastAPI server with REST + SSE + WebSocket, bearer auth (header
  + browser-subprotocol fallback), 409 single-turn-per-session, a permission-approval bridge
  that pauses a turn over WS, and a parity test asserting in-process and over-server event
  streams match. **Server transport quality rivals goose/claw-code.**
- **Sessions:** atomic temp+rename writes, versioned, crash-safe, corruption-guarded (only
  the query/search/fork and CLI-resume wiring is missing — tracked above).
- **Ops:** layered pydantic-settings config with secrets isolation and a docs-enforcing
  compliance test (incl. the version-sync guard); a CI-gated deterministic behavioral eval
  harness that **exceeds claw-code (which has none)**; a typed `AgentEvent` stream as a
  structured observability backbone with a published JSON Schema; one-command verification
  (`poe check`); `uv.lock` committed with per-wheel SHA-256 hashes.

---

## Reading the priorities

- **Start with the dead-wired cluster (#1, #2, #5, #9, plus #4).** Highest ROI: the seams
  exist, so these are wiring jobs that close real reliability/cost gaps, and they make
  Zak-Code visibly better on real (non-local) models immediately.
- **Then the operability gate (#3 headless `run`).** It unblocks scheduling (#13),
  CI use, and ACP (#21) — a force multiplier.
- **The two safety holes (#6, #7) are cheap and worth doing early** — single-chokepoint,
  conservative, and they make advertised controls actually true.
- **The XL items (#14 isolation, #21 ACP) are real but additive** new surfaces with no
  current in-repo consumer — schedule deliberately, not reflexively.

## Method note

Generated by the `zakcode-parity-review` workflow (script preserved under the session's
`workflows/scripts/`). 8 lanes × 4 sources = 32 Explore-agent readers producing structured
capability inventories; 8 gap-synthesis judges; adversarial verifiers re-checking every
"behind" claim against Zak-Code's real code (this is why some inventory-stage gaps do not
appear here — they were refuted as already-present). The clean-room rule was enforced in
every claw-code reader's prompt; `[CLEAN-ROOM]` items must be re-expressed in our own design.

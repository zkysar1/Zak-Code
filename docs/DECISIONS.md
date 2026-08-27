# Architecture Decision Records (ADRs)

Chronological log of significant, hard-to-reverse decisions. Newest at the bottom.
Format: each ADR has Context, Decision, Consequences, and Status.

---

## ADR-0001 — Clean-room implementation (no forking of prior art)

- **Status:** Accepted (2026-05-30)
- **Context:** We have access to a community reverse-engineering of Claude Code
  (`claw-code`, incl. a Rust port + snapshots of Claude Code's tool/command surface),
  plus open-source agents Hermes (Python) and goose (Rust). Forking the Rust port means
  inheriting an admitted MVP and a steep language; using leaked TypeScript source is
  legally/ethically unacceptable and not vendor-agnostic.
- **Decision:** Build Zak Code **clean-room**. We study architecture and public docs of
  prior art and re-express ideas in our own design. We never copy leaked/proprietary code.
- **Consequences:** Full ownership and a clean IP story; more up-front design work. The
  reverse-engineered material is kept **outside** the repo as read-only study material.

## ADR-0002 — Language: Python

- **Status:** Accepted (2026-05-30)
- **Context:** Choices were Python, TypeScript/Node, or Rust. Hard requirement:
  vendor-agnostic. Soft requirement: readable/extensible for building-and-learning.
- **Decision:** **Python 3.11+**, `src/` layout, package `zakcode`, managed with `uv`.
- **Consequences:** Immediate access to `litellm` (100+ providers) and the richest agent
  ecosystem; most legible for iteration. We forgo Rust's raw performance and TS's tightest
  Claude-Code mirroring.

## ADR-0003 — License: MIT

- **Status:** Accepted (2026-05-30)
- **Context:** Public repo; owner wants freedom for future ("special things later"),
  including potential commercial use.
- **Decision:** **MIT.**
- **Consequences:** Maximal permissiveness and simplicity; no patent grant (acceptable).

## ADR-0004 — Provider abstraction via litellm; first-class Ollama + OpenAI

- **Status:** Accepted (2026-05-30)
- **Context:** Must be vendor-agnostic. Owner has access to **local Ollama** and
  **OpenAI** (not Anthropic) — which conveniently forces a genuinely provider-neutral core.
- **Decision:** All model access flows through a thin provider layer
  (`zakcode/providers/`) built on **litellm**. Default dev model is a local Ollama model;
  OpenAI is the first cloud target. Any litellm provider works via config.
- **Consequences:** Switching providers is config-only. We must handle provider feature
  gaps (e.g. local models lacking native tool-calling) in the provider layer.

## ADR-0005 — Three-layer architecture (core engine / API server / clients)

- **Status:** Accepted (2026-05-30)
- **Context:** Owner wants an API-based core (Claude Code SDK shape) so the CLI is one of
  several possible interfaces (web app, other apps).
- **Decision:** **(1)** `zakcode` core engine as an importable library; **(2)**
  `zakcode-server` (FastAPI, SSE/WebSocket) exposing it over HTTP; **(3)** thin clients,
  CLI first. Clients call the core in-process or the server over HTTP.
- **Consequences:** Clean reuse across interfaces; a small amount of indirection. Detailed
  in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## ADR-0006 — Build via multi-agent workflow orchestration

- **Status:** Accepted (2026-05-30)
- **Context:** A large, multi-day, high-token initiative. Owner asked to "manage a team"
  and keep documentation under central control.
- **Decision:** Build through orchestrated **workflows** (fan-out research, per-subsystem
  implementation pipelines, adversarial verification), with the orchestrator owning the
  docs. Process is defined in [`WORKFLOW.md`](WORKFLOW.md).
- **Consequences:** Parallelism and rigor at scale; requires disciplined scoping and
  verification so parallel work stays coherent.


## ADR-0007 — Reabsorb the provider package into the core (one src tree)

- **Status:** Accepted (2026-06-10)
- **Context:** Post-M11 extracted the provider contract into an in-repo editable package
  (`packages/zds-llm-provider`, provider track "M-7/8/9") to make portability a packaging
  fact. In practice zak-code was its only consumer (verified 2026-06-10: no other local
  repo references it; never published to an index), and the split's costs were real:
  four re-export shim modules, a second pyproject/mypy/pytest surface, the package's 92
  tests collected by neither local pytest nor CI, and "where does X live?" confusion for
  the owner. Owner asked for the lowest-cognitive-load structure.
- **Decision:** Move the package's modules into the core where their shims already
  pointed — `zakcode/messages.py`, `zakcode/usage.py`, and
  `zakcode/providers/{base,text_tools,structured,bitnet,claude_code}.py` — merge its
  tests into `tests/`, and delete `packages/`. The vendor-agnostic boundary is preserved
  **by contract test instead of packaging**:
  `tests/test_contracts.py::test_no_vendor_sdk_imports_outside_provider_layer` bans
  litellm outside its two named provider modules and bans vendor SDKs everywhere else.
- **Consequences:** One package, one test suite (1403 tests, all collected by default),
  no path-dependency machinery, bare `pytest` works via `pythonpath`. `BitNetProvider`
  (local OpenAI-compatible llama.cpp/BitNet servers) and `ClaudeCodeProvider` remain
  fully supported as `zakcode.providers.*` modules. If an external consumer ever
  materializes, re-extraction is mechanical — the modules stay pydantic-only by enforced
  contract, so the boundary survives the merge.


## ADR-0008 — Hierarchical task-network planning in the cognitive core

- **Status:** Accepted (2026-06-14)
- **Context:** The engine had a strong *reactive* cognition stack (recipe gate, stuck
  ladder, doom-loop guard, write-grounding, lessons) but **no task-management substrate** —
  no decomposition of a goal into steps, no progress tracking, no "what's next?" Worse,
  `ARCHITECTURE.md` claimed a "live TODO list re-injected near the end of context" that did
  not exist in code. The owner wanted Zak Code to be a drop-in replacement for Claude Code's
  domain-agnostic cognitive core (skills carry domain knowledge), with planning bolstered at
  the **first/near-term layer** — "what am I doing immediately next", decompose-to-primitives
  then execute — explicitly NOT the long-horizon, multi-layer planning a higher "mind"
  (ayoai-mind) owns. The owner had researched HTN/PDDL and asked that decomposition, "one of
  the most basic forms," be done "extremely well, not lazy."
- **Decision:** Add a **hierarchical task network** as the engine substrate (`zakcode/tasks.py`):
  every node is **compound** (a goal that must be decomposed into `children` before it is
  actionable; its status is *derived* from its children, never set directly) or **primitive**
  (a directly-executable step). The network computes the **frontier** (next actionable
  primitive), enforces a single focused `in_progress` step, and renders the live checklist.
  The model authors it via the **`update_plan` tool** (full-replace, TodoWrite-style — robust
  for weak models; the harness re-numbers ids and re-derives invariants each edit). The loop
  **re-injects** the plan near the end of context each iteration (ephemeral, non-persisted —
  cache-safe) and runs a **bounded, self-arming plan gate** that refuses to quietly finish a
  turn with open steps (nudge → complete `degraded`, never deadlock). The plan **persists on
  `Session.task_network`** so it spans turns. A `task_update` `AgentEvent` lets clients render
  the list. **The engine owns decomposition structure + discipline; the *method* for how to
  decompose a given kind of task is domain knowledge that stays in skills**, preserving the
  clean-room, vendor/domain-agnostic boundary.
  - Owner-chosen design forks: **hybrid enforcement** (model-driven tool + harness
    re-injection + self-arming gate, not a hard up-front planning gate); **persist in session**
    (not per-turn); **real hierarchy, done rigorously** (not a flat checklist; but a
    lightweight HTN — no PDDL solver/world-model in the core, which would require a domain
    model the architecture places in skills).
- **Consequences:** The engine now plans proactively, not just recovers reactively, and the
  long-aspirational re-injection claim is real. Decomposition method is `kind`-inferred (a step
  with subtasks is compound, else primitive), so the model expresses hierarchy by nesting, not
  by a flag — there is no model-settable "compound-but-undecomposed" state (`undecomposed()`
  and its advisory are defensive, reachable only by constructing a network directly). The plan
  gate adds at most `_MAX_PLAN_NUDGES` (2) extra iterations to a struggling turn. An unfinished
  plan deliberately carries across turns (and is re-nudged) until completed or cleared — correct
  HTN behavior (don't silently abandon a plan), at the cost of a stale incomplete plan haunting
  later turns until the model clears it with `update_plan`. A future enhancement could make
  `kind` model-settable to activate the under-decomposition gate, and route a `planner` role
  model (`Settings.model_roles`) to a cheaper model for decomposition.

  - **Update (2026-06-14):** the research survey (`docs/research/task-planning-decomposition.md`)
    confirmed the design and prioritized follow-ups; the two **P0**s are now implemented on this
    branch — **R1** a domain-agnostic project-verifier gate (`agent/verify.py` +
    `Settings.verify_command`: a turn that changed code can't finish until the configured
    tests/lint command passes; ends `verification_failed` (degraded) after a bounded number of
    attempts; inert when unset), and **R2** optional task **dependencies**
    (`Task.blocked_by`, sanitized into a DAG, frontier-aware). The undecomposed-gate /
    model-settable-`kind` and planner-role-model ideas remain future (P1/P2) work.

  - **Update (2026-06-14, follow-ups):** the P1/P2 research recommendations are now implemented on
    this branch too — **R3** capability-triggered decomposition (a stuck primitive step gets a
    "break it into sub-steps" nudge; prompt complexity floor + anti-over-decomposition guidance),
    **R4** the read-only planner can emit a *structured* plan (`update_plan` added to its toolset;
    planner-role model routing already existed), **R5** an opt-in, off-by-default `require_plan`
    gate (withhold the first mutating tool until a plan exists; bounded, fail-open), and **R6** a
    shared structured-handoff instruction on every sub-agent. **R7** (a facts/assumptions ledger)
    stays DEFERRED, per its own "watch, don't build" recommendation — it belongs to the higher-level
    mind, not the near-term core. The single-threaded inline design was kept (no planner/executor
    split) per the report's "keep it sharp; simple beats agentic" caveat.

  - **Update (2026-06-15, issue #32 — stale-plan auto-clear):** generalized the turn-start plan
    reset from "drop only a COMPLETED plan" to also drop an **abandoned** one — an unfinished plan
    that has sat byte-identical across `_MAX_PLAN_IDLE_TURNS` (3) consecutive turn-starts (the model
    neither advanced nor edited it). Without it, a plan the model forgot to clear re-injected into
    context + spent the plan-gate's nudges + flagged the turn `degraded` on *every* later turn — a
    recurring tax for weak local models. Deterministic (a `(id, status, title)` progress signature
    compared across turns, via the new pure `TaskNetwork.progress_signature()`), conservative (ANY
    edit resets the idle counter, so an active plan is never auto-cleared; a cleared plan is freely
    re-creatable), and bounded (a constant threshold; the per-turn plan gate already bounded each
    turn). Two append-only `Session` fields (`plan_idle_turns`, `plan_signature`); no schema bump;
    wired identically on both the buffered and streaming loop paths.

  - **Update (2026-06-15, primitiveness criteria — HTN cross-system survey):** surveyed three
    sibling HTN/decomposition implementations for transferable ideas — the Ayoai-Environment-Processor
    (a dual **HTN + A\*** planner over a STRIPS world-model, archive-informed cost weighting,
    LLM-grounded decomposition); **ayoai-mind** (the higher mind's aspirations→goals layer, a
    22-criterion goal-selector, scope classification, per-goal verification + blocker-TTL); and the
    omni continual-learning framework's `/decompose` skill (a model-driven HTN *protocol*:
    5-criterion primitiveness test, idempotency gate, verification-as-schema). **Headline:** the lean
    design here already *structurally* neutralizes most of what those systems add machinery for —
    full-replace `update_plan` moots idempotency back-references (ayoai-mind's own report says so
    verbatim), `kind`-inferred-from-nesting makes hierarchical cycles and "compound-but-empty" states
    unrepresentable, and the issue-#32 idle auto-clear subsumes stale-blocker TTLs. The one genuinely
    additive, philosophy-fitting idea (convergent across `/decompose` AND ayoai-mind) was implemented:
    the primitiveness **stopping-rule** now names the two criteria a single-action floor omits — a
    **clear done-condition** and **no approach decision still hidden in the step** — in both the
    `_PLANNING` system-prompt section and the `update_plan` tool description. Model-facing guidance
    only: no schema, no infra, cache-stable, and it directly targets weak local models (the reason the
    HTN exists); maps to this ADR's own noted future work (model-settable `kind` / under-decomposition
    gate). Deliberately kept OUT as higher-mind / domain-coupled territory: the STRIPS world-model + A\*
    ordering, archive-weighted goal scoring, scope classification, the 22-criterion selector, and the
    facts/assumptions ledger (**R7**). The cross-system evidence *reinforces* R7's "watch, don't build"
    deferral — every sibling keeps belief/world state at the long-horizon layer, never the near-term
    core, which is exactly the boundary this ADR draws.

  - **Update (2026-06-15, done-conditions via `note`):** completed the verification-as-schema half of
    the decomposition story WITHOUT adding a field. The survey's "every step carries a checkable
    done-condition" idea was already structurally present — `Task.note` has always been the
    "acceptance criterion" slot (parsed by `update_plan`, rendered into the re-injected plan), so a
    parallel `done_when` field would only have duplicated it (single-source-of-truth; "simple beats
    agentic" — the explicit push-back-on-over-building call). Instead: sharpened `note`'s role to *the
    step's checkable done-condition* (in the `Task` docstring AND the `update_plan` schema field
    description) and wired the `_PLANNING` guidance to record each primitive step's done-condition
    there. So the just-shipped "steps must be checkable" rule now names WHERE to capture the check, and
    that check rides in the plan the model re-reads every turn. No data-model change, no migration;
    pinned by `test_step_note_is_the_checkable_done_condition`.

## ADR-0009 — zakpick: task-category model routing via per-category `(model, source)` assignment

- **Status:** Accepted (2026-06-14)
- **Context:** Using one capable (often cloud) model for *every* internal prompt is the costly
  default — a one-line summary, a JSON gate, and a hard refactor all pay the same per-token price,
  and a small/local model would serve the easy ones fine. The seam for fixing this was reserved from
  day one: `ModelResolver.resolve(task=...)` (PKG-AUTO / D21) carries a `task` parameter the v1
  availability resolver ignores, explicitly so "zakpick" task-category routing could land later
  without API breakage. `Settings.model_roles` (planner / subagent / summarizer → a cheaper model)
  already proved the pattern for three named roles; zakpick is its general form across *every*
  internal call site. The question was the **interface**: what does the user actually turn?
- **Decision:** Add a third `default_model` sentinel, `"zakpick"` (alongside `"auto"` and a concrete
  model string). Under it the engine routes each internal prompt to the model the user assigned to
  that prompt's **task category** rather than using one model for everything. The interface is the
  **categories, not a dial**: `quick_code`, `deep_code`, `summarize`, `plan`, `delegate`, `classify`
  — mapped to the real call sites (deep_code/quick_code → the main generation turn; summarize →
  compaction; plan → the plan sub-agent; delegate → the general sub-agent / `task` tool; classify is
  a reserved seam with no live caller yet). The user parks a `(model, source)` pair on each category
  via the new `Settings.zakpick_models` (env `ZAKCODE_ZAKPICK_MODELS`, a JSON object keyed by
  category, each value `{model, source}`, `source` defaulting to `"groq"`). `source` is **deliberately
  separate from `model`** because a model name alone is ambiguous — `qwen3-32b` runs at Groq *or*
  locally — so the user states both; `source="local"` maps to the `ollama_chat` backend, any other
  source is the litellm provider prefix (`zakcode.providers.routing.ZakpickModel.litellm_string`
  joins them). Out of the box every category has a built-in default drawn from **Groq's published
  lineup** (Groq serves only open-source models, so the defaults double as a "which open-source model
  to download to run this category locally" guide), graduated by cost/capability
  (`DEFAULT_CATEGORY_MODELS`): classify→`llama-3.1-8b-instant`; summarize & quick_code→`gpt-oss-20b`;
  plan→`qwen3-32b`; deep_code & delegate→`gpt-oss-120b` (tools-**reliable**, the strongest).
  `llama-3.3-70b-versatile` is deliberately avoided (the registry flags it `tools_unreliable`).
  **The one automatic decision** is the quick-vs-deep coder split: a cheap, deterministic, offline
  classifier (`classify_main_turn` — a pure function of request length + context fraction + a latched
  struggle signal; **no** model call, **no** iteration-count input) picks which of the user's *two*
  coder models drives each main turn, and a one-way **soft latch** flips to `deep_code` the moment a
  real struggle signal fires (a stuck-ladder action, a doom-loop, or a verify-gate failure). This
  escalation only ever switches between the two coder models the user already configured — never a
  model Zak Code chose.
  - **The user owns the consequences.** Zak Code never substitutes a model the user didn't assign: a
    slow local model on a weak GPU is slow (their choice); a rate-limited or failing cloud model is
    handled by the existing `fallback_model` seam like any other provider error. Under zakpick
    `model_failover` uses *only* an explicit `fallback_model` (else the turn ends `provider_error`),
    and `model_resolution` stays `None` (no availability re-resolution) — there is no tier-based
    active escalation and no source-masking.
- **Rejected alternative (the key decision):** an earlier "dial" design — a local/cloud/save/max knob
  over a curated **4-tier ladder** with runtime source-masking and degrade-down logic. It was rejected
  because it made Zak Code **own a local-vs-cloud tradeoff and tier curation that belong to the user**.
  The category→model map moves every model-choice back to the user: they say exactly which model (and
  source) runs each category, and they own the result. A dial would have Zak Code guessing which
  provider to prefer and quietly degrading between them — precisely the ownership it should not take.
- **Clean-room provenance:** this is a **Zak-Code-original, beyond-parity** feature — none of the
  reference harnesses (Claude Code, Hermes, goose) ship task-category model routing, so there is no
  competitor design to mirror or re-express. Provenance is first-principles reasoning, recorded here.
  `providers/routing.py` imports **no** litellm/vendor SDK (the contract test stays green); the model
  strings in `DEFAULT_CATEGORY_MODELS` are *data* (like the candidate lists in
  `resolve._EXTERNAL_SOURCES`), not a hardcoded vendor sort.
- **Consequences:** Cost becomes a per-category lever the user controls without leaving the engine
  vendor-agnostic; the reserved `resolve(task=...)` seam is now realized end-to-end (config validator,
  `ZAKPICK_SENTINEL` + a `describe_resolution` branch in `resolve.py`, `Agent._resolve_task_provider` /
  `_main_provider_for` reusing the existing `_provider_for` cache, a `main_provider_for` seam + the
  classifier wiring + the soft latch in both loop paths, and a `category` field + `provider_for_task`
  hook on sub-agents). The CLI info panel, `/model`, and banner show a friendly per-category table
  (`model (source)`, never a raw slug); only categories with a live call site are advertised, so
  `classify` is never shown. **Deferred seams** (future work, with triggers): a cheap difficulty-
  classifier *model* for the `classify` category (today the split is heuristic-only — the category +
  its structured-output shape are pre-wired; build when a real gray-zone classification call site
  appears, **or** when an offline eval shows `classify_main_turn`'s routing accuracy sits well below
  ~80% on a representative coding set — the yardstick from *Intelligence per Watt* (arXiv 2511.07885;
  [`references/intelligence-per-watt.md`](references/intelligence-per-watt.md)), which finds an
  80%-accurate router captures ~80% of the achievable savings, so a crude heuristic near that ceiling
  is not worth upgrading); cost/price metadata on `Capabilities` (today the defaults encode cost by
  *curation*, not a price field; build when the engine needs to reason about price at runtime, e.g. a
  budget-aware router — also what powers the `/cost` "vs all-deep" savings estimate); and an
  `embeddings` category (build when an embedding call site exists).

## ADR-0010 — `deep_think`: opt-in best-of-N self-fusion deliberation

- **Status:** Accepted (2026-06-15)
- **Context:** zakpick (ADR-0009) is a cost-*down* axis — route each prompt to a cheaper model.
  The complementary axis is spending *more* deliberately on a genuinely hard sub-problem. The
  owner's original brief gestured at this ("maybe do a generate, then filter, then a critic"), and
  two public results sharpened it: OpenRouter's *Fusion beats Frontier*
  ([`references/fusion-beats-frontier.md`](references/fusion-beats-frontier.md)) — a panel + judge +
  synthesizer beats any single model, and crucially a model paired with **itself** ("self-fusion")
  jumped **+6.7 points**, so *the synthesis step itself* carries most of the lift — and *Intelligence
  per Watt*. OpenRouter's own guidance is that a coding agent should not run Fusion on everything,
  but **invoke it selectively, when the model judges a question worth the extra time and cost.**
- **Decision:** Add `deep_think`, an **opt-in tool** the model calls on one hard, self-contained
  question. It samples the agent's strongest model several times (best-of-N, diversity temperature),
  then makes one synthesis pass that reads the candidates and writes the single best answer
  (generate → critique → synthesize, in one bounded tool). Scoped to Zak Code's constraints:
  - **Self-fusion, one model, the user's own.** It samples the agent's *strongest configured* model
    — under zakpick the `deep_code` category, else `default_model` (via `Agent._resolve_task_provider`
    + the new `Sampler` seam on `ToolContext`). It never reaches for a model the user did not assign,
    so it owns no provider tradeoff (the line ADR-0009 draws). We deliberately did **not** build the
    multi-provider panel + separate judge model: the self-fusion finding says synthesis carries most
    of the lift, and single-model best-of-N keeps the feature small and vendor-agnostic (no
    cross-model orchestration layer; a possible future seam if a measured need appears).
  - **The model decides, never automatic.** It is a normal tool — visible in the transcript, gated by
    permissions, `READ_ONLY` tier — invoked at the model's discretion, with a description that flags
    it as EXPENSIVE and for hard sub-problems only. Zak Code never fires it on the agent's behalf.
  - **Cost visible + bounded.** Its extra calls are attributed in the per-model `/cost` breakdown and
    folded into the shared turn-tree budget, so the 2–3× spend is never hidden, and `max_cost_usd` /
    `max_iterations` bound it. An operator can disable it with `tool_exposure_deny=["deep_think"]`.
  - **Degrades gracefully.** No `Sampler` wired (a bare/delegated loop) → a clean "unavailable" error,
    never a crash; a failed sample is tolerated; a failed synthesis falls back to the fullest
    candidate. Handlers never raise (the tool contract).
- **Clean-room provenance:** the *idea* (ensemble/self-consistency + synthesis) is public
  (self-consistency decoding; OpenRouter's Fusion); the implementation is original and uses only the
  vendor-agnostic `Provider` abstraction — `deep_think.py` imports no litellm/vendor SDK.
- **Consequences:** the agent gains a deliberate "think harder" rung that composes with zakpick's
  cheap→capable escalation (quick_code → deep_code → *deep_think on deep_code*), entirely within the
  user's configured models and spend controls. **Deferred seams** (future, with triggers): a
  multi-provider Fusion panel + judge (build only if single-model best-of-N proves insufficient on a
  measured task set — it is a 2–3× cost, opposite the cost-down motivation, so it stays opt-in and
  late); a `deep_think` zakpick category so the user can assign a *distinct* model to deliberation
  (build when someone wants deliberation on a different model than `deep_code`).

## ADR-0011 — The quality engine: small-model fan-out for quality (off-by-default seams)

- **Status:** Accepted (shipped, 2026-06).
- **Context:** zakpick routes per task and `deep_think` deliberates on one hard question, but neither
  raises the floor on the SMALL models the cost-down axis leans on. The bet (measured, not assumed):
  a fixed small-model ceiling is beaten by STRUCTURE — decompose → fan out → judge/score → iterate —
  not by a bigger model, so ~10 cheap calls + selection can beat one big call.
- **Decision:** Build a vendor-agnostic library of quality primitives (`src/zakcode/quality/`) —
  LLM-as-judge (binary / pairwise / N-judge vote), best-of-N generation, **oracle-first** selection
  ("oracle for *works*, judge for *good*"; external sound verification beats LLM self-critique), IAUS
  rubric scoring + a ship/iterate cost-gate, a bounded refine loop, and a `best_attempt` core. Wire it
  into the live agent through two seams, both **OFF by default** so the default path is byte-identical:
  **seam A** (a quality gate in `agent/loop.py` — scores the written diff after the completion critic)
  and **seam B** (best-of-N retry in the `Agent` — on a STALLED turn, fan out N isolated attempts and
  adopt the first that verifies, by DIFF, never a blind overwrite).
- **Why this shape:**
  - **Oracle-first, judge-second.** The measured failure mode was selection, not generation: a single
    judge mis-ranks. Filter by a deterministic oracle (the verifier), then judge-rank the survivors —
    and pairwise beats absolute scoring (RankPrompt).
  - **Off by default, proven byte-identical.** Every seam guards on its flag first, so the default
    path costs nothing and the suite stays green unchanged. Measure-before-integrate: each seam
    shipped with a bench (`run_quality`, `run_bestof`) before being trusted.
  - **Safe adoption.** Seam B copies only source (size-capped), verifies in isolation, and adopts by
    diff with a TOCTOU guard — never a blind overwrite of the user's workspace.
- **Measured:** the quality gate is a NICHE tool (neutral on a task whose failures are stalls, not
  weak completions); **best-of-N is the win** (4/5 vs 1-big 3/5 across the suite, the edge on hard
  tasks), so seam B deploys it where it pays. Validated live: seam B fired on a real stall, ran 3
  isolated attempts, and adopted a verified one by diff.
- **Consequences:** an opt-in quality layer that composes with zakpick (cheap models) and `deep_think`
  (deliberation). **Deferred:** best-of-N *plans* (seam C — low value; the HTN in `tasks.py` already
  decomposes); a wired cost-fraction cap (today bounded by `best_of_attempts` + the per-turn budget);
  rolling seam B's retry spend into the parent session's cost report.

## ADR-0012 — Model-facing skill invocation (`use_skill`) + skill chaining

- **Status:** Accepted (shipped, 2026-06).
- **Context:** Skills (M7) shipped L0 discovery, L1 lazy bodies, the `ON_SKILL_SELECTED` learning
  signal, `save_skill`, and the human `/<name>` path — but the catalog told the model to "invoke a
  skill by name" with **no tool to do so**. Only a human could invoke a skill; the model couldn't,
  and skills could not chain. `Agent.invoke_skill`'s own docstring named the gap ("a future
  model-facing tool").
- **Decision:** Add the model-facing **`use_skill`** tool. It loads a skill's body by name and returns
  it as the **tool result** (not a session message), fires `ON_SKILL_SELECTED` with `source="tool"`,
  and is `READ_ONLY` / `NEVER_PARALLEL`. It reads a `SkillResolver`/`SkillLoad` seam off the
  `ToolContext` (mirroring `sampler`/`spawner`); both invocation paths share one `_load_skill_body`
  core. Registered **only when `enable_skills`**, so the default tool surface is byte-identical.
- **Why this shape:**
  - **Result, not session surgery.** Returning the body as the tool result (vs. injecting a user
    message mid-turn) is the natural context-injection point and can't reorder the
    assistant/tool-result exchange the loop is mid-assembly on.
  - **Chaining falls out for free.** Because invocation is a tool call, a skill body whose step says
    "now use the X skill" is carried out by the model calling `use_skill` again — no new machinery.
  - **One core, two paths.** CLI `/<name>` and the tool both run `_load_skill_body` (resolve → lazy
    read → defang → fire signal); only delivery and `source` differ, so behavior stays consistent.
  - **Read-only + off by default.** The tool only reads a file and fires an observe-only hook; any
    writes its instructions call for go through the ordinary file tools' own permission gates.
- **Validated:** a live 3-skill relay (`bench/run_skill_chain.py`) — `gpt-4o-mini` invoked
  `relay-start → relay-middle → relay-finish` in one turn, every call `source=tool`, all three
  execution markers landed, `completed` for ~$0.003.
- **Consequences:** skills are now a model-driven, composable capability surface, and the
  `source` field lets a learning mind weight model- vs. operator-driven selections.
- **Follow-ups (shipped 2026-06-20):**
  - **Sub-agents can invoke + chain skills.** The parent's resolver is threaded into the
    `SubAgentRunner` → each child `AgentLoop`, and `use_skill` is registered on the child registry.
    The general-purpose delegate (full toolset) gets it; the read-only **planner** does not (its
    tool subset omits `use_skill`). A child resolves against the same registry and draws from the
    same per-turn budget. Attribution is correct per caller: the `use_skill` tool reads a
    `caller_query` off the `ToolContext` (each loop stamps its own turn's prompt), so a sub-agent's
    `ON_SKILL_SELECTED` records the *child's* task, not the parent's originating turn — while the
    shared registry/budget still come from the parent's resolver.
  - **Per-turn skill-invocation budget** (`skill_invocation_budget`, `0` = unlimited): each
    model-driven (`source="tool"`) `use_skill` draws one unit, shared across the whole turn-tree and
    reset per top-level turn; over the cap the tool returns a `denied_reason` (no body, no signal),
    bounding a runaway/cyclic chain (A→B→A) more tightly than `max_iterations`. A human `/<name>` is
    never throttled. The running count is logged and surfaced in `/skills`.
  - **Validated live:** branching routing (`bench/run_skill_branch.py`) — one skill reads input and
    calls a different next-skill per case (urgent vs. routine), correct both ways; and the budget
    capping the relay at N invocations while the model finishes gracefully past the denial.
  - **CLI `/<skill>` runs the turn immediately (2026-08-19).** The REPL used to load the body
    lazily and print "describe your task and it will apply" — a second message was needed before
    anything happened, which is not Claude Code's slash semantics and confused the first live
    Claude-Mind boot (`/start sera` loaded and then sat at the prompt). Core gained
    `Agent.compose_skill_turn` (load + defang + signal, **no session mutation**, returns
    `turn_text`); the REPL streams that text through the same path as any typed message, so the
    slash command IS the turn. `Agent.invoke_skill` is retained, rewritten over compose, as the
    deferred stage-context variant for embedders. First defect surfaced by the Serene
    dogfooding engagement.
  - **The composed slash turn carries invocation provenance (2026-08-19, same live boot, one
    fix later).** Running the skill immediately was not enough: the composed text
    (`[skill: start]` + body) never said WHO invoked it, and frameworks ship skills whose own
    rules forbid model self-invocation (a Mind's "control skills: Claude MUST NOT invoke
    /start"). A rule-following model (Gemini, no Claude-Code training priors) therefore
    refused the operator's own keystroke: "user-only command, please run this yourself in the
    terminal" — answered TO the terminal. Two-part fix, both in core: `compose_skill_turn` now
    emits Claude Code's command-expansion frame (`<command-message>` / `<command-name>` /
    `<command-args>`, echoing the TYPED token — under `triggers:` routing that may differ from
    the resolved skill name), and `SkillRegistry.render_catalog` states the contract in the
    system prompt (a user message BEGINNING with `<command-name>` = the human typed it; rules
    limiting a skill to user invocation are satisfied). The `use_skill` tool path keeps its
    distinct `[arguments: …]` frame deliberately — the asymmetry is what makes the provenance
    signal informative, and it is conformance-pinned in both directions. Vendor-agnostic by
    design: the explicit prompt contract does the work for models with no Claude-Code priors;
    the CC-shaped markers do it for models with them.
  - **Headless one-shot slash dispatch (2026-08-20, closes #148).** `chat -p "/start sera"`
    used to hand the slash line to the model as prose — the REPL dispatched, the one-shot
    path did not, and cron/systemd boots are one-shots. Now the one-shot path routes through
    the same `_skill_command_turn` helper (same compose, same rendering, same provenance
    frame). Exit-code semantics chosen for scripts: a discovered-but-refused or unreadable
    skill exits **1** — a scripted boot must fail loudly, because the silent alternative is
    the model politely conversing with a cron job; an UNKNOWN `/token` (or a thin `--server`
    agent with no skills surface) falls through as plain text, since a one-shot prompt may
    legitimately begin with a slash-path.
  - **Block-form frontmatter lists (2026-08-20).** A pre-deployment smoke over a live
    Claude-Mind tree (78 skills, zero parse errors) found 60 of 78 declare lists in YAML
    block form (`triggers:` + `- "/start"` lines) — which the minimal parser returned as an
    EMPTY STRING, silently no-opping trigger routing (masked by name-matching) and breaking
    the extras-preservation promise for the majority spelling. `parse_frontmatter` now does a
    block-sequence lookahead; a bare `key:` with no items keeps its empty-string behavior,
    and a `- name: x` list-of-maps item survives as the string `"name: x"`.

## ADR-0013 — Workspace hook adoption: folder-trust ask-once in the interactive CLI

- **Status:** Accepted (shipped, 2026-08-19).
- **Context:** `settings_hooks` defaulted to a hard `false`, and NOTHING was said when a
  workspace carried a `.claude/settings.json` hooks block that was being ignored. The library
  principle ("every compatibility surface off by default — a foreign workspace never changes
  behavior un-opted-in", INTEGRATIONS.md) is right for embedders: the SDK passes
  `enable_settings_hooks=True` explicitly and works. But the interactive CLI inherited the
  library default *silently*, so a Claude-Mind workspace loaded, discovered skills, ran them —
  and dropped every hook. The first live Mind boot failed four layers downstream (`/start`
  refused on a missing hook-injected `MIND_SID`), and the in-session model diagnosed it as an
  unfixable environment problem. Second defect surfaced by the Serene dogfooding engagement;
  the operator's design review named the root cause: same core, different defaults per door,
  with the risky door silent.
- **Decision:** Make `settings_hooks` **tri-state** (`bool | None`, default `None` = unset).
  Explicit `true`/`false` (env or `.env`) is honored silently everywhere, unchanged. UNSET
  resolves per host: library/server → off (principle intact, embedder behavior unchanged);
  the **interactive CLI** → Claude Code **folder-trust semantics** — when the workspace
  actually declares loadable hooks, ask the operator once (`always for this workspace / this
  session only / never for this workspace`) and remember `always`/`never` per resolved
  workspace path in `~/.zakcode/workspace-trust.json`. Headless (`-p`, no tty) never prompts:
  one dim pointer line, hooks stay off, scripted behavior deterministic. Every remembered
  state still prints one line at startup — the defect was SILENCE, so even "off, as you chose"
  says so. Policy + persistence live in core (`zakcode.workspace_trust`:
  `resolve_hooks_adoption`, `hooks_decision`, `remember_hooks_decision`;
  `summarize_settings_hooks` in the settings loader is the detection half); the CLI supplies
  only the ask-UI (`_ask_hooks_adoption`) — UI in the interface, policy in the core, per the
  repo's core/interface non-negotiable.
- **Alternatives rejected:** flipping the default to `true` globally (executes workspace shell
  hooks by surprise in every embedder/server — a real security regression, and it breaks the
  documented library principle); a warning banner that tells the operator to export an env var
  (that is a workaround shipped as a fix — the operator explicitly rejected it); prompting
  every session without persistence (nags the common case; Claude Code's folder trust is
  remembered, so ours is).
- **Consequences:** `zakcode chat` in a Claude-Mind workspace is now zero-config: answer `y`
  once and the Stop-hook loop, PreToolUse injection, and SessionStart hooks all fire from then
  on. The trust store is keyed per surface (`{path: {settings_hooks: …}}`) so permissions /
  statusLine / output-styles can join the same one-decision flow later without re-asking.

## ADR-0014 — Provider 429s are survivable: fixed backoff horizon, smoothing, resumable stop

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** A 429 RESOURCE_EXHAUSTED mid-run on Vertex AI (gemini-2.5-flash, dynamic
  shared quota) was retried 3 times over ~6 seconds, then the run hard-stopped with a
  mid-stream fallback error and the operator read a 42-iteration / 2.45M-token run as lost.
  Google's own guidance for dynamic shared quota is that a 429 is *temporary contention*:
  the remedy is minutes-scale exponential backoff with jitter plus traffic smoothing — a
  3-attempt counter is the wrong shape entirely. Separately, the engine already persisted
  every iteration at message boundaries, but discarded the final partial streamed text and
  never *said* the session was resumable, so a survivable stop looked like a total loss.
- **Decision:** Three changes in the core engine, none of them knobs (no-knobs ruling —
  the task brief proposed `ZAKCODE_PROVIDER_RETRY_ATTEMPTS` / `..._MAX_SECONDS` envs and
  they were deliberately not added; the former `provider_max_retries` setting was removed):
  1. **Fixed retry policy** (`agent/loop.py`): a pure rate limit (429 / transient 5xx)
     retries with equal-jitter exponential backoff (base 2s, per-wait cap 60s), honoring
     `Retry-After` up to 120s, for as long as wall-clock elapsed since the first 429 stays
     inside a 300s horizon (`_RATE_LIMIT_RETRY_HORIZON`). Wall clock — not an attempt count
     or a sum of sleeps — so zero-delay `Retry-After` sequences cannot retry unboundedly.
     Timeouts and provider-rejected tool calls keep a small fixed bound
     (`_MAX_INTERRUPT_RETRIES = 3`): waiting minutes on a hung backend helps nobody.
  2. **Traffic smoothing** (`providers/litellm_provider.py`): request STARTS on one
     provider instance are paced ≥1s apart (`_pace`). Invisible on real calls (every model
     call takes longer); it only shaves the burst edge that dynamic shared quota punishes.
  3. **Resumable stop** (`agent/loop.py`): when retries genuinely exhaust, the turn ends
     `provider_error` as before — but a mid-stream failure's partial *text* is now persisted
     (with an "interrupted partway; continue from where it left off" rail; partial tool-call
     fragments stay discarded — unexecutable), and the stop status names the recovery:
     the session is saved, the next message or `/resume` continues it. No parallel store —
     this is the existing #184 session/transcript machinery doing what it already did,
     plus the two missing pieces (the tail, and saying so).
- **Alternatives rejected:** the brief's env-var knobs (cognitive load; one way of doing
  things); retrying mid-stream 429s (re-issuing re-yields text the client already rendered);
  litellm-level `num_retries` (two compounding retry layers — the loop stays THE mechanism).
- **Consequences:** a quota storm now costs up to ~5 minutes of jittered waiting instead of
  killing the run at 6 seconds; fleets desynchronize instead of re-spiking in lockstep; and
  when the horizon is genuinely exhausted the operator is told, truthfully, that nothing
  was lost.

## ADR-0015 — Stuck ladder gets a step-back rung: attack the premise before giving up

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** A field turn (serene) burned 17 iterations trying to add a knowledge-tree
  node: every attempt varied the METHOD (`tree` as a shell command, `tree.sh`, `tree.py`
  with guessed flags) while sharing one wrong PREMISE — that `world/knowledge/tree/`
  existed relative to the cwd (the real tree lived under an external `.mind-data/` root).
  The ladder nudged, narrowed to read-only, and stopped; the read-only rung even had the
  right tools in hand and still probed the wrong assumed path. The operator then typed
  "take a step back, and think about what the right path is, and try again" — and the
  model recovered in two probes: one failing List on the assumed path, then a List from a
  root it could verify, walking down to the real location. The intervention worked because
  it attacked the assumption, not the effort; and notably its first probe FAILED.
- **Decision:** A fourth rung, `STEP_BACK`, between narrow and stop (defaults now
  nudge@3 → narrow@4 → step-back@5 → stop; no knobs). Once per turn, the loop injects a
  reassessment rail modeled on the operator's message: state the goal in one sentence,
  name the assumption every failed attempt shared, verify it from the ground up with
  read-only probes (list a directory you KNOW exists and walk down; run the command with
  `--help`; read the file you believe is there), and only then act. Firing it RESETS the
  streak and the per-call failure counts — the field recovery's first post-prompt probe
  failed, and without the reset that honest probe would have tripped the stop mid-recovery.
  A second climb re-runs nudge and narrow and the second arrival at the rung stops the
  turn, so a genuinely doomed turn ends at ~10 stuck iterations instead of 5.
- **Alternatives rejected:** making the rung read-only-restricted (the field intervention
  was prompt-only and the model probed voluntarily; restriction would block a
  legitimately-correct immediate action); resetting only partially (any post-reassessment
  failure would then compound leftover streak and stop before discovery completes);
  re-arming step-back after a TURN_END veto (unbounded ladders; one reassessment per turn).
- **Consequences:** turns that die on a wrong premise — wrong path, missing tool, changed
  interface — now get one explicit chance to re-derive it before stopping; doomed turns
  cost up to 5 more iterations; every step-back turn reports `degraded` so a recovery is
  never mistaken for a clean run.

## ADR-0016 — One Ctrl-C gesture can never kill the session (or the cockpit)

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** Field incident (serene): the operator hammered Ctrl-C at a turn that looked
  hung (a model call chewing a 529-line `git status` prompt), and the whole cockpit died —
  every tmux pane, one gesture. Three stacked defects: (1) presses landing in the gaps of
  the mid-turn interrupt teardown (during the drain pump, the wait-line stop, the notice
  print) escaped as raw KeyboardInterrupts and unwound the REPL; (2) the freshly-armed
  double-press window then read any surviving rapid press as the deliberate "yes, exit"
  second press; (3) the REPL's turn call caught only ProviderError, so ANY other escape —
  interrupt or crash — exited the REPL. And the cockpit chat pane deliberately chains the
  tmux session teardown onto REPL exit (#210, the requested double-press-closes-everything
  affordance), which turned each of those escapes into a full cockpit kill.
- **Decision:** Three layers, no knobs:
  1. **Atomic teardown** (`_absorb_interrupts`): every step of the mid-turn interrupt
     teardown retries through further presses; between the interrupt and the next live
     prompt a Ctrl-C can only mean "still mashing at the hung thing". The drain also
     switched from `run_until_complete(task)` to `asyncio.wait(timeout=5s)` — immune to
     the bpo-22429 hang AND to a tool that ignores cancellation (state is persisted at
     message boundaries, so abandoning a stuck drain is safe).
  2. **Gesture refractory** (`_ctrl_c_disposition`): a press within 0.35s of a
     MID-TURN-armed window is the same hammer gesture and is absorbed — for as long as
     presses stay rapid. A deliberate second press after reading the notice still exits:
     two presses total, the requested affordance. At an idle prompt nothing changes — a
     rapid double-press there remains the documented exit gesture.
  3. **REPL survival arms**: both prompt loops absorb an escaped KeyboardInterrupt and
     survive any turn-level exception ("turn failed: …" + prompt back). The session is
     persisted either way; a bad turn must never cost the transcript or the cockpit.
- **Alternatives rejected:** SIG_IGN during teardown (platform-uneven, and a stuck teardown
  would make the CLI Ctrl-C-immune); dropping the cockpit's exit chain (double-press
  close-everything is the requested behavior); a refractory at the idle prompt too (breaks
  the documented rapid double-press exit and its tests).
- **Consequences:** interrupting a hung-looking turn is now always safe regardless of how
  many times the key is hammered; deliberate exit still takes exactly two presses; a
  crashed turn reports and returns to the prompt; the only way a REPL (and thus a cockpit)
  ends is EOF, `/exit`, or a deliberate double-press.

## ADR-0017 — Compound requests decompose into plan steps; a coverage backstop guards the finish

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** Field incident (serene): one message asked for two skills — a
  /fresh-eyes-code review AND an /encode-session. A mid-turn interjection plus a session
  replay evicted the first ask from conversation memory, and the turn ended "done" having
  run only the second. Conversation memory is the wrong home for a multi-part ask: it is
  exactly what interruptions and resumes destroy. The plan (TaskNetwork) is the right
  home — it lives in session state, `_reset_stale_or_completed_plan` carries unfinished
  plans across turns, and the plan gate already refuses to let a turn quietly end with
  open steps. The gap was purely that nothing SEEDED the plan from the request, and the
  prompt's old "skip planning under three steps" guidance actively suppressed planning
  for two-part asks.
- **Decision:** Two mechanical layers plus one prompt line, no knobs:
  1. **Arrival-time seeding** (`_seed_plan_from_request`): when a user message names >=2
     skills that resolve against the live registry (slash tokens only, unknown names and
     path-like slashes never match), each not-already-planned skill gets one primitive
     step ("run /<name>") appended at turn entry, in both the buffered and streaming
     paths. Enforcement is then the EXISTING plan gate's job — no new nag machinery.
     Deliberately mechanical: no NL decomposition of every message (ceremony on "go
     ahead", latency, drifting judgment); the model still authors real plans itself.
  2. **Completion-time coverage backstop** (`_skill_coverage_nudge`): when a turn is
     about to complete, any skill the request explicitly named that was neither invoked
     via use_skill (errored loads don't count) nor mentioned by the plan in ANY state
     gets one nudge naming it ("The request also asked for /x — run it now with
     use_skill, or say explicitly why it should be skipped"). One-shot per turn; a plan
     mention in a terminal state is a deliberate decision and is not re-litigated.
  3. **Prompt guidance:** the planning section now says a request asking for MORE THAN
     ONE thing records each part as its own step before starting, and the skip line
     changed from a step-count threshold to "one straightforward thing" — the old
     "fewer than three steps" wording was WHY two-part asks got no plan.
- **Alternatives rejected:** decomposing every say via an extra model pass (latency +
  ceremony + judgment drift — the seeding layer is deterministic and covers the observed
  failure); a coverage HARD-stop instead of a nudge (a model can have a legitimate reason
  to decline — the nudge forces it to say so out loud); seeding for N=1 requests (the
  backstop already covers single-skill drops without adding plan ceremony to every
  "/encode-session please").
- **Consequences:** multi-skill asks survive interruptions and replays as open plan
  steps; a request-named skill can no longer be silently dropped — it either runs or the
  model explicitly declines it in text; single-part requests stay ceremony-free.

## ADR-0018 — Degeneration is contained: model-default temperature, a per-completion output cap, and a repetition guard

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** Field incident (serene, the morning zakpick put Gemini on the quick
  categories): gemini-2.5-flash-lite fell into the documented Gemini 2.5 repetition
  attractor — "I will now provide the information you requested." streamed once a second,
  indefinitely — and only the operator's Ctrl-C ended the turn. Three harness facts made
  the model's failure OUR incident: the config defaulted `temperature` to 0.0 (Google
  explicitly warns Gemini 2.5+ below 1.0 loops), no per-completion output cap was sent
  (Gemini's own cap is ~65k tokens, all billed), and nothing recognized repetition. The
  empty-completion give-up gate fired correctly first (flash-lite's paired failure mode)
  — its nudge then prompted the apology spiral.
- **Decision:** Three pieces, no knobs:
  1. **Temperature default is None — send nothing.** Every backend runs at its own
     intended default (Gemini 1.0, Claude 1.0, llama.cpp ~0.8). The harness-wide 0.0 was
     fake determinism inherited from local-model habits; it second-guessed every model's
     tuning and is a documented loop trigger on Gemini 2.5+. An explicitly configured
     value — 0.0 included — is still sent verbatim. (The structured side-call's forced
     per-call 0.0 for schema extraction is unchanged: short, validated, retried.)
  2. **Per-completion output cap** (`max_tokens` 8192 on every call, per-call overrides
     win, `drop_params` drops it where unsupported). Bounds any degeneration — and its
     bill — to one bounded completion; a legit long answer continues through the loop's
     existing length-continuation path (capped at 3).
  3. **Repetition guard** (`agent/degeneration.py` + both turn paths): a pure tail
     detector (dominant-line branch, mutation-tolerant, 12-of-15 convicts; exact-period
     branch for no-newline/control-char floods) probed periodically during streaming
     (cutting a runaway stream within seconds) and on every buffered no-tool-call
     completion. First conviction: the garbage is discarded BEFORE the transcript (same
     recovery contract as ModelOutputRejected — the attempt is still billed), one
     corrective rail, fresh retry. Second: honest `stop_reason="degenerated"` (degraded,
     non-vetoable like recipe_stalled — re-prompting a twice-collapsed model produces
     more of the same). Tool-calling completions are never judged: a batch doing work
     rides along.
- **Alternatives rejected:** frequency/presence penalties (per-backend support is uneven
  and Google's own penalty implementations have caused other degeneration); a
  temperature FLOOR for Gemini only (special-casing one vendor hides the general
  contract — model defaults are the one honest no-knobs rule); detector-only without the
  cap (an undetected loop shape would still stream 65k billed tokens); a hard-stop on
  first conviction (a single bad sample deserves one fresh chance — state is already
  persisted at message boundaries, so the retry is free).
- **Consequences:** a degenerating model now costs seconds and at most ~8k tokens, twice,
  then ends honestly; operators see "response degenerated into repetition" instead of a
  screen of garbage; every backend runs at its vendor-intended temperature unless the
  operator says otherwise; `zakcode config` renders "(model default)" for the unset case.

## ADR-0019 — A missing optional capability is an executable remedy, not an absence; web egress carries privacy floors

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** Field observation (same deployment as ADR-0018): an agent asserted "web
  search is unavailable (missing ddgs)" from memory, baked the gap into a goal chain as a
  standing blocker, and never attempted the fix. Two harness facts fed that. (1) The
  dependency gate reads only WORKSPACE manifests, so the self-fix command
  `pip_install_hint` emits — targeting zakcode's own interpreter and zakcode's own
  declared `[web]` packages — was itself hard-DENIED in autonomous mode: the harness
  suggested a remedy its own gate then refused. (2) The fix strings were informational,
  and a remedy phrased as information gets narrated into plans instead of executed.
  Separately, that deployment operates under a strict data boundary, and web queries /
  fetch requests are the one egress surface where private text can be pasted out with no
  mechanical check.
- **Decision:** Two halves, no knobs:
  1. **Self-serviceable capability.** The dependency gate's declared set unions
     `harness_declared_packages()` — zakcode's own distribution requirements, every extra
     — so the exact remedy the harness emits passes its own gate. Every
     missing-optional-dep fix string is now the DIRECTIVE `install_now_fix()`: install it
     NOW by running the command, then retry; do not report the capability as unavailable,
     do not plan around it, do not hand the install back. Web tools stay
     always-registered — a missing dep surfaces at call time as this remedy, never as
     tool absence.
  2. **Web privacy floors.** `web_search` refuses (never truncates) a query over 400
     chars — a distilled question, not a paste — and refuses a query carrying a saved
     secret VALUE (the shared `SecretsProvider` scrub is the detector) or
     credential-SHAPED text (`redact_secrets`, with `{{secret:NAME}}` placeholders
     stripped first: the safe form must not be the refused form). The semantic half —
     no proprietary code, client names, personal data, internal hostnames — rides the
     tool description plus a Safety bullet in the system prompt. `web_fetch` refuses a
     raw saved-secret value in the url or a header (the placeholder is the sanctioned
     form) and credential-shape-screens header values (not the url: query strings
     legitimately match `token=…` assignment shapes).
- **Alternatives rejected:** hiding web tools when deps are missing (absence teaches
  "narrate the blocker"; presence-with-remedy teaches "run the fix"); truncating
  overlong queries (still sends the head of the paste); shape-screening fetch URLs
  (benign `?page_token=…` params match the assignment shape); a deployment-specific
  privacy mode (the floors are universal; site policy belongs to the site).
- **Consequences:** a missing `[web]` dep is a one-command self-fix the gate allows even
  in autonomous mode; the union widens nothing else (only names zakcode itself
  declares); a secret value or credential-shaped paste cannot leave via a search query
  or a fetch header even by accident; the sanctioned `{{secret:NAME}}` path keeps
  working everywhere.

## ADR-0020 — The anomaly rail: a write over a failed read carries the question

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** Field observation (same deployment): a knowledge-tree index said a node
  existed, the read of its file failed ("File not found"), and the model silently wrote
  a fresh file and moved on — never noting that two sources of truth had just
  disagreed. The underlying cause could have been index drift (yesterday's write lost)
  or a path-resolution split (the framework's scripts resolve a virtual prefix to an
  external directory; the harness's raw file tools resolve the same string
  workspace-relative — silently creating a SHADOW tree no script will ever read).
  Either way the recovery looked clean and the contradiction died unexamined. The
  general failure: an error that CONTRADICTS prior evidence deserves one diagnostic
  beat, and models pave over it with the most plausible next action instead.
- **Decision:** A single, narrow, mechanical tripwire in the shared tool-execution seam
  (both turn paths funnel through `_execute_tool_call`): the loop remembers, per turn,
  every path whose `read_file` errored; a later SUCCESSFUL `write_file` to the same
  path (relative/absolute spellings canonicalized) appends `[harness] a read of this
  exact path failed earlier this turn…` to the write's result — asking the model to
  either diagnose the mismatch in one sentence or confirm the create was intentional.
  Fires once per path per turn; per-turn memory only; zero extra iterations (the note
  rides the result the model reads anyway, at the exact moment it decides what to
  build on the new file).
- **Alternatives rejected:** vetoing the write pending confirmation (create-if-missing
  is a common LEGITIMATE pattern — read to check, create when absent — and a veto
  would tax every instance with a round-trip); a plan-gate integration (plan nudges
  fire at turn end, after the pave-over already happened; the decision moment is the
  tool result); a generic "diagnose every error" nudge (fires constantly, becomes
  noise, and the model tunes it out — the value is in flagging the CONTRADICTION
  shape specifically); tracking every tool pair (read-then-write same path is the
  measured incident shape; broader pairs invite false positives without evidence).
- **Consequences:** the pave-over now costs the model one explicit sentence of
  diagnosis instead of zero; intentional creates lose nothing (the note tells them to
  carry on); shadow-tree and stale-index bugs surface at creation time instead of
  weeks later when a script cannot find content the model swears it wrote.

## ADR-0021 — Injected nudges carry provenance; the footer shows the prompt-cache share

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** Two small field wobbles from the same transcripts. (1) Loop-injected
  nudges (plan gate, empty-completion, stuck, critic, …) are delivered as USER-ROLE
  messages opening with a bare "Hint:" — and a field model attributed one to the human:
  "I have received your request to continue with the plan" when no user had spoken. The
  same model apologized reflexively after every nudge. The bracket provenance idiom
  (`[harness]`/`[hook]`/`[plan]`) already existed for observations, but directives
  lacked it and nothing DEFINED the tags for the model. (2) A 27-iteration turn showed
  "4861.1k tokens" with no way to see how much of that was discounted cache re-sends —
  the accounting existed (`Usage.cache_read_tokens`, surfaced only in `/cost`) but the
  per-turn footer hid it, so a big number read as a big bill and a caching failure
  (0% on a caching-capable backend) was invisible.
- **Decision:** (1) `_control_rail` renders every loop-injected directive as
  `[harness] Hint: …` — one constant, covering every nudge site — and the system
  prompt's Behavior section defines the tag family once: bracket-tagged output is
  automated runtime output, never the user; never attribute it to the user, never
  apologize in response, and more generally state-and-continue instead of apologizing.
  Tool-result rails stay bare `Hint:`/`Fix:` (they ride inside a tool frame, already
  unambiguous). (2) The turn footer's token item appends `(N% cached)` — cache-read
  share of prompt tokens — whenever the backend reports cache reads; silent at zero.
- **Alternatives rejected:** a wordier per-nudge preamble ("this is an automated
  message…" — token tax on every nudge; the one-time system-prompt definition + a
  9-char tag does the same work); renaming the rail word (the model already learns
  `Hint:` from tool rails — provenance is an orthogonal axis, so it composes as
  bracket + rail instead of replacing it); a footer cost breakdown (the `/cost`
  command already itemizes; the footer needs one glanceable share, not a table).
- **Consequences:** no injected nudge can read as the user speaking, on any model;
  the apology reflex is addressed at its trigger; prompt-cache health is visible on
  every turn footer, so "is caching hitting?" is answered at a glance instead of by
  archaeology.

## ADR-0022: Compaction that survives small windows — chunked summarization, the post-compact hook, honest triggers, visible notices

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** A 131k-window local pod session overflowed mid-turn (an uncapped
  2,776-line tool result) and the reactive recovery compacted-and-retried. The deep
  dive found four defects around that one moment. (1) The summarize call carried the
  ENTIRE old history raw in one request — but the reactive path fires only AFTER an
  overflow, so its input is oversized by construction: the recovery could throw the
  very `ContextWindowExceeded` it was recovering from, and `compact_now()` had no
  exception guard, so that second overflow would have escaped from inside an `except`
  handler and killed the turn ungracefully. (2) Claude Code fires
  `SessionStart(source="compact")` after every compaction — the seam frameworks use to
  restore serialized state — but the loop deliberately folded "compact" into "resume",
  i.e. the post-compact event NEVER fired mid-session, so a framework's
  PreCompact-serialized state had no restore moment. (3) The reactive recovery reused
  `compact_now()` and therefore reported `trigger="manual"` to PreCompact hooks for an
  automatic recovery. (4) Proactive threshold compaction was silent — a transcript
  rewrite the operator only discovered as apparent memory loss ("casual auto compact?!").
- **Decision:** `_summarize_for_compaction` stays raw-single-call while the history fits
  0.7× the summarizer's window (the common case, byte-identical behavior); above that it
  renders the history to labeled plain text (tool calls as one compact line, results by
  their output — no orphan structured tool blocks a provider API could reject), slices
  at 0.5×window using a conservative 2-chars-per-token floor, summarizes each slice, and
  folds the part-summaries once if the join is itself oversized. `compact_now(*,
  trigger)` takes the honest trigger ("auto" from both recovery paths, "manual" from
  `/compact`) and returns False on summarize failure instead of raising. Every
  successful compaction — proactive, manual, reactive — now fires
  `SessionStart(source="compact")` right after the rewrite (Claude Code parity; the
  LifecyclePayload doc gains the third value). `_maybe_compact` returns a notice string
  ("context near the window — compacted N → M messages") that the streaming path yields
  as an AgentStatus and the trace records; the reactive notice now carries the before →
  after counts.
- **Alternatives rejected:** a strict "only retry if strictly smaller" recovery guard
  (rejected once in review #1 — the `_MAX_CONTEXT_RECOVERY` bound already terminates);
  token-exact chunk splitting via per-slice tokenizer calls (the 2-chars-per-token
  floor never overshoots and costs nothing); firing a bespoke `PostCompact` event
  (Claude Code already has a name for this moment — `SessionStart(source="compact")` —
  and frameworks already branch on `source`, so inventing a second vocabulary would
  orphan existing hooks); rendering the single-call path to text too (uniform but
  changes what the summarizer sees for every session; the raw path is proven and stays).
- **Consequences:** compaction can no longer overflow itself, on any window size; a
  framework's serialize→restore pair (PreCompact → SessionStart:compact) works under
  this harness exactly as under Claude Code; PreCompact hooks can trust `trigger`; and
  every compaction is visible in the transcript with its before → after counts.

## ADR-0023: Seam-level tool-output clamp — no single result may swamp the window

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** The 131k-pod overflow (ADR-0022's incident) had a root cause upstream of
  compaction: a `use_skill` call returned a 2,776-line skill body whole, and nothing
  between a tool and the transcript bounds result size. Per-tool caps exist only where a
  tool knows its own shape (`read_file`'s 100KB); `bash` output, skill bodies, and grep
  sweeps are unbounded. Compaction cannot save this case by itself — the newest
  messages are kept verbatim (`preserve_recent`), so one giant recent result survives
  every compaction and re-overflows the retry.
- **Decision:** `_execute_tool_call` — the single seam both turn paths funnel through —
  clamps each result's model-facing text to a window-proportional ceiling:
  `context_window × 0.25 × 3 chars/token` (≈25% of the window at code-heavy density;
  32k assumed when a provider declares no window). Head-heavy head+tail keep (2/3 + 1/3:
  openings carry structure, endings carry verdicts) with an elision note naming the loss
  and the remedy ("re-run narrower — filter, page, or slice"). The clamp runs BEFORE
  hook notes and rails are appended, so appended guidance is never lost to the elision;
  PostToolUse hooks still see the full output (they are subprocesses, not context).
- **Alternatives rejected:** per-tool caps on `bash`/`use_skill` (every future tool
  re-solves it, and none of them knows the model's window — the loop does); a fixed
  byte cap (wrong on both ends: starves 1M-window models, still kills 8k ones); refusing
  oversized results as errors (a partial result plus a remedy beats a hard failure —
  the create-if-missing lesson from ADR-0020 applied to size); token-exact measurement
  via the provider tokenizer per result (cost on every call; the 3-chars/token floor is
  conservative in the right direction).
- **Consequences:** a small-window session can no longer be sunk by one verbose command,
  skill body, or grep; big-window models are effectively untouched (150k chars at 200k
  window); the model is always told when it is looking at a partial result and how to
  get the rest.

## ADR-0024: Small-model containment — degenerate tool arguments are vetoed; a completion that announces work is not a completion

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** Two failures from one small-model session. (1) A python -c payload whose
  arguments had collapsed into repetition ("import json; " ×28, "YOUR_" ×38 mid-string)
  executed unjudged: the ADR-0018 repetition guard judges COMPLETION TEXT and
  deliberately skips tool-call batches ("a batch calling tools is doing work"), so
  degeneration inside arguments — where small models actually put it — had no detector.
  (2) A later turn ENDED on "Now I will use the `create_file` command … I will then use
  `mv` …" and the loop accepted the completion: the empty-completion gate needs empty
  text, the plan gate needs an open plan (none existed), the quality gate scores only
  runnable writes — announced-but-unperformed work fell between every gate. The session
  then printed the zakpick "no struggle signal" advisory while visibly struggling,
  because neither failure latched anything.
- **Decision:** Three pieces, one ADR, because they share a cause (a small model losing
  the thread) and a consumer (zakpick's struggle latch). (1) `burst_repetition()` in
  the degeneration module: convicts a run of ONE short unit (2–64 chars, ≥2 distinct
  characters, non-whitespace) repeated ≥12× consecutively for ≥150 chars, ANYWHERE in
  the text — thresholds no legitimate command contains; single-char runs (dividers,
  padding, newline floods) are never convicted. `_execute_tool_call` runs it over the
  JSON-encoded arguments BEFORE the permission gate (the operator is never prompted to
  approve garbage) and vetoes with a bare `Fix:` rail naming the fragment and the
  remedy. (2) The false-done guard: when a turn is about to complete and the completion
  TAIL announces future work (first-person future + an ACTION verb — "I will use / run /
  create…", "let me now…"; "I'll let you know" and "I will need you to provide" do not
  match), one `[harness]` nudge asks for the work or a plain finish. Once per turn; a
  model that was only describing says so and finishes, so a false positive costs one
  bounded iteration. (3) Both degeneration strikes and the argument veto latch the
  per-turn struggle flag, folded into `signal_latched` each iteration — zakpick now
  escalates on degeneration, and the "cheaper model may keep up" advisory can no longer
  fire on a session that degenerated.
- **Alternatives rejected:** scanning tool arguments with `repeated_tail` (tail-anchored
  by design; argument degeneration is buried mid-payload); regex backreference matching
  (unpredictable backtracking on adversarial input; the windowed self-overlap scan is
  bounded C-speed slice compares); killing the turn on a degenerate call (the veto is a
  recovery rail — the stuck ladder and doom-loop guard already own repeated failure);
  gating "done" on a todo/plan (the plan gate already does that when a plan exists; this
  guard covers exactly the planless case); an LLM judge for "is this done?" (a model
  call per completion to catch a failure mode of small models — the regex is free and
  surgical).
- **Consequences:** degenerate arguments never execute and never reach a permission
  prompt; a turn can no longer end on an announcement with nothing behind it; struggle
  is visible to model routing the moment it happens, on both turn paths.

## ADR-0025: Workspace hooks always load — the adoption flag is gone

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** Settings-hooks ingestion was tri-state (`ZAKCODE_SETTINGS_HOOKS` unset/true/false), resolved in the interactive CLI by a one-time folder-trust prompt remembered per workspace. On a Mind deployment box the ask was never answered, so the framework ran with ZERO of its declared hooks — no agent-env injection, no Stop-hook veto, none of its 43 PreToolUse gates — and the miss was invisible until a transcript showed the model hand-typing what the inject hook should have added. This is the operator's standing rule made concrete: no feature flags, one way of doing things — a framework whose protections ride on hooks must never silently run unprotected.
- **Decision:** Hooks declared in `<workspace>/.claude/settings.json`, `.claude/settings.local.json`, and `.zakcode/settings.json` ALWAYS load at Agent construction. The `settings_hooks` setting, the `enable_settings_hooks` parameter, the folder-trust prompt, and the whole `workspace_trust` module are deleted. What remains is not a flag but the security floor, unchanged: every hook command is scanned against the catastrophic blocklist (matches hard-denied in autonomous mode, registered-with-warning otherwise), and provider keys are scrubbed from every hook child's environment.
- **Alternatives rejected:** keeping the prompt but defaulting to "always" (still a flag, still a divergent path, and a dismissed prompt still meant an unprotected session); folding hooks under a broader workspace-trust gate (same silent-failure shape, one level up); auto-enabling only when a Mind-style framework is detected (special-casing one consumer inside a generic engine).
- **Consequences:** a workspace's committed automation is live the moment zakcode opens it — the Claude Code parity story is now unconditional; opening an unfamiliar repo executes its declared hooks (mitigated by the danger scan and key scrubbing — the same posture Claude Code lands on after one keystroke); the sibling opt-ins (`settings_permissions`, `output_style`, `status_line`) are deliberately untouched — they reshape permission posture and voice rather than run declared automation, and each needs its own decision.

## ADR-0026: The broken-record guard, and skill references must be request-shaped

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** Two more small-model transcripts. (1) A Mind loop session re-sent ONE
  closing paragraph ("The plan shows all 3 steps are complete … No further action is
  needed") five times: the framework's Stop hook correctly vetoed each finish, the veto
  re-prompt went out, and the model answered it by re-emitting its previous message
  verbatim — an unbounded veto↔parrot cycle, each lap billing a full context. Nothing
  watched for it: the degeneration guard judges repetition INSIDE one completion, the
  doom-loop guard watches tool batches, and text-only completions repeat across
  iterations without tripping either. (2) The skill-coverage backstop fired "request
  named a skill that never ran" demanding EIGHT skills after a turn whose pasted prompt
  merely DISCUSSED them — `_skill_refs` accepted backticks, quotes, parens, brackets,
  and colons as token prefixes, so documentation mentions counted as invocation
  requests (and plan seeding would seed them as steps).
- **Decision:** (1) Broken-record guard: each no-tool-call completion ≥80 chars is
  counted per turn by its whitespace-normalized text; a re-send gets one escalating
  `[harness]` rail ("you have already said exactly this — take the next CONCRETE
  action, or state in one sentence of NEW information what is blocking"; sharper from
  the third occurrence), checked FIRST in the completion branch so a parrot never
  re-buys the completion critic or the quality gate, and latching the ADR-0024 struggle
  flag so zakpick escalates. (2) `_skill_refs` counts only request-shaped tokens:
  fenced blocks and `>`-quoted lines are stripped, and the prefix class narrows to
  start-of-text/whitespace/`,;!` — backticked, quoted, parenthesized, bracketed, and
  colon-glued tokens are mentions. Both the coverage backstop and plan seeding
  inherit the precision.
- **Alternatives rejected:** ending the turn after N parrots (under a framework Stop
  hook the turn ending IS the failure — the loop dies until an operator returns; the
  guard's job is to change the stimulus, and the budget remains the hard bound);
  fuzzy/similarity matching for repeats (verbatim normalization catches the attractor
  that exists; similarity thresholds invite false positives on legitimately incremental
  status lines); NLU-grade request detection for skill refs (the mention shapes are
  structural — code spans, quotes, parens — and structure is enough).
- **Consequences:** a veto↔parrot cycle now breaks on its first repeat instead of
  running unbounded; parroting is visible to model routing; pasted documentation can no
  longer conscript the coverage backstop or the plan seeder; `[/name]`-bracketed tokens
  no longer count as requests (they read as optional-syntax documentation).

## ADR-0027: Long skill bodies are decomposed into the plan, not held in the model's head

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** The harness already decomposes USER requests (plan seeding for multi-skill
  asks, the plan gate holding open steps) — but a skill body arrives as one wall of
  instructions and the model is simply told to follow it. Capable models can; the small
  models in field testing cannot: a 2,776-line skill body was followed for two steps
  and then narrated instead of executed, and after a mid-turn compaction or the seam
  clamp the un-executed remainder of the wall is partially GONE. Operator ruling: the
  risk asymmetry favors decomposition — the worst case of decomposing is a few extra
  plan steps, while the worst case of NOT decomposing is silent non-execution.
- **Decision:** `use_skill` fires a decompose rail whenever a loaded body is ≥2,000
  chars: the hint tells the model to FIRST record the concrete steps THIS request needs
  with `update_plan` (folding in the context it already has), then execute them in
  order, marking each done — and the result data carries `decompose: true` for clients.
  The tool description teaches the pattern once ("for a LONG skill, first decompose its
  steps into your plan"). Short bodies keep the plain follow hint, byte-identical. The
  existing machinery does the rest: the plan gate holds the finish on open steps, plan
  state survives compaction where instruction recall does not, and ADR-0024's
  false-done guard catches a narrated-but-unexecuted ending.
- **Alternatives rejected:** mechanically parsing skill bodies into plan steps (bodies
  are heterogeneous prose/pseudocode — the MODEL holds the task context and must write
  the steps, which is exactly the operator's framing); a completion-time backstop
  nudging when a long skill ran without a plan (nags capable models that legitimately
  execute without ceremony; the hint fires at the one moment the model has both the
  instructions and its context in hand, and the false-done guard already covers the
  failure ending); injecting the body as plan steps via session surgery (violates the
  tool's "result, not session surgery" contract).
- **Consequences:** long skills become checked-off steps instead of recalled prose on
  every model size; skill execution survives compaction and clamping through the plan;
  worst case on a capable model is one short extra hint line.

## ADR-0028: The protected-path floor is tier-aware — reads of the agent's own config are not writes

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** The protected-path floor (ADR self-remediation Step 2, PR #27) scanned the
  file-path arguments of EVERY tool against the protected patterns without consulting the
  tool's permission tier. Field incident (2026-08-26, sera on a Mind deployment): a
  `read_file` of the agent's OWN `.claude/skills/google-drive-access/SKILL.md` was hard-denied
  in autonomous with "blocked write to a protected path: agent config (.claude/)" — a read,
  refused by a write gate, with a message describing it as a write. The wrongly-refused first
  step derailed the model (a small Gemini-class model) into a hallucinated-tool-syntax /
  repetition collapse that ended in a false "done". The bug was latent for months because
  models normally read skills via `use_skill` (internal resolver, no permission gate) and read
  repo files via bash (shell args deliberately unscanned); a direct `read_file` on a
  `.claude/` path was the one shape that tripped it — and every READ_ONLY file tool
  (read_file, list_dir, glob, grep, pdf, office, images) was equally affected.
- **Decision:** `decide()` passes `read_only` (tier is READ_ONLY) into the floor. For
  READ_ONLY-tier tools the three write-sensitive BUILT-INS — `.git/`, the venv,
  `.claude/` — do not bind: reading them is normal operation (a skill file is made to be
  read). Two classes still bind reads: `.env` (reading a secrets file is itself the leak —
  the floor's charter says "read/rewrite secrets"), and ALL operator/settings-ingested
  extras (a CC `deny Read(glob)` permission rule ingests as a protected-path regex, so
  extras must bind reads to honor it). Denial/escalation messages now say "read of" /
  "write to" accurately. The loop's hook-rewrite re-check passes the same flag. An unknown
  tool (no spec) stays fail-closed (tier defaults to most-dangerous → no read exemption).
  The grant fast-paths are unchanged: a read-exempt call falls through to `decide()` and
  allows; a secrets read still re-decides and can never ride a session grant.
- **Consequences:** Mind-deployment agents can read their own skills, rules, and settings
  with file tools again; `.env` reads still hard-deny in autonomous and re-prompt
  interactively; write behavior is byte-for-byte unchanged. Residual (unchanged, deliberate):
  autonomous agents still cannot WRITE `.claude/` — deployments whose framework expects
  self-editing config need an operator decision before that posture changes.

## ADR-0029: No built-in agent-config restriction — the settings files are the sole authority over `.claude/`

- **Status:** Accepted (shipped, 2026-08-26). Operator ruling, verbatim intent: "get rid of
  the restriction 100% and let agents edit what they want, but follow settings file …
  of course they should be able to edit settings file like claude.md and the actual settings
  files, as long as the settings allow."
- **Context:** The protected-path floor hardcoded `agent config (.claude/)` as a built-in
  class, so an autonomous agent could never write its own skills, rules, or settings — even
  though frameworks built on this engine (the Mind framework) are DESIGNED around agents
  evolving their own config, with git as the safety net and a deliberate, narrow
  constitutional-anchor deny (`settings.local.json` protecting itself) as the only hard line.
  Meanwhile the mechanism that could express that policy — CC `permissions.{allow,deny,ask}`
  ingestion from `.claude/settings.json` + `settings.local.json` — sat behind an opt-in flag
  (`ZAKCODE_SETTINGS_PERMISSIONS`, off by default), the exact tri-state shape ADR-0025
  eliminated for hooks. Net effect: the engine enforced a blanket opinion the operator never
  asked for, while ignoring the specific permissioning the operator actually wrote down.
- **Decision:** Two-fold, and the halves only work together. (1) The
  `agent config (.claude/)` built-in protected pattern is DELETED — agents read and write
  their own config, `CLAUDE.md`, and the settings files freely by default. (2) Settings
  permissions ingestion is UNCONDITIONAL — no setting, no `Agent(enable_settings_permissions)`
  param (a stale env var is inert via `extra="ignore"`); skipped only when the host injects
  its own `permission_policy`. The workspace's `permissions` block IS the authorization
  posture and the ONLY authority over `.claude/`: a framework that wants any of its config
  protected declares `deny Edit|Write(glob)` (or `Read`) rules there, which ingest as extra
  protected paths and bind reads AND writes, hard-denying in autonomous. The `.git/`, `.env`,
  and venv built-ins are unchanged (not named by the ruling; `.env` read-blocking and the
  catastrophic-command floor remain the never-waivable secrets/safety spine).
- **Alternatives rejected:** keeping `.claude/` built-in but exempting Mind-style workspaces
  (special-casing one consumer); a config flag to disable the built-in (a flag — the standing
  rule is one way of doing things); ingestion-on-only-when-permissions-present (still a
  divergent path, and indistinguishable from silent non-loading when a file has a typo).
- **Consequences:** Mind agents self-evolve their skills/rules/settings under zakcode exactly
  as the framework intends, and Mind's constitutional anchor (the self-referential deny in
  `settings.local.json`) is now actually ENFORCED by the engine instead of approximated by a
  blanket block. A workspace with no permissions block leaves `.claude/` fully open — that is
  the ruling's default, not an oversight. An agent can edit settings files to loosen its own
  next-session posture unless the settings deny it (settings load at Agent construction, so a
  self-edit never takes effect mid-session); operators who care pin the anchor deny. Ingested
  allows still cannot cross the catastrophic or protected floors (unchanged invariant, proven
  in test_cc_permissions.py).

## ADR-0030: Ingested path denies keep their verb — an Edit/Write deny never blocks a read

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** Fresh-eyes review of ADR-0029 (same day, before any deployment updated). CC
  `deny Read|Edit|Write|MultiEdit(glob)` gestures all ingested into ONE protected-path list,
  and settings-ingested extras were deliberately never read-exempt (ADR-0028, so a
  `deny Read(glob)` would bind reads). The two decisions composed into a regression: every
  Edit/Write-only deny also blocked READS. Measured against the real Ayoai Mind settings pair
  (the shape sera runs): 36 of 44 deny gestures are Edit/Write-only, so under always-on
  ingestion `read_file` on `world/knowledge/tree/_tree.yaml`, the `start`/`stop`/`boot`
  control skills, and `settings.local.json` itself would hard-deny in autonomous — paths the
  framework REQUIRES agents to read (Tier-1 retrieval reads the tree; the anchor tripwire
  assumes the anchor is readable). The ADR-0028 bug class (a read refused by a write gate),
  reintroduced through the settings channel; also a CC-conformance divergence, since a CC
  Edit deny leaves the path readable.
- **Decision:** The verb is retained through ingestion. `IngestedPermissions` carries two
  lists: `protected_path_regexes` (from `Read` denies — strict, bind reads and writes) and
  `protected_path_regexes_write_only` (from `Edit`/`Write`/`MultiEdit` denies). The Agent
  compiles the second with `compile_protected_paths(..., write_only=True)`, which suffixes
  each compiled description with the `" (write-only)"` mark; `_protected_path_reason`
  skips marked patterns for READ_ONLY-tier tools exactly as it skips the write-sensitive
  built-ins. Operator `ZAKCODE_PROTECTED_PATHS` regexes have no verb and stay strict.
- **Consequences:** Verified against the live Mind settings on absolute paths — the tree,
  control skills, anchor, and validator all read=ALLOW / write=DENY; `.env.local` denied
  both ways (a `Read` deny plus the built-in). Write behavior is unchanged: every Edit/Write
  deny binds writes exactly as before. Residual found by the same probe and fixed next
  (ADR-0031): a RELATIVE path argument bypasses a `*/`-prefixed glob because the decision
  runs on the raw argument.

## ADR-0031: The protected-path scan resolves relative arguments against the workspace

- **Status:** Accepted (shipped, 2026-08-26).
- **Context:** Found by the ADR-0030 verification probe. The permission decision runs on
  the RAW tool arguments (step 1 of the seam, before the tool resolves anything), while
  Claude Code matches path rules against absolute paths — its file tools only take absolute
  paths. A framework's deny globs are written for that world: the Mind's
  `Edit(*/.claude/skills/start/*)` needs a parent segment. So the same file was denied when
  spelled `/opt/ayoai-mind/.claude/skills/start/SKILL.md` and ALLOWED when spelled
  `.claude/skills/start/SKILL.md` — and models spell paths relative to the workspace
  constantly. Pre-existing, but ADR-0029 made the settings the SOLE authority over
  `.claude/`, which turned a quirk into a bypass of every `*/`-prefixed rule.
- **Decision:** `PermissionPolicy` takes an optional `workspace_root` (the Agent passes
  its workspace; `child_view` propagates it). `_protected_path_reason` scans a tighten-only
  union of candidates: the raw argument, plus — for a relative path when a root is known —
  `normpath(join(root, value))`. Either matching binds. String math only, no filesystem
  call, so `decide()` stays pure; a `..`-laden path normalizes before matching. The
  read-only exemption applies identically to both forms.
- **Alternatives rejected:** rewriting the glob translation so a leading `*/` becomes
  optional (fixes one idiom, silently diverges from CC glob semantics, and leaves absolute
  patterns like `//C:/...` unreachable from relative spellings); resolving in each file tool
  (the decision has already been made by then).
- **Consequences:** Verified against the live Mind settings: the relative spelling of a
  control-skill file now denies exactly like the absolute one; the verb semantics of
  ADR-0030 are unchanged (read still ALLOW). A policy constructed without a root (library
  callers, tests) behaves exactly as before.

## ADR-0032: The served mind's session store lives under the workspace — conversations travel with the mind, not the host

- **Status:** Accepted (shipped, 2026-08-27). Found live by the Vinheim presence work
  (g-369-15): a served mind's conversations vanished with the host that served it.
- **Context:** `SessionStore()` defaulted to the per-user home (`~/.zakcode/sessions`),
  which is right for the terminal client — one human, one machine, many projects — and
  wrong for `zakcode webapp`, whose topology is one container per served workspace. A
  served workspace IS one mind's home: its identity (`self.md`), rules, skills, the
  `.say` inbox, the `.current-session` marker and its uploads already live there. Only the
  transcripts those things point at lived somewhere else, on the serving host's disk. So
  the marker (workspace-scoped, durable) and its target (host-scoped, ephemeral) were on
  different lifetimes: recycle the host and every conversation is gone while the
  workspace still names one — the dangling-marker case `GET /sessions/current` heals, but
  healing an amnesia is not the same as not having it. There is one way of doing things
  here, so this is a decision about where the store IS, not a knob for where it may be.
- **Decision:** `SessionStore.for_workspace(root)` re-roots the user store's layout at
  `<workspace>/.zakcode/sessions` — the same directory family as the project-level
  `.zakcode/settings.json`, one JSON document per session, unchanged format. `create_app()`
  uses it whenever no store is injected. The terminal client (`zakcode`, the cockpit,
  `-s` resume) keeps `~/.zakcode/sessions` untouched. The workspace store writes a
  self-ignoring `.gitignore` (`*`) once, so a workspace that is also a git checkout never
  commits its transcripts. No flag, no env var.
- **Alternatives rejected:** a `--sessions-dir` flag / `ZAKCODE_SESSIONS_DIR` env (a knob for
  a question with one right answer, and every deployment would have to remember to set
  it); repointing `ZAKCODE_HOME` at the workspace (`zakcode_home()` is a config home only
  and must never be treated as a workspace — D20 — and it would drag `.env` along with
  it); a host-side symlink from `~/.zakcode/sessions` into the workspace (leaves the
  lifetime split in place and, with several minds served from one host user, makes every
  daemon share one store).
- **Consequences:** Conversations survive the host: stop the container, start another
  against the same workspace, `/sessions` lists the same transcripts and
  `/sessions/current` resolves. Two served workspaces are isolated by construction. A
  local `zakcode webapp` run without `--workspace` now stores under `<cwd>/.zakcode/sessions`
  rather than the home dir — sessions created by the old default are not migrated (the
  format is identical; copy the files if they matter). The mind's own file tools can see
  its transcripts, since `.zakcode/` is not in the default ignore set — deliberate (a mind
  may read its own history); `.zakcodeignore` hides it where that is unwanted. Markers
  written before this ADR point at ids the new store does not have and self-heal through
  `/sessions/current` exactly as any dangling marker does.

## ADR-0033: Small-model containment II — fuzzy repetition, claim-vs-action, directive nudges, text-only stall, resume safety

- **Status:** Accepted (shipped, 2026-08-27). Field incident 2026-08-26 (serene:
  gemini-2.5-flash-lite on `quick_code`, a `/resume`d transcript, on a process still
  running the pre-`zakcode update` build): "finish forging this skill" produced a 20-line
  "Let's try again. I will try to create the skill correctly." spiral, then "I have updated
  world/forged-skills.yaml … I have registered the skill" with nothing written — and the
  turn completed. No plan, no skill read, no visible reason.
- **Context:** Every rail from ADR-0018 / 0024 / 0026 was on and none fired, each for a
  measurable reason. (1) The degeneration guard convicts 12 of 15 IDENTICAL lines; the
  spiral mutated a word per line ("add" → "create", "this again" → "again") and topped out
  at 3, while 10–11 of those 15 lines shared ≥ 60% of their words with one sentence. (2) The
  false-done regex required the action verb to follow "will" directly — "I will *try to*
  create" never matched. (3) Nothing compared a completion's CLAIM of a change against the
  tool calls that ran. (4) The critic was scoped off `quick_code` turns entirely. (5) The
  empty-completion nudge invited narration ("say what you tried, what failed, and what
  should happen next") and the model obliged with apologies. (6) zakpick's only
  escalation paths were the guards that stayed silent, so five text-only completions ran on
  the cheap model. (7) The route was a trace-only note, invisible at the terminal. (8) The
  resumed transcript carried the old build's collapse into the new session, and nothing
  recorded which build had written it or how its last turn ended.
- **Decision:** one change, no flags, both turn paths:
  1. `repeated_tail` gains a **near-duplicate branch**: ≥ 8 of the last 15 lines at ≥ 0.6
     word-Jaccard to one short line, with ≥ 3 distinct variants (an identical-only window
     stays the exact branch's call at its 12-line bar — a fuzzier branch must not undercut
     it) and fewer than half of the similar lines introducing a token seen nowhere else in
     the window (a listing adds vocabulary on every line; a spiral recycles a closed
     vocabulary). Same discard-retry-then-`degenerated` contract as ADR-0018.
  2. `_FUTURE_INTENT_RE` admits `try to / attempt to / go ahead and / proceed to` between
     the future form and the verb.
  3. **Claim-vs-action guard**: a completion whose tail reports a change (first-person
     past/perfect change verb tied to a file-ish object in the same sentence) in a turn
     with zero executed non-`READ_ONLY` tool calls gets one `_CLAIM_NUDGE` per turn and
     latches the struggle flag. The critic also runs on a `quick_code` turn whose
     completion claims a change.
  4. `_EMPTY_COMPLETION_NUDGE` is a directive: one tool call, the answer, or one blocking
     sentence — nothing else.
  5. **Text-only stall**: the second no-tool-call completion in a turn (necessarily after a
     nudge or veto) with no open plan latches the struggle flag; a tool batch resets the
     count.
  6. **Transparency**: `route: <category> → <model>` on every route change and
     `text-only completion #N (no tool calls)` as status lines (streaming path; the CLI
     renders them as dim `·` lines).
  7. **Resume safety**: `Session.build` (stamped at every save from `build_commit()`) and
     `Session.last_stop_reason` (stamped at turn end). `-s <id>` and `/resume` print why and
     run `compact_now(trigger="resume")` when the build differs — an unstamped document
     counts as older — or the last turn ended `gave_up` / `degenerated` / `doom_loop`.
- **Alternatives rejected:** lowering the exact branch's bar or swapping it for a
  similarity threshold alone (a numbered listing scores the same as the spiral — the
  vocabulary predicate is what separates them); a `--strict-done` flag or per-model gating
  (one way of doing things); refusing `/resume` across builds outright (the transcript is
  still the human's context — compaction keeps its summary); ending the turn on an
  unbacked claim (a nudge is bounded, and a model that did the work in an EARLIER turn can
  say so and finish).
- **Consequences:** the real wall convicts at the mid-stream probe, not at the end of the
  turn; a numbered listing, a file listing, code, a checklist and eleven identical lines
  stay legal (all pinned). "I have added some context below" is conversation, not a claim
  (no file-ish object); "I have not updated" and "you have updated" never match. The claim
  nudge costs one bounded iteration on a text-only turn. Sessions saved by this build carry
  the stamp; every older transcript compacts once on its first resume. Residual: the web
  client's resume (`server/app.py` load sites) does not yet run the notice, and
  `zakcode update` still tells the operator to restart running sessions — both belong to
  the in-place update change (PR-C), which retires the restart line by restarting the
  process itself (the terminal half shipped as ADR-0034).

## ADR-0034: A running chat restarts itself into a newly installed build at its next idle prompt

- **Status:** Accepted (shipped, 2026-08-27). Resolves the ADR-0033 residual for the
  terminal client. Field incident 2026-08-26 (serene): `zakcode update` printed
  `c4edaa4 → 0c28c8b` and "running chat sessions keep the old build until restarted"; the
  chat was not restarted, and the next turn collapsed on code that had already been fixed.
- **Context:** A Python process runs the modules it imported; a reinstall changes the disk,
  not the process. Claude Code's answer is a passive banner ("Update installed · Restart to
  apply") that the human acts on — which is exactly the step that was skipped. The harness
  already persists the whole conversation every turn (`SessionStore`) and resumes it by id,
  so a restart costs the human nothing but a banner; the only thing missing was a process
  that notices the new install and performs the restart itself. The one moment that is
  safe is the idle prompt: no turn in flight, no tool running, no permission prompt open.
- **Decision:** (1) `build_info.install_identity()` reads the install FRESH — the recorded
  commit for a git-URL install, the checkout's HEAD for a local-path install — plus an
  **install marker**, the mtime of the install's own `direct_url.json`, which every
  (re)install rewrites regardless of shape. `running_build()` is that reading frozen at
  import; `install_changed()` compares markers (never labels, so a dev checkout whose HEAD
  moves without a reinstall never trips it). (2) The input mux polls an `idle_probe` every
  5 s while — and only while — the REPL is truly idle (`next_input(idle=True)`); a mid-turn
  wait such as a permission prompt never consults it. A True verdict surfaces as a new
  input kind, `restart`. (3) The REPL handles `restart` by stamping the session document
  with the build that will read it (so the resumed session is an upgrade, not an ADR-0033
  cross-build compaction), flushing, and `os.execv(sys.executable, ["-m", "zakcode", …])`
  with the original arguments and `--session <id>` pinned (`chat` named explicitly when
  the original invocation was the bare command, whose root callback takes no chat
  options). A failed exec is reported and the old process keeps serving. (4) `zakcode
  update` says so instead of asking for a restart. `_persist` stamps `Session.build` from
  `running_build()`, so a reinstall landing mid-session can never re-label a document the
  old process wrote.
- **Alternatives rejected:** hot-reloading modules (mixed old/new objects in one process —
  the class of bug no test catches); a passive "restart to apply" banner (the incident);
  restarting mid-turn on a timer (a tool or prompt could be in flight); keying the probe on
  the commit label (a local-path install records none, and a dev HEAD moves without a
  reinstall); a `--no-auto-restart` flag (one way of doing things — the idle restart is
  invisible when nothing changed and one banner when it did).
- **Consequences:** After `zakcode update`, an idle chat restarts within ~5 s and resumes
  the same session with a one-line banner; a busy chat restarts at its next idle prompt. A
  no-op reinstall of the same commit also restarts once (the marker moved) — harmless. The
  web server (`zakcode serve`) does not yet restart itself: it has no idle boundary the
  harness owns, and the web client's resume notice is the same residual — tracked, not
  hidden. The probe costs one metadata read every 5 s of idle time.

## ADR-0036: Every server door dispatches a leading slash like the CLI

- **Status:** Accepted (shipped, 2026-08-27). Field finding 2026-08-27 (Vinheim, bravo): a
  served Mind could never be STARTED. Its framework's boot command (`/start <agent> --mode
  assistant`) is a user-invocable-only skill, and the server's three doors — the say consumer,
  `POST /chat`, `POST /chat/stream` — passed raw text to the turn, so the model either refused
  its own boot command as self-invocation or free-associated over the slash line. A headless
  deployment (systemd `mind-serve@`, a provisioning recipe, no terminal) has ONLY those doors.
- **Context:** the 2026-08-19 CLI work gave a typed slash Claude Code semantics through
  `Agent.compose_skill_turn` — the command-expansion frame at the very START of the user
  message is how a user-invocable-only skill learns a human typed it. The REPL,
  `chat --say-inbox` and `-p` (#148) all ride it, and the docs already called the inbox "one
  input rule, two doors, ONE reader" — but the served surface never joined; it was a third
  door with a different rule.
- **Decision:** `dispatch_slash` in the server, called first by all three doors. The first
  whitespace token (lower-cased, slash stripped) is the skill and the remainder its args,
  exactly the one-shot parse. Invoked → the framed `turn_text` IS the turn; a queued nudge is
  left queued rather than folded in front of the frame (the frame's position is the signal).
  Denied / unreadable → NO turn: `/chat` answers 403 / 500 with the reason; `/chat/stream` and
  the say consumer publish `status` + `done(stop_reason="skill_refused", degraded, error)` so
  no watcher hangs. Unknown `/token`, or a thin `AgentLike` without `compose_skill_turn` →
  prose, unchanged. The watch `user_message` marker keeps the TYPED text, not the skill body.
  No flag — it is the CLI's existing rule applied to the doors that lacked it.
- **Alternatives rejected:** auto-`/start` on session create inside the server (the boot
  command belongs to the deployment; zakcode stays framework-agnostic); a dedicated
  `POST /skill` endpoint (a second input contract beside the say inbox — the inbox IS the
  contract, and a recipe writes a file, not HTTP); letting a refused slash fall through as
  prose (the CLI already decided a scripted boot fails loudly, never silently).
- **Consequences:** a recipe can end provisioning by writing `/start tricks --mode assistant`
  into `.say`, and the Mind boots itself; the operator sees the typed command on the watch
  page and the framework's own "Assistant mode active" reply. The inbox is now a COMMAND
  lane: any surface that forwards untrusted user text to `/say` must neutralize a leading
  slash at ITS trust boundary (Vinheim: the gateway's `sanitizeSay`). Open: no telemetry
  distinguishes a served slash from a typed one (same gap the CLI has).

## ADR-0035: The classify side-call also names the skill a request implies

- **Status:** Accepted (shipped, 2026-08-27). Field incident 2026-08-26 (serene): "finish
  forging this skill" — the skill-forging skill was the whole task, but the request carried no
  `/slash` token, so the harness never knew: no plan step was seeded, the coverage backstop
  stayed unarmed, and the model collapsed without ever reading the skill.
- **Context:** ADR-0026 deliberately made skill references REQUEST-shaped (`/name` tokens
  only) so pasted documentation could not conscript the plan seeder and the backstop; the
  cost was that a request naming its skill in prose gets none of that scaffolding. A regex
  over prose would re-open the false-positive door ADR-0026 closed. But the harness already
  spends ONE cheap classify-model call per ambiguous turn to judge scope
  (`should_consult_classifier`); the same call can be asked to name the catalogued skill
  the request is unmistakably asking to run.
- **Decision:** `DIFFICULTY_SCHEMA` gains an optional `skill` (string or null; `difficulty`
  stays the only required field, so a classifier that omits it is still valid). The prompt,
  when the workspace has a skill catalog, lists it (name: description, capped at 60 entries
  / 100 chars) with the rule *by name, or by an unmistakable description of what that skill
  does — otherwise null; never guess*. `parse_skill` accepts only an exact (case-insensitive,
  leading-slash-tolerant) match against the catalog, so a guessed name is dropped rather
  than seeded. The side-call returns a `DifficultyVerdict(category, skill)`; the loop feeds
  `category` to `classify_main_turn` as before and, for a named skill, `_adopt_implied_skill`
  arms the coverage backstop and seeds the same `run /<skill>` plan step a typed `/name`
  gets, then rebuilds the call so the model sees the step on that very iteration (the
  streaming path also says so in a status line). No flag: the classifier already ran; it
  now answers one more question.
- **Alternatives rejected:** a prose regex over the request (the ADR-0026 false-positive
  class — a pasted prompt discussing eight skills demanded all eight); a second, dedicated
  side-call (double the spend for the same judgment); trusting the classifier's name without
  the catalog match (a hallucinated skill would seed a step `use_skill` cannot load).
- **Consequences:** "finish forging this skill" on a workspace whose catalog carries the
  forging skill now runs with a seeded `run /forge-skill` step and an armed backstop — a
  text-only finish is nudged to run it or say why not. Detection rides the side-call, so it
  reaches exactly the turns the classifier judges (short request, small context, zakpick
  on, not latched); a long request or a single-model install gets no skill detection, as
  before. A wrong-but-catalogued name costs one plan step the model can cancel with a
  sentence. Each prompt grows by the catalog text (bounded).

## ADR-0036: A typed /skill is not a request to parse; a blocker is not a finding until a tool call fails

- **Status:** Accepted (shipped, 2026-08-27). Field incident 2026-08-27 (serene, first
  typed `/start sera` on a build carrying ADR-0026's compound-request seeder): the harness
  seeded `run /start, /stop, /boot, /prime` from the start skill's own prose — a plan telling
  the model to STOP the agent it was starting — and the coverage backstop made it re-load
  the 1,200-line skill through `use_skill` (492k tokens in 7 iterations). The model then read
  a hook's source, declared the session id it injects "not available in this execution
  environment", and ended three turns on that sentence without ever running the skill's own
  one-line check (which passes). A follow-up "then make one" was classified as implying
  `/create-aspiration` — a guess. And every status line read `route: … →
  TextToolCallingProvider`, naming the adapter, not the model.
- **Context:** `Agent.compose_skill_turn` (Claude Code slash semantics) makes the typed
  skill's WHOLE BODY the user message, behind a command-expansion frame at the very start.
  ADR-0026's seeder and backstop read the user message for request-shaped `/name` tokens;
  a skill body is documentation, and the framework's skills mention each other constantly.
  Separately, the harness's sanctioned finish shapes include "one sentence stating what is
  blocking you" (ADR-0033) with no requirement that anything demonstrated the block.
- **Decision:** Four rules, no flags. (1) A message that begins with the command frame is a
  composed skill turn (`_composed_skill_name`): it is never seeded from, and its
  `requested_skills` are empty — the body IS the invocation, so no second load is demanded.
  A frame anywhere else is just text (the seeder still works on pastes). (2) Blocker-
  without-evidence guard: `_execute_tool_call` counts failed calls per turn; a completion
  whose tail declares the model blocked in the first person ("I am blocked", "I cannot
  proceed", "I cannot … without", "please provide") while that count is zero gets ONE
  directive nudge — run the check that would fail and show it, or continue — and latches
  the struggle flag. Answers that mention absence ("two fields are missing") are not
  claims. (3) The classifier's implied skill must share a content word (4-char stem, small
  stopword list, name or description) with the request or it is dropped, category kept —
  the deterministic floor under "never guess". (4) `_provider_label` unwraps `.inner`
  chains so the route status names the model.
- **Alternatives rejected:** teaching `_skill_refs` to skip prose (it cannot know a body
  from a paste; the frame can); blocking every blocker claim (a demonstrated one is the
  right finish); asking the classifier for a confidence (a guessing model reports high
  confidence); a per-skill "may be implied" flag (the user forbids flags; the anchor is one
  rule for all).
- **Consequences:** `/start sera` runs the start skill with an empty plan and the body loaded
  once. A hallucinated environment blocker costs one nudge instead of a dead turn — and the
  nudge names the fix (probe it). A genuine blocker still ends the turn on the second
  completion. A request that describes a skill in unrelated words ("do the thing that
  files work into the queue") no longer gets the skill seeded; that is the accepted price.

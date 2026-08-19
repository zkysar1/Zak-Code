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

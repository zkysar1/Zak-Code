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

## Hermes architecture digest

Research synthesized from the NousResearch/hermes-agent repo, AGENTS.md, the official docs (architecture, agent-loop, hooks, providers, plugin, curator, python-library pages), DeepWiki, and release notes (v0.3.0–v0.15.0). Note: Hermes is a large, multi-surface system (CLI/TUI, messaging gateway, cron, ACP/IDE, kanban). Zak Code should adopt the *core agent patterns* and deliberately skip the surface sprawl.

---

### Provider transport abstraction

**How Hermes does it.** Format conversion and HTTP transport are pulled out of the main agent into `agent/transports/` behind a `ProviderTransport` ABC. Concrete subclasses — `AnthropicTransport`, `ChatCompletionsTransport`, `ResponsesApiTransport` (OpenAI Codex/Responses), `BedrockTransport` (AWS Converse API) — each own four responsibilities: (1) message conversion to the provider's wire shape, (2) tool-schema conversion, (3) kwargs/request assembly, (4) response normalization back to a canonical internal form. Crucially, cross-cutting concerns stay on the agent, *not* the transport: **streaming, retries, prompt-cache breakpoint management, and credential refresh remain on `AIAgent`.** A separate runtime resolver (`hermes_cli/runtime_provider.py` + `auth.py` with a `PROVIDER_REGISTRY`) maps a `(provider, model)` string to `(api_mode, api_key, base_url)`; the `api_mode` (`anthropic_messages` / `chat_completions` / `codex_responses` / `bedrock_converse`) selects which transport instance is used. Model providers are themselves pluggable (`register_provider(ProviderProfile(...))`, last-writer-wins so user plugins override bundled ones).

**ADOPT.** This is the single most valuable pattern. Define a `ProviderTransport` ABC with exactly these methods (`convert_messages`, `convert_tools`, `build_request_kwargs`, `parse_response`/`normalize_response`) and a canonical internal message format that everything else speaks. Ship Anthropic + a generic OpenAI-ChatCompletions transport first; add Responses/Bedrock later behind the same seam. Keep streaming/retries/caching/credential-refresh on the agent core, not duplicated per transport — this is what keeps each transport small and prevents drift. Separate "resolve provider→(mode, key, url)" from "transport does the conversion"; the indirection lets you switch models via config with zero code change.

**SKIP.** The 35+ provider profiles, OAuth credential pools, and per-provider alias tables. Start with API-key env vars only and grow the registry as needed.

---

### Agent-loop modularization

**How Hermes does it.** The historical monolith is `run_agent.py` (`AIAgent`, ~50–60 constructor params, ~12k LOC) with the public loop method `run_conversation()`. Around it sits `agent/` (~14+ focused modules), each with one job:
- `prompt_builder.py` — assembles the system prompt in ordered tiers **stable → context → volatile** (identity/tool guidance, then context files, then memory blocks) so the cacheable prefix never moves.
- `prompt_caching.py` — manages provider cache breakpoints.
- `context_engine.py` (ABC) + `context_compressor.py` — pluggable context management; default does lossy summarization at threshold.
- `conversation_compression.py` — compresses history.
- `tool_executor.py` — tool dispatch + concurrency (see below).
- `tool_guardrails.py` — `before_call()` safety gating.
- `memory_manager.py` / `memory_provider.py` (ABC) — memory orchestration.
- `skill_commands.py` / `skill_preprocessing.py` — skills.
- `curator.py` (+ `curator_backup.py`) — skill lifecycle.
- `auxiliary_client.py` — cheap side-model calls (vision, summarization, titles).
- `model_metadata.py` — context lengths / token estimation.
- `trajectory.py` — ShareGPT export.
- `error_classifier.py`, `rate_limit_tracker.py`, `redact.py`, `display.py` — utilities.

The loop itself: build/cache system prompt → compression gate → interruptible API call → parse → dispatch tool calls → append results → repeat until a non-tool response or `max_iterations` (default 90). A thread-safe **`IterationBudget`** is shared across parent and subagents, with refunding so programmatic tool calls don't prematurely exhaust the budget. All surfaces (CLI, gateway, cron, ACP) converge on the *one* `AIAgent.run_conversation()`.

**ADOPT.** The module split by responsibility, and especially the **three-tier prompt ordering for cache stability** and "never mutate past context / toolsets / memory mid-conversation" rule — this directly protects prompt-cache hit rate (and cost). Adopt the single-core-loop-with-many-surfaces principle, an `IterationBudget` object, and wrapping tool exceptions as tool-result messages so the loop never crashes. The `auxiliary_client` pattern (route cheap meta-work to a small model with its own provider/model/limits) is worth copying.

**SKIP.** The 50-param constructor and 12k-LOC `AIAgent` — that's tech debt Hermes is actively decomposing, not a model to imitate. Build the loop modular from day one. Skip `trajectory.py` unless you're training models.

---

### Tool system & parallel execution

**How Hermes does it.** Tools self-register at import time via a dependency-free `tools/registry.py` (`registry.register(name, toolset, schema, handler, check_fn, requires_env)`). Auto-discovery imports tool modules, but a tool is only *exposed* if its name appears in a toolset (`toolsets.py`). Handlers take `(args: dict, **kwargs)` and **must return a JSON string and never raise** (catch-all → `{"error": ...}`).

Parallel execution lives in `agent/tool_executor.py` with two paths: `execute_tool_calls_concurrent()` and `execute_tool_calls_sequential()`. A heuristic `_should_parallelize_tool_batch()` decides which to use, classifying tools into three buckets:
1. **Never-parallel** (interactive/stateful, e.g. `clarify`) → forces sequential.
2. **Read-only safe** (allowlist: `read_file`, `search_files`, `web_search`, `web_extract`, `session_search`, `skill_view`, `skills_list`, `vision_analyze`, …) → always parallel-safe.
3. **Path-scoped** (`read_file`, `write_file`, `patch`) → parallel *only* if no path is a prefix of another's (same-subtree check serializes conflicting file ops).

Concurrent path: `max_workers = min(len(runnable_calls), _MAX_TOOL_WORKERS)` with `_MAX_TOOL_WORKERS = 8`, `concurrent.futures.ThreadPoolExecutor`, each call a future via `_run_tool()`. ContextVars are propagated into worker threads (`propagate_context_to_thread`); **results are reassembled in original tool-call order regardless of completion order.** Interrupts cancel unstarted futures and set per-thread interrupt flags for in-flight ones; ~30s heartbeats keep gateways from timing out. Pre-execution (both paths): tool-search unwrap → scope gate → `pre_tool_call` hook block check → guardrails → checkpoint snapshot before file mutations/destructive commands. (Subagent-level parallelism is separate: `delegate_task(tasks=[...])` runs child `AIAgent`s in a pool, default `max_concurrent_children=3`, plus a v0.11 file-coordination layer.)

**ADOPT.** The self-registering registry + JSON-string/never-raise handler contract (clean, testable, decoupled). Most importantly the **read-only-safe / write-path-serialized concurrency model**: allowlist read-only tools for unconditional parallelism, serialize file writes by path-prefix overlap, force interactive tools sequential, cap workers (~8), and reorder results to call order. Checkpoint before mutations. This gives big latency wins on multi-read turns with bounded risk.

**SKIP.** 70+ tools across ~28 toolsets and 6 terminal backends (Docker/SSH/Modal/Daytona/Singularity). Zak Code should ship a small core toolset (read/write/patch/search/shell/web) and let plugins add the rest. The subagent file-coordination layer is advanced; defer it.

---

### Plugin/hook system

**How Hermes does it.** Plugins are discovered from `~/.hermes/plugins/`, project `./.hermes/plugins/`, and pip entry points (`PluginManager` in `hermes_cli/plugins.py`). Each plugin has a `plugin.yaml` manifest and a single `register(ctx)` entrypoint. Hard rule: **plugins must never modify core files — you extend the generic plugin surface instead.** `ctx` exposes: `register_tool(...)`, `register_hook(event, cb)`, `register_cli_command(...)` (a `hermes <plugin>` subcommand), `register_command(name, handler, description)` (in-session `/slash` command), `dispatch_tool(...)` (call another tool with approval/budget inherited), plus specialized single-select registrars (`register_memory_provider`, `register_context_engine`, `register_image_gen_provider`, `register_platform`, etc.).

Lifecycle hooks fire through a unified dispatcher that **catches errors and continues** (one bad hook never breaks the loop). Signatures and return semantics:
- `pre_tool_call(tool_name, args, task_id, **kw)` → return `{"action":"block","message":...}` to veto; else ignored.
- `post_tool_call(tool_name, args, result, task_id, duration_ms, **kw)` → observer.
- `pre_llm_call(session_id, user_message, conversation_history, is_first_turn, model, platform, **kw)` → return a string or `{"context": "..."}` to **inject ephemeral context into that turn's user message only** (never the system prompt — preserves cache); multiple plugins' outputs join with double newlines.
- `post_llm_call(session_id, user_message, assistant_response, conversation_history, model, platform, **kw)` → observer (only on successful turns).
- `on_session_start(session_id, model, platform, **kw)` / `on_session_end(session_id, completed, interrupted, model, platform, **kw)` → observers.
- Plus transform hooks: `transform_tool_result`, `transform_terminal_output`, `transform_llm_output` (first non-empty return wins), and `subagent_stop`.

Hooks must accept `**kwargs` for forward compatibility.

**ADOPT.** The whole shape: `register(ctx)` entrypoint, `ctx`-based registration, the unified error-isolating dispatcher, and the six core hooks with these exact signatures. The **`pre_llm_call`-injects-into-user-message-not-system-prompt** design is the key insight (it's how memory/RAG/guardrails plug in without breaking cache) — adopt it verbatim. The "only `pre_*` returns affect behavior, everything else is fire-and-forget observer" contract keeps the system predictable. Require `**kwargs` on all callbacks.

**SKIP.** The dozen+ specialized single-select registrars (video gen, gateway platforms, TTS/transcription). Start with `register_tool`, `register_hook`, `register_command`, and one provider hook; add specialized surfaces only when a real need appears.

---

### Skills & learning loop & memory

**How Hermes does it.** *Skills* are markdown docs (`SKILL.md`) with YAML frontmatter, surfaced via **progressive disclosure**: Level 0 = name+description always in the system prompt; Level 1 = full SKILL.md loaded on demand; Level 2 = referenced `scripts/`/`references/`/`templates/` pulled as needed. Built-ins live in `skills/` (auto-loaded); heavier ones in `optional-skills/` (explicit install). The **learning loop**: after solving a non-trivial task the agent creates/updates a skill via the `skill_manage` tool (provenance `created_by: "agent"`), and skills self-improve during use. A background **Curator** (`agent/curator.py`) tracks per-skill usage (`~/.hermes/skills/.usage.json`), moves skills through active → stale → archived, and periodically spawns an auxiliary-model review proposing consolidations/patches. Invariants: only touches agent-created skills, never deletes (max action = archive), pinned skills exempt. Skills are invokable as `/<skill-name>` slash commands. Memory has three independent layers: (1) **frozen-snapshot files** `MEMORY.md`/`USER.md`/`SOUL.md` read once at session start and embedded immutably in the system prompt (char-capped, ~2200/~1375) so they never mutate mid-conversation; (2) **cross-session recall** via SQLite + FTS5 full-text search (`session_search` tool, LLM summarization); (3) **pluggable provider** behind a `MemoryProvider` ABC (`prefetch()`, `sync_turn()`, `shutdown()`) with backends like honcho/mem0/supermemory. Periodic "nudges" prompt the agent to persist knowledge.

**ADOPT.** Skills-as-markdown with **progressive disclosure** (name/desc cheap in prompt, body lazy-loaded) — efficient and provider-agnostic; align with the agentskills.io frontmatter format for portability. The **frozen-snapshot memory** pattern (read MEMORY.md/USER.md once, embed immutably, cap size) is excellent for cost/cache and simple to build. Add FTS5 session search early — cheap, high-value cross-session recall. Define a `MemoryProvider` ABC so external backends plug in later. The Curator's "never delete, only archive, only agent-created, track usage" invariants are a smart guardrail if you do auto-generated skills.

**SKIP / be cautious.** The full autonomous skill-creation + curator loop is the riskiest part to copy — auto-generated skills accumulate near-duplicates and need active maintenance. For Zak Code, start with **user/manually-authored skills + progressive disclosure**; add agent-authored skills + curator only once the catalog grows. Skip Honcho dialectic user-modeling and the 8 memory backends initially.

---

### Public library / embedding API

**How Hermes does it.** `from run_agent import AIAgent`. Two entry points: `agent.chat(message: str) -> str` (one-shot, returns final text) and `agent.run_conversation(user_message, system_message=None, conversation_history=None, task_id=None) -> dict` with `{final_response, messages}` for full control. Key constructor flags for embedding: `quiet_mode=True`, `skip_context_files=True`, `skip_memory=True`, `enabled_toolsets`/`disabled_toolsets`, `max_iterations`, `api_key`, `base_url`, `provider`, `api_mode`, `model`, `ephemeral_system_prompt`. Multi-turn works by passing the prior `messages` back as `conversation_history` (the original list is not mutated). **Hard constraint: `AIAgent` is not thread-safe — create one instance per thread/task** (e.g. `concurrent.futures` for batch, or the bundled `batch_runner.py`). Documented embeddings: FastAPI endpoint, Discord bot, CI step.

**ADOPT.** Expose exactly this two-tier API: a dead-simple `chat(str)->str` and a `run_conversation(...)->{final_response, messages}` for control. Make `conversation_history` round-trippable and non-mutating. Provide embedding-friendly flags (`quiet`, `skip_context_files`, `skip_memory`) so the agent can run headless in apps/CI. Document and enforce "one instance per thread."

**SKIP.** Importing from a top-level `run_agent` module name is awkward — Zak Code should expose a clean package import (`from zak_code import Agent`). Consider making it thread-safe (or clearly immutable-config + per-call state) rather than shipping the "not thread-safe" footgun.

---

### Packaging (uv)

**How Hermes does it.** `uv`-managed: `pyproject.toml` + `uv.lock`, Python 3.11+, extras like `[all]`/`[dev]`. Install via a curl/PowerShell bootstrap script (`scripts/install.sh` / `install.ps1`) that sets up the venv and symlinks the executable; dev setup via `./setup-hermes.sh`. **Dependency policy: every dependency must have an upper bound** (`>=floor,<next_major`; git URLs pinned to commit SHA — never bare `>=`). State/config live under `~/.hermes/` resolved through `get_hermes_home()` (profile-aware via `HERMES_HOME`), with `config.yaml` for settings and `.env` for secrets only. Tests run through `scripts/run_tests.sh` (hermetic: unset creds, `TZ=UTC`, `-n auto` xdist, subprocess-per-test isolation).

**ADOPT.** `uv` + `pyproject.toml` + committed lockfile, Python 3.11+, extras for optional deps. **Pin upper bounds on all deps** — cheap insurance against breakage. Separate `config.yaml` (settings) from `.env` (secrets only), and route all state through one `get_home()` helper rather than hardcoding paths (makes multi-profile/testing trivial). Hermetic test runner with subprocess isolation is good hygiene if you have import-time registration (which the registry pattern implies).

**SKIP.** The OS-specific curl/PowerShell installer scripts and MinGit bundling — for a dev-focused coding agent, `uv tool install` / `pipx` / `uv pip install` is enough. Skip the full multi-profile `HERMES_HOME` machinery until you actually need concurrent instances; just keep paths centralized so you *can* add it.

---

**Net recommendation for Zak Code (priority order):** (1) `ProviderTransport` ABC with cross-cutting concerns on the core; (2) modular agent loop with three-tier cache-stable prompt + `IterationBudget`; (3) self-registering tool registry with read-safe/write-path-serialized parallel execution; (4) `register(ctx)` plugin system with the six hooks and `pre_llm_call` user-message injection; (5) two-tier `chat()`/`run_conversation()` library API; (6) frozen-snapshot memory + FTS5 search + `MemoryProvider` ABC; (7) progressive-disclosure markdown skills (manual first, curator later); (8) uv packaging with pinned deps and config/secrets split. Deliberately skip the surface sprawl (multi-platform gateway, 6 terminal backends, 35+ providers, autonomous-skill curator) until the core is proven.

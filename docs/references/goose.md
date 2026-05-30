## goose architecture digest

Block's `goose` is an open-source, on-machine AI agent (Apache 2.0, now under the Linux Foundation's Agentic AI Foundation) built as a Rust Cargo workspace plus a TypeScript/Electron desktop app. Its defining architectural bets are provider-agnosticism, MCP-as-the-only-extension-mechanism, and a single core agent loop shared across all UIs. The notes below extract concrete patterns and translate each into lessons for **Zak Code** (Python coding agent).

---

### 1. Provider abstraction

**How goose does it:**
- A single `Provider` trait is the runtime interface every backend implements: `complete()` (single request/response with usage metrics), `stream()` (SSE streaming), `fetch_supported_models()`, `generate_session_name()`, plus metadata/capability methods. A companion `ProviderDef` trait handles metadata + instantiation (`from_env()` reads a global `Config` singleton). This keeps the agent layer fully provider-agnostic — concrete structs (`openai.rs`, `anthropic.rs`, `google.rs`, `databricks.rs`, `ollama.rs`, etc.) absorb all per-API quirks.
- **15–25+ providers** across three families: API-based (Anthropic/Claude, OpenAI, Google Gemini, xAI Grok, Mistral), cloud platforms (Amazon Bedrock, GCP Vertex AI, Azure OpenAI, Databricks, Snowflake), and local (Ollama, Ramalama, Docker Model Runner). Plus CLI pass-through "providers" (Claude Code, OpenAI Codex, Cursor Agent, Gemini CLI) and ACP agents-as-providers.
- **Factory + registry:** `create_with_named_model()` / `create()` instantiates by name + model. A **build-time "canonical models registry" of ~1,700 models** auto-resolves context windows, token limits, and capability flags (streaming, native tool support, vision) so the agent doesn't hardcode model facts.
- **`ModelConfig`** is the central struct carrying model id, token/context limits, and capability flags.
- **`AuthMethod` enum** abstracts credentials: API-key headers (OpenAI `OPENAI_API_KEY`, Anthropic `x-api-key` + `anthropic-version`), OAuth device flow (GitHub Copilot, Databricks, with `DiskCache` token caching), static tokens, subprocess-managed creds (Claude Code/Gemini CLI), or none (local Ollama).
- **Unified error + retry:** `ProviderError` categorizes failures (`Authentication`, `ContextLengthExceeded`, `RateLimitExceeded`, `RequestFailed`). Default backoff = 3 retries, 1s initial, 2x multiplier, 60s cap; per-provider overrides (Ollama uses 10 retries for slow local model loading; Google parses server `retryDelay`; Databricks has large-model configs).
- **Tool shim:** models lacking native tool-calling get a shim that serializes tool schemas into the prompt as text and parses tool calls back out of generated text — so the agent's tool-calling path is identical regardless of model.
- **Lead/Worker pattern:** `create()` detects env config and can wrap two providers in a `LeadWorkerProvider`, routing expensive "lead" reasoning vs. cheap "worker" execution.
- Custom providers must speak OpenAI-, Anthropic-, or Ollama-compatible formats; custom headers allowed for tenant ids/extra auth.

**Lessons for Zak Code:**
- Define one narrow `Provider` Protocol/ABC in Python (`complete`, `stream`, `list_models`, `capabilities`) and make the agent depend only on it — never on a vendor SDK directly. Adapters live in `zak/providers/<name>.py`.
- Ship a **declarative model registry** (a YAML/JSON data file, not code) mapping model id → context window, max output, supports_tools, supports_vision, supports_caching. This is goose's single highest-leverage decoupling move; it lets you add models without code and lets context-management logic query limits generically.
- Centralize **auth as a strategy/enum** (api-key header, bearer, OAuth device flow with on-disk token cache, env-only, none) so a new provider only declares which method it uses.
- Centralize **retry/backoff and a normalized error taxonomy** (`AuthError`, `ContextLengthExceeded`, `RateLimited(retry_after)`, `RequestFailed`) in the base layer; providers only translate their wire errors into it. Honor server `Retry-After`.
- Implement a **tool shim** early so non-tool-native or local models still work via prompt-encoded tool schemas + parsing — keeps the agent loop uniform.
- Steal the **lead/worker** idea: allow a config where a strong model plans and a cheap/local model executes routine steps; wrap both behind the same `Provider` interface so the agent is unaware.

---

### 2. Extension system via MCP

**How goose does it (the core design bet — every integration is an MCP server):**
- goose is an **MCP host/client**; each extension is an **MCP server** with a 1:1 client connection. The `ExtensionManager` owns a map of extension-name → `McpClient`, and is the sole orchestrator that (a) discovers extensions, (b) initializes them over the right transport, and (c) executes tool calls against them. This decouples all capabilities from the core — ~70+ extensions exist without touching core code.
- Extensions expose the three MCP primitives: **tools** (JSON-Schema functions), **resources** (URI-addressable data), **prompts** (templates).
- **Six extension types:**
  - `stdio` — local MCP server as a child process (most third-party integrations; `cmd` + `args` + env), via env `env_keys`/`envs`.
  - `builtin` — compiled into the goose binary, in-process, referenced by name (e.g. `developer`).
  - `platform` — in-process system features (search, subagents).
  - `streamable_http` — remote MCP server over HTTP/SSE (URI + optional auth headers).
  - `frontend` — tools handled by the Desktop UI renderer.
  - `inline_python` — Python embedded in a recipe, run via `uvx`.
- **Builtin vs external:** builtin (`developer` = file ops, shell, text editor, search, tree-sitter, screenshots; `computer_controller` = GUI automation; `memory` = `remember_memory`/`retrieve_memories`/`remove_memory_category`, with `.goose/memory` local vs `~/.config/goose/memory` global) ship in-process with zero subprocess overhead. External (stdio/streamable_http) come from package managers or remote endpoints.
- **Tool namespacing (the key detail):** every extension's tools are prefixed `{extension_name}__{tool}` (double underscore). `developer.write_file` → `developer__write_file`; `memory.search` → `memory__search`; also `developer__shell`, `developer__text_editor`, `memory__remember_memory`. `get_prefixed_tools()` applies the prefix at registration; `dispatch_tool_call()` reverses it — splits on `__`, looks up the extension, strips the prefix, invokes the underlying tool. This lets multiple servers expose same-named tools collision-free and gives O(1) routing.
- **Tool budgeting:** an `available_tools` config field narrows which tools are exposed to the LLM (token savings); goose recommends keeping it near **~25 tools** because large tool sets degrade selection quality.
- **Security:** OSV vulnerability scanning, `GOOSE_ALLOWLIST` URL allowlisting, command whitelist for deeplink installs (only `npx`, `uvx`, `docker`, `cu`, `jbang`, `goosed`; blocks `npx -c` injection), and `env_keys` that prompt the user for secrets rather than embedding them. Install via `goose://extension?...` deeplinks (`cmd`, `arg`, `url`, `name`, `env`, `header`, `timeout` default 300s).

**Lessons for Zak Code (our MCP strategy):**
- **Adopt MCP as the one extension contract.** Make Zak Code an MCP host; every integration — including your own first-party tools — is an MCP server. This is goose's biggest structural win: no second plugin API to maintain, and you inherit the entire MCP ecosystem (GitHub, DBs, browsers, Drive) for free.
- Build a Python **`ExtensionManager`** holding name→client, responsible for discovery, lifecycle (init/shutdown), and tool dispatch. Keep the agent ignorant of transports.
- **Namespace tools as `extension__tool` with `__`** and route by splitting the prefix. Match MCP's name regex `^[a-zA-Z0-9_-]{1,64}$` — do **not** use `:` or `.` as separators (goose/Docker-gateway hit MCP-spec violations doing so). Watch total prefixed length against the 64-char limit.
- Support at minimum **stdio + streamable-HTTP** transports, plus **in-process "builtin"** servers for hot-path tools (file/shell/edit/search) to avoid subprocess latency. Python makes in-process trivial — register them through the same MCP interface so they're indistinguishable to the agent.
- Ship a small set of **builtins mirroring goose**: a `developer`-style server (read/write/edit files, run shell, search, tree-sitter-based code structure) and a `memory` server with local-vs-global scoping. These are the must-haves for a coding agent.
- Implement **a tool budget / `available_tools` filter** and keep the default exposed set small (~25). Provide per-recipe/per-session tool allow-lists so users can prune noisy MCP servers.
- Mirror the **security posture**: prompt for secrets via `env_keys` (never inline), maintain a command allowlist for auto-installing stdio servers, optional URL allowlist for HTTP extensions, and consider OSV scanning of installed servers.

---

### 3. Agent loop

**How goose does it:**
- The entry point is `Agent::reply(user_message, session_id)`, a single loop that all UIs route through. Each turn: (1) load active extensions + assemble the tool set via `ExtensionManager`, (2) build the system prompt dynamically (`PromptManager` / `prompt_manager`) from extension info + mode, (3) check/apply context management, (4) stream the LLM response via the provider, (5) inspect + dispatch any tool calls, (6) feed results back, (7) persist to session. Repeat until the model stops requesting tools or hits a limit.
- It emits **`AgentEvent`** items (text, tool_call, tool_result) so CLI, HTTP server, and ACP all consume the same event stream.
- **Tool inspection before execution:** calls pass through a `ToolInspectionManager` chaining a `SecurityInspector` (prompt-injection detection), a `PermissionInspector`, and a repetition/loop detector (`ToolMonitor`) to prevent infinite loops. Approved calls run **concurrently** via `dispatch_tool_call()`.
- **Permission modes (`GooseMode`):** `Auto` (run everything — current default), `Approve` (confirm every call), `Chat` (no tools, pure chat), `SmartApprove` (read-only auto-runs; writes/destructive require approval, decided by an LLM-based `PermissionJudge`/`PermissionInspector` 5-layer hierarchy). Switchable mid-session via `/mode`; user+cached decisions persist in `~/.config/goose/permission.yaml`. On approval-needed, the agent suspends and emits `ActionRequired` → UI renders a `ToolConfirmation`.
- **Context management:** `context_mgmt` module with `check_if_compaction_needed()` + `compact_messages()`, token-counted via `tiktoken-rs`. **Auto-compaction fires at ~80% of the token limit** (tunable via `GOOSE_AUTO_COMPACT_THRESHOLD`); strategies = summarize / truncate / clear / prompt. Older tool outputs get summarized in the background while recent calls stay full-detail (kicks in past ~10 tool calls). `/compact` triggers a `HistoryReplaced` event. **Max Turns** default = 1000 consecutive turns without user input.

**Lessons for Zak Code:**
- Build **one `reply()`-style async loop** that emits a typed event stream (`text` / `tool_call` / `tool_result` / `action_required`). Every front-end (CLI, server, future IDE) consumes events — never duplicate loop logic per UI.
- Put a **pluggable inspection pipeline** in front of tool dispatch: permission check, prompt-injection/security check, and a repetition guard. Make inspectors composable so policy is data, not branching code.
- Implement **four permission modes** equivalent to Auto/Approve/Chat/SmartApprove. SmartApprove (auto-allow reads, gate writes/deletes/network) is the right default for a coding agent — safer than goose's Auto default, which the community flagged as dangerous. Persist per-tool decisions to a `permission.yaml`-style cache.
- **Dispatch independent tool calls concurrently** (Python `asyncio.gather`) for latency.
- Add **auto-compaction at a configurable token threshold (~80%)** with summarize/truncate/clear strategies, summarizing stale tool outputs while preserving recent detail. Use the model registry's context limits to know when to fire. Add a **max-turns** safety cap and a manual `/compact`.

---

### 4. Core / CLI / Server / Desktop separation

**How goose does it:** a Cargo workspace with clean layering:

| Crate / dir | Role | Tech |
|---|---|---|
| `goose` | Core library: Agent loop, Provider system, ExtensionManager, SessionManager (SQLite), Recipe engine, config | Rust lib |
| `goose-mcp` | Built-in MCP servers (developer, computer_controller, memory) | Rust |
| `goose-cli` | Interactive CLI (`goose` binary) — calls the core library directly | Rust |
| `goose-server` | Backend daemon `goosed` (HTTP + WebSocket) | Rust + Axum |
| `ui/desktop` | Rich chat/settings GUI | Electron + React → talks to `goosed` |

- All UIs funnel into the **same core `Agent`**; the CLI links the library in-process, while Desktop talks to `goosed` over HTTP/WS. Provider/extension/session logic lives once in core and is never reimplemented per surface. There's also ACP protocol support as another consumer of the same loop.

**Lessons for Zak Code:**
- Structure as a **`zak-core` library + thin frontends**: `zak-cli`, `zak-server` (FastAPI/Starlette with WebSocket streaming of the AgentEvent stream), and any future GUI. Core must have **zero UI dependencies**.
- Split **builtin MCP servers into their own package** (`zak-mcp`) so they're reusable and independently testable — and could even run standalone for other MCP hosts.
- Have the CLI **call core in-process** (fast, simple) and the server **wrap core behind HTTP/WS** for remote/desktop use — both emitting the identical event stream so a future desktop/IDE client is a thin renderer, not a fork of the agent logic.
- Keep **session persistence in core** (SQLite is a fine default — file-based, zero-ops, resumable) so every surface gets resumable sessions for free.

---

### 5. Recipes / sessions

**How goose does it:**
- **Sessions** are the persistent backing of every conversation, stored in **SQLite** via `SessionManager`. A session records conversation history, token usage, working directory, associated recipe(s), provider metadata, and extension state. Sessions are resumable; `persist_extension_state()` / `load_extensions_from_session()` restore extensions concurrently on resume.
- **Recipes** are declarative **YAML** workflows run through a Recipe engine using **minijinja templating**. Core fields: `title`, `description`, `instructions` (agent role/context), `prompt`, `activities` (suggested starting actions), `parameters` (typed placeholders, referenced as `{{ param }}`, substituted into instructions/prompt/activities before run), `extensions` (which MCP servers to load — same six types as above), and an optional `response` with a `json_schema` to **enforce structured JSON output** (validated against the schema).
- **Sub-recipes:** a main recipe registers `sub_recipes` (each with `name` → used to generate a tool name, and `path`). Sub-recipes are exposed to the agent as callable tools, run in **isolated worker processes**, can execute **in parallel** (batch/fan-out), and can't nest (one level only). Each subagent/worker gets its **own ExtensionManager, ToolMonitor, channels, and isolated context** (`Agent::new()`).
- A **scheduler** runs recipes on **cron** schedules, creating automated unattended sessions.

**Lessons for Zak Code:**
- Make **sessions first-class and persistent** (SQLite): store history, token usage, cwd, provider/model, active extensions, and recipe linkage — and make them **resumable**, restoring extension state on resume. This is essential for a long-running coding agent.
- Adopt a **declarative recipe format** (YAML + Jinja2 — natural in Python) with `title`/`description`/`instructions`/`prompt`/`parameters`/`extensions`/`activities`. Recipes bundle prompt + tool/extension selection + settings into one shareable, version-controllable artifact — ideal for reproducible coding workflows ("run our PR-review pipeline").
- Support a **`response.json_schema`** to force structured output, and validate against it — critical when recipes feed downstream automation/CI.
- Implement **sub-recipes as agent-callable tools running in isolated subprocesses with their own ExtensionManager/context**, enabling **parallel fan-out** (e.g. lint N files, analyze M modules concurrently) and clean context isolation. Enforce the **one-level no-nesting** rule to bound complexity.
- Add a **cron scheduler** that materializes recipes into unattended sessions for recurring tasks (nightly dependency upgrades, scheduled audits).

---

**Top transferable principles for Zak Code:** (1) one `Provider` interface + a data-driven model registry = painless multi-provider support; (2) MCP as the *single* extension contract, with `extension__tool` namespacing and in-process builtins for hot paths; (3) one shared agent loop emitting a typed event stream that every UI consumes; (4) a `zak-core` library with thin CLI/server frontends; (5) SQLite-backed resumable sessions plus declarative Jinja2 recipes with schema-validated output and parallel isolated sub-recipes.

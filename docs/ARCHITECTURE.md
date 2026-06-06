# Zak Code — Architecture

## Overview

Zak Code is a clean-room, vendor-agnostic Python coding agent. Its defining bet is a hard split into three layers: a **core engine** (`zakcode`) that is an importable, UI-free library implementing the whole agent — provider abstraction, agent loop, tools, sessions, context management, and the extension model; a **server** (`zakcode-server`) that exposes the core over HTTP with SSE and WebSocket streaming; and **thin clients** (a Typer CLI today, a web UI later) that render a typed event stream and never re-implement agent logic. Every provider is normalized through **LiteLLM** behind one small `Provider` abstraction so Ollama and OpenAI are first-class and any of LiteLLM's 100+ backends is reachable by changing a model string. The core is opinionated about the things that decide whether a coding agent is usable in practice: real token counting and auto-compaction, deny-first permissions enforced on a code path the model cannot reach, parallel execution of read-only tools with path-conflict serialization for writes, structured (non-string) tool I/O, atomic session writes, and a stable cacheable system-prompt prefix.

## Layered diagram

```
+-----------------------------------------------------------------------------+
|                            THIN CLIENTS                                      |
|                                                                             |
|   zakcode/cli/ (Typer)            web client (later)        IDE (later)      |
|   - REPL + one-shot               - browser UI              - editor plugin  |
|   - renders AgentEvent stream     - consumes SSE/WS         - consumes WS    |
|   - in-process (calls core)       - over HTTP               - over HTTP      |
+--------------------|--------------------------|-----------------|------------+
                     | in-process import         |  HTTP/SSE/WS    |  HTTP/WS
                     |                           v                 v
                     |     +-------------------------------------------------+
                     |     |        API SERVER  (zakcode-server)             |
                     |     |        zakcode/server/  (FastAPI)               |
                     |     |  - REST: /sessions, /tools, /config             |
                     |     |  - POST /chat            (buffered turn)        |
                     |     |  - POST /chat/stream     (SSE AgentEvents)      |
                     |     |  - WS   /ws/{session}    (bidi: input+events    |
                     |     |                           +approval prompts)    |
                     |     |  - serializes AgentEvent -> SSE/WS frames       |
                     |     +-----------------------|-------------------------+
                     |                             | imports
                     v                             v
+-----------------------------------------------------------------------------+
|                       CORE ENGINE  (zakcode, importable library)            |
|                                                                             |
|   agent/        loop.py  prompt.py  compact.py     <- the ReAct loop        |
|       |              emits AgentEvent (text | tool_call | tool_result |     |
|       |              action_required | usage | done)                        |
|       v                                                                     |
|   providers/   LiteLLM-backed Provider (sync+async, stream, normalize)      |
|   tools/       base.py registry + builtins/ (read/write/edit/glob/grep/...) |
|   session/     persistence + in-memory state (atomic, resumable)           |
|   commands/    slash-command dispatch (/compact, /model, /resume, ...)      |
|   hooks/       PreToolUse/PostToolUse/pre_llm_call ... lifecycle dispatcher |
|   plugins/     register(ctx) extensions; contribute tools/hooks/commands    |
|   skills/      progressive-disclosure markdown skills (SKILL.md)            |
|   config.py    pydantic-settings layered config + permission model          |
|                                                                             |
|   external:  LiteLLM ---> OpenAI | Ollama | Anthropic | ... (100+)          |
|              MCP clients (stdio + streamable-HTTP) ---> MCP servers          |
+-----------------------------------------------------------------------------+
```

Rule: dependencies point **inward**. `cli/` and `server/` depend on the core; the core depends on nothing in the outer layers. The core has zero UI dependencies and runs headless (in a script, in CI, in a FastAPI handler) with identical behavior.

## Module map (`src/zakcode/*`)

### `config.py`
Layered application configuration via **pydantic-settings**, plus the permission model types. Responsibilities: resolve config (highest precedence first) from explicit overrides → `ZAKCODE_`-prefixed environment variables → a local `.env` → defaults, and validate eagerly (fail fast on bad config/credentials at construction, not first call). A later iteration deep-merges JSON scopes (user → project → local). State routes through one home-dir helper (honoring `ZAKCODE_HOME`) so paths are never hardcoded and multi-profile/testing is trivial. Settings and **secrets are deliberately separated**: provider API keys are *not* modeled here — LiteLLM reads them from their standard env vars (`OPENAI_API_KEY`, etc.), keeping secrets out of the config surface.
Key types: `Settings` (root `BaseSettings`: `default_model` — named so because pydantic reserves the bare `model` — `fallback_model`, `temperature`, `ollama_base_url`, `max_iterations`, `permission_mode`, `workspace_root`, plus a `provider` property deriving the prefix), `load_settings()`, `ModelConfig` (model string, `api_base`, temperature, `max_tokens`, timeout, `num_retries`, `extra`), `PermissionMode` (ordered enum), `PermissionPolicy`, `PermissionRule` (input-pattern aware), `ToolBudget`.

### `providers/`
The vendor-agnostic LLM layer built **on top of LiteLLM**. Nothing outside this package imports `litellm` directly, so the model client is swappable, mockable, and the place where cost/retry/normalization are centralized. Exposes a narrow `Provider` (sync `complete` + async `acomplete`, sync `stream` + async `astream`, `count_tokens`, `capabilities`, `list_models`). Translates the canonical internal message/tool shape to LiteLLM's OpenAI-shaped request and normalizes the `ModelResponse` (and streaming chunks) back to `LLMResult`/`AssistantEvent`. Owns retry/backoff, error-taxonomy mapping, prompt-cache breakpoints, and token/cost accounting.
Key types: `Provider` (ABC), `LiteLLMProvider`, `ModelConfig`, `LLMResult` (`text`, `tool_calls: list[ToolCall]`, `finish_reason`, `usage`, `cost_usd`, `raw`), `ToolCall` (`id`, `name`, `arguments: dict`), `AssistantEvent` (flat normalized stream enum: `TextDelta | ToolCallDelta | ToolCallComplete | Usage | MessageStop`), `Capabilities` (`supports_tools`, `supports_vision`, `supports_caching`, `context_window`, `max_output`), and the error taxonomy (`AuthError`, `ContextWindowExceeded`, `RateLimited(retry_after)`, `RequestFailed`).

### `agent/`
The provider-agnostic engine. Pure, testable, and the single code path every surface runs through.
- `loop.py` — `AgentLoop.run_turn()` / `arun_turn()`: the ReAct cycle. Holds zero provider or transport knowledge (depends only on `Provider`, the tool executor, and the permission policy via dependency injection). Emits a typed `AgentEvent` stream. Key types: `AgentLoop`, `TurnResult` (`assistant_messages`, `tool_results`, `iterations`, `usage`), `AgentEvent` (`text | tool_call | tool_result | action_required | usage | done`), `IterationBudget` (shared across parent and sub-agents, supports refunds), `StopReason`.
- `prompt.py` — `SystemPromptBuilder`: assembles the system prompt in ordered tiers **stable → context → volatile** around a `DYNAMIC_BOUNDARY` marker so the cacheable prefix never moves. Stable = identity, safety policy, tool guidance. Context = environment, git status/diff snapshot, discovered memory files (`ZAK.md` ancestor-chain discovery, root→cwd, content-hash dedup, per-file + total char budgets). Volatile = ephemeral per-turn injections. Key types: `SystemPromptBuilder`, `PromptSection`, `MemoryFile`.
- `compact.py` — `Compactor`: context management. `should_compact()` uses **real token counts** (`provider.count_tokens`, not `len/4`) against the model's context window; auto-fires the LLM-written summarization pass; preserves the last N turns verbatim; collapses older history into a single leading system summary; merges with any prior summary (idempotent re-compaction) and emits a "resume directly, do not acknowledge the summary" continuation instruction. Key types: `Compactor`, `CompactionConfig` (`preserve_recent`, `threshold_fraction`), `CompactionResult`.

### `tools/`
- `base.py` — the self-registering tool registry and execution seam. Tools are **data + a handler**: a declarative spec (name, description, JSON-schema parameters, required permission tier, concurrency class) separate from an async `execute(args: dict) -> ToolResult`. Handlers never raise — failures are wrapped into an error `ToolResult` so the loop never crashes. Provides `definitions(allowed)` (schemas sent to the model, filtered by tool budget), `permission_specs(allowed)`, name aliasing, and one canonical `execute(name, args)` dispatch. MCP tools register here too under qualified names. Key types: `ToolSpec`, `Tool` (ABC), `ToolRegistry`, `ToolResult` (`output`, `is_error`, optional structured `data`/images, and **rails** `hint`/`fix`), `ToolContext`, `ConcurrencyClass` (`ReadOnlySafe | PathScoped | NeverParallel`). **Result rails:** a tool may return a `hint` (next step on success) or `fix` (remedy on error); the loop renders whichever is set as a trailing `Hint:`/`Fix:` line in the model-facing text and mirrors it into `data`. Naming the next action is the biggest help for a small model (which is weak at planning the next step and at error recovery) — e.g. a permission denial states which `permission_mode` unblocks the call, and `remember` hints the model to end the turn once the write succeeds.
- `builtins/` — the small, sharp core toolset: `read`, `write`, `edit`, `glob`, `grep`, `bash`, `powershell`, `todo_write`, plus daily-driver tools (`web_fetch`, `web_search`, `task` sub-agent, plan-mode enter/exit). Each enforces timeouts and output caps, emits truncation hints, and keeps tool input/output as structured `dict`/`ToolResult`, never a re-parsed string.

### `mcp/`
The Model Context Protocol **client** subsystem (M5) — a clean-room, no-SDK implementation that makes Zak Code an MCP host. `MCPClient` speaks JSON-RPC 2.0 over a pluggable `Transport`; `StdioTransport` frames messages as **newline-delimited JSON** (one object per line — *not* LSP `Content-Length`) over an asyncio subprocess, with lazy spawn + initialize-once. `ExtensionManager` discovers each server's tools and adapts them to plain `McpTool`s registered into the **same** `ToolRegistry` under qualified names `mcp__<server>__<tool>`, so the loop dispatches and permission-gates them exactly like builtins (default tier `DangerFullAccess`). `config.py` declares servers in JSON, resolves secrets via `${VAR}` env refs (never stored in config), and enforces a command allowlist. Discovery is **lazy**: a `tool_search` builtin plus a `ToolRegistry` active-set keep MCP schemas out of the prompt beyond a tool budget (~25), surfacing them on demand. The facade wires it opt-in (`Agent(enable_mcp=True)` → `connect_mcp()`); a bad/unsupported server is recorded as data, never fatal.
Key types: `Transport` (Protocol), `StdioTransport`, `MCPClient`, `McpToolDef`, `McpCallResult`, `McpTool`, `ExtensionManager`, `DiscoveryReport`, `McpServerConfig`. _Deferred: streamable-HTTP transport, MCP resources/prompts, OAuth._

### `session/`
Session persistence and in-memory state. One file per session keyed by a UUID, stored under the home dir; **atomic writes** (temp file + rename). Persists after every turn and every mutating command. Per-message `usage` stored inline so cumulative cost is reconstructable on resume without a side file. Tool input is stored as parsed JSON. Versioned document for forward migration.
Key types: `Session` (`version`, `id`, `cwd`, `model`, `messages`, `created_at`), `Message` (`role: system|user|assistant|tool`, `blocks`), `ContentBlock` (`Text | ToolUse{id,name,input:dict} | ToolResult{tool_use_id,output,is_error,data} | Thinking{text,signature}`), `SessionStore` (`save`/`load`/`list`/`resume`), `UsageTracker`.

### `commands/`
Slash-command dispatch for interactive surfaces. A command is a name + handler + description, registered in a table (plugins can add more). Persists the session after any mutating command.
Key types: `Command`, `CommandRegistry`, `CommandResult`. Built-ins: `/help`, `/clear`, `/exit`, `/model`, `/compact`, `/cost`, `/context`, `/resume`, `/session`, `/config`, `/permissions`, `/memory`, `/init`, `/diff`, `/plan`, `/mcp`, `/skills`, `/hooks`, `/agents`.

### `hooks/`
The lifecycle hook engine — user-configurable shell hooks and in-process plugin callbacks behind one dispatcher that **catches errors and continues** (one bad hook never breaks the loop). Shell hooks receive a JSON payload on stdin and use an exit-code protocol (0 = allow, 2 = deny/veto, other = warn-and-continue); a denied tool yields an error `ToolResult` and the model sees the feedback. Hooks are invoked with argv arrays, not a shell string, to avoid injection.
Key types: `HookEvent` (`PreToolUse | PostToolUse | pre_llm_call | post_llm_call | on_session_start | on_session_end`), `HookManager`, `HookResult` (`action: allow|block`, `message`, optional mutated args). Only `pre_*` returns affect behavior; the rest are observers. `pre_llm_call` injects ephemeral context into **that turn's user message only** — never the system prompt — to preserve cache.

### `plugins/`
The first-party Python extension model. Plugins are discovered from `~/.config/zakcode/plugins/`, `./.zakcode/plugins/`, and pip entry points; each has a manifest and a single `register(ctx)` entrypoint and **must never modify core files**. `ctx` exposes `register_tool`, `register_hook`, `register_command`, and select single-slot providers. Plugin tools may also run as subprocesses fed JSON on stdin with `ZAKCODE_*` env vars (language-agnostic).
Key types: `Plugin`, `PluginManifest`, `PluginContext` (`ctx`), `PluginManager`.

### `skills/`
Progressive-disclosure skills as markdown (`SKILL.md` with YAML frontmatter). Level 0 = name+description always cheaply in the prompt; Level 1 = full body loaded on demand; Level 2 = referenced scripts/templates pulled as needed. Loaded from bundled, user, and project dirs; invokable as `/<skill-name>`. Manual/user-authored first; agent-authored skills and a curator are deferred.
Key types: `Skill`, `SkillFrontmatter`, `SkillRegistry`, `SkillLoader`.

### `server/`
FastAPI app wrapping the core (see API surface below). Serializes the core's `AgentEvent` stream to SSE and WebSocket frames; bridges interactive permission prompts to clients over the WebSocket `action_required` channel; manages one shared asyncio event loop for the process. The default agent factory builds each request's `Agent` with a full **mind** — operator identity (`self.md`), always-on rules, cross-session memory (one shared SQLite provider), and skills — loaded from the workspace root (`serve --workspace` / `ZAKCODE_WORKSPACE_ROOT`); the topology is one container per customer env.
Key types: `create_app()`, request/response Pydantic models (`ChatRequest`, `ChatResponse`, `SessionInfo`, `ToolInfo`), `EventSerializer`.

### `cli/`
The Typer CLI — a thin client that **calls the core in-process** (no HTTP). Runs a REPL and one-shot mode, renders the `AgentEvent` stream with stream-safe-boundary flushing (never render half a code fence), shows a live status line (iterations, tokens, compaction, active model, pending approvals), and prompts for permission escalations.
Key types: the Typer `app`, `Renderer`, `CliPermissionPrompter`, `StatusLine`.

## Data flow of one agent turn

The loop is a ReAct cycle wrapped in layered stop conditions. One turn:

1. **User input.** The client (CLI in-process, or server via `/chat`/WS) calls `AgentLoop.arun_turn(user_text)`. The text is appended to `session.messages` as a `user` message and the session is persisted.
2. **Compaction gate.** `Compactor.should_compact(session, provider)` compares real token counts (`provider.count_tokens`) against the model's context window (from `provider.capabilities`). At ~70–80% it runs an LLM-written summary, preserves the last N turns verbatim, replaces older history with one leading system summary, and emits a `compaction` event.
3. **`pre_llm_call` hooks.** Plugin/shell `pre_llm_call` hooks may return text that is joined and appended to **this turn's user message only** (never the system prompt — preserves the cache prefix).
4. **Prompt assembly.** `SystemPromptBuilder.build()` produces the ordered system prompt (stable cacheable prefix, then context: env + git snapshot + memory files, then volatile) with the `DYNAMIC_BOUNDARY` marker positioned for the provider's cache breakpoint.
5. **Provider call (via LiteLLM).** The loop builds the canonical request (system prompt, `session.messages`, `tools = registry.definitions(allowed)` filtered by the tool budget, `tool_choice="auto"`) and calls `provider.astream(...)`. `LiteLLMProvider` translates to LiteLLM's OpenAI-shaped `acompletion(stream=True, stream_options={"include_usage": True})`, targeting OpenAI, Ollama, or any backend by model-string/`api_base` with no loop changes.
6. **Stream parse / normalization.** LiteLLM chunks are normalized to the flat `AssistantEvent` enum: `TextDelta` is forwarded as a `text` `AgentEvent` for live rendering; tool-call argument fragments are accumulated **by chunk `index`** (only the first delta carries `id`/`name`; later deltas carry partial `function.arguments`) into complete `ToolCall`s with parsed `dict` arguments; usage arrives in the final chunk. The full assistant message is reconstructed and appended to the session (text block + any `ToolUse` blocks + thinking blocks if present). Usage and `cost_usd` are recorded.
7. **Tool-call extraction & stop check.** Collect the assistant's `ToolUse` blocks. **If there are none → natural completion: emit `done` and stop.** Otherwise continue. The loop also enforces the other terminators here: iteration cap (default ~20, hard error/summary on exceed — never unbounded), token/cost budget, a **doom-loop detector** (same tool + identical args within N iterations → halt as `doom_loop`), and a broader **multi-signal stuck detector** (`agent/stuck.py`) that votes across several no-progress signals (all-failing batch, repeatedly-failing call, near-repeat with no progress) and, on a sustained streak, runs an **escalating recovery ladder** — inject a nudge, then narrow the next iteration to read-only tools, and only then stop as `stuck`. A thin `degraded` flag rolls up onto `TurnResult`/`AgentDone` when a turn engaged recovery or ended non-cleanly. Cooperative cancellation is checked each iteration.
8. **Permission + parallelism planning.** Each pending call passes through the inspection pipeline before execution: permission check (`PermissionPolicy.authorize(name, args)`), `PreToolUse` hooks, and a dangerous-pattern check. The loop then partitions the approved calls by concurrency class:
   - `ReadOnlySafe` (e.g. `read`, `glob`, `grep`, `web_fetch`) → run concurrently via `asyncio.gather`, capped at ~8 workers.
   - `PathScoped` (e.g. `write`, `edit`) → parallel **only** when no path is a prefix of another's; conflicting subtrees are serialized.
   - `NeverParallel` (interactive/stateful, e.g. plan-mode, ask-user) → forced sequential.
   A call needing escalation (active mode `ask`) emits an `action_required` `AgentEvent`; the client (or WS) supplies a decision (and may "allow for the rest of the session"). A denied call still produces an error `ToolResult` so the model can recover — the turn never aborts.
9. **Permissioned tool execution.** Approved calls run through `registry.execute(name, args)` (builtin, plugin, or MCP by qualified name). Before file mutations or destructive commands a checkpoint snapshot is taken. Handlers never raise; results are `ToolResult` (structured `data`/images preserved, not coerced to a single text string). `PostToolUse` and `transform_tool_result` hooks observe/transform. Results are reassembled **in original call order** regardless of completion order, and each is emitted as a `tool_result` `AgentEvent`.
10. **Append results & loop.** Each `ToolResult` is appended to `session.messages` as a `tool` message (keyed by `tool_use_id`); the session is persisted. The loop returns to step 5 — the next provider call naturally includes the tool results as input. There is no separate tool-result queue. Repeat until a terminator from step 7 fires, then emit `done` with a `TurnResult`.

## Provider abstraction (on top of LiteLLM)

The design goal is that the loop, session, tools, and context manager are **completely provider-agnostic**; all vendor quirks live in `providers/`.

- **One canonical wire model + LiteLLM as the translator.** LiteLLM already presents one OpenAI-shaped API across 100+ providers. Zak Code wraps it in a thin `Provider` so the rest of the app speaks the canonical internal shape and never imports `litellm`. Request `tools` are OpenAI JSON-schema for all providers; LiteLLM translates per-backend.
- **Sync + async, both streaming.** `complete`/`acomplete` map to `litellm.completion`/`acompletion`; `stream`/`astream` map to the same with `stream=True`. The server uses the async path on one shared event loop (no per-request loop creation). `litellm.drop_params=True` prevents hard failures when a local model rejects OpenAI-only params.
- **Streaming + tool-call normalization.** A streaming accumulator builds `ToolCall`s by chunk `index`, parsing the JSON-string `arguments` into a `dict` (lossy fallback `{"_raw": ...}` on parse error). We use the manual index accumulator rather than `litellm.stream_chunk_builder`, which has known gaps with content-then-tool_calls. `stream_options={"include_usage": True}` is always set so streamed turns still report usage/cost.
- **Model selection by string.** The model string is `provider/model` (e.g. `openai/gpt-4o`, the default dev model `ollama_chat/llama3.1`). A new provider is a new prefix, no code change. `api_base` is passed per call; API keys come from LiteLLM's standard env vars, never the config surface.
- **Ollama vs OpenAI (first-class, opinionated).** OpenAI: `openai/<model>` + `OPENAI_API_KEY`, native tool calling. Ollama: prefer `ollama_chat/<model>` (hits `/api/chat`) over `ollama/`; default host `http://localhost:11434`. **For local tool calling, prefer the native `ollama_chat/<model>` path in `auto` mode**, which routes to the **text protocol** (below): this sidesteps the `ollama_chat/` + native-tools bugs (mixed content+tool_calls errors, tool results landing in `content`) *and* gets the `num_ctx` lift + window clamp. The OpenAI-compatible shim (`model="openai/<ollama-model>"` + `api_base="http://localhost:11434/v1"`, with `OLLAMA_API_BASE` also set) remains an option, but on it the model gets **neither** the text protocol **nor** the `num_ctx` lift (both are gated on the `ollama`/`ollama_chat` prefix), so it is not recommended for weak local models. Gate behavior with `litellm.supports_function_calling(model=...)`; for non-tool-native models, fall back to prompt-injected tool schemas (`add_function_to_prompt`) or route to an OpenAI fallback — all behind the `Provider` so callers stay agnostic. In `auto` mode the facade routes Ollama models (`ollama`/`ollama_chat`) to the **text protocol** outright (`zakcode._resolve_tool_calling_mode`): they advertise tool support via `litellm.supports_function_calling`, but their native tool path is unreliable through litellm (empty responses for common local models such as `qwen2.5:3b`), while the text protocol drives tools reliably; `tool_calling_mode=native` overrides this. On the text path the wrapper hardens weak local models **mechanically** (not by prompt-pleading): it passes **stop sequences** (protocol/template sentinels) so the model cannot autoregress into a fabricated `<tool_result>` or leak chat-template tokens; defaults to **one tool call per turn** (`single_tool_per_turn`, stopping at `</tool_call>`); defangs any forged frames left in the residual; and (in the litellm provider) lifts Ollama's `num_ctx` to the model's real window, clamped — with `capabilities().context_window` clamped to the same value so the compactor never lets Ollama truncate the prompt tail before compaction fires.
- **Reliability.** Built-in retry via `num_retries` (exponential backoff on 408/429/5xx, honoring `Retry-After`); a normalized error taxonomy (`AuthError`, `ContextWindowExceeded`, `RateLimited`, `RequestFailed`) mapped from LiteLLM/OpenAI exceptions; optional cross-provider `fallbacks`. `ContextWindowExceeded` is not retried — it signals the loop to compact.
- **Cost/token accounting.** `count_tokens` (LiteLLM `token_counter`), `get_max_tokens` for the context budget, and per-call `cost_usd` from `response._hidden_params["response_cost"]`, accumulated into a session total. Local models report `cost=0` (treated as free, or registered with custom pricing).

## Context management & compaction

Context is a first-class, actively managed resource — quality decays well before the technical limit ("context rot"), so the effective window is treated as smaller than advertised.
- **Cache-stable prompt ordering.** The system prompt is built **stable → context → volatile** with a `DYNAMIC_BOUNDARY`. The stable prefix (identity, safety policy, tool schemas) is cached at the provider level; toolsets, memory, and past context are never mutated mid-conversation, protecting cache hit rate and cost. The **identity** slot is the operator-authored `self.md` when present (`.zakcode/self.md` > `<ws>/self.md` > `~/.config/zakcode/self.md`), which REPLACES the default identity line — this is how a "mind" gives the runtime its persona; absent a `self.md` the prompt is byte-for-byte the default.
- **Just-in-time context.** Prefer lightweight identifiers + on-demand tool calls over pre-loading files. Tool results are summarized per type (bash vs. file read), large outputs are capped with truncation hints, and the live TODO list is re-injected near the end of context to counter instruction fade-out.
- **Real-token-triggered auto-compaction.** `Compactor` fires at ~70–80% of the model's real context window (configurable). It preserves the last N (3–5) turns verbatim and replaces older history with a single leading **LLM-written** system summary (the heuristic structured summary is kept only as an offline fallback). Re-compaction detects and merges any prior summary (idempotent), and the summary carries a "resume directly, do not acknowledge" instruction. Compaction pairs with Git-based checkpoints so state is recoverable.

## Sessions & persistence

Sessions are first-class and resumable. Each session is one JSON document under the home dir, keyed by a UUID, written **atomically** (temp file + rename) after every turn and every mutating slash command. The document is versioned for migration. It stores conversation history (with parsed-JSON tool input and structured tool results), per-message `usage` (so cumulative cost is reconstructable on resume without a side file), cwd, provider/model, and active extensions. `/resume` and `/session` restore a session and its extension state; `UsageTracker.from_session` rebuilds cumulative usage by summing inline per-message usage. Round-trip serialization is reversible and unit-tested. SQLite is a candidate backend later for cross-session FTS search; the JSON-per-session format is the simple, zero-ops default.

## Extension model (tools vs. MCP vs. plugins vs. skills) and hooks

Four distinct mechanisms, each for a different job:
- **Tools** — first-party capabilities the model calls directly. Declarative spec + never-raising handler in `tools/builtins/`. Keep the core set small and sharp (read/write/edit/glob/grep/shell + a few). Prefer composing capability through the shell over hand-built specialty tools.
- **MCP** — the contract for third-party and remote integrations (CLI-less SaaS, OAuth/per-user auth, audit trails). Zak Code is an MCP host supporting **stdio + streamable-HTTP** transports. MCP tools register into the same `ToolRegistry` under qualified names `server__tool` (double underscore, matching MCP's `^[a-zA-Z0-9_-]{1,64}$` rule — never `:` or `.`), with O(1) routing by splitting the prefix. Discover tools **lazily** (a `tool_search` surface; load schemas on demand) and gate activation behind approval so schemas never bloat the base prompt. Keep the exposed tool budget small (~25). Default to CLI/shell for local/known tools; add MCP only where auth or a missing CLI requires it.
- **Plugins** — first-party Python/subprocess extensions via a single `register(ctx)` entrypoint; they contribute tools, hooks, and commands without touching core files.
- **Skills** — progressive-disclosure markdown (`SKILL.md`); cheap name/description always present, body lazy-loaded; invokable as `/<skill-name>`.

**How hooks fire.** A unified `HookManager` dispatches lifecycle events and isolates errors (one bad hook never breaks the loop). In a turn: `pre_llm_call` runs before the provider call and may inject ephemeral context into the user message only; `PreToolUse` runs before each tool (can veto via exit code 2 / `{"action":"block"}`, a denial becoming an error `ToolResult`); `PostToolUse` and `transform_tool_result` observe/transform results; `on_session_start`/`on_session_end` bracket the session. Only `pre_*` returns affect behavior; all callbacks must accept `**kwargs` for forward compatibility.

## Server API surface

`zakcode-server` is FastAPI over the core, exposing one shared async event loop.

- `POST /chat` — run one turn (or full task) and return the buffered result. Body `ChatRequest { session_id?, message, model?, tools? }` → `ChatResponse { session_id, text, tool_results, usage, cost_usd }`.
- `POST /chat/stream` — same input, returns **Server-Sent Events**: each frame is a serialized `AgentEvent` (`text` deltas, `tool_call`, `tool_result`, `action_required`, `usage`, `done`). For one-directional streaming clients.
- `POST /complete` — a **raw schema-valid completion**: no tools, no agent loop, no session, no permission gate. Body `CompleteRequest { prompt? | messages?, system?, schema?, model? }` → `CompleteResponse { data, text, usage, cost_usd, repaired }`. For non-agent callers that need bounded structured output (e.g. a semantic extractor): a `schema` is enforced via the provider's structured output + a bounded repair retry, returning a `{error, detail, raw_text}` 502 if the model cannot satisfy it. Builds a raw `LiteLLMProvider` directly (the text-tool wrapper is bypassed; `temperature=0` on the schema path).
- `WS /ws/{session_id}` — bidirectional channel: client sends user input and approval decisions; server streams `AgentEvent`s including `action_required` permission prompts (the interactive permission bridge for browser/IDE clients).
- `GET /sessions` / `GET /sessions/{id}` / `POST /sessions` / `DELETE /sessions/{id}` — list, fetch (history + usage), create, delete sessions.
- `GET /tools` — list registered tool definitions (builtins + plugins + active MCP) with schemas and required permission tiers.
- `GET /config` / `PATCH /config` — read and update non-secret settings.
- `GET /health` — liveness.

**Auth & multi-tenant hardening.** The server is unauthenticated by default and `zakcode serve` binds `127.0.0.1`, so the loopback-dev posture is unchanged. For hosted/multi-tenant use (e.g. one container per customer env behind a router), set `ZAKCODE_AUTH_TOKEN`: every route except `GET /health` then requires `Authorization: Bearer <token>` (constant-time compared), enforced by an HTTP middleware that is **only registered when a token is configured**. The WebSocket authenticates the handshake before `accept()`, taking the token from the `Authorization` header or, for browsers (which cannot set handshake headers), the `Sec-WebSocket-Protocol: bearer, <token>` subprotocol — deliberately **not** a `?token=` query param, which would be persisted in uvicorn's access log. `serve` **refuses a non-loopback `--host`** when no token is set unless `--insecure` is passed. An optional `ZAKCODE_ALLOWED_MODELS` allowlist rejects (400) any per-request `model` override not on the list. The token is `exclude=True`, so it never appears in `GET /config`.

All surfaces (CLI in-process, SSE, WS) emit the **identical `AgentEvent` stream**, so a future web/IDE client is a thin renderer, never a fork of agent logic.

## Config & permission model

- **Layered config.** pydantic-settings resolves overrides → `ZAKCODE_` env vars → `.env` → defaults, validated eagerly, with JSON scope merging (user → project → local) layered on later. Settings and secrets are separated (provider keys live in LiteLLM's env vars, not the config); all paths flow through the home-dir helper.
- **Permission modes.** The configured `permission_mode` is one of `ask | acceptEdits | allow | deny`, mapped onto ordered tiers `ReadOnly < WorkspaceWrite < DangerFullAccess`, plus the session mode `Plan` (read-only, write tools absent from the schema). Each tool declares a required tier in its spec (`read=ReadOnly`, `write`/`edit=WorkspaceWrite`, `bash`/`powershell`=DangerFullAccess). Authorization: allow if the active mode satisfies the required tier; otherwise prompt at the single sensible boundary (workspace-write → danger) or deny.
- **Deny-first, enforced outside the model.** Default mode is gated (not full-access). Unknown tools default to the **strongest** required tier (deny by default — fail-closed). Enforcement runs on a code path the model cannot reach, so a prompt-injected model still cannot override a deny rule. Rules are **input-pattern aware** (e.g. allow `bash` only for `git status`), backed by a `DANGEROUS_PATTERNS` blocklist (`rm -rf /`, `sudo`, DB drops). A denied tool returns an error `ToolResult` so the model can recover. Approval decisions persist per session ("allow for the rest of the conversation") to fight approval fatigue. Sandboxing (filesystem + network egress allowlist via a proxy) is a planned mode; it is not advertised until implemented.

## Design principles

1. **One core, many thin surfaces.** Every UI runs the same `AgentLoop` and consumes the same typed `AgentEvent` stream — never re-implement agent logic per client.
2. **Provider-agnostic by construction.** The loop never imports a vendor SDK; everything goes through `Provider` over LiteLLM, normalized to one canonical shape. Ollama and OpenAI are first-class.
3. **Context is a managed resource.** Cache-stable prompt ordering, real-token-triggered auto-compaction, just-in-time retrieval, and per-type result summarization — more tokens make agents worse, not better.
4. **Few sharp tools, CLI-first.** A small core toolset, lean on the shell and mature CLIs, MCP only for the 20% (auth/remote/CLI-less). Keep the exposed tool budget small.
5. **Deny-first safety enforced outside the model.** Permission checks, dangerous-pattern blocklists, and hooks live in the harness on a path the model can't reach; denials are recoverable, not fatal.
6. **Parallel reads, serialized conflicting writes.** Read-only tools run concurrently (capped); writes serialize on path-prefix overlap; interactive tools stay sequential.
7. **Layered stop conditions.** Natural completion (no tool calls) plus iteration cap, cost/token budget, doom-loop detection, and cooperative cancellation — never an unbounded loop.
8. **Structured everything, lossless.** Tool input/output are `dict`/`ToolResult`, not re-parsed strings; thinking blocks persist; tool results keep structured data and images.
9. **Pure, injectable core.** `Provider`, tool executor, and permission policy are injected — the loop is a pure, trivially testable function. Construct eagerly; fail fast on bad config/credentials.
10. **Crash-safe, resumable state.** Atomic session writes after every turn and mutating command; inline per-message usage; versioned documents.
11. **Extend without forking.** Tools, MCP, plugins (`register(ctx)`), and skills (markdown) extend the agent without modifying core files; hooks are error-isolated and only `pre_*` can change behavior.
12. **Implementable now, not aspirational.** Prefer the strongest proven patterns; defer the surface sprawl (multi-agent teams, remote/bridge, cron, autonomous skill curation) until the core loop is solid.

## claw-code (Rust) architecture digest

Reference base: `C:\ZakNoCloud\_zakcode_research\claw-code\claw-code-main\rust\crates`. This is a clean-room reverse engineering of Claude Code. It is split into crates: `api` (provider/transport), `runtime` (agent loop, session, config, prompt, MCP, permissions, hooks), `tools` (tool registry + execution), `commands` (slash commands), `plugins`, and `rusty-claude-cli` (the wiring/CLI/TUI). The cleanest architectural idea to steal: a **provider-agnostic, sync `ApiClient` trait at the runtime boundary** plus a **provider-specific async layer in `api`** that normalizes everything to Anthropic-shaped streaming events. Below, each section gives concrete decisions to adopt and pitfalls to avoid.

---

### 1. Agent loop control flow

The loop lives in `runtime/src/conversation.rs:ConversationRuntime::run_turn` (lines 153-263). One turn:

1. Push the user message onto `session.messages` as `ConversationMessage::user_text` (`conversation.rs:158`).
2. `loop { iterations += 1; if iterations > max_iterations { Err } ... }` (`conversation.rs:166-172`).
3. Build `ApiRequest { system_prompt, messages: session.messages.clone() }` and call `self.api_client.stream(request)?` — a **synchronous trait call returning `Vec<AssistantEvent>`** (`conversation.rs:174-178`). The whole stream is drained into a vector before the loop continues; the loop itself is not async.
4. `build_assistant_message(events)` (`conversation.rs:291-328`) folds events into one assistant `ConversationMessage`: concatenates `TextDelta` into a single `Text` block, emits `ToolUse` blocks, captures `Usage`, and requires a terminal `MessageStop` (errors "stream ended without a message stop event" otherwise; also errors on empty content).
5. Record usage into `UsageTracker` (`conversation.rs:180`), push assistant message to session + summary.
6. **Stop condition**: collect `ToolUse` blocks; if none, `break` (`conversation.rs:197-199`). This is the only natural terminator — the assistant produced text with no tool calls.
7. For each pending tool use (sequential `for` loop, `conversation.rs:201`): run permission check → PreToolUse hook → execute → PostToolUse hook → append a `tool_result` `ConversationMessage` (role `Tool`) to session and summary. Then the loop repeats (next API call now sees the tool results).
8. Return `TurnSummary { assistant_messages, tool_results, iterations, usage }`.

**Streaming event handling** is two-layered. The runtime sees only the flat enum `AssistantEvent { TextDelta, ToolUse { id, name, input }, Usage, MessageStop }` (`conversation.rs:19-29`). The translation from raw SSE to that enum lives in the CLI bridge `AnthropicRuntimeClient::stream` (`rusty-claude-cli/src/main.rs:2902-3045`), which `block_on`s the async stream, accumulates `InputJsonDelta` partial-JSON fragments into `pending_tool.input`, flushes a `ToolUse` event on `ContentBlockStop`, and renders text deltas live via `MarkdownStreamState`.

**Max iterations**: default is `usize::MAX` (`conversation.rs:141`) — effectively unbounded; sub-agents override to 32 (`tools/src/lib.rs:1503 DEFAULT_AGENT_MAX_ITERATIONS`). PARITY.md line 209 flags "unlimited max_iterations" as a deliberate decision.

ADOPT:
- The **flat normalized event enum** (`AssistantEvent`) as the runtime/transport contract — it decouples the loop from SSE wire details and from which provider produced them.
- **Accumulate the full assistant message before deciding** whether to continue (parse-then-decide). Clean and testable; the loop has zero provider knowledge.
- **Tool results re-enter as the next turn's input** simply by being appended to `session.messages` — no separate "tool result queue"; the next `stream()` call naturally includes them.
- Stop condition = "assistant emitted no tool calls" is the single correct terminator.
- The scripted-`ApiClient` unit tests (`conversation.rs:413-522`) show the loop is trivially testable because `ApiClient`/`ToolExecutor` are traits — replicate this seam in Python (inject the model client and tool executor).

AVOID:
- **`usize::MAX` iterations** — in Python set a real cap (e.g. 50-100) with a clear terminal error/summary; an infinite tool loop will burn tokens and hang. The Rust default is a known foot-gun.
- **Tools run strictly sequentially** in the `for` loop (`conversation.rs:201`); there is **no parallel tool execution** despite the model emitting multiple `tool_use` blocks. The real Claude Code parallelizes read-only tools. Plan for an async gather of independent tool calls in Python.
- **No thinking blocks survive into history** — `ThinkingDelta`/`SignatureDelta` are parsed in `api/types.rs` and the CLI bridge but dropped (`main.rs:2978-2979`); the session `ContentBlock` has no thinking/redacted variant (`session.rs:17-33`). If you want extended thinking with tool use, you must persist `thinking`+`signature` blocks and replay them, which this port cannot do.
- **The whole stream is buffered into a `Vec` before the loop proceeds** — fine for correctness, but you lose true incremental processing at the loop level. Streaming UX is bolted on inside the bridge only.
- `build_assistant_message` errors if there are zero content blocks; an empty assistant turn will crash the loop rather than gracefully ending.

---

### 2. Tool registry & execution model

Two registries exist. `tools/src/lib.rs:mvp_tool_specs()` (lines 216-537) is a `Vec<ToolSpec { name, description, input_schema: serde_json::Value, required_permission: PermissionMode }>` — a flat, hand-written list (bash, read_file, write_file, edit_file, glob_search, grep_search, WebFetch, WebSearch, TodoWrite, Skill, Agent, ToolSearch, NotebookEdit, Sleep, SendUserMessage, Config, StructuredOutput, REPL, PowerShell). `GlobalToolRegistry` (`lib.rs:59-199`) wraps the builtins plus `Vec<PluginTool>` and provides:
- `definitions(allowed_tools)` → `Vec<ToolDefinition>` sent to the API (`lib.rs:141`).
- `permission_specs(allowed_tools)` → `Vec<(name, PermissionMode)>` used to build the policy (`lib.rs:165`).
- `execute(name, &Value)` → dispatches builtins via `execute_tool` (a big `match name` at `lib.rs:539-564`) or finds a plugin tool by name (`lib.rs:188-198`).
- `normalize_allowed_tools` (`lib.rs:94`) with aliasing (`read`→`read_file`, `glob`→`glob_search`, etc.) so `--allowedTools` accepts friendly names.

Dispatch + execution at runtime is via the `ToolExecutor` trait (`conversation.rs:35-37`); the live impl `CliToolExecutor` (`main.rs:3606-3662`) re-checks `allowed_tools`, parses the input string to JSON, calls `tool_registry.execute`, and renders the result. Tool input flows through the system as a **raw JSON string** (`ContentBlock::ToolUse.input: String`), parsed lazily at each boundary.

Permission check happens **per tool, before execution** in the loop (`conversation.rs:202-251`): `PermissionPolicy::authorize(tool_name, input, prompter)`. Each builtin declares a `required_permission`; the policy is built by folding `permission_specs` into `PermissionPolicy::with_tool_requirement` (`main.rs:3664-3671`).

ADOPT:
- **Tools as data**: a declarative spec (name, description, JSON schema, required permission) separate from the execute function. Easy to register, list, and gate.
- **One canonical executor seam** (`ToolExecutor.execute(name, input) -> Result<String, ToolError>`) so the loop never knows tool internals.
- **Per-tool permission tier baked into the spec** (read_file=ReadOnly, write/edit=WorkspaceWrite, bash/REPL/PowerShell/Agent=DangerFullAccess) — see `lib.rs:233,248,262,278,…`. This is a clean, auditable model.
- **Name aliasing** for `--allowedTools` ergonomics.
- Plugin tools execute as **subprocesses fed JSON via stdin + env vars** (`plugins/src/lib.rs:297-339`) — a simple, language-agnostic extension mechanism worth keeping for an MVP.

AVOID:
- **No parallel tool dispatch** (same gap as the loop).
- **Tool input as `String` not structured JSON** in the session model (`session.rs:24-32`) forces repeated parse/serialize round-trips and lossy fallbacks (`convert_messages` at `main.rs:3689` does `serde_json::from_str(input).unwrap_or_else(|_| json!({"raw": input}))`). In Python keep tool input as a `dict`.
- **Default required mode is `DangerFullAccess`** when a tool is unknown to the policy (`permissions.rs:81-86 required_mode_for`) — fail-open. A safer default is to require the highest privilege (deny) for unknown tools.
- The `Agent` tool spawns an **OS thread running a whole new `ConversationRuntime`** (`tools/src/lib.rs:1586-1620`) and writes manifest/output files; there's no real isolation, cancellation, or result piping back into the parent conversation. Treat sub-agents as a first-class async task with structured handoff in Python, not a detached thread.
- Tool results are always coerced to a single text block on the way back to the API (`main.rs:3699-3701`); structured/JSON tool results and images are lost.

---

### 3. Provider / API abstraction

The `api` crate cleanly separates providers behind a `Provider` trait (`api/src/providers/mod.rs:12-24`) with `send_message` and `stream_message` returning boxed futures. Concrete impls: `providers/anthropic.rs:AnthropicClient` and `providers/openai_compat.rs:OpenAiCompatClient` (used for both xAI and OpenAI via `OpenAiCompatConfig::xai()/openai()`). `ProviderClient` (`client.rs:21-107`) is the runtime-facing enum that dispatches; `MessageStream` (`client.rs:86-107`) is an enum unifying the two stream types behind `next_event() -> Option<StreamEvent>`.

The key normalization decision: **everything is expressed in Anthropic's shape**. `MessageRequest`/`InputMessage`/`InputContentBlock`/`ToolDefinition`/`StreamEvent` (`api/src/types.rs`) are Anthropic-modeled. The OpenAI-compat provider **translates inbound** (`openai_compat.rs:build_chat_completion_request`, `translate_message`, `openai_tool_definition`, `openai_tool_choice`, lines 634-748) and **synthesizes Anthropic-style `StreamEvent`s outbound** from OpenAI chat-completion chunks via a `StreamState` machine (`openai_compat.rs:299-469`): it fabricates `MessageStart`, opens a text block on first content, maps OpenAI `tool_calls[].function.arguments` deltas to `InputJsonDelta`, and emits a synthetic `MessageDelta`+`MessageStop` at finish, normalizing finish reasons (`stop`→`end_turn`, `tool_calls`→`tool_use`, line 913).

Provider selection: `ProviderClient::from_model` → `resolve_model_alias` + `detect_provider_kind` (`providers/mod.rs:117-179`). Detection prefers the model-name family (claude*→Anthropic, grok*→Xai) and falls back to which credential env var is present. `max_tokens_for_model` (`mod.rs:182-189`) hardcodes 32k for opus, 64k otherwise.

SSE parsing: `api/src/sse.rs:SseParser` buffers bytes, splits frames on `\n\n` or `\r\n\r\n` (`sse.rs:40-60`), strips `event:`/`data:` lines, ignores `ping` and `[DONE]`, joins multi-line `data:` and `serde_json`-parses into `StreamEvent`. The Anthropic stream reader (`anthropic.rs:570-601`) pulls `response.chunk()`, feeds the parser, and queues events.

Auth: `AnthropicClient` supports `x-api-key`, bearer (OAuth), or both (`anthropic.rs:AuthSource`, lines 24-91). `resolve_startup_auth_source` (`anthropic.rs:399-437`) resolves env → saved OAuth token → refresh-if-expired. Retries: exponential backoff (`send_with_retry`, `anthropic.rs:278-312`), default 2 retries, retryable on 408/409/429/5xx and connect/timeout errors (`is_retryable_status`, `is_retryable`).

How a single loop targets either: the runtime only knows the `ApiClient` trait; the bridge picks the provider client and converts `AssistantEvent`s identically regardless of source, because the OpenAI path already emits Anthropic-shaped `StreamEvent`s.

ADOPT:
- **Normalize on one canonical (Anthropic) wire model and adapt other providers into it** — both inbound request translation and outbound event synthesis. This keeps the loop, session, and tools provider-agnostic. This is the single most valuable pattern for the Python port.
- **The streaming state machine for OpenAI-compat** (`StreamState`/`ToolCallState` in `openai_compat.rs:299-531`): map OpenAI tool-call argument deltas to incremental `partial_json`, track per-index tool-call blocks, synthesize start/stop events. Reproduce this if you support OpenAI-style providers.
- **Incremental SSE parser** that buffers partial frames and tolerates JSON split across `data:` lines (`sse.rs` + its tests at lines 199-218).
- **Retry policy** with bounded exponential backoff and an explicit retryable-status set.
- **Tri-state auth** (api-key, bearer/OAuth, both) and startup auth resolution with auto-refresh.
- `request_id` capture from `request-id`/`x-request-id` headers (`anthropic.rs:535-541`) for debugging.

AVOID:
- **`tokio::runtime::Runtime::new()` constructed per `AnthropicRuntimeClient` and `block_on` per turn** (`main.rs:2859,2879,2920`), plus a *separate* `Runtime::new()` inside `resolve_saved_oauth_token_set` (`anthropic.rs:482`). Creating runtimes ad hoc is wasteful and a known anti-pattern. In Python, run one event loop (`asyncio`) for the whole session; don't spin up a loop per request.
- **OpenAI-compat drops cache token accounting** (always 0 for cache_creation/read, `openai_compat.rs:350-355,788-794`) and the Anthropic streaming bridge also zeroes cache tokens in `MessageDelta` (`main.rs:3002-3003`) even though the type carries them — cost reporting is wrong for cached requests. Wire cache tokens through from `message_start`/`message_delta`.
- **`max_tokens_for_model` is duplicated and divergent**: `api/providers/mod.rs:182` vs a second local copy in `main.rs:44-50`. Single-source this in Python.
- **No prompt caching** is sent (no `cache_control` on system/tools/messages). Real Claude Code caches the system prompt and tool defs; add `cache_control` to cut cost — this port leaves big savings on the table.
- **`tool_choice` is hardcoded to `Auto`** whenever tools are enabled (`main.rs:2916`); there's no forced-tool or no-tool control surfaced to the loop.
- System prompt is flattened to a single string with `"\n\n".join` (`main.rs:2912`) rather than an array of cacheable blocks.

---

### 4. Context window & compaction strategy

`runtime/src/compact.rs`. `CompactionConfig { preserve_recent_messages: 4, max_estimated_tokens: 10_000 }` defaults (`compact.rs:14-21`). Token estimation is **`len/4 + 1` per block** (`estimate_message_tokens`, lines 391-403) — a crude heuristic, no tokenizer. `should_compact` (lines 37-47) ignores any existing summary prefix, then compacts only if remaining messages exceed `preserve_recent_messages` AND estimated tokens ≥ threshold.

`compact_session` (lines 89-131): keeps the last `preserve_recent_messages` verbatim, summarizes everything older into a **synthetic `MessageRole::System` message** placed at index 0, containing a structured, locally-generated summary (not an LLM summary). The summary (`summarize_messages`, lines 143-228) counts roles, lists tool names used, recent user requests, inferred "pending work" (keyword scan for todo/next/pending/follow up/remaining), key files referenced (path-with-extension scan), current work, and a truncated per-message timeline. There's a `<summary>`/`<analysis>` tag convention (`format_compact_summary`, lines 50-62) and a continuation preamble + "resume directly" instruction (lines 3-6, 65-86). Re-compaction merges the prior summary's highlights with new ones (`merge_compact_summaries`, lines 230-263) so context accretes across compactions. It's invoked from `/compact` (`commands/src/lib.rs:1164`) and `ConversationRuntime::compact`.

ADOPT:
- **Preserve N recent messages verbatim + collapse the rest into a single leading system summary** — the standard, sound compaction shape.
- **Idempotent re-compaction that detects and merges a prior summary** (`extract_existing_compacted_summary`, `merge_compact_summaries`) so you don't re-summarize the summary or lose earlier context.
- The **continuation message with an explicit "resume directly, do not acknowledge the summary" instruction** (`COMPACT_DIRECT_RESUME_INSTRUCTION`, line 6) — prevents the model from narrating the compaction.
- Structured summary sections (scope, tools, recent requests, pending work, key files, timeline) give a deterministic, testable summary even without an LLM call.

AVOID:
- **`len/4` token estimation** is inaccurate; use a real tokenizer (`tiktoken`-equivalent / the Anthropic token-count endpoint) for the trigger, especially near the true context limit.
- **The summary is generated locally by heuristics, not by the model** — keyword scans for "pending work" and naive file detection are brittle and lossy. Real Claude Code asks the model to write the summary. In Python, do an LLM summarization pass for fidelity; keep the heuristic version only as an offline fallback.
- **Compaction is manual-only** (`/compact`); `should_compact` exists but the loop never auto-triggers it. Wire auto-compaction into the turn loop when estimated tokens approach the window.
- Summary truncation at 160 chars/block (`truncate_summary`) can discard the load-bearing part of long tool results.

---

### 5. System-prompt construction & CLAUDE.md / memory discovery

`runtime/src/prompt.rs`. `SystemPromptBuilder::build()` (lines 134-156) assembles an ordered `Vec<String>` of sections: intro → optional output-style → system rules → "doing tasks" rules → "executing actions with care" → a **`SYSTEM_PROMPT_DYNAMIC_BOUNDARY` marker** → environment context → project context → instruction files (CLAUDE.md content) → runtime config section → appended sections. The static prose is hand-written (lines 441-490) and notably includes guidance about permission modes, `<system-reminder>` tags, prompt-injection flagging, hooks behaving like user feedback, and auto-compaction — mirroring real Claude Code's system prompt themes.

Memory discovery: `discover_instruction_files` (lines 192-225) walks **from filesystem root down to cwd** (ancestors reversed, so root-most first) and at each directory checks `CLAUDE.md`, `CLAUDE.local.md`, `.claw/CLAUDE.md`, `.claw/instructions.md`. Files are deduped by **normalized-content hash** (`dedupe_instruction_files`, lines 326-341) so identical rules across scopes appear once. Rendering (`render_instruction_files`, lines 303-324) enforces a **per-file budget of 4,000 chars and total budget of 12,000 chars** (lines 39-40), truncating with `[truncated]`/budget-exhausted markers, and labels each file with a scope. Git context (`read_git_status`, `read_git_diff`, lines 227-275) is captured via `git --no-optional-locks status --short --branch` and staged/unstaged diffs and injected as a "Project context" section. `load_system_prompt` (lines 404-418) ties cwd discovery + config load + OS into the builder.

ADOPT:
- **Ordered sections joined with a dynamic boundary marker** — the marker (`SYSTEM_PROMPT_DYNAMIC_BOUNDARY`) cleanly separates the stable, cacheable prefix from per-session dynamic content; this is exactly where you'd split for prompt caching.
- **Ancestor-chain CLAUDE.md discovery (root→cwd) with content-hash dedup** and **per-file + total char budgets** — prevents memory files from blowing the context and from duplicating across nested scopes.
- **Injecting git status + diff snapshot** as project context.
- The system-prompt themes (permission-mode awareness, `<system-reminder>` semantics, prompt-injection flagging, blast-radius/reversibility guidance) are good content to port verbatim-in-spirit.

AVOID:
- **The static prose is `FRONTIER_MODEL_NAME = "Claude Opus 4.6"` hardcoded** (`prompt.rs:38`) and dumped into the prompt (`environment_section`, line 175); keep model identity data-driven.
- **The whole merged config JSON is rendered into the system prompt** (`render_config_section`, lines 420-439, called at `build()` line 152) — this leaks settings (and any secrets in `env`) into the model context every turn and wastes tokens. Do not put raw config JSON in the system prompt.
- Discovery only looks for `CLAUDE.md`/`.claw/*` under the cwd ancestry; there's **no user-global memory dir** merged here (the `/memory` command and skills use different roots — `~/.claude`, `~/.codex`, `$CODEX_HOME` in `commands/src/lib.rs:686-802`), so memory paths are inconsistent across subsystems. Unify your memory roots.
- Truncation is by raw char count, not token-aware.

---

### 6. Session persistence format

`runtime/src/session.rs`. `Session { version: u32, messages: Vec<ConversationMessage> }`; each `ConversationMessage { role: System|User|Assistant|Tool, blocks: Vec<ContentBlock>, usage: Option<TokenUsage> }`; `ContentBlock` is `Text{text}` | `ToolUse{id,name,input:String}` | `ToolResult{tool_use_id,tool_name,output,is_error}` (lines 9-46). Serialized via a **hand-rolled JSON** layer (`json.rs`, used through `to_json`/`from_json`, lines 99-325) to a `{version, messages:[{role, blocks:[{type,…}], usage?}]}` document; saved with `Session::save_to_path` (plain `fs::write`, no atomic rename) and loaded with `load_from_path`.

The CLI manages sessions on disk under `.claw/sessions/<session-id>.json` where id is `session-<unix_millis>` (`main.rs:1752-1771`); it lists/sorts by mtime (`list_managed_sessions`, 1791-1822), supports `/resume`, `/session list|switch`, and **persists after every turn and every mutating slash command** (`persist_session`, `main.rs:1252-1255`). `UsageTracker::from_session` (`usage.rs:176-184`) reconstructs cumulative usage by summing per-message `usage` on load — so usage survives resume.

ADOPT:
- **Per-message `usage` stored inline** so cumulative cost/usage is reconstructable on resume without a side file.
- **Versioned session document** (`version` field) for forward migration.
- **Persist after every turn and every mutating command** for crash safety; one file per session keyed by a sortable id.
- Round-trip equality is unit-tested (`session.rs:386-431`) — keep persistence reversible and tested.

AVOID:
- **Hand-rolled JSON (`json.rs`)** — needless; use the stdlib `json` in Python.
- **`save_to_path` is a plain `fs::write`, not atomic** (contrast `oauth.rs:write_credentials_root` which *does* write-temp-then-rename, lines 357-366). Use temp-file + atomic rename for sessions to avoid corruption on crash.
- **Tool input stored as a `String`**, not parsed JSON (lossy, see §2).
- **No thinking/redacted-thinking block type** — can't persist extended-thinking turns (see §1).
- Session id is millisecond-timestamp only — collision-prone for rapid programmatic creation; use a UUID.

---

### 7. Config & permission model

Config: `runtime/src/config.rs`. `ConfigLoader::discover` (lines 197-224) defines precedence (later overrides earlier): user `~/.claw.json` (legacy) → user `~/.claw/settings.json` → project `.claw.json` → project `.claw/settings.json` → local `.claw/settings.local.json`. `load()` deep-merges JSON objects (`deep_merge_objects`, lines 911-925), merges MCP servers by name with scope tracking, and parses typed `RuntimeFeatureConfig` (hooks, plugins, mcp, oauth, model, permission_mode, sandbox). `default_config_home()` honors `CLAW_CONFIG_HOME` then `$HOME/.claw` (lines 421-427). Permission mode is parsed from either `permissionMode` or `permissions.defaultMode` with synonym mapping (`parse_permission_mode_label`, lines 625-637): `default|plan|read-only`→ReadOnly, `acceptEdits|auto|workspace-write`→WorkspaceWrite, `dontAsk|danger-full-access`→DangerFullAccess.

Permissions: `runtime/src/permissions.rs`. `PermissionMode` is an **ordered enum** `ReadOnly < WorkspaceWrite < DangerFullAccess < Prompt < Allow` (derives `PartialOrd`, lines 3-10). `PermissionPolicy { active_mode, tool_requirements: BTreeMap<name,mode> }`. `authorize` (lines 89-134): allow if `active==Allow` or `active >= required`; otherwise, if `active==Prompt` or (`active==WorkspaceWrite` and `required==DangerFullAccess`), call the `PermissionPrompter` (interactive y/N in `CliPermissionPrompter`, `main.rs:2822-2856`); else deny with a reason. The denied tool still produces a `ToolResult{is_error:true}` so the loop continues and the model sees the denial (`conversation.rs:248-250`).

Defaults: CLI default permission mode is **`DangerFullAccess`** (`args.rs:15`, `main.rs:378-384 default_permission_mode`, env override `RUSTY_CLAUDE_PERMISSION_MODE`), flagged in PARITY.md lines 203-206 as a deliberate choice. There's also a parsed-but-not-enforced `SandboxConfig` (`config.rs:639-662`, `sandbox.rs`).

ADOPT:
- **Layered config with explicit precedence + deep-merge** (user → project → local) and a typed feature-config extracted from merged JSON.
- **Ordered permission enum with `>=` comparison** for "does the active mode satisfy the tool's required mode" — concise and correct.
- **Per-tool required mode + escalation-prompt only for the one sensible boundary** (workspace-write → danger), everything else auto-allow-or-deny.
- **Denied tools return an error tool-result rather than aborting the turn** — the model can recover/explain. Excellent pattern.
- Synonym mapping for permission-mode labels (accept the names users/other tools write).

AVOID:
- **Default `DangerFullAccess`** — for a real product default to a gated mode and prompt on escalation. The repo set this for convenience; don't inherit it.
- **Unknown-tool required mode defaults to `DangerFullAccess`** = fail-open (`permissions.rs:81-86`); make unknown tools require the strongest privilege (i.e., deny by default).
- **Permission rules are per-tool *mode tiers only*** — there is **no path/command-pattern allowlisting** (e.g., "allow `bash` only for `git status`"). Real Claude Code matches on tool input patterns. Add input-aware rules.
- `SandboxConfig` is parsed but never enforced (`config.rs` only reads it; PARITY-style gap) — don't advertise sandboxing you don't implement.
- Hooks/plugins config is merged but the runtime only conditionally runs hooks (see §8 note) — keep config and enforcement in lockstep.

---

### 8. MCP integration

Config → bootstrap → client → manager, all in `runtime/src/`:
- **Config** (`config.rs:85-134`, `parse_mcp_server_config` lines 700-735) supports transports `stdio`, `sse`, `http`, `ws`, `sdk`, `claudeai-proxy` (ManagedProxy), each typed; merged by server name with scope.
- **Bootstrap/identity** (`mcp.rs`, `mcp_client.rs`): `normalize_name_for_mcp` and `mcp_tool_name` produce qualified tool names `mcp__<server>__<tool>` (`mcp.rs:26-37`); `mcp_server_signature`/`scoped_mcp_config_hash` give stable identity/dedup keys (FNV-style hash); CCR-proxy URL unwrapping (`unwrap_ccr_proxy_url`, lines 40-62). `McpClientBootstrap::from_scoped_config` (`mcp_client.rs:57-67`) precomputes normalized name, tool prefix, signature, and a typed `McpClientTransport`.
- **stdio transport + JSON-RPC** (`mcp_stdio.rs`): spawns the server (`McpStdioProcess::spawn`, lines 581-605) with `stdin/stdout` piped, `stderr` inherited, env applied. Framing is **LSP-style `Content-Length` headers** (`read_frame`/`write_frame`, lines 640-675; `encode_frame` line 787). Typed JSON-RPC request/response structs (`JsonRpcRequest`/`JsonRpcResponse`, `McpInitializeParams`, `McpListToolsResult`, `McpToolCallResult`, etc.) and helpers `initialize`/`list_tools`/`call_tool`/`list_resources`/`read_resource` (lines 711-749). Protocol version `2025-03-26` (line 796).
- **Manager** (`mcp_stdio.rs:311-571`): `McpServerManager::from_runtime_config` builds managed servers **only for stdio** (other transports recorded as `unsupported_servers`, lines 334-343). `discover_tools` lazily spawns + initializes each server once (`ensure_server_ready`, lines 507-570), paginates `tools/list` via cursor, and builds a `tool_index: name → ToolRoute{server,raw_name}`. `call_tool(qualified_name, args)` routes to the right server, reusing the already-spawned process; `shutdown` kills children idempotently. Monotonic `next_request_id`.

ADOPT:
- **Qualified tool names `mcp__<server>__<tool>`** with a central routing index (`tool_index`) so MCP tools slot into the same flat tool registry as builtins.
- **Lazy spawn + initialize-once + reuse the process** across discovery and calls (`ensure_server_ready`); verified by a test asserting exactly one `initialize` (`mcp_stdio.rs:1618-1661`).
- **Typed JSON-RPC** structs with `Content-Length` framing for stdio; pagination via `next_cursor`.
- **Stable config signature/hash** for change detection and dedup of identically-configured servers.
- **Record unsupported transports as data rather than failing** the whole manager (graceful degradation).
- Per-server env injection and `stderr` inherited for debuggability.

AVOID:
- **Only stdio is actually wired** — sse/http/ws/sdk/managed-proxy are parsed and bootstrapped into transport enums but the manager refuses them (`mcp_stdio.rs:334-343`). If you need remote MCP, implement those transports.
- **MCP tools are not connected into the live tool registry/loop** in this port — `McpServerManager` exists and is tested, but `GlobalToolRegistry`/`CliToolExecutor` only know builtins + plugin tools; there's no code path feeding discovered MCP tool defs into `definitions()` or routing `execute` to `call_tool`. Close this gap in Python: merge MCP tool defs into the model's tool list and dispatch by qualified name.
- **`read_available` (raw 4096-byte read) coexists with the framed reader** (`mcp_stdio.rs:633-638`) — only used by tests; don't mix raw and framed reads on the same stream in production.
- Manager is **single-threaded sequential** over servers in `discover_tools`; parallelize startup for many servers.

---

### Cross-cutting notes for the Python port

- **Hooks**: `runtime/src/hooks.rs` *does* execute PreToolUse/PostToolUse shell hooks (exit 0=allow, 2=deny, other=warn; stdin gets a JSON payload, env vars carry tool name/input/output), wired in `conversation.rs:211-238`. (PARITY.md lines 65-70 calls hooks "config-only", but the worktree's `conversation.rs` clearly runs them — trust the code: hooks run, denial yields an error tool-result, feedback is appended to output.) ADOPT the exit-code protocol and JSON-over-stdin payload. AVOID running hooks via a shell with the raw command string (`hooks.rs:239-255` uses `sh -lc`/`cmd /C`) — injection risk; prefer argv arrays.
- **Plugins** (`plugins/src/lib.rs`): builtin/bundled/external kinds, manifest at `.claude-plugin/plugin.json`, install/enable/disable/update with a registry file, subprocess tools, and aggregated hooks/tools. Solid extension model; the subprocess-tool contract (JSON via stdin + `CLAWD_*` env, `lib.rs:297-339`) is the part worth copying.
- **TUI/streaming render** (`rusty-claude-cli/src/render.rs`): a streaming-markdown renderer (`MarkdownStreamState` only flushes on safe boundaries — blank lines / closed code fences, `render.rs:638-666`) plus syntect highlighting and a spinner. The **stream-safe-boundary flushing** idea is worth adopting so you don't render half a code fence.
- **Biggest structural wins to carry over**: (1) the `ApiClient`/`ToolExecutor`/`PermissionPrompter` trait seams that make the loop pure and testable; (2) Anthropic-shaped normalization with per-provider adapters; (3) ordered system prompt with a cache boundary; (4) ancestor CLAUDE.md discovery with dedup+budget; (5) the permission tier model with recoverable denials.
- **Biggest things to fix vs. this port**: real token counting + auto-compaction + LLM-written summaries; prompt caching with `cache_control`; parallel tool execution; structured (non-string) tool I/O and thinking-block persistence; one shared async event loop instead of per-request runtimes; atomic session writes; actually connecting MCP tools into the loop; safer permission defaults (deny-unknown, gated default mode, input-pattern rules).

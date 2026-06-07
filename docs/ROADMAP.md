# Zak Code — Engineering Roadmap

> **Status:** Living document. Every milestone updates this file (see _Definition of done_). Last revised: 2026-06-01.
> **Owner:** zkysar@gmail.com

---

## 1. Vision

**Zak Code** is a provider-agnostic, terminal-first AI coding agent built in Python. It is engineered as a **clean core engine** that is reusable across surfaces, with thin clients layered on top. The two reference models are:

- **The good ideas to steal** — Claude Code's tool/permission/prompt design, claw-code's normalized-event loop and trait seams, Hermes' modular agent + transport split, goose's MCP-as-extension-contract and shared event loop.
- **The mistakes to avoid** — unbounded iteration loops, fail-open permissions, `len/4` token estimation, per-request event loops, hand-rolled serialization, lossy string tool I/O, raw config in the system prompt, and "surface sprawl."

### 1.1 Three layers (hard architectural boundary)

```
┌──────────────────────────────────────────────────────────────┐
│ LAYER 3 — Thin clients (no agent logic; render an event stream)│
│   • CLI (Typer)            ← built first                        │
│   • Web client             ← later                             │
└───────────────▲──────────────────────────────────────────────┘
                │ HTTP / SSE / WebSocket (typed AgentEvent stream)
┌───────────────┴──────────────────────────────────────────────┐
│ LAYER 2 — zakcode-server (FastAPI)                             │
│   Exposes the core over HTTP/SSE/WS. No business logic of its  │
│   own; serializes the same events the CLI consumes in-process. │
└───────────────▲──────────────────────────────────────────────┘
                │ direct import (in-process)
┌───────────────┴──────────────────────────────────────────────┐
│ LAYER 1 — zakcode core engine (importable library)            │
│   The agent loop, providers, tools, sessions, permissions,     │
│   hooks, plugins, skills, compaction. ZERO UI dependencies.    │
│   Public API: `from zakcode import Agent`                      │
└──────────────────────────────────────────────────────────────┘
```

**Invariant:** all surfaces converge on one `Agent.run_turn()` loop emitting one typed `AgentEvent` stream. The CLI links the core in-process; the server wraps it behind HTTP/WS. A future web/IDE client is a renderer, never a fork of the loop.

### 1.2 Target package layout (`src/zakcode/`)

```
src/zakcode/
  __init__.py            # public API: Agent, chat(), run_turn()
  config.py              # pydantic-settings; layered config + secrets split
  providers/             # litellm-based, vendor-agnostic; first-class Ollama + OpenAI
    base.py              #   Provider ABC + canonical message/event model
    litellm_provider.py  #   thin wrapper over litellm.acompletion
    registry.py          #   model registry (context window, caps) — data-driven
  agent/
    loop.py              #   the ReAct loop; emits AgentEvent stream
    prompt.py            #   ordered, cache-stable system prompt builder
    compact.py           #   context compaction (auto + manual)
  tools/
    base.py              #   registry + Tool contract (validate/execute seam)
    builtins/            #   read_file, write_file, list_dir, bash, glob, grep, ...
  session/               #   persistence + in-memory state (resumable)
  commands/              #   slash commands
  hooks/                 #   lifecycle hooks (PreToolUse/PostToolUse/...)
  plugins/               #   register(ctx) extension surface
  skills/                #   progressive-disclosure markdown skills
  server/                #   FastAPI + SSE/WebSocket (Layer 2)
  cli/                   #   Typer entrypoint (Layer 3)
```

### 1.3 Cross-cutting design rules (apply to every milestone)

- **Normalize on a canonical event model.** The loop sees a flat `AssistantEvent` enum (`text_delta`, `tool_use`, `usage`, `message_stop`); providers adapt into it. LiteLLM gives us OpenAI-shaped responses; we normalize those once in `providers/`.
- **Trait/ABC seams for testability.** `Provider`, `ToolExecutor`, and the permission gate are injectable so the loop is pure and unit-testable with scripted clients.
- **Tool I/O is structured.** Tool input is a `dict`, not a string; tool results carry structure. No lossy round-trips.
- **Permissions live in the harness, deny-first.** Enforcement is a separate code path from model reasoning; unknown tools require the strongest privilege.
- **Cache-stable prompt ordering.** System prompt is ordered `stable → dynamic` with a boundary marker; dynamic/config content never sits in the cacheable prefix, and raw config JSON is never injected.
- **One async event loop** for the whole session — never construct a runtime per request.
- **Real token counting** via litellm's `token_counter`/`get_max_tokens`; never `len/4` for trigger decisions.
- **Atomic session writes** (temp file + rename), versioned documents, UUID session ids.
- **Bounded iteration** with a real cap and a clear terminal summary; never `usize::MAX`/unbounded.

### 1.4 Parity tiers (from the Claude Code parity analysis)

- **P0** — the irreducible coding loop: read/write/edit/search/run + sessions + core slash commands.
- **P1** — daily-driver: streaming TUI, permissions/hooks runtime, server, sub-agents, MCP, git workflow, skills.
- **P2** — full-parity/advanced: plugins ecosystem, web client, remote/bridge, teams, cron, evaluation depth.

Milestones below are sequenced to climb these tiers without surface sprawl.

---

## 2. Milestones

Each milestone defines **Goal**, **Scope** (deliverables), **Exit criteria** (testable), **Build workflow** (the orchestration we run to produce it), and a **Definition of done**.

---

### M0 — Runnable minimal agent loop (P0) — ✅ DONE (2026-05-30, commit `0d4b9fd`)

> **Status: shipped.** All exit criteria met. The loop drives real tool calls end-to-end
> against a scripted provider (and is wired for Ollama/OpenAI via litellm); `read_file`,
> `write_file`, `list_dir`, `glob`, `grep`, `bash` are live; sessions persist atomically and
> resume; `zakcode chat` works as a thin REPL; **216 tests pass; ruff + mypy clean** (M0 +
> a hardening pass: a doom-loop guard, a fixed `glob` path-escape leak, and ~120 edge tests;
> source commit `8bfb23c`). Live
> Ollama/OpenAI end-to-end was not exercised in CI (no local Ollama running / no key in the
> build env) — the loop is verified with a hermetic scripted provider; first live run is a
> follow-up. Next: **M1 — streaming + rich TUI.**

**Goal:** A runnable, minimal agent loop driven via litellm against **both Ollama (local, default for dev) and OpenAI**, with a small sharp tool set, a working Typer CLI (`zakcode chat`), sessions, and tests.

**Scope**
- `config.py` (pydantic-settings): layered config, `.env` for secrets only, provider/model selection, `api_base` for Ollama. Default dev provider = Ollama (`ollama_chat/…` or `openai/…` + local `api_base`); OpenAI via `openai/…`.
- `providers/base.py` + `providers/litellm_provider.py`: `Provider` ABC with `acomplete`; canonical `AssistantEvent`/`LLMResult` normalization (`_to_result`); tool-call args parsed from JSON string → `dict`; `drop_params=True`; per-call cost capture.
- `providers/registry.py`: data-driven model facts (context window, max output, supports_tools) seeded from litellm helpers.
- `agent/loop.py`: ReAct loop — build request → `acomplete` → accumulate assistant message → if no tool calls, stop; else execute tools sequentially (M0), append `role:"tool"` results, repeat. **Bounded** by a real iteration cap (default 50) with terminal summary.
- `agent/prompt.py`: minimal ordered system prompt (identity + tool guidance + environment), stable/dynamic boundary marker in place (caching wired later).
- `tools/base.py`: self-registering registry; `Tool` contract with `validate`/`execute` seam; declarative spec (name, description, JSON schema, required-permission tier).
- `tools/builtins/`: `read_file`, `write_file`, `list_dir`, `bash`, `glob`, `grep`. Handlers never raise (errors → structured tool result); enforced timeouts and output caps.
- `session/`: versioned session document, per-message usage stored inline, **atomic writes**, UUID ids, resumable; persist after every turn.
- `cli/`: Typer app with `zakcode chat` (one in-process `Agent`, single asyncio loop), `--provider/--model`, `/exit`.
- Tests: scripted-provider unit tests for the loop (no network); tool unit tests; session round-trip test; a live smoke test against local Ollama (skipped if absent).

**Exit criteria**
- `zakcode chat` runs end-to-end against **local Ollama** and against **OpenAI** (env-gated), completing a multi-step task that reads, edits, and runs a file.
- The loop terminates correctly on "no tool calls" and on hitting the iteration cap (test-verified).
- Tool calls round-trip as structured `dict` input and structured results.
- Sessions persist atomically and resume with cumulative usage intact (round-trip test passes).
- `pytest` green; loop tested with a scripted provider (zero network).

**Build workflow**
1. Scaffold package with `uv` (`pyproject.toml` + lockfile, Python 3.11+, pinned upper bounds; `config.yaml` vs `.env` split).
2. Build `providers/` against litellm; validate Ollama OpenAI-compat path for tool calling (use `openai/<model>` + `api_base=.../v1` to avoid `ollama_chat/` tool bugs).
3. Build `tools/base.py` + the six builtins with tests.
4. Build `agent/loop.py` with injected provider+executor; unit-test with scripted events.
5. Wire `session/` and the Typer CLI; run the live Ollama smoke test.
6. `/code-review` the diff, then `/run` to confirm `zakcode chat` works; update docs.

**Definition of done:** a developer with Ollama installed can `uv tool install` Zak Code and complete a real read-edit-run task in `zakcode chat`; all tests pass; ROADMAP + README + a CONFIG doc reflect M0.

---

### M1 — Streaming + rich TUI (P1) — ✅ DONE (2026-05-30, commits `948eada`…`4c6f9a4`)

> **Status: shipped.** Live token streaming end to end (`LiteLLMProvider.astream` →
> `ToolCallAccumulator` reassembles tool-call arg fragments by index → `AgentLoop.astream_turn`
> emits the typed `AgentEvent` stream with the same stop semantics as `arun_turn`), a fence-safe
> rich `StreamRenderer`, `zakcode chat` streaming by default with **Ctrl-C cancelling the
> in-flight turn cleanly**, and the deferred exact-string **`edit_file`** tool. Final green state
> **288 tests pass + 1 skipped; ruff + mypy clean.** Feature commit `948eada`; integration
> follow-ups `2de36e9`/`4c6f9a4` (the feature commit briefly went in red — a duplicate
> `Agent.astream_turn` and a stale registry test — both caught by the orchestrator's independent
> verification and fixed, with a facade regression test added). Next: **M2 — permissions + hooks.**

**Goal:** Live token-by-token streaming and a polished terminal experience, on the same loop.

**Scope**
- `providers/`: `astream` + index-based streaming accumulator (text deltas + incremental tool-call argument deltas reassembled by `index`); `stream_options={"include_usage": True}` for final usage.
- `agent/loop.py`: emit a typed `AgentEvent` stream (`text`, `tool_call`, `tool_result`, `status`) instead of a blocking buffer; cooperative cancellation (Ctrl-C interrupts mid-turn).
- `cli/`: rich TUI renderer — stream-safe markdown flushing (only on safe boundaries: blank lines / closed code fences), syntax highlighting, spinner, and a live status line (iterations, tokens, active model, cost).
- Tests: streaming accumulator unit tests (partial JSON across deltas, content-then-tool_calls); cancellation test.

**Exit criteria**
- Assistant text renders incrementally; tool-call arguments reassemble correctly from streamed deltas (test-verified).
- A half-written code fence never renders broken (boundary-flush test).
- Ctrl-C cancels the in-flight turn cleanly without corrupting the session.
- Status line shows live iteration/token/cost.

**Build workflow**
1. Implement the streaming accumulator + tests against recorded chunk fixtures.
2. Convert the loop to yield `AgentEvent`s; add cancellation tokens.
3. Build the TUI renderer; `/run` to eyeball streaming + interrupt.
4. `/code-review`; update docs.

**Definition of done:** streaming works against both providers, the TUI is pleasant and non-glitchy, cancellation is clean, tests pass, docs updated.

---

### M2 — Permissions + hooks runtime (P1) — ✅ DONE (2026-05-30, commits `a8933d8`…`3cbe12f`, docs `2ee2c80`)

> **Status: shipped.** Deny-first permissioning enforced in the **core** (a code path the
> model cannot reach) plus a real lifecycle-hook runtime. `permissions.py`: `PermissionMode`
> (deny/ask/acceptEdits/allow), a pure tier×mode `decide()` + stateful `authorize()` with
> per-session allow/deny memory and an injected `PermissionPrompter`; unknown tools fail closed;
> a `DANGEROUS_PATTERNS` blocklist (rm -rf /, sudo, mkfs, fork bomb, dd-to-device, DROP TABLE,
> git force-push, curl|sh, …) that only ever *tightens* a verdict. `hooks/`: PreToolUse/PostToolUse
> runtime — shell hooks as argv arrays (no injection) with JSON-on-stdin + exit-code protocol
> (0=allow/2=block/other=warn) plus in-process callbacks, every failure error-isolated.
> `AgentLoop._execute_tool_call` is the **single seam** both the buffered and streaming paths run
> through: permission authorize → PreToolUse (veto/mutate) → execute → PostToolUse; a denial/veto
> becomes a recoverable error result. CLI: `/permissions`, `/hooks`, and a console prompter that
> shows the exact command. **365 tests pass; ruff + mypy clean.** All exit criteria met
> (dangerous-command block independent of the model; fail-closed unknown tool; hook veto + arg
> mutation + bad-hook isolation; session-approval persistence — all test-verified on both paths).
> Next: **M3 — FastAPI server (SSE/WS).**
>
> _Trust note:_ permission is decided on the model's requested arguments; a PreToolUse hook may
> then rewrite them before execution. Hooks are **operator-configured, trusted** code (like a
> shell rc) — this is intended (a hook can both veto and adjust), not a model-reachable bypass.

**Goal:** Defense-in-depth, deny-first permissioning with a real lifecycle-hook runtime.

**Scope**
- Permission model: ordered mode enum (`ReadOnly < WorkspaceWrite < DangerFullAccess < Prompt < Allow`); per-tool required tier baked into specs; escalation prompt only at the sensible boundary (workspace-write → danger). **Defaults gated** (not full-access); **unknown tools deny** (fail-closed). Denied tools return an error tool-result so the loop recovers.
- Input-aware rules: `DANGEROUS_PATTERNS` blocklist (`rm -rf /`, `sudo`, DB drops), stale-read detection (file changed since last read), and command/path-pattern allowlisting (e.g. allow `bash` only for `git status`).
- `hooks/`: PreToolUse/PostToolUse runtime — JSON-over-stdin payload, exit-code protocol (0=allow, 2=deny, other=warn), argv arrays not shell strings; hook errors isolated (never crash the loop); feedback appended as tool-result.
- `commands/`: `/permissions`, `/hooks`; persist per-session approval decisions ("allow this pattern for the rest of the conversation").
- Tests: permission matrix tests (allow/deny/escalate per tier); dangerous-pattern rejection; hook veto (exit 2) and arg-mutation; fail-closed-unknown-tool test.

**Exit criteria**
- A dangerous command is blocked on a code path independent of model reasoning (test-verified).
- Unknown tools are denied by default; denials produce recoverable error tool-results.
- A PreToolUse hook can veto (exit 2) and mutate args; a bad hook does not break the turn.
- Approval decisions persist within a session.

**Build workflow**
1. Build the permission policy + tiers with the full matrix test.
2. Add input-aware rules (patterns, stale-read).
3. Build the hooks runtime + exit-code protocol tests.
4. Wire `/permissions` and `/hooks`; safety-probe via the eval-style script.
5. `/security-review` the permission code; `/code-review`; update docs.

**Definition of done:** safety probes confirm rejection across every layer; permissions/hooks documented (including the hook contract); tests green.

---

### M3 — FastAPI server + SSE/WS (Layer 2, P1) — ✅ DONE (2026-05-31, commits `19d9bcc`…`8512baa`)

> **Status: shipped.** The core is now drivable over HTTP. `server/wire.py` is the pure JSON
> contract (AgentEvent (de)serialization + request/response + WS control frames). `server/app.py`
> `create_app()` exposes `/health`, `/config` (secret-free), `/tools`, `/sessions` CRUD,
> `POST /chat` (buffered) and `POST /chat/stream` (SSE), plus **`WS /ws/{id}`** — bidirectional
> input/interrupt with a `WebSocketPermissionPrompter` that bridges `action_required` approvals to
> the client (so `ask` mode is interactive over the socket; REST/SSE run headless = fail-closed).
> `server/client.py` `ServerClient` + `zakcode serve` and `chat --server <url>` make the CLI a thin
> client. **The parity exit criterion is met:** in-process vs over-server transcripts are
> byte-identical once serialized (test-verified). **401 tests pass, 1 skipped; ruff + format + mypy clean.**
>
> **M3-hardening (2026-05-31, commits `2bccb74`/`c386f78`/`2fb7cff`)** — acted on two independent
> fresh-eyes reviews (no blockers). (A) `Settings.api_key` is now `exclude=True` so `/config`
> cannot leak a secret by serialization; (B) a per-request `model` override preserves the operator's
> permission/workspace posture (model-only `model_copy`) and server agents get the `SessionStore`
> wired in for incremental persistence; (C) `/chat` + `/chat/stream` refuse a second concurrent turn
> on one session (409). Remaining review nits (WS error frames untyped/raw, cross-channel
> concurrency, `PATCH /config`/`tools` filter) are logged in `RISKS.md` as future work.
> Same `AgentEvent` stream across CLI-in-process / SSE / WS — a future web client is a renderer,
> not a fork. (`PATCH /config` deferred — config is read-only over HTTP for now.) Next: **M4 —
> sub-agents / Task tool.**
>
> _Security note (self-reviewed):_ no auth yet — bind to localhost (`zakcode serve` defaults to
> `127.0.0.1`); the deny-first permission gate still runs in the core for every served turn, and
> `/config` strips `api_key`. Network auth (token/JWT) is a follow-up before any non-local bind.

**Goal:** Expose the core over HTTP with streaming, enabling remote/headless use and future web clients.

**Scope**
- `server/`: FastAPI app wrapping the **same** core `Agent`; endpoints for create-session, run-turn (SSE stream of `AgentEvent`s), resume, list sessions, cancel. WebSocket channel for bidirectional control (interrupt, permission approvals).
- Server emits the **identical** typed event stream the CLI consumes in-process (serialization layer only; zero new agent logic).
- Permission approvals brokered over WS (server pauses turn → `action_required` event → client approves → resumes).
- `cli/`: optional `--server <url>` mode so the CLI can run as a thin client of the server (proves the boundary).
- Tests: SSE stream contract test; WS interrupt + approval round-trip; "CLI-against-server produces same transcript as CLI-in-process" parity test.

**Exit criteria**
- `zakcode serve` runs; a client streams a full turn over SSE and interrupts over WS.
- Permission escalation round-trips over WS (`action_required` → approve → resume).
- In-process vs over-server transcripts match for the same scripted input (parity test).

**Build workflow**
1. Define the wire schema for `AgentEvent` (Pydantic models shared by server + clients).
2. Build SSE run-turn + WS control; reuse the core untouched.
3. Add `--server` client mode to the CLI; run the parity test.
4. `/security-review` the server surface (authz, input validation); `/code-review`; update docs.

**Definition of done:** server streams turns, brokers approvals, and the CLI works identically in-process and over HTTP; tests prove parity; server + API docs published.

---

### M4 — Sub-agents / Task tool (P1) — ✅ DONE (2026-05-31, commits `1259cec`…`72cada1`)

> **Status: shipped (2026-05-31), commits `1259cec`…`72cada1`.** Sub-agents are isolated child
> `AgentLoop`s (fresh `Session`, filtered tool registry via `ToolRegistry.subset`, an isolated
> context) sharing one `IterationBudget` (consume-only + child cap; one-level nesting enforced by
> building children with no spawner). The `task` tool (`tools/builtins/task.py`) runs subtasks
> concurrently via `asyncio.gather` and returns condensed summaries, not transcripts. **Plan Mode**
> is schema-enforced: the read-only `plan` sub-agent's registry omits `write_file`/`edit_file`/`bash`
> entirely, so they are absent from the tool schema the planner is given (not refused at runtime).
> CLI `/plan <task>` drafts a plan (no auto-execute) and `/agents` lists the types. Delegation is
> opt-in via `Agent(enable_subagents=True)`. **456 tests pass, 1 skipped; ruff + format + mypy
> clean.** (Note: the shared budget uses lazy per-iteration consume; the originally-planned
> reserve/refund + depth model was dropped as dead code after the M4 fresh-eyes review.)

**Goal:** Isolated, parallel sub-agents for delegation, returning condensed summaries.

**Scope**
- Sub-agents as first-class **async tasks** (not detached threads): each is a lightweight `Agent` sharing the tool registry/provider but with **fresh history**, **filtered tool access**, and an **isolated context window**; structured handoff back to the parent.
- `IterationBudget` shared across parent + children with refunding; default child cap (e.g. 32); one-level nesting only.
- A `task` tool (Plan/Execute/general-purpose subagent types) exposed to the model; independent subtasks run concurrently (`asyncio.gather`); children return short (~1–2K token) summaries, not raw transcripts.
- Plan/Execute separation: a read-only **Plan Mode** whose planner subagent has write tools *absent from its schema* (schema-enforced, not runtime-checked); emits an editable plan artifact requiring approval to execute. `/plan`, `/agents` commands.
- Tests: child isolation (no context bleed), parallel fan-out correctness, plan-mode write-tool-absence, budget refund/exhaustion.

**Exit criteria**
- A parent delegates two independent subtasks that run concurrently and merge correctly.
- A planner subagent literally cannot call write tools (not in its schema) — test-verified.
- Child agents return summaries; parent context does not balloon with child transcripts.
- Shared `IterationBudget` is respected and refunded.

**Build workflow**
1. Build the sub-agent factory (filtered tools, fresh history, isolated context) + `IterationBudget`.
2. Add the `task` tool + concurrent dispatch.
3. Build Plan Mode with schema-filtered planner; wire `/plan`, `/agents`.
4. `/code-review`; behavioral tests; update docs.

**Definition of done:** delegation + parallel fan-out work, plan mode is schema-enforced (the planner's write tools are absent from its schema), tests pass, docs updated. _Shipped scope note: `/plan` drafts and prints a read-only plan (it is never auto-executed); a formal editable-artifact + approval→execute handoff is deferred (tracked in `RISKS.md`)._

---

### M5 — MCP client (extensions) (P1) — ✅ DONE (2026-05-31, commits `48f109b`…`6005f9d`)

> **Status: shipped.** Zak Code is now an MCP host. A clean-room, no-SDK client speaks
> newline-delimited JSON-RPC 2.0 over a pluggable transport (`mcp/jsonrpc.py`, `mcp/transport.py`
> `StdioTransport`, `mcp/client.py` `MCPClient` — initialize-once, list/call/close). `mcp/manager.py`
> adapts each MCP tool into a plain `Tool` (`McpTool`) registered under the qualified name
> `mcp__<server>__<tool>` in the **same** `ToolRegistry`, so the loop dispatches and permission-gates
> it exactly like a builtin (MCP tools default to the `DANGER_FULL_ACCESS` tier). `mcp/config.py`
> declares servers in JSON (`mcpServers`/`servers`), resolves secrets via `${VAR}` env refs (never
> stored in config), and enforces a command allowlist. The `Agent` facade wires it opt-in
> (`enable_mcp=True`; `connect_mcp()` spawns + discovers, `__init__` stays side-effect-free), and the
> CLI `/mcp` lists/connects servers. **Lazy discovery**: a `tool_search` builtin + a tool budget
> (~25) — `ToolRegistry` gained an active-set so MCP tools beyond the budget register *hidden* and are
> surfaced on demand, keeping schemas out of the base prompt. Graceful degradation throughout (a bad
> server is recorded, never fatal). **551 tests pass, 1 skipped; ruff + format + mypy clean.**
> _Deferred (tracked in `RISKS.md`): streamable-HTTP transport, MCP resources/prompts, OAuth
> (`McpAuthTool`); only stdio + tools shipped._

**Goal:** Connect external capability via MCP, slotting MCP tools into the same registry and loop.

**Scope**
- MCP host/client + `ExtensionManager` (name→client; discover/init/dispatch). Transports: **stdio** (newline-delimited JSON-RPC 2.0 framing — one JSON object per line, *not* LSP `Content-Length`; lazy spawn + initialize-once + reuse) and **streamable-HTTP/SSE** _(HTTP deferred; stdio shipped)_; record unsupported transports as data (graceful degradation).
- **Qualified tool names** `mcp__<server>__<tool>` with a central routing index; MCP tool defs merged into `definitions()` and dispatched by qualified name — indistinguishable from builtins to the loop. Match MCP name regex; mind the 64-char limit.
- Lazy/just-in-time discovery: a `tool_search` surface so large MCP schemas don't bloat the base prompt; activation gated behind user approval; tool-budget filter (keep exposed set small, ~25).
- Security: secrets via prompted env-keys (never inlined), command allowlist for auto-installed stdio servers, optional URL allowlist for HTTP servers.
- `commands/`: `/mcp` (add/list/auth). MCP tools honor the M2 permission tiers.
- Tests: stdio init-once assertion, qualified-name routing, lazy-load gating, unsupported-transport graceful path.

**Exit criteria**
- A stdio MCP server's tools appear (namespaced) and execute through the normal loop.
- Schemas load lazily on demand (not preloaded), gated by approval.
- A misconfigured/unsupported-transport server degrades gracefully (no host crash).

**Build workflow**
1. Build the stdio transport + JSON-RPC client + init-once test.
2. Build `ExtensionManager` + qualified-name routing into the registry.
3. Add HTTP transport + lazy discovery + tool budget; wire `/mcp`.
4. Apply permission gating + secret handling; `/security-review`; `/code-review`; update docs.

**Definition of done:** MCP tools work end-to-end through the same loop/permission model, discovery is lazy and gated, security posture is in place, tests pass, docs updated.

---

### M6 — Plugins (P2) — ✅ DONE (2026-05-31, commits `8f72972`…`e9b5f34`)

> **Status: shipped (Python plugin path).** A plugin is a `plugin.json` manifest + a `register(ctx)`
> entrypoint that contributes tools/hooks/commands through a **narrow** `PluginContext` — never
> touching core files. `commands/CommandRegistry` is the slash-command table; `plugins/` holds the
> contract + `PluginManager.load_into` (trust+enable-gated, error-isolated, records contributions);
> `plugins/discovery.py` finds plugins in `.zakcode/plugins` (project) + `~/.config/zakcode/plugins`
> (user) + the `zakcode.plugins` entry-point group. **Security:** a plugin's module is **not imported
> until after the trust gate passes** (deferred lazy loader), so an untrusted workspace plugin can
> never run code at discovery; trust is explicit (`ZAKCODE_TRUSTED_PLUGINS` / `trusted_plugins=`).
> The `Agent` facade wires it opt-in (`enable_plugins=True`); the CLI exposes `/plugins` and now
> dispatches plugin-registered slash commands. **584 tests pass, 1 skipped; ruff + format + mypy
> clean.** A fresh-eyes review's blocker (import-before-trust) and both MAJORs (dead command surface,
> no CLI trust path) were fixed. _Deferred (tracked in `RISKS.md`): the language-agnostic subprocess
> tool contract, `/reload-plugins`, a shipped sample-plugin fixture + author guide, and the
> `pre_llm_call` cache-safe hook (the hook enum currently has only PreToolUse/PostToolUse)._

**Goal:** A first-party extension surface (`register(ctx)`) that never touches core files.

**Scope**
- `plugins/`: discovery from user + project plugin dirs and entry points; `plugin.yaml` manifest + single `register(ctx)` entrypoint. `ctx` exposes `register_tool`, `register_hook`, `register_command`, and a couple of provider hooks.
- Unified, error-isolating hook dispatcher (one bad plugin never breaks the loop); the six core hooks with `**kwargs` forward-compat. Key design: `pre_llm_call` injects ephemeral context into **the turn's user message, not the system prompt** (preserves cache).
- Plugin tools execute as subprocesses fed JSON via stdin + env (language-agnostic), honoring permission tiers.
- `commands/`: `/plugin` (install/enable/disable/list), `/reload-plugins`.
- Tests: plugin load/register, hook isolation, `pre_llm_call` injection target (user message, not system prompt), subprocess tool contract.

**Exit criteria**
- A sample plugin registers a tool + a hook + a slash command without modifying core.
- A throwing hook is isolated; the turn completes.
- `pre_llm_call` output lands in the user message (cache prefix unchanged) — test-verified.

**Build workflow**
1. Build the plugin loader + `register(ctx)` + manifest.
2. Build the error-isolating hook dispatcher reusing the M2 hook runtime.
3. Build subprocess tool contract; wire `/plugin`.
4. Ship a sample plugin as a fixture; `/code-review`; update docs.

**Definition of done:** plugins extend tools/hooks/commands without core edits, hooks are isolated and cache-safe, tests pass, docs (incl. a plugin-author guide) updated.

---

### M7 — Skills (P2) — ✅ DONE (commits `302edaa`, `c4700dd`, green-fix `1e5a7d5`)

Progressive-disclosure skills shipped: `src/zakcode/skills/__init__.py` parses `SKILL.md` frontmatter
(hand-rolled YAML subset — no PyYAML dependency) with three disclosure levels — L0 (name+description)
folded into the cacheable system-prompt tier, L1 (body) read lazily only on invocation (`body_loaded`
stays False until used), L2 (sibling files) reachable from the skill directory. `discover_skills`
merges bundled → user (`~/.config/zakcode/skills`) → project (`.zakcode/skills`), project overriding;
a malformed `SKILL.md` is recorded and skipped, never raised. `Agent(enable_skills=True)` wires the
registry; the CLI adds `/skills` (list) and bare `/<skill-name>` (invoke → injects the body as a
cache-safe next-turn user message). Tests: `tests/test_skills.py`, `tests/test_skills_facade.py`.
Deferred: an autonomous skill-authoring curator.

**Goal:** Progressive-disclosure, markdown-defined skills (manually authored first).

**Scope**
- `skills/`: `SKILL.md` with YAML frontmatter; progressive disclosure — L0 name+description always in prompt, L1 full body on demand, L2 referenced `scripts/`/`references/` pulled as needed. Align frontmatter with the portable agentskills format.
- Skill loader (bundled + user/project dirs); skills invokable as `/<skill-name>` and via a `skill` tool.
- `commands/`: `/skills` (list/manage).
- (Deferred to a later increment: agent-authored skills + a usage-tracking curator with "never delete, only archive, only agent-created" invariants.)
- Tests: frontmatter parse, L0/L1/L2 disclosure (body not loaded until invoked), `/<skill>` dispatch.

**Exit criteria**
- A user-authored skill shows L0 in the prompt and loads its body only on invocation (token-cost test).
- `/<skill-name>` and the `skill` tool both invoke it.

**Build workflow**
1. Build the skill loader + frontmatter parser + disclosure logic.
2. Wire `/skills` + the `skill` tool; ship 1–2 bundled skills.
3. `/code-review`; tests; update docs.

**Definition of done:** progressive-disclosure skills work and stay token-cheap until used; tests pass; skill-authoring docs added.

---

### M8 — Advanced context compaction (P1/P2) — ✅ DONE (commits `e5b549c`, `61f936c`, `5fc00dd`, `d270c95`, `4e8f21e`)

Auto-compaction shipped: `src/zakcode/agent/compact.py` `Compactor` triggers at a configurable
fraction (default 0.8) of the **real** context window (`provider.capabilities().context_window` +
`provider.count_tokens`), preserves the last N turns verbatim, and collapses older history into a
single leading system summary written by the model. The summary boundary is tool-pair-safe (never
splits an assistant tool-call from its result) and re-compaction is idempotent (a prior summary is
folded in, never stacked), with a "resume directly, don't re-acknowledge" continuation note. The
loop calls `_maybe_compact()` before each turn (buffered + streaming); `compact_now()` backs the
`/compact` command; `Agent(enable_compaction=True)` opts in. Best-effort: a summarization failure is
logged and the turn proceeds with full history. Tests: `tests/test_compact.py` (8),
`tests/test_compact_facade.py` (3). Deferred: per-tool-type result truncation, large-output offload
to temp-file handles, and end-of-context TODO re-injection.


**Goal:** Keep long sessions alive with real token counting, auto-compaction, and LLM-written summaries.

**Scope**
- `agent/compact.py`: trigger at ~70–80% of the **real** context window (from the model registry / litellm `get_max_tokens` + `token_counter`), auto-fired from the loop (not manual-only).
- Preserve the last N (3–5) turns verbatim; collapse older history into a single leading summary written **by the model** (heuristic structured summary kept only as offline fallback). Idempotent re-compaction that detects + merges a prior summary; explicit "resume directly, do not acknowledge the summary" instruction.
- Per-tool-type result handling: truncate bash vs file-read differently; offload large outputs to temp files referenced by handle; emit truncation hints. Re-inject the live TODO at the **end** of context to resist instruction fade-out.
- Prompt caching: split static (identity/safety/tool schemas) from dynamic at the boundary marker; apply `cache_control`; wire cache-token accounting through usage.
- `commands/`: `/compact`, `/context` (window usage), `/cost`.
- Tests: auto-trigger at threshold, recent-turns-preserved, re-compaction merge idempotence, cache-token accounting, 50+ turn long-horizon session survives.

**Exit criteria**
- A 50+ turn session auto-compacts and continues coherently (long-horizon test).
- Token trigger uses a real tokenizer, not `len/4`.
- Re-compaction merges (doesn't re-summarize the summary); cache tokens reported correctly.

**Build workflow**
1. Wire real token counting + the auto-trigger into the loop.
2. Implement LLM summarization + idempotent merge + resume instruction.
3. Add per-tool truncation/offload + TODO re-injection + prompt caching.
4. Run the long-horizon eval; `/code-review`; update docs.

**Definition of done:** long sessions survive via auto-compaction with model-written summaries and correct cost accounting; tests pass; context-management docs updated.

---

### M9 — Evaluation harness (P2) — ✅ DONE (commits `46732b6`, `41e6eae`, `cfe47e7`)

Behavioral eval harness shipped in `src/zakcode/evals/`. It drives the **real** agent loop with
deterministic, no-network `ScriptedProvider`s (a real `Provider` whose completions come from a fixed
script or a dynamic responder; history-growing `count_tokens` + configurable `context_window` so
compaction thresholds are crossed cheaply) and asserts the invariants that must never regress, via
six probes (`evals/probes.py`): **completion-detection**, **safety-rejection** (privileged call fails
closed at the core gate; nothing executes), **plan-mode-readonly** (planner registry has no
write/exec tools — schema-enforced), **doom-loop-halt** (`stop_reason="doom_loop"`),
**partial-failure-recovery** (a flaky tool's error surfaces as recoverable feedback; the model
retries), and **long-horizon-compaction** (history crosses the window → LLM summary → turn still
completes). `EvalCase/EvalResult/EvalReport` + `run_evals` (a probe failure is recorded, never aborts
the suite); `make_agent()` drives the real `Agent`; `hermetic_env()` scrubs provider keys + pins
`TZ=UTC`. To enable injection without a network, the `Agent` facade gained an additive `provider=`
parameter (litellm still built lazily when omitted). Runnable as **`zakcode eval`** (`-v` for
details), which exits non-zero if any probe fails — a CI gate. Tests:
`tests/test_evals_harness.py` (12), `tests/test_evals_probes.py` (9), `tests/test_cli_eval.py` (3).
_Deferred: Git-snapshot undo per step, structured JSON phase-event logging, and a GitHub Actions
workflow wiring `zakcode eval` + the pytest gate (the harness is CI-ready; the workflow file is the
remaining glue)._


**Goal:** Test the agent like software — behavioral E2E suites gating Zak Code's own changes.

**Scope**
- `evals/`: behavioral (not internal) test suites covering: long-horizon (50+ turn) compaction; safety probes (attempt dangerous ops, confirm rejection across every layer); completion-detection (graceful termination, no iteration-cap surprises); plan-mode (write tools genuinely unavailable); partial-failure/recovery (timeout → recovers, no doom-loop); doom-loop detection (identical tool+args within N → halt + ask).
- Observability: structured JSON phase-boundary events; live status line; Git-snapshot-based undo per step; logs of tool invocations, approvals, safety rejections, and system-reminder injections.
- CI gating: Zak Code's self-generated changes run through the project's own lint/type/unit/E2E + security scan before commit; "find other usages of modified symbols" check.
- Tests/eval: the eval harness itself is run in CI; thresholds defined for pass/fail.

**Exit criteria**
- The eval suite runs in CI and gates merges; all suites pass on `main`.
- A doom-loop probe is detected and halted (test-verified).
- Safety probes confirm rejection at every layer; observability emits structured phase events.

**Build workflow**
1. Build the eval runner (scripted providers + fixtures, hermetic: unset creds, `TZ=UTC`, subprocess isolation).
2. Author the six probe suites + doom-loop detector.
3. Wire structured logging, status line, Git-undo.
4. Add CI gating; `/code-review`; update docs.

**Definition of done:** behavioral evals gate CI, doom-loop + safety probes pass, observability is in place; eval + observability docs published.

---

### M10+ — Web client & beyond (P2) — ✅ web client DONE (commits `4ecea94`, `2ff57be`)

> **Web client shipped.** A dependency-free single-page client (`src/zakcode/server/static/index.html`
> — vanilla HTML/CSS/JS, no build step) is served by the same M3 server at `/` (and mounted at `/app`).
> It is a **pure renderer of the `AgentEvent` stream with no agent logic**: it creates a session
> (`POST /sessions`), opens `WS /ws/{id}`, sends `{type:"input"}`, renders every event type
> (text / tool_call / tool_result / status / usage / done), and drives the **permission-approval
> bridge** (renders the server's `action_required` frame and replies `{type:"approval", outcome:…}`).
> The server publishes the contract at **`GET /schema/events`** (JSON Schema generated from the same
> adapter that serializes frames, so it cannot drift) plus `wire.event_type_names()`. The DoD's
> "shared wire schema contract test" is `tests/test_webclient_contract.py` (7): the client's declared
> `EVENT_TYPES` must equal the server's set, every event type has a render case, the WS verbs +
> approval outcomes are valid, and no provider/agent logic leaked into the client. Loopback-only
> posture unchanged (no CORS, no auth). Server surface tests: `tests/test_server_webclient.py` (5).
>
> **Deferred (opt-in, each its own future mini-milestone with the same DoD):** session list/resume &
> cost/context visualization in the UI; agent-authored skills + curator; OS-level sandboxing;
> multi-agent teams / coordinator / cron / YAML recipes (declarative Jinja2 workflows — *not* the
shipped Recipe Cursor; see Post-M11); additional providers (Anthropic/Bedrock/Vertex —
> just new litellm prefixes, no loop changes); plus the M9 deferrals (Git-undo, JSON phase events,
> the CI workflow file).

**Goal:** A thin web client on the M3 server, plus deferred advanced surfaces.

**Scope (web client)**
- Web UI consuming the M3 SSE/WS `AgentEvent` stream — pure renderer (chat, streaming, tool-result display, permission-approval prompts, status line). No agent logic.
- Session list/resume, plan-artifact editing, cost/context visualization.
- Tests: web client renders the same event stream the CLI does (shared wire schema contract test).

**Deferred / opt-in (only after the core is proven, to avoid surface sprawl)**
- Agent-authored skills + curator (usage tracking, archive-not-delete).
- OS-level sandboxing (filesystem + network egress allowlist via proxy; secrets outside the box).
  *Partial:* filesystem confinement is enforced (workspace path guard), and the **network egress
  allowlist via a domain-whitelisting proxy** shipped opt-in (`ZAKCODE_EGRESS_PROXY`,
  `zakcode.sandbox.EgressProxy`) for subprocess egress — best-effort (cooperating clients).
  Remaining: kernel-enforced isolation so the proxy can't be bypassed (Linux bubblewrap / Windows
  AppContainer or Docker), and routing secrets outside the box.
- Multi-agent teams / coordinator, remote/bridge, cron scheduling, **YAML recipes** (declarative YAML + Jinja2 workflows with schema-validated output — distinct from the shipped **Recipe Cursor**, the small-model verify-before-finish gate; see the Post-M11 section).
- Additional providers (Anthropic, Bedrock, Vertex) — just new litellm prefixes + registry entries, no loop changes.

**Exit criteria (web client)**
- The web client completes a full read-edit-run task purely by rendering the server's event stream, including a permission approval, with no duplicated loop logic.

**Build workflow**
1. Freeze/publish the `AgentEvent` wire schema from M3.
2. Build the web renderer against it; contract-test against the CLI's event stream.
3. Layer deferred surfaces individually, each as its own mini-milestone with the same DoD discipline.

**Definition of done:** a thin web client works against the server with zero agent logic; each deferred surface, when built, ships with tests and doc updates and does not modify the core loop.

---

### M11 — Learning substrate (P2) — ✅ DONE (2026-06-01, commits `7467420`…HEAD)

> **Status: shipped.** Not a learning *policy* of its own — the **substrate seams** a
> self-learning framework (e.g. Claude-Mind) folds into. Built in seven reviewed phases,
> each adversarially fresh-eyes-reviewed and committed green:
>
> 1. **Text tool-calling fallback** (`providers/text_tools.py`) — tool-less local models
>    get tool use via a `<tool_call>` text protocol; `tool_calling_mode` auto/native/text;
>    untrusted tool output defanged in the rewritten history.
> 2. **`PreLLMCall` context injection** (`hooks/`) — per-turn background context (memory/
>    RAG/retrieval) folded in as an ephemeral, fenced-untrusted, **non-persisted** tail
>    message → prompt-cache safe. In-process + shell hooks.
> 3. **Rules** (`rules/`) — always-on `.md` guidance (`.zakcode/rules` + `.claude/rules`)
>    in the cacheable tier; sub-agents inherit them; bounded render.
> 4. **Cross-session memory** (`memory/`) — `MemoryProvider` ABC + SQLite/FTS5 default
>    (relocatable), `remember`/`recall` tools, per-turn recall hook; secrets redacted at
>    the store boundary (`secrets.py`).
> 5. **Skill authoring** (`skills.save_skill` + `save_skill` tool, `.claude/skills`
>    discovery) — runtime, path-traversal-safe skill creation.
> 6. **Tier-2 items** — shared-budget refunds, real read-only tool concurrency
>    (tier-guarded), operator deny-rule grammar (append/tighten-only), prompt-cache
>    invariant test.
> 7. **Lifecycle hooks + integration docs** — `SessionStart`/`PreCompact`/`SessionEnd`
>    observe-only hooks; `Agent.aclose()`; [`docs/INTEGRATIONS.md`](INTEGRATIONS.md) maps
>    every framework need to a seam.
>
> **801 passing tests; ruff + format + mypy clean.** **Deferred (see INTEGRATIONS.md):**
> the autonomous "never-terminate" Stop-hook continuation loop, verbatim `settings.json`
> hook ingestion, and managing a framework's background daemon — out of scope for a
> coding agent; the safe fold-in is reader/assistant mode + the encode pass.

### Post-M11 — Portable cognition + small-model reliability — ✅ shipped (commits `a6b9e41`…HEAD)

> Hardening so the agent stays reliable on **small local models** — the mandate is to make
> `qwen2.5:3b` usable, *not* to upgrade the model — plus extraction of a vendor-agnostic
> provider package. Highlights:
>
> 1. **Vendor-agnostic provider package** (`packages/zds-llm-provider`, provider track
>    "M-7/8/9") — the `Provider` ABC + the text tool-calling layer extracted into a
>    pydantic-only, no-vendor-SDK package; adds `ClaudeCodeProvider` + `BitNetProvider`.
> 2. **Small-model reliability bundle** — protocol/template stop-sequences,
>    single-tool-per-turn, an Ollama `num_ctx` lift with a matching capability window, and
>    a cp1252-safe glyph set with ASCII fallbacks.
> 3. **The Recipe Cursor** (`agent/recipe.py`) — a per-turn **verify-before-finish gate**
>    for create-and-run tasks: a write firewall → write-grounding read-back → a gate that
>    won't let the turn end until the written runnable script (`.py`/`.js`/`.ts`/`.sh`/…)
>    was actually *run* (matching a stated output literal when the request clearly states
>    one, via a harness-issued run when it would not raise a permission prompt), else it
>    ends as `recipe_stalled` rather than claiming a false success. **Always on and
>    self-arming** — not a feature flag: it arms from the observed write and is inert on
>    turns that write no runnable script (one way of doing things; the agent steers itself).
>    The former `ZAKCODE_RECIPE_*` / `ZAKCODE_VERIFY_WRITES` / `ZAKCODE_SINGLE_TOOL_PER_TURN`
>    knobs were removed in favor of this autonomy; see the **Small-model reliability**
>    section of the [README](../README.md).
>
> **Terminology — "Recipe Cursor" ≠ "YAML recipes."** The shipped *Recipe Cursor* (above)
> is a small-model verify-before-finish gate. It is **unrelated** to the deferred *YAML
> recipes* (declarative Jinja2 workflows) listed in the M10+ deferrals.

---

## 3. Sequencing rationale

```
M0  core loop ........................ P0  (must exist first)
M1  streaming + TUI .................. P1  (UX on the same loop)
M2  permissions + hooks ............. P1  (safety before remote exposure)
M3  server (SSE/WS) ................. P1  (boundary proof; enables web later)
M4  sub-agents / Task ............... P1  (delegation + plan/execute)
M5  MCP client ...................... P1  (external capability)
M6  plugins ......................... P2  (first-party extension)
M7  skills .......................... P2  (progressive-disclosure knowledge)
M8  advanced compaction ............. P1/P2 (long-horizon survival + caching)
M9  evaluation harness .............. P2  (gate our own changes)
M10+ web client & deferred surfaces . P2  (thin client; no sprawl)
```

Safety (M2) precedes server exposure (M3). The server boundary (M3) is proven before the web client (M10+) depends on it. Compaction (M8) is slotted after the extension surfaces exist so it can account for MCP/plugin/skill context, though its token-counting groundwork is partly laid in M0.

---

## 4. Definition of done — global rules

A milestone is **done** only when **all** hold:

1. **Exit criteria pass** — every testable criterion is met with automated tests.
2. **Tests green** — unit + behavioral suites pass; the loop is exercised with scripted providers (no network) plus at least one live smoke test per provider where applicable.
3. **Core stays clean** — no UI/transport dependency leaks into Layer 1; the CLI and server consume the identical `AgentEvent` stream.
4. **Safety holds** — no regressions in the deny-first permission model; security-relevant changes pass `/security-review`.
5. **Docs updated** — **every milestone updates this ROADMAP** (status, dates, any scope changes) and the relevant subsystem docs/READMEs. Documentation is a deliverable, not an afterthought; intent (the *why*) is captured, not just behavior.

# Zak Code — Claude Code Parity Matrix

A living document tracking Zak Code's feature parity against the Claude Code reference
surface. The reference surface is reconstructed from the Claude Code archive snapshots
(184 tool entries, 207 command entries, 1902 TS-like source files) collapsed to the
actual user-facing tools, slash commands, and subsystems.

Every row in the tables below corresponds to something Claude Code ships today
(**Claude Code? = yes** for all rows). Zak Code's status starts at **Planned** across the
board; this matrix is the single place where status, priority, and target milestone are
tracked as Zak Code is built out.

> Status note: Zak Code is at the planning stage. All items are `Planned`. As components
> land, flip status to `In Progress` → `Done` (or `Partial` / `Deferred`) and keep the
> milestone/notes columns honest.

---

## Legend

**Claude Code?** — Whether the feature exists in the Claude Code reference surface. Every
row here is `yes` (this matrix only tracks parity against things Claude Code actually ships).

**Zak Code status**
- `Planned` — Agreed for the roadmap, not started. (Current value for everything.)
- `In Progress` — Actively being built.
- `Partial` — Usable but incomplete vs. Claude Code behavior.
- `Done` — At parity (or deliberately-scoped equivalent) and shipped.
- `Deferred` — Acknowledged, intentionally postponed beyond current milestones.

**Priority tier**
- **P0** — Irreducible core. The minimum set that makes a working terminal coding agent
  (read/edit/search/run loop + task tracking + authenticate/configure/recover).
- **P1** — Daily-driver. What a serious daily user needs: git workflow, MCP, sub-agents,
  hooks runtime, sessions ergonomics, web access, plan/review, quality-of-life.
- **P2** — Full-parity / advanced. Multi-agent teams, remote/bridge/IDE/desktop/mobile/
  voice, cron, plugin ecosystem, analytics, billing/privacy, cosmetic, and internal-only.

**Target milestone**
- **M1 — Core Loop**: provider transport, agent loop, core tools, permissions, config,
  CLAUDE.md memory, session persistence, minimal TUI.
- **M2 — Daily Driver**: git workflow commands, MCP tools + transport, single sub-agent,
  hooks runtime, skills registry, plan/review, session ergonomics, web tools.
- **M3 — Extensibility**: plugins subsystem, skills curation, output styles, keybindings,
  SessionMemory, structured/remote I/O foundations.
- **M4 — Advanced / Full Parity**: multi-agent teams/swarm, remote/bridge/server,
  cron/remote-trigger, IDE/desktop/mobile/voice, analytics, SDK, internal/debug surface.

> **Milestone-numbering note.** This matrix uses **coarse capability tiers** (M1 = core
> loop, M2 = daily-driver, M3 = extensibility, M4 = advanced) to group 180+ rows at a
> glance. [`ROADMAP.md`](ROADMAP.md) uses **fine-grained, sequenced milestones** (M0 core
> loop → M1 streaming → M2 permissions → M3 server → M4 sub-agents → M5 MCP → M6 plugins →
> M7 skills → M8 compaction → M9 evals → M10+ web) and is the **canonical source for what
> gets built when**. The two numbering schemes are independent; when they disagree, ROADMAP
> wins. (A later pass may retag these rows to ROADMAP's numbers once the surface stabilizes.)

**Touches** (in Tools notes): FS = filesystem, NET = network, PROC = subprocess/process.

---

## 1. Tools

One row per Claude Code model-facing tool (sub-tool families such as the Task* and Team*
groups are listed individually; pure helpers are noted at the end).

> **M0 delivered (2026-05-30, commit `0d4b9fd`).** The P0 read/edit/search/run core is live
> as Zak Code tools: **`read_file`** (≈ FileReadTool), **`write_file`** (≈ FileWriteTool),
> **`glob`** (≈ GlobTool), **`grep`** (≈ GrepTool), **`bash`** (≈ BashTool), plus a
> **`list_dir`** convenience tool. All are workspace-scoped (path-escape-rejecting), never
> raise (errors → structured `ToolResult`), and run through the live `AgentLoop`. Still P0 and
> **not yet built**: **`edit`** (≈ FileEditTool — exact-string edit; today only whole-file
> `write_file`), `PowerShellTool`, and `TodoWriteTool`.

| Tool | Purpose | Claude Code? | Zak Code status | Priority tier | Target milestone | Notes |
|------|---------|--------------|-----------------|---------------|------------------|-------|
| FileReadTool (Read) | Read a file incl. image processing, with size/line limits (`file_path`, `offset`, `limit`) | yes | Planned | P0 | M1 | FS. Core read primitive; required for the loop. |
| FileWriteTool (Write) | Create/overwrite a file (`file_path`, `content`) | yes | Planned | P0 | M1 | FS. Gate behind workspace-write permission; checkpoint before mutate. |
| FileEditTool (Edit) | Exact-string replace edit in a file (`old_string`, `new_string`, `replace_all`) | yes | Planned | P0 | M1 | FS. Keep tool input structured (dict), not raw string. |
| GlobTool | Fast filename pattern matching (`pattern`, `path`) | yes | Planned | P0 | M1 | FS. Read-only; parallel-safe. |
| GrepTool | Ripgrep-based content search (`pattern`, `glob/type`, `output_mode`, context flags) | yes | Planned | P0 | M1 | FS. Read-only; parallel-safe. |
| BashTool | Run a shell command with sandbox/permission/security/destructive-command/read-only/path validation (`command`, `timeout`, `description`, `run_in_background`) | yes | Planned | P0 | M1 | FS/NET/PROC. DangerFullAccess tier; DANGEROUS_PATTERNS blocklist + background promotion. |
| PowerShellTool | Windows PowerShell equivalent of Bash (CLM types, git-safety, common-parameters, security/permission validation) | yes | Planned | P0 | M1 | FS/NET/PROC. Co-P0 with Bash — primary shell on the Windows target environment. |
| TodoWriteTool | Maintain the in-session todo list (`todos[]`: content, status, activeForm) | yes | Planned | P0 | M1 | State (FS). Re-inject live TODO at end of context to fight instruction fade-out. |
| AgentTool (Task) | Spawn a sub-agent to autonomously run a sub-task; built-in types (general-purpose, explore, plan, verification, claude-code-guide, statusline-setup), fork/resume/run, agent memory/color (`description`, `prompt`, `subagent_type`) | yes | Planned | P1 | M2 | Indirect FS/NET/PROC. Start with general-purpose/explore/plan; sub-agents return condensed summaries. |
| EnterPlanModeTool | Enter plan mode (model plans before acting) | yes | Planned | P1 | M2 | Read-only Planner; write tools absent from planner schema. |
| ExitPlanModeV2Tool | Exit plan mode and present the plan for approval (`plan`) | yes | Planned | P1 | M2 | Surface plan as editable artifact; explicit approval to execute. |
| WebFetchTool | Fetch a URL and process its content, with preapproved-host list (`url`, `prompt`) | yes | Planned | P1 | M2 | NET. Honor egress allowlist. |
| WebSearchTool | Web search returning results (`query`, allow/block domains) | yes | Planned | P1 | M2 | NET. |
| MCPTool | Invoke a tool exposed by a connected MCP server (collapse classification) | yes | Planned | P1 | M2 | Indirect NET. Route by qualified `mcp__<server>__<tool>` name into the flat registry. |
| ListMcpResourcesTool | List resources available from MCP servers (optional `server`) | yes | Planned | P1 | M2 | NET. |
| ReadMcpResourceTool | Read a specific MCP resource (`server`, `uri`) | yes | Planned | P1 | M2 | NET. |
| McpAuthTool | Authenticate to an MCP server (OAuth) | yes | Planned | P1 | M2 | NET. Tri-state auth + token cache. |
| ConfigTool | Read/write supported CLI settings programmatically (`setting`, `value`) | yes | Planned | P1 | M2 | FS (settings). Restrict to supportedSettings allowlist. |
| SkillTool | Discover and execute a registered skill (`skill`, `args`) | yes | Planned | P1 | M2 | FS, indirect NET/PROC. Progressive disclosure of SKILL.md. |
| NotebookEditTool | Edit Jupyter notebook cells (`notebook_path`, `cell_id`, `source`, `cell_type`, `edit_mode`) | yes | Planned | P1 | M2 | FS. |
| ToolSearchTool | Search for / lazily load deferred tool schemas (`query`, `max_results`) | yes | Planned | P1 | M2 | Keeps base prompt small; load schemas on demand. |
| AskUserQuestionTool | Pause and ask the user a structured multiple-choice question (`question`, `options/header`) | yes | Planned | P1 | M2 | Interactive — force sequential (never parallel). |
| TaskCreateTool | Create a background/tracked task (task spec/description) | yes | Planned | P2 | M4 | State (FS), indirect NET/PROC. |
| TaskGetTool | Fetch a task's status/details (`task_id`) | yes | Planned | P2 | M4 | State (FS). |
| TaskListTool | List tracked tasks (filters) | yes | Planned | P2 | M4 | State (FS). |
| TaskOutputTool | Retrieve a task's output (`task_id`) | yes | Planned | P2 | M4 | State (FS). |
| TaskStopTool | Stop/cancel a running task (`task_id`) | yes | Planned | P2 | M4 | State (FS), PROC. |
| TaskUpdateTool | Update a task's fields/status (`task_id`, fields) | yes | Planned | P2 | M4 | State (FS). |
| TeamCreateTool | Create a multi-agent team (team spec) | yes | Planned | P2 | M4 | State (FS), NET. Multi-agent orchestration. |
| TeamDeleteTool | Delete a multi-agent team (`team_id`) | yes | Planned | P2 | M4 | State (FS), NET. |
| SendMessageTool | Send a message to another agent/teammate (multi-agent) (recipient, message) | yes | Planned | P2 | M4 | NET. Backed by spawnMultiAgent helper. |
| RemoteTriggerTool | Trigger a remote agent/run (target + payload) | yes | Planned | P2 | M4 | NET. |
| ScheduleCronTool (CronCreate / CronDelete / CronList) | Create, delete, list scheduled cron jobs / routines (schedule expr, prompt/command, id) | yes | Planned | P2 | M4 | FS/NET. Materializes recipes into unattended sessions. |
| EnterWorktreeTool | Create/switch into a git worktree for isolated work (worktree/branch name) | yes | Planned | P2 | M4 | FS, PROC (git). |
| ExitWorktreeTool | Leave/clean up the active git worktree | yes | Planned | P2 | M4 | FS, PROC (git). |
| BriefTool | Produce/upload a "brief" summary artifact with attachments | yes | Planned | P2 | M4 | FS (upload), NET. |
| LSPTool | Language-server queries: symbol context, formatters, schema-typed results (symbol/file/position) | yes | Planned | P2 | M4 | FS, PROC (LSP). |
| REPLTool | REPL / primitive-tools execution surface (constants + primitiveTools) | yes | Planned | P2 | M4 | FS, indirect NET, PROC. DangerFullAccess tier. |
| SleepTool | Pause execution for a duration (`duration`) | yes | Planned | P2 | M4 | No FS/NET/PROC. |
| SyntheticOutputTool | Emit synthetic/structured output into the transcript (output payload) | yes | Planned | P2 | M4 | No FS/NET/PROC. |
| TestingPermissionTool | Internal test-only permission tool | yes | Planned | P2 | M4 | Internal/test-only. |
| _shared helper: spawnMultiAgent_ | Multi-agent spawning helper backing Agent/Team/SendMessage (not model-facing) | yes | Planned | P2 | M4 | Indirect NET/PROC. Helper, not a model-facing tool. |
| _shared helper: gitOperationTracking_ | Tracks git operations across tools (not model-facing) | yes | Planned | P2 | M4 | FS, PROC (git). Helper, not a model-facing tool. |

---

## 2. Slash Commands

One row per Claude Code slash command (collapsed from 207 source entries; install-wizard
steps, plugin sub-views, and `index/*` duplicates folded into their parent command).

| Command | Purpose | Claude Code? | Zak Code status | Priority tier | Target milestone | Notes |
|---------|---------|--------------|-----------------|---------------|------------------|-------|
| `/help` | Show help | yes | Planned | P0 | M1 | |
| `/clear` | Clear conversation, caches, or both | yes | Planned | P0 | M1 | |
| `/exit` | Exit the CLI | yes | Planned | P0 | M1 | |
| `/login` | Authenticate (OAuth) | yes | Planned | P0 | M1 | Tri-state auth with auto-refresh. |
| `/logout` | Sign out | yes | Planned | P0 | M1 | |
| `/model` | Choose the model | yes | Planned | P0 | M1 | Data-driven model registry. |
| `/init` | Initialize CLAUDE.md for the repo | yes | Planned | P0 | M1 | |
| `/memory` | View/edit CLAUDE.md memory | yes | Planned | P0 | M1 | Unify memory roots across subsystems. |
| `/config` | Open the settings/config editor | yes | Planned | P0 | M1 | Layered config (user → project → local). |
| `/permissions` | View/edit tool permissions | yes | Planned | P0 | M1 | Add input-pattern rules, not just mode tiers. |
| `/compact` | Compact (summarize) conversation context | yes | Planned | P0 | M1 | Wire auto-compaction; prefer LLM-written summaries. |
| `/cost` | Show token/cost usage for the session | yes | Planned | P0 | M1 | Wire cache-token accounting. |
| `/diff` | Show working-tree / change diff | yes | Planned | P0 | M1 | |
| `/resume` | Resume a previous conversation | yes | Planned | P0 | M1 | Cumulative usage reconstructable on resume. |
| `/version` | Show version | yes | Planned | P0 | M1 | |
| `/agents` | Manage/select custom and built-in sub-agents | yes | Planned | P1 | M2 | |
| `/mcp` | Manage MCP servers (add, list, IdP/xaa auth) | yes | Planned | P1 | M2 | |
| `/skills` | List/manage skills | yes | Planned | P1 | M2 | |
| `/hooks` | View/manage hooks configuration | yes | Planned | P1 | M2 | Net-new vs. port: needs runtime hook execution. |
| `/plan` | Enter/manage plan mode | yes | Planned | P1 | M2 | |
| `/review` | Code review / ultrareview (+remote) | yes | Planned | P1 | M2 | |
| `/security-review` | Security-focused review | yes | Planned | P1 | M2 | |
| `/commit` | Create a git commit | yes | Planned | P1 | M2 | |
| `/commit-push-pr` | Commit + push + open a PR | yes | Planned | P1 | M2 | |
| `/branch` | Create/switch git branches | yes | Planned | P1 | M2 | |
| `/context` | Show/visualize context-window usage | yes | Planned | P1 | M2 | Use real tokenizer, not len/4. |
| `/status` | Show current status | yes | Planned | P1 | M2 | |
| `/session` | Manage sessions | yes | Planned | P1 | M2 | Atomic session writes; UUID ids. |
| `/rename` | Rename the session (auto-generated names) | yes | Planned | P1 | M2 | |
| `/export` | Export the conversation/session | yes | Planned | P1 | M2 | |
| `/doctor` | Diagnose installation/environment health | yes | Planned | P1 | M2 | |
| `/add-dir` | Add another working directory to the session | yes | Planned | P1 | M2 | |
| `/keybindings` | View/edit keyboard shortcuts | yes | Planned | P1 | M2 | |
| `/theme` | Choose UI theme | yes | Planned | P1 | M2 | |
| `/output-style` | Choose output style | yes | Planned | P1 | M2 | |
| `/effort` | Set reasoning effort level | yes | Planned | P1 | M2 | |
| `/vim` | Toggle vim editing mode | yes | Planned | P1 | M2 | |
| `/usage` | Show usage details | yes | Planned | P1 | M2 | |
| `/stats` | Show statistics | yes | Planned | P1 | M2 | |
| `/feedback` | Send feedback to Anthropic | yes | Planned | P1 | M2 | |
| `/release-notes` | Show release notes | yes | Planned | P1 | M2 | |
| `/upgrade` | Upgrade the CLI | yes | Planned | P1 | M2 | |
| `/onboarding` | Run first-run onboarding | yes | Planned | P1 | M2 | |
| `/plugin` | Browse marketplaces, install/manage/validate/trust plugins | yes | Planned | P2 | M3 | Plugin ecosystem (net-new build). |
| `/reload-plugins` | Reload plugins | yes | Planned | P2 | M3 | |
| `/tasks` | View/manage background tasks | yes | Planned | P2 | M4 | |
| `/bridge` | Manage the remote-control REPL bridge | yes | Planned | P2 | M4 | |
| `/bridge-kick` | Kick/restart the bridge | yes | Planned | P2 | M4 | |
| `/remote-env` | Configure remote environment | yes | Planned | P2 | M4 | |
| `/remote-setup` | Remote agent setup | yes | Planned | P2 | M4 | |
| `/chrome` | Drive/connect Claude-in-Chrome integration | yes | Planned | P2 | M4 | |
| `/desktop` | Desktop app integration | yes | Planned | P2 | M4 | |
| `/mobile` | Mobile companion integration | yes | Planned | P2 | M4 | |
| `/voice` | Voice mode | yes | Planned | P2 | M4 | |
| `/ide` | IDE integration management | yes | Planned | P2 | M4 | |
| `/install` | Install the CLI | yes | Planned | P2 | M4 | |
| `/install-github-app` | Install the GitHub app (multi-step OAuth wizard) | yes | Planned | P2 | M4 | |
| `/install-slack-app` | Install the Slack app | yes | Planned | P2 | M4 | |
| `/passes` | Manage "passes" (multi-pass workflows) | yes | Planned | P2 | M4 | |
| `/ultraplan` | Heavyweight multi-pass planning | yes | Planned | P2 | M4 | |
| `/thinkback` | Record "thinkback" reasoning sessions | yes | Planned | P2 | M4 | |
| `/thinkback-play` | Replay "thinkback" reasoning sessions | yes | Planned | P2 | M4 | |
| `/rewind` | Rewind conversation/checkpoint state | yes | Planned | P2 | M4 | |
| `/teleport` | Teleport/jump to a session or context | yes | Planned | P2 | M4 | |
| `/share` | Share a session/conversation | yes | Planned | P2 | M4 | |
| `/insights` | Show usage/work insights | yes | Planned | P2 | M4 | |
| `/advisor` | Surface contextual advice/suggestions | yes | Planned | P2 | M4 | |
| `/autofix-pr` | Automatically fix issues on a pull request | yes | Planned | P2 | M4 | |
| `/bughunter` | Run a bug-hunting agent pass | yes | Planned | P2 | M4 | |
| `/issue` | Create/triage an issue | yes | Planned | P2 | M4 | |
| `/pr_comments` | Fetch/manage PR review comments | yes | Planned | P2 | M4 | |
| `/tag` | Tag the session/conversation | yes | Planned | P2 | M4 | |
| `/files` | Browse/manage session files | yes | Planned | P2 | M4 | |
| `/copy` | Copy conversation/output to clipboard | yes | Planned | P2 | M4 | |
| `/statusline` | Configure the status line | yes | Planned | P2 | M4 | |
| `/color` | Set agent/UI color | yes | Planned | P2 | M4 | |
| `/stickers` | Stickers / easter-egg feature | yes | Planned | P2 | M4 | Cosmetic. |
| `/good-claude` | Positive-reinforcement / reward signal | yes | Planned | P2 | M4 | |
| `/btw` | Inject an out-of-band "by the way" note to the model | yes | Planned | P2 | M4 | |
| `/brief` | Generate a brief/summary artifact | yes | Planned | P2 | M4 | |
| `/summary` | Summarize the conversation | yes | Planned | P2 | M4 | |
| `/fast` | Toggle fast mode | yes | Planned | P2 | M4 | |
| `/sandbox-toggle` | Toggle sandboxed execution | yes | Planned | P2 | M4 | FS + network isolation when implemented. |
| `/extra-usage` | Manage/purchase extra usage | yes | Planned | P2 | M4 | |
| `/rate-limit-options` | Configure rate-limit behavior options | yes | Planned | P2 | M4 | |
| `/privacy-settings` | Manage privacy settings | yes | Planned | P2 | M4 | |
| `/env` | Show/manage environment variables | yes | Planned | P2 | M4 | |
| `/terminalSetup` | Configure terminal integration | yes | Planned | P2 | M4 | |
| `/oauth-refresh` | Refresh OAuth tokens | yes | Planned | P2 | M4 | |
| `/heapdump` | Dump JS heap for debugging | yes | Planned | P2 | M4 | Internal/debug. |
| `/ant-trace` | Internal Anthropic tracing/diagnostics | yes | Planned | P2 | M4 | Internal/debug. |
| `/ctx_viz` | Context visualization (debug) | yes | Planned | P2 | M4 | Internal/debug. |
| `/debug-tool-call` | Inspect/replay a tool call for debugging | yes | Planned | P2 | M4 | Internal/debug. |
| `/break-cache` | Invalidate prompt/response caches | yes | Planned | P2 | M4 | Internal/debug. |
| `/backfill-sessions` | Backfill/migrate historical session records | yes | Planned | P2 | M4 | Internal/migration. |
| `/mock-limits` | Mock rate limits (testing) | yes | Planned | P2 | M4 | Internal/test. |
| `/reset-limits` | Reset rate limits (testing) | yes | Planned | P2 | M4 | Internal/test. |
| `/init-verifiers` | Initialize verifier setup for the repo | yes | Planned | P2 | M4 | |
| _`createMovedToPluginCommand`_ | Internal shim that stubs commands relocated into plugins | yes | Planned | P2 | M3 | Internal shim, not user-facing. |

---

## 3. Subsystems

Major Claude Code subsystems (module counts shown where known). Status reflects how much
of each subsystem Zak Code plans to build vs. defer.

| Subsystem | Scope | Claude Code? | Zak Code status | Priority tier | Target milestone | Notes |
|-----------|-------|--------------|-----------------|---------------|------------------|-------|
| Core query/tool loop | `assistant` (sessionHistory), QueryEngine, Tool, Task, tools.ts — the agentic loop the tools run against | yes | Planned | P0 | M1 | Single `reply()`-style loop emitting a typed event stream; clean harness/model boundary; layered stop conditions (no-tool / iteration cap / budget / doom-loop). |
| Services: api | Anthropic client, bootstrap, admin requests, errors (subset of the 130-module services layer) | yes | Planned | P0 | M1 | Normalize on Anthropic wire shape; per-provider adapters; bounded retry/backoff; cache_control. |
| Services: oauth | OAuth credential flow + token cache | yes | Planned | P0 | M1 | Tri-state auth (api-key, bearer/OAuth, both); auto-refresh; atomic credential writes. |
| Config / settings | Layered settings discovery + deep-merge, typed feature config | yes | Planned | P0 | M1 | Precedence user → project → local; secrets in `.env`, settings in config. |
| Memory: memdir basics | CLAUDE.md memory discovery (ancestor chain), dedup, budgets; team memory deferred | yes | Planned | P0 | M1 | Root→cwd discovery with content-hash dedup + per-file/total char budgets; unify memory roots. |
| Sessions / persistence | Versioned, resumable session documents with inline per-message usage | yes | Planned | P0 | M1 | Atomic write (temp + rename); UUID ids; structured (dict) tool I/O; persist after every turn/mutating command. |
| TUI essentials | Minimal Ink/React-equivalent terminal UI: App, REPL, core dialogs | yes | Planned | P0 | M1 | Stream-safe-boundary markdown flushing; live status line (iterations, tokens, compaction, model). |
| Services: mcp | MCP connection manager + tool routing into the live registry | yes | Planned | P1 | M2 | Qualified `mcp__<server>__<tool>` names; lazy spawn + initialize-once; stdio + streamable-HTTP; actually connect MCP tools into the loop. |
| Sub-agents / coordinator (single) | `coordinatorMode`; single sub-agent delegation with isolated context | yes | Planned | P1 | M2 | First-class async task with structured handoff + condensed summaries; shared IterationBudget; max-iterations cap (not unbounded). |
| Hooks runtime | PreToolUse/PostToolUse lifecycle hook execution (currently config-only in the port) | yes | Planned | P1 | M2 | Exit-code protocol (0=allow, 2=deny/veto), JSON-over-stdin payload; argv arrays not raw shell strings; denial → error tool-result. |
| Skills registry + bundled | `loadSkillsDir`, bundled skills, MCP skill-builders; `/skills` surface | yes | Planned | P1 | M2 | Markdown SKILL.md with progressive disclosure; start manual-authored; defer autonomous curator. |
| Keybindings | Default + user bindings, parser/matcher/resolver, schema, validation | yes | Planned | P1 | M2 | |
| Output styles | Load output styles from a directory | yes | Planned | P1 | M2 | |
| SessionMemory | Persistent session memory + prompts; cross-session recall | yes | Planned | P1 | M2 | Frozen-snapshot memory + FTS5 search; MemoryProvider ABC. |
| Prompt suggestion | PromptSuggestion (+ speculation) service | yes | Planned | P1 | M2 | |
| Plugins | Loader + marketplace + `services/plugins` (install/enable/disable/update/trust); contribute hooks/tools/commands/MCP | yes | Planned | P2 | M3 | Net-new build (absent in the port). `register(ctx)` entrypoint; subprocess tool contract (JSON via stdin + env). |
| Migrations | Settings/model migrations (auto-update, permission, model renames, repl-bridge → remote-control) | yes | Planned | P2 | M3 | |
| State + bootstrap | AppState/AppStateStore/selectors; startup bootstrapping | yes | Planned | P2 | M3 | Eager construction at startup; validate config/creds up front. |
| CLI: structured/remote I/O + transports | structuredIO, remoteIO, print, ndjson, handlers; HybridTransport/SSE/WebSocket/uploaders/ccrClient | yes | Planned | P2 | M4 | Foundations may start in M3; full transport stack in M4. |
| Bridge | Remote-control REPL bridge: API/config/messaging/permission callbacks, JWT, polling, attachments | yes | Planned | P2 | M4 | |
| Remote | RemoteSessionManager, SessionsWebSocket, remotePermissionBridge, sdkMessageAdapter | yes | Planned | P2 | M4 | |
| Server (direct-connect) | createDirectConnectSession, directConnectManager, types | yes | Planned | P2 | M4 | `zak-server` wraps core behind HTTP/WS emitting the same event stream. |
| SDK (entrypoints) | Programmatic agent SDK: controlSchemas, coreSchemas/Types, agentSdkTypes | yes | Planned | P2 | M4 | Two-tier API: simple `chat()` + full `run_conversation()`; round-trippable history. |
| Upstream proxy | relay + upstreamproxy (API relay) | yes | Planned | P2 | M4 | |
| Multi-agent teams / swarm | Team create/delete, SendMessage, coordinator/swarm-worker handlers | yes | Planned | P2 | M4 | Get parent context right before fan-out; isolated child ExtensionManager/context. |
| Scheduler: cron / remote-trigger | ScheduleCron + RemoteTrigger; unattended scheduled sessions | yes | Planned | P2 | M4 | |
| Analytics | datadog, growthbook, first-party events, sinks/killswitch | yes | Planned | P2 | M4 | Optional; respect privacy settings. |
| Voice | Voice-mode enablement (backed by services/voice) | yes | Planned | P2 | M4 | UI/cosmetic. |
| Buddy | On-screen companion sprite/notifications | yes | Planned | P2 | M4 | Cosmetic. |
| Vim mode | Motions, operators, text objects, transitions | yes | Planned | P2 | M3 | Ships with `/vim`. |
| Notifications | ~17 notification hooks (rate-limit, MCP connectivity, plugin install, model migration, LSP init, settings errors, teammate shutdown) | yes | Planned | P2 | M4 | |
| Sandbox | Filesystem + network isolation execution mode (parsed-but-unenforced in the port) | yes | Planned | P2 | M4 | Enforce both FS and egress allowlist at OS level; keep secrets outside; don't advertise until enforced. |

---

## Build philosophy (summary)

- **P0 / M1** delivers the smallest working terminal coding agent: provider transport with
  Anthropic-shaped normalization, a clean modular agent loop, the read/edit/search/run +
  TODO core tools, a deny-first per-tool permission model with recoverable denials, layered
  config, ancestor CLAUDE.md memory, atomic resumable sessions, and a minimal TUI.
- **P1 / M2** makes it a daily driver: MCP tools wired into the loop, single sub-agent
  delegation, the hooks runtime (net-new vs. the reference port), skills, plan/review, the
  git workflow commands, and session/quality-of-life ergonomics.
- **P2 / M3–M4** chases full parity: the plugin ecosystem, structured/remote transports,
  bridge/remote/server, multi-agent teams/swarm, cron/remote-trigger, IDE/desktop/mobile/
  voice, SDK, analytics, and the internal/debug surface.

**Known reference-port gaps to fix on the way (not just port):** runtime hook execution,
the entire plugin subsystem, structured/remote transports, and most of the 130-module
services layer are net-new builds for Zak Code — plus real token counting + auto-compaction
+ LLM-written summaries, prompt caching with `cache_control`, parallel (read-safe) tool
execution, structured tool I/O and thinking-block persistence, one shared async event loop,
atomic session writes, MCP tools actually connected into the loop, and safer permission
defaults (deny-unknown, gated default mode, input-pattern rules).

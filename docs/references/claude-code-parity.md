## Claude Code parity surface

This digest is reconstructed faithfully from `tools_snapshot.json`, `commands_snapshot.json`, the 29 subsystem files under `reference_data/subsystems/`, and `PARITY.md`. The snapshots list one entry per archived source module (e.g. `prompt.ts`, `UI.tsx`, `constants.ts`), so they over-count; below each tool/command directory is collapsed to its actual user-facing surface. Source counts from `archive_surface_snapshot.json`: 184 tool entries, 207 command entries, 1902 TS-like files.

---

### (1) Built-in TOOLS

Grouped by the `tools/<Dir>/` families in `tools_snapshot.json`. "Touches" columns are inferred from each tool's responsibility/name (FS = filesystem, NET = network, PROC = subprocess/process).

| Tool | One-line purpose | Key inputs / params (discernible) | FS | NET | PROC |
|------|------------------|-----------------------------------|----|-----|------|
| **AgentTool** (Task) | Spawn a sub-agent to autonomously run a sub-task; supports built-in agent types (general-purpose, explore, plan, verification, claude-code-guide, statusline-setup), fork/resume/run, agent memory + color/display management | `description`, `prompt`, `subagent_type` | indirect | indirect | indirect |
| **AskUserQuestionTool** | Pause and ask the user a structured multiple-choice question | `question`, `options/header` | no | no | no |
| **BashTool** | Run a shell command (with sandbox, permission, security, sed-edit, destructive-command, read-only & path validation) | `command`, `timeout`, `description`, `run_in_background` | yes | yes | yes |
| **PowerShellTool** | Windows PowerShell equivalent of Bash with CLM types, git-safety, common-parameters, security/permission validation | `command`, `timeout`, `description`, `run_in_background` | yes | yes | yes |
| **BriefTool** | Produce/upload a "brief" summary artifact with attachments | brief content, attachments | yes (upload) | yes | no |
| **ConfigTool** | Read/write supported CLI settings programmatically | `setting`, `value` (from supportedSettings) | yes (settings) | no | no |
| **EnterPlanModeTool** | Enter plan mode (model plans before acting) | (mode toggle) | no | no | no |
| **ExitPlanModeV2Tool** | Exit plan mode and present the plan for approval | `plan` | no | no | no |
| **EnterWorktreeTool** | Create/switch into a git worktree for isolated work | worktree/branch name | yes | no | yes (git) |
| **ExitWorktreeTool** | Leave/clean up the active git worktree | — | yes | no | yes (git) |
| **FileEditTool** (Edit) | Exact-string replace edit in a file | `file_path`, `old_string`, `new_string`, `replace_all` | yes | no | no |
| **FileReadTool** (Read) | Read a file incl. image processing, with size/line limits | `file_path`, `offset`, `limit` | yes | no | no |
| **FileWriteTool** (Write) | Create/overwrite a file | `file_path`, `content` | yes | no | no |
| **GlobTool** | Fast filename pattern matching | `pattern`, `path` | yes | no | no |
| **GrepTool** | Ripgrep-based content search | `pattern`, `glob/type`, `output_mode`, context flags | yes | no | no |
| **LSPTool** | Language-server queries: symbol context, formatters, schema-typed results | symbol/file/position | yes | no | yes (LSP proc) |
| **NotebookEditTool** | Edit Jupyter notebook cells | `notebook_path`, `cell_id`, `source`, `cell_type`, `edit_mode` | yes | no | no |
| **MCPTool** | Invoke a tool exposed by a connected MCP server (with collapse classification) | server/tool name + tool args | indirect | yes | indirect |
| **ListMcpResourcesTool** | List resources available from MCP servers | optional `server` | no | yes | no |
| **ReadMcpResourceTool** | Read a specific MCP resource | `server`, `uri` | no | yes | no |
| **McpAuthTool** | Authenticate to an MCP server (OAuth) | server, auth params | no | yes | no |
| **RemoteTriggerTool** | Trigger a remote agent/run | trigger target + payload | no | yes | no |
| **ScheduleCronTool** (CronCreate / CronDelete / CronList) | Create, delete, list scheduled cron jobs / routines | schedule expr, prompt/command, id | yes | yes | no |
| **SendMessageTool** | Send a message to another agent/teammate (multi-agent) | recipient, message | no | yes | no |
| **SkillTool** | Discover and execute a registered skill | `skill` name, `args` | yes | indirect | indirect |
| **SleepTool** | Pause execution for a duration | `duration` | no | no | no |
| **SyntheticOutputTool** | Emit synthetic/structured output into the transcript | output payload | no | no | no |
| **TaskCreateTool** | Create a background/tracked task | task spec/description | yes (state) | indirect | indirect |
| **TaskGetTool** | Fetch a task's status/details | `task_id` | yes (state) | no | no |
| **TaskListTool** | List tracked tasks | filters | yes (state) | no | no |
| **TaskOutputTool** | Retrieve a task's output | `task_id` | yes (state) | no | no |
| **TaskStopTool** | Stop/cancel a running task | `task_id` | yes (state) | no | yes |
| **TaskUpdateTool** | Update a task's fields/status | `task_id`, fields | yes (state) | no | no |
| **TeamCreateTool** | Create a multi-agent team | team spec | yes (state) | yes | no |
| **TeamDeleteTool** | Delete a multi-agent team | `team_id` | yes (state) | yes | no |
| **TodoWriteTool** | Maintain the in-session todo list | `todos[]` (content, status, activeForm) | yes (state) | no | no |
| **ToolSearchTool** | Search for / lazily load deferred tool schemas | `query`, `max_results` | no | no | no |
| **WebFetchTool** | Fetch a URL and process its content (with preapproved-host list) | `url`, `prompt` | no | yes | no |
| **WebSearchTool** | Web search returning results | `query`, allow/block domains | no | yes | no |
| **REPLTool** | REPL/primitive-tools execution surface (constants + primitiveTools) | code/primitives | yes | indirect | yes |
| **TestingPermissionTool** | Internal test-only permission tool | — | no | no | no |
| _shared: gitOperationTracking_ | Tracks git operations across tools (helper, not a model-facing tool) | — | yes | no | yes (git) |
| _shared: spawnMultiAgent_ | Multi-agent spawning helper backing Agent/Team/SendMessage (helper) | — | no | yes | yes |

---

### (2) SLASH COMMANDS

Collapsed from `commands_snapshot.json` (one row per command directory; install-github-app step components, plugin sub-views, and `index/*.tsx` duplicates folded in).

| Command | Purpose |
|---------|---------|
| `/add-dir` | Add another working directory to the session (with validation) |
| `/advisor` | Surface contextual advice/suggestions |
| `/agents` | Manage/select custom and built-in sub-agents |
| `/ant-trace` | Internal Anthropic tracing/diagnostics |
| `/autofix-pr` | Automatically fix issues on a pull request |
| `/backfill-sessions` | Backfill/migrate historical session records |
| `/branch` | Create/switch git branches |
| `/break-cache` | Invalidate prompt/response caches |
| `/bridge`, `/bridge-kick` | Manage the remote-control REPL bridge; kick/restart it |
| `/brief` | Generate a brief/summary artifact |
| `/btw` | Inject an out-of-band "by the way" note to the model |
| `/bughunter` | Run a bug-hunting agent pass |
| `/chrome` | Drive/connect Claude-in-Chrome integration |
| `/clear` | Clear conversation, caches, or both |
| `/color` | Set agent/UI color |
| `/commit`, `/commit-push-pr` | Create a git commit; commit + push + open a PR |
| `/compact` | Compact (summarize) conversation context |
| `/config` | Open the settings/config editor |
| `/context` | Show/visualize context window usage (interactive + noninteractive) |
| `/copy` | Copy conversation/output to clipboard |
| `/cost` | Show token/cost usage for the session |
| `/ctx_viz` | Context visualization (debug) |
| `/debug-tool-call` | Inspect/replay a tool call for debugging |
| `/desktop` | Desktop app integration |
| `/diff` | Show working-tree / change diff |
| `/doctor` | Diagnose installation/environment health |
| `/effort` | Set reasoning effort level |
| `/env` | Show/manage environment variables |
| `/exit` | Exit the CLI |
| `/export` | Export the conversation/session |
| `/extra-usage` | Manage/purchase extra usage (interactive + noninteractive) |
| `/fast` | Toggle fast mode |
| `/feedback` | Send feedback to Anthropic |
| `/files` | Browse/manage session files |
| `/good-claude` | Positive-reinforcement / reward signal |
| `/heapdump` | Dump JS heap for debugging |
| `/help` | Show help |
| `/hooks` | View/manage hooks configuration |
| `/ide` | IDE integration management |
| `/init`, `/init-verifiers` | Initialize CLAUDE.md / verifier setup for the repo |
| `/insights` | Show usage/work insights |
| `/install`, `/install-github-app`, `/install-slack-app` | Install the CLI / GitHub app (multi-step OAuth wizard) / Slack app |
| `/issue` | Create/triage an issue |
| `/keybindings` | View/edit keyboard shortcuts |
| `/login`, `/logout` | Authenticate / sign out (OAuth) |
| `/mcp` | Manage MCP servers (add, list, IdP/xaa auth) |
| `/memory` | View/edit CLAUDE.md memory |
| `/mobile` | Mobile companion integration |
| `/mock-limits`, `/reset-limits` | Mock / reset rate limits (testing) |
| `/model` | Choose the model |
| `/oauth-refresh` | Refresh OAuth tokens |
| `/onboarding` | Run first-run onboarding |
| `/output-style` | Choose output style |
| `/passes` | Manage "passes" (multi-pass workflows) |
| `/perf-issue` | File a performance issue |
| `/permissions` | View/edit tool permissions |
| `/plan` | Enter/manage plan mode |
| `/plugin`, `/reload-plugins` | Browse marketplaces, install/manage/validate/trust plugins; reload plugins |
| `/pr_comments` | Fetch/manage PR review comments |
| `/privacy-settings` | Manage privacy settings |
| `/rate-limit-options` | Configure rate-limit behavior options |
| `/release-notes` | Show release notes |
| `/remote-env`, `/remote-setup` | Configure remote environment / remote agent setup |
| `/rename` | Rename the session (auto-generated names) |
| `/resume` | Resume a previous conversation |
| `/review`, `/security-review` | Code review / ultrareview (+ remote); security-focused review |
| `/rewind` | Rewind conversation/checkpoint state |
| `/sandbox-toggle` | Toggle sandboxed execution |
| `/session` | Manage sessions |
| `/share` | Share a session/conversation |
| `/skills` | List/manage skills |
| `/stats` | Show statistics |
| `/status` | Show current status |
| `/statusline` | Configure the status line |
| `/stickers` | Stickers/easter-egg feature |
| `/summary` | Summarize the conversation |
| `/tag` | Tag the session/conversation |
| `/tasks` | View/manage background tasks |
| `/teleport` | Teleport/jump to a session or context |
| `/terminalSetup` | Configure terminal integration |
| `/theme` | Choose UI theme |
| `/thinkback`, `/thinkback-play` | Record/replay "thinkback" reasoning sessions |
| `/ultraplan` | Heavyweight multi-pass planning |
| `/upgrade` | Upgrade the CLI |
| `/usage` | Show usage details |
| `/version` | Show version |
| `/vim` | Toggle vim editing mode |
| `/voice` | Voice mode |
| _`createMovedToPluginCommand`_ | Internal shim that stubs commands relocated into plugins |

---

### (3) Major SUBSYSTEMS

From the subsystem snapshots (`module_count` shown) + PARITY.md descriptions.

- **assistant** (1) — `sessionHistory.ts`; the agentic conversation/session-history surface that the core tool loop runs against.
- **query / QueryEngine / Tool / Task / tools.ts** (root files) — the core query engine, tool abstraction, and task model that drive the agentic loop.
- **services** (130) — the largest backend layer. Families: `api/` (Anthropic client, bootstrap, admin requests, errors), `oauth/`, `mcp/` (connection manager + UI), `SessionMemory/` (persistent session memory + prompts), `analytics/` (datadog, growthbook, first-party event logging, sinks/killswitch), `PromptSuggestion/` (+ speculation), `AgentSummary/`, `MagicDocs/`, plus plugin operations, settings sync, policy/rate limits, team-memory sync, notifier, and voice. PARITY: only api/oauth/mcp core exist in the port; everything else is missing.
- **tools orchestration** (in services/tools) — `StreamingToolExecutor`, `toolExecution`, `toolOrchestration`, `toolHooks`; the layered streaming/tool-call/hook execution pipeline.
- **hooks** (104) — two distinct things: (a) the user-configurable **PreToolUse/PostToolUse** lifecycle hooks (schema in `schemas/hooks.ts`, `types/hooks.ts`; executed via `services/tools/toolHooks.ts`); and (b) a large body of React UI hooks — `toolPermission/` (permission context + interactive/coordinator/swarm-worker handlers + permission logging), `notifs/` (~17 notification hooks: rate-limit, MCP connectivity, plugin install/autoupdate, model migration, LSP init, settings errors, teammate shutdown, etc.), and file/unified suggestions. PARITY: config parsed but no runtime hook execution in the port.
- **plugins** (2) — `builtinPlugins.ts` + `bundled/index.ts`; plugin scaffolding. Backed by `services/plugins/` (PluginInstallationManager, pluginOperations), `types/plugin.ts`, and the `/plugin` marketplace UI (browse/add-marketplace/discover/manage/validate/trust/settings). Plugins can contribute hooks, tools, commands, and MCP servers. PARITY: entirely missing in the port.
- **skills** (20) — `loadSkillsDir.ts`, `bundledSkills.ts`, `mcpSkillBuilders.ts` plus bundled skills: batch, claudeApi(+content), claudeInChrome, debug, keybindings, loop, loremIpsum, remember, scheduleRemoteAgents, simplify, skillify, stuck, updateConfig, verify(+content). Skills are markdown-defined capabilities loaded from dirs + MCP. PARITY: port has local SKILL.md loading only — no bundled registry, no `/skills`, no MCP skill-builder.
- **memdir** (8) — memory directory: `findRelevantMemories`, memory scan/age/types, paths, and **team memory** (`teamMemPaths`, `teamMemPrompts`). Underpins persistent + team-shared memory.
- **cli** (19) — `structuredIO.ts`, `remoteIO.ts`, `print.ts`, `ndjsonSafeStringify.ts`, handler split (`handlers/auth|agents|mcp|plugins|autoMode|util`), and the **transports** stack (`HybridTransport`, `SSETransport`, `WebSocketTransport`, `SerialBatchEventUploader`, `WorkerStateUploader`, `ccrClient`). PARITY: port has none of the structured/remote transport layer.
- **bridge** (31) — the remote-control REPL bridge: bridge API/config/messaging/permission callbacks, JWT utils, poll config, inbound messages/attachments, capacity wake, code-session API, remote bridge core. Lets a remote controller drive a local REPL session.
- **remote** (4) — `RemoteSessionManager`, `SessionsWebSocket`, `remotePermissionBridge`, `sdkMessageAdapter`; remote session lifecycle + permission bridging over WS.
- **server** (3) — `createDirectConnectSession`, `directConnectManager`, types; direct-connect server sessions.
- **upstreamproxy** (2) — `relay.ts`, `upstreamproxy.ts`; an upstream API proxy/relay.
- **coordinator** (1) + **multi-agent** — `coordinatorMode.ts`; coordinator/swarm orchestration tying together Agent/Team/SendMessage tools and `toolPermission` swarm/coordinator handlers.
- **entrypoints** (8) — `cli.tsx`, `init.ts`, `mcp.ts`, sandbox types, and the **SDK** (`sdk/controlSchemas`, `coreSchemas`, `coreTypes`, `agentSdkTypes`); the programmatic agent SDK surface.
- **state** (6) — `AppState`, `AppStateStore`, store, selectors, onChange, teammate view helpers; central app state.
- **bootstrap** (1) — startup state bootstrapping.
- **keybindings** (14) — full keybinding engine: default + user bindings, parser, matcher, resolver, schema, validation, reserved shortcuts, display formatting.
- **vim** (5) — vim editing: motions, operators, text objects, transitions.
- **outputStyles** (1) — load output styles from a directory.
- **migrations** (11) — settings/model migrations (auto-update, bypass-permissions, model renames Opus/Sonnet versions, repl-bridge → remote-control, auto-mode opt-in resets).
- **buddy** (6) — the on-screen companion sprite/notifications (cosmetic).
- **voice** (1) — voice-mode enablement flag (UI feature backed by services/voice).
- **components** (389) / **screens** (3) — Ink/React TUI: App, dialogs (bridge, bypass-permissions, cost-threshold, oauth, channel-downgrade), context visualization, coordinator status, plugin hint menu; screens Doctor/REPL/ResumeConversation.
- **constants** (21) / **types** (11) / **schemas** (1) / **utils** (564) / **native-ts** (4) — shared constants (api/tool limits, betas, oauth, prompts, system-prompt sections), type defs (command, hooks, permissions, plugin, ids, generated protobuf event types), hook schema, a very large utility layer (Shell, Cursor, QueryGuard, agent context/id, swarm enablement, attachments, attribution, auth, ansi→png/svg, etc.), and native TS helpers (color-diff, file-index, yoga-layout).

---

### (4) Proposed PRIORITY TIERING for Zak Code

**Build philosophy:** P0 = the smallest set that makes a working terminal coding agent; P1 = what a daily driver needs (git workflow, MCP, sub-agents, hooks, sessions, web); P2 = full-parity / advanced (multi-agent teams, remote/bridge, cron, analytics, cosmetic, internal-only). Aligns with PARITY.md ("strong core loop; plugins/hooks-runtime/CLI-breadth/services are the big gaps").

#### Tools

| Tier | Tools | Rationale |
|------|-------|-----------|
| **P0** | FileReadTool, FileWriteTool, FileEditTool, GlobTool, GrepTool, BashTool, TodoWriteTool | The irreducible read/edit/search/run + task-tracking loop; nothing useful exists without these. |
| **P0 (platform)** | PowerShellTool | Required as the shell on the target Windows environment; co-P0 with Bash. |
| **P1** | AgentTool (general-purpose/explore/plan), WebFetchTool, WebSearchTool, EnterPlanModeTool, ExitPlanModeV2Tool, MCPTool, ListMcpResourcesTool, ReadMcpResourceTool, McpAuthTool, ConfigTool, SkillTool, NotebookEditTool, ToolSearchTool, AskUserQuestionTool | Daily-driver capabilities: sub-agent delegation, web access, plan mode, the MCP tool surface (port already has MCP transport), settings/skills, interactive questions, and tool lazy-loading. |
| **P2** | TaskCreate/Get/List/Output/Stop/Update, TeamCreateTool, TeamDeleteTool, SendMessageTool, RemoteTriggerTool, ScheduleCronTool (Cron*), EnterWorktreeTool, ExitWorktreeTool, BriefTool, LSPTool, REPLTool, SleepTool, SyntheticOutputTool, TestingPermissionTool, shared/spawnMultiAgent, shared/gitOperationTracking | Advanced/full-parity: background-task + multi-agent/team orchestration, remote triggers, cron scheduling, worktrees, LSP, and internal/cosmetic tools. None block a usable coding agent. |

#### Commands

| Tier | Commands | Rationale |
|------|----------|-----------|
| **P0** | `/help`, `/clear`, `/exit`, `/login`, `/logout`, `/model`, `/init`, `/memory`, `/config`, `/permissions`, `/compact`, `/cost`, `/diff`, `/resume`, `/version` | Minimum to authenticate, configure, run, manage context/cost, and recover sessions. (Port already implements most of these.) |
| **P1** | `/agents`, `/mcp`, `/skills`, `/hooks`, `/plan`, `/review`, `/security-review`, `/commit`, `/commit-push-pr`, `/branch`, `/context`, `/status`, `/session`, `/rename`, `/export`, `/doctor`, `/add-dir`, `/keybindings`, `/theme`, `/output-style`, `/effort`, `/vim`, `/usage`, `/stats`, `/feedback`, `/release-notes`, `/upgrade`, `/onboarding` | The git workflow, extensibility config (agents/mcp/skills/hooks), review/plan, session ergonomics, and quality-of-life that define a daily driver. |
| **P2** | `/plugin`, `/reload-plugins`, `/tasks`, `/bridge`, `/bridge-kick`, `/remote-env`, `/remote-setup`, `/chrome`, `/desktop`, `/mobile`, `/voice`, `/install-github-app`, `/install-slack-app`, `/install`, `/ide`, `/passes`, `/ultraplan`, `/thinkback`, `/thinkback-play`, `/rewind`, `/teleport`, `/share`, `/insights`, `/advisor`, `/autofix-pr`, `/bughunter`, `/issue`, `/pr_comments`, `/tag`, `/files`, `/copy`, `/statusline`, `/color`, `/stickers`, `/good-claude`, `/btw`, `/brief`, `/summary`, `/fast`, `/sandbox-toggle`, `/extra-usage`, `/rate-limit-options`, `/privacy-settings`, `/env`, `/terminalSetup`, `/oauth-refresh`, `/heapdump`, `/ant-trace`, `/ctx_viz`, `/debug-tool-call`, `/break-cache`, `/backfill-sessions`, `/mock-limits`, `/reset-limits`, `/init-verifiers`, `/insights` | Full-parity / advanced: plugin ecosystem, remote/bridge/IDE/desktop/mobile/voice integrations, GitHub/Slack install wizards, advanced planning/replay, billing/privacy, and internal debug/test commands. Defer until core + daily-driver are solid. |

#### Subsystems (build order, mirrors the command tiers)

- **P0:** core query/tool loop (assistant + QueryEngine + Tool), services/api + oauth, config/settings, CLAUDE.md memory (memdir basics), session persistence, TUI essentials (components/screens minimal).
- **P1:** services/mcp + mcp tools, AgentTool/coordinator (single sub-agent), **hooks runtime** (PreToolUse/PostToolUse — currently config-only per PARITY), skills registry + bundled skills, keybindings, output styles, SessionMemory, prompt suggestion.
- **P2:** plugins subsystem (loader + marketplace + services/plugins), cli transports/structuredIO/remoteIO, bridge, remote, server (direct-connect), upstreamproxy, multi-agent teams/swarm, ScheduleCron/RemoteTrigger, analytics, buddy/voice, migrations, entrypoints SDK.

**Key parity caveat (from PARITY.md):** the existing Rust port already covers the P0 tool/loop foundation but explicitly lacks runtime hook execution, any plugin subsystem, structured/remote transports, and most of the 130-module services layer — so for Zak Code those P1 hooks-runtime and P2 plugins/transports items are net-new builds, not ports.

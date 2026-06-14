# Configuration reference

Every `Settings` field, its environment variable, default, and meaning.

## Resolution order

Lowest → highest precedence; closer to the invocation wins, explicit env always wins:

| Layer | Where | Notes |
| --- | --- | --- |
| built-in defaults | the tables below | |
| user config home | `~/.zakcode/.env` | per-user, follows you to any directory (D20) |
| workspace `.env` | the invocation cwd | per-project; shadows the user file |
| process environment | real env vars | always wins over both files |
| explicit overrides | `load_settings(**kw)` / CLI flags | always wins |

The user config home is `~/.zakcode` (`%USERPROFILE%\.zakcode` on Windows); the
`ZAKCODE_HOME` env var overrides the directory (tests / portable installs). It is a
**config home only** — it is never treated as a workspace root. v1 contents: a
single `.env` file.

Provider API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`,
`TAVILY_API_KEY`) are deliberately **not** settings — litellm reads them from the
environment, which either `.env` populates. Put keys in `~/.zakcode/.env` once and
every `zakcode` invocation on the machine has them; a workspace `.env` overrides
per-project. `zakcode info` names each key's source (`env` / `workspace .env` /
`user .env`) so "why is it using that key on this machine" is always answerable.

A completeness test (`tests/test_config_docs.py`) asserts every field is documented
here — adding a Settings field without documenting it fails CI.

## Model / provider

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `default_model` | `ZAKCODE_DEFAULT_MODEL` | `ollama_chat/llama3.1` | Primary litellm model string (`provider/model`), or **`auto`** to resolve by availability at startup: local Ollama if up, else the first viable external per `auto_model_preference` (read-only probes — never a chat call; cached, re-probed on failure; tools-unreliable models skipped; nothing viable = loud startup failure with a per-source + key-provenance diagnosis). |
| `fallback_model` | `ZAKCODE_FALLBACK_MODEL` | unset | Model to switch to (once per turn) when the primary call fails with a non-rate-limit error. With `default_model=auto` it is the explicit override of the auto chain — tried before auto re-resolution. |
| `auto_model_preference` | `ZAKCODE_AUTO_MODEL_PREFERENCE` | `groq, openai, anthropic` | External provider order the `auto` resolver tries after local (comma/space/JSON list). |
| `model_roles` | `ZAKCODE_MODEL_ROLES` | `{}` | Per-role overrides (JSON; keys `planner` / `subagent` / `summarizer`) so cheap roles can use a cheap model. |
| `temperature` | `ZAKCODE_TEMPERATURE` | `0.0` | Sampling temperature, 0.0–2.0. |
| `tool_calling_mode` | `ZAKCODE_TOOL_CALLING_MODE` | `auto` | `auto` \| `native` \| `text` — how tools reach the model; `auto` self-resolves per provider. |
| `ollama_base_url` | `ZAKCODE_OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint. |
| `api_base` | `ZAKCODE_API_BASE` | unset | Any OpenAI-compatible endpoint override (llama.cpp / BitNet / vLLM / LM Studio). |
| `api_key` | `ZAKCODE_API_KEY` | unset | Placeholder key for local servers that require one; never a real cloud key (those use the standard env vars). Excluded from every `model_dump()`. |
| `provider_max_retries` | `ZAKCODE_PROVIDER_MAX_RETRIES` | `3` | Retries (with `retry_after`-aware backoff) after a rate-limited model call; `0` disables. Only 429s retry. |

## Agent behavior & permissions

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `max_iterations` | `ZAKCODE_MAX_ITERATIONS` | `50` | Hard cap on agent-loop iterations per turn. |
| `max_cost_usd` | `ZAKCODE_MAX_COST_USD` | _(unset)_ | Stop the turn (and its whole sub-agent tree) once cumulative model cost in USD reaches this ceiling (`stop_reason="budget_exhausted"`). Unset = no cost bound. |
| `max_tokens` | `ZAKCODE_MAX_TOKENS` | _(unset)_ | Stop the turn-tree once cumulative total tokens reach this ceiling (`stop_reason="budget_exhausted"`). Unset = no token bound. A cumulative spend guard, not a per-call output cap. |
| `turn_end_veto_budget` | `ZAKCODE_TURN_END_VETO_BUDGET` | `0` | Max times per turn a `TURN_END` hook may veto a vetoable stop (`completed` / `doom_loop` / `stuck`) and re-enter the loop with its continuation prompt (the Claude-Code-Stop-hook seam). `0` disables the gate entirely. `max_iterations` / `budget_exhausted` / `provider_error` / `recipe_stalled` are never vetoable. |
| `permission_mode` | `ZAKCODE_PERMISSION_MODE` | `ask` | `ask` \| `acceptEdits` \| `allow` \| `autonomous` \| `deny`. `autonomous` never prompts; catastrophic commands hard-deny. |
| `tool_trust_overrides` | `ZAKCODE_TOOL_TRUST_OVERRIDES` | `{}` | Per-tool mode overrides (JSON, tool → mode), loosen or tighten. Cannot loosen the dangerous floor in an autonomous session. |
| `subprocess_inherit_provider_keys` | `ZAKCODE_SUBPROCESS_INHERIT_PROVIDER_KEYS` | `false` | When false (default), `*_API_KEY` vars are scrubbed from bash/powershell children. |
| `dependency_gate` | `ZAKCODE_DEPENDENCY_GATE` | `true` | When true (default), a shell command that installs a package the project's manifests/lockfile don't declare (pip/uv/poetry/npm…) escalates to a prompt — and hard-denies in `autonomous`. Tighten-only; `uv sync`/`npm ci`/declared/editable installs pass through. See [SELF-REMEDIATION.md](SELF-REMEDIATION.md). |
| `denied_commands` | `ZAKCODE_DENIED_COMMANDS` | `[]` | Extra deny regexes appended to the dangerous-command blocklist (newline-separated or JSON array); tighten-only. |
| `protected_paths` | `ZAKCODE_PROTECTED_PATHS` | `[]` | Extra protected-path regexes appended to the built-in floor (`.git/`, `.env`, the venv, `.claude/`). A write matching one escalates to a prompt — and hard-denies in `autonomous` — even under `allow`/`acceptEdits` or a grant. Tighten-only. See [SELF-REMEDIATION.md](SELF-REMEDIATION.md) Step 2. |
| `tool_exposure_allow` | `ZAKCODE_TOOL_EXPOSURE_ALLOW` | `[]` | Per-task tool filter (Step 4). If non-empty, ONLY tools whose canonical name matches one of these globs are exposed to the model (and invocable). Empty = no allow restriction. Comma/space-separated or JSON. |
| `tool_exposure_deny` | `ZAKCODE_TOOL_EXPOSURE_DENY` | `[]` | Tool-name globs NEVER exposed to the model (wins over the allow list), e.g. `bash,powershell,mcp__*`. Narrows attack surface (a tool the model can't see can't be hijacked by injected content); exposure-only, never loosens the permission gate. See [SELF-REMEDIATION.md](SELF-REMEDIATION.md) Step 4. |
| `workspace_root` | `ZAKCODE_WORKSPACE_ROOT` | current dir | Root directory the agent operates within. |

## Web tools & egress

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `search_backend` | `ZAKCODE_SEARCH_BACKEND` | `ddgs` | `ddgs` (free, no key) \| `tavily` (needs `TAVILY_API_KEY`) \| `searxng`. |
| `searxng_url` | `ZAKCODE_SEARXNG_URL` | unset | Self-hosted SearXNG base URL (when `search_backend=searxng`). |
| `web_allowed_domains` | `ZAKCODE_WEB_ALLOWED_DOMAINS` | `[]` | When non-empty, `web_fetch` may only reach these domains (+ subdomains), enforced per redirect hop. |
| `web_fetch_confirm` | `ZAKCODE_WEB_FETCH_CONFIRM` | `false` | Escalate every `web_fetch` to a confirmation prompt (denied outright in `deny`/`autonomous`). |
| `egress_proxy` | `ZAKCODE_EGRESS_PROXY` | `false` | Route bash/powershell egress through a localhost domain-allowlisting proxy. |
| `egress_allowed_domains` | `ZAKCODE_EGRESS_ALLOWED_DOMAINS` | `[]` | Domains the egress proxy permits; empty + proxy on = deny all subprocess egress. |

## Cross-session memory

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `settings_hooks` | `ZAKCODE_SETTINGS_HOOKS` | `false` | Load shell hooks from `<workspace>/.claude/settings.json` + `.zakcode/settings.json` (event names mapped; `Stop` → `TurnEnd`; dangerous commands hard-denied in autonomous mode; provider keys scrubbed from hook env). Off by default so workspaces configured for other hook runtimes don't half-fire here. |
| `memory_db_path` | `ZAKCODE_MEMORY_DB_PATH` | `<workspace>/.zakcode/memory.db` | SQLite store for opt-in cross-session memory. |
| `memory_recall_limit` | `ZAKCODE_MEMORY_RECALL_LIMIT` | `5` | Memories the recall hook injects per turn. |
| `memory_recall_min_overlap` | `ZAKCODE_MEMORY_RECALL_MIN_OVERLAP` | `1` | Distinctive-word overlap floor for auto-recall; `0` disables the floor. |

## HTTP server (`zakcode serve`)

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `auth_token` | `ZAKCODE_AUTH_TOKEN` | unset | Bearer token required on every route except `/health` when set; unset = loopback-only dev (non-loopback bind needs `--insecure`). Excluded from every `model_dump()`. |
| `allowed_models` | `ZAKCODE_ALLOWED_MODELS` | `[]` | When non-empty, the only model strings a request may override to. |

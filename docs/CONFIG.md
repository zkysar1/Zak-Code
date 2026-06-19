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
| `default_model` | `ZAKCODE_DEFAULT_MODEL` | `ollama_chat/llama3.1` | Primary litellm model string (`provider/model`); **`auto`** to resolve by availability at startup (local Ollama if up, else the first viable external per `auto_model_preference` — read-only probes, cached + re-probed on failure, tools-unreliable models skipped, nothing viable = loud startup failure); or **`zakpick`** to route each prompt to the model you assigned to its task **category** (see `zakpick_models`). |
| `zakpick_models` | `ZAKCODE_ZAKPICK_MODELS` | `{}` | Per-category model assignments for `default_model=zakpick` (inert otherwise). JSON object; keys `quick_code` / `deep_code` / `summarize` / `plan` / `delegate` / `classify`; each value `{model, source}` where `source` defaults to `groq` (use `local` for Ollama, or any litellm prefix like `openai`/`anthropic`). Unset categories use built-in Groq defaults (so it works out of the box; the defaults also tell you which open-source model to download to run a category locally). Zak Code never substitutes a model you didn't assign and owns no local/cloud tradeoff — a slow local model is slow; a failing cloud model uses `fallback_model` like any other. The cheap **quick_code** vs capable **deep_code** split for the main turn is chosen automatically per turn (short/easy → quick; long/hard or on a struggle signal → deep). Example: `ZAKCODE_ZAKPICK_MODELS={"deep_code":{"model":"qwen3:32b","source":"local"},"plan":{"model":"gpt-4o","source":"openai"}}`. |
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
| `completion_review_attempts` | `ZAKCODE_COMPLETION_REVIEW_ATTEMPTS` | `0` | When a turn CHANGED code (wrote a runnable file) and the model tries to finish, send it back this many times to re-read the request and verify every requirement against what is actually on disk — and finish any abandoned/failed operation — before completing. Bounded so it converges (an unbounded "don't finish until perfect" loops forever on a model that can't reach it). Scoped to **complex** (non-`quick_code`) turns under zakpick, so it never slows a simple one-line fix. `0` (default) disables it; `2` is a good value for higher autonomous quality on hard, multi-part tasks. |
| `trace_dir` | `ZAKCODE_TRACE_DIR` | _(unset)_ | If set, write a structured per-turn JSONL **decision trace** to this directory — one `turn_<n>.jsonl` per turn recording how the loop routed and every gate/recovery intervention it fired, ending with the stop (observability; complements the session transcript). Best-effort: a write error never affects the turn. Unset = no trace files (the trace is still attached to the in-memory `TurnResult`/`AgentDone`). |
| `permission_mode` | `ZAKCODE_PERMISSION_MODE` | `ask` | `ask` \| `acceptEdits` \| `allow` \| `autonomous` \| `deny`. `autonomous` never prompts; catastrophic commands hard-deny. |
| `tool_trust_overrides` | `ZAKCODE_TOOL_TRUST_OVERRIDES` | `{}` | Per-tool mode overrides (JSON, tool → mode), loosen or tighten. Cannot loosen the dangerous floor in an autonomous session. |
| `subprocess_inherit_provider_keys` | `ZAKCODE_SUBPROCESS_INHERIT_PROVIDER_KEYS` | `false` | When false (default), `*_API_KEY` vars are scrubbed from bash/powershell children. |
| `dependency_gate` | `ZAKCODE_DEPENDENCY_GATE` | `true` | When true (default), a shell command that installs a package the project's manifests/lockfile don't declare (pip/uv/poetry/npm…) escalates to a prompt — and hard-denies in `autonomous`. Tighten-only; `uv sync`/`npm ci`/declared/editable installs pass through. See [SELF-REMEDIATION.md](SELF-REMEDIATION.md). |
| `denied_commands` | `ZAKCODE_DENIED_COMMANDS` | `[]` | Extra deny regexes appended to the dangerous-command blocklist (newline-separated or JSON array); tighten-only. |
| `protected_paths` | `ZAKCODE_PROTECTED_PATHS` | `[]` | Extra protected-path regexes appended to the built-in floor (`.git/`, `.env`, the venv, `.claude/`). A write matching one escalates to a prompt — and hard-denies in `autonomous` — even under `allow`/`acceptEdits` or a grant. Tighten-only. See [SELF-REMEDIATION.md](SELF-REMEDIATION.md) Step 2. |
| `verify_command` | `ZAKCODE_VERIFY_COMMAND` | _(unset)_ | Shell command that verifies the workspace (e.g. `uv run poe check`, `pytest -q`, `npm test`). When set, a turn that **changed code** may not finish until this command passes — the harness runs it itself when it would auto-allow (allow/autonomous or a prior grant), else nudges the model; after a bounded number of attempts a still-failing turn ends `verification_failed` (degraded). Domain-agnostic: the engine never guesses the command. Unset = no project gate (the always-on recipe gate that verifies a freshly written script still applies). |
| `require_plan` | `ZAKCODE_REQUIRE_PLAN` | `false` | Opt-in "plan before you act": when true, the harness won't run a **mutating** tool (write/edit/shell) until the model has laid out a plan with `update_plan`. Read-only investigation is never gated, and the gate is bounded (after a couple of nudges the action runs anyway — fail-open, never deadlocks). Off by default because forcing a plan on trivial turns is counterproductive. |
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

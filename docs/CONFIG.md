# Configuration reference

Every `Settings` field, its environment variable, default, and meaning. Resolution
order (highest precedence first): explicit overrides → `ZAKCODE_*` environment
variables → the project `.env` (loaded by `load_settings()`; existing environment
always wins) → the defaults below. Provider API keys (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`) are deliberately **not**
settings — litellm reads them from the environment, which `.env` populates.

A completeness test (`tests/test_config_docs.py`) asserts every field is documented
here — adding a Settings field without documenting it fails CI.

## Model / provider

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `default_model` | `ZAKCODE_DEFAULT_MODEL` | `ollama_chat/llama3.1` | Primary litellm model string (`provider/model`). |
| `fallback_model` | `ZAKCODE_FALLBACK_MODEL` | unset | Model to retry with if the primary errors (wiring is internal-package scope — audit P0-3b). |
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
| `permission_mode` | `ZAKCODE_PERMISSION_MODE` | `ask` | `ask` \| `acceptEdits` \| `allow` \| `autonomous` \| `deny`. `autonomous` never prompts; catastrophic commands hard-deny. |
| `tool_trust_overrides` | `ZAKCODE_TOOL_TRUST_OVERRIDES` | `{}` | Per-tool mode overrides (JSON, tool → mode), loosen or tighten. Cannot loosen the dangerous floor in an autonomous session. |
| `subprocess_inherit_provider_keys` | `ZAKCODE_SUBPROCESS_INHERIT_PROVIDER_KEYS` | `false` | When false (default), `*_API_KEY` vars are scrubbed from bash/powershell children. |
| `denied_commands` | `ZAKCODE_DENIED_COMMANDS` | `[]` | Extra deny regexes appended to the dangerous-command blocklist (newline-separated or JSON array); tighten-only. |
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
| `memory_db_path` | `ZAKCODE_MEMORY_DB_PATH` | `<workspace>/.zakcode/memory.db` | SQLite store for opt-in cross-session memory. |
| `memory_recall_limit` | `ZAKCODE_MEMORY_RECALL_LIMIT` | `5` | Memories the recall hook injects per turn. |
| `memory_recall_min_overlap` | `ZAKCODE_MEMORY_RECALL_MIN_OVERLAP` | `1` | Distinctive-word overlap floor for auto-recall; `0` disables the floor. |

## HTTP server (`zakcode serve`)

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `auth_token` | `ZAKCODE_AUTH_TOKEN` | unset | Bearer token required on every route except `/health` when set; unset = loopback-only dev (non-loopback bind needs `--insecure`). Excluded from every `model_dump()`. |
| `allowed_models` | `ZAKCODE_ALLOWED_MODELS` | `[]` | When non-empty, the only model strings a request may override to. |

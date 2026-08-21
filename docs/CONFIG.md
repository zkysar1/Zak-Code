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

`zakcode info` names the source for the settings that decide **cost and routing** too —
`api_base`, `permission_mode`, `local_only` — annotating anything that came from a file
with `(from user .env)` / `(from workspace .env)`. A real environment variable always
outranks `~/.zakcode/.env`, so a value edited in the user file may never have applied;
the annotation makes that shadowing visible instead of silent. `info` also reports
`local_only`, `local_api_bases`, `extra_body`, and `extra_headers` (header **names**
only — never values, per [GUARDRAILS](GUARDRAILS.md)), because those fail quietly: an
older build has no such field at all, and every other row still reads correct.

`zakcode version` reports the commit for a VCS install — `0.0.1 (git ab2c313e0482)` —
recovered from PEP 610 install metadata. The `version` string in `pyproject.toml` is
hand-maintained, so without this two checkouts weeks apart are indistinguishable from
inside the process.

A completeness test (`tests/test_config_docs.py`) asserts every field is documented
here — adding a Settings field without documenting it fails CI.

## Model / provider

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `default_model` | `ZAKCODE_DEFAULT_MODEL` | `ollama_chat/llama3.1` | Primary litellm model string (`provider/model`); **`auto`** to resolve by availability at startup (local Ollama if up, else the first viable external per `auto_model_preference` — read-only probes, cached + re-probed on failure, tools-unreliable models skipped, nothing viable = loud startup failure); or **`zakpick`** to route each prompt to the model you assigned to its task **category** (see `zakpick_models`). |
| `zakpick_models` | `ZAKCODE_ZAKPICK_MODELS` | `{}` | Per-category model assignments for `default_model=zakpick` (inert otherwise). JSON object; keys `quick_code` / `deep_code` / `summarize` / `plan` / `delegate` / `classify`; each value `{model, source}` where `source` defaults to `groq` (use `local` for Ollama, or any litellm prefix like `openai`/`anthropic`). Unset categories use built-in Groq defaults (so it works out of the box; the defaults also tell you which open-source model to download to run a category locally). Zak Code never substitutes a model you didn't assign and owns no local/cloud tradeoff — a slow local model is slow; a failing cloud model uses `fallback_model` like any other. The cheap **quick_code** vs capable **deep_code** split for the main turn is chosen automatically per turn (short/easy → quick; long/hard or on a struggle signal → deep). Example: `ZAKCODE_ZAKPICK_MODELS={"deep_code":{"model":"qwen3:32b","source":"local"},"plan":{"model":"gpt-4o","source":"openai"}}`. |
| `fallback_model` | `ZAKCODE_FALLBACK_MODEL` | unset | Model to switch to (once per turn) when the primary call fails with a non-rate-limit error. With `default_model=auto` it is the explicit override of the auto chain — tried before auto re-resolution. |
| `auto_model_preference` | `ZAKCODE_AUTO_MODEL_PREFERENCE` | `groq, openai, anthropic` | External provider order the `auto` resolver tries after local (comma/space/JSON list). |
| `model_roles` | `ZAKCODE_MODEL_ROLES` | `{}` | Per-role overrides (JSON; keys `planner` / `subagent` / `summarizer` / `judge`) so cheap roles can use a cheap model. |
| `temperature` | `ZAKCODE_TEMPERATURE` | `0.0` | Sampling temperature, 0.0–2.0. |
| `tool_calling_mode` | `ZAKCODE_TOOL_CALLING_MODE` | `auto` | `auto` \| `native` \| `text` — how tools reach the model; `auto` self-resolves per provider. |
| `ollama_base_url` | `ZAKCODE_OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint. |
| `api_base` | `ZAKCODE_API_BASE` | unset | Any OpenAI-compatible endpoint override (llama.cpp / BitNet / vLLM / LM Studio). |
| `api_key` | `ZAKCODE_API_KEY` | unset | Placeholder key for local servers that require one; never a real cloud key (those use the standard env vars). Excluded from every `model_dump()`. |
| `extra_body` | `ZAKCODE_EXTRA_BODY` | `{}` | JSON merged verbatim into every completion request body (litellm `extra_body`). The escape hatch for server-specific knobs litellm has no first-class parameter for — notably llama.cpp thinking control: `ZAKCODE_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":false}}`. Thinking tokens are billed against `max_tokens`, so switching it off on bounded work is a real saving (measured on Qwen3.8-27B: 36 completion tokens → 4, same answer). Per-**category** control is `zakpick_models[...].thinking`, which merges over this. There is no per-request thinking *depth*: a `reasoning_budget` in the body is ignored by llama.cpp (measured — 64 and 256 both produced ~13k reasoning chars against a 13,130 baseline); it is a server startup flag only. |
| `extra_headers` | `ZAKCODE_EXTRA_HEADERS` | `{}` | HTTP headers added to every completion request (litellm `extra_headers`). Values may contain `{hostname}` and `{pid}`, expanded per process — which is what lets ONE config line identify an arbitrary number of terminals: `ZAKCODE_EXTRA_HEADERS={"X-ZDS-Instance":"{hostname}-{pid}"}` makes each Zak Code process report to a self-hosted pod as its own caller (`zak-code/cc-04-40213`) while every terminal shares a single provisioned API key. Without the placeholders you would need one distinct value per terminal — a registration step before every launch, which in practice means they all share one label and attribution collapses to a single row. An unknown `{placeholder}` or a stray brace passes through untouched: a header value is not a format string, and a typo must not take inference down. |
| `local_only` | `ZAKCODE_LOCAL_ONLY` | `false` | **Cost guarantee.** Refuse any call that would reach a metered API rather than degrading to one — for running many agents against your own hardware, where a silent failover to a paid provider is the thing you cannot allow. "Local" means no third-party billing, not "on this machine": `ollama_chat/*`, or a generic-OpenAI model (`openai/…`, a bare name, `openai_like/…`, `hosted_vllm/…`) served through `api_base`. A named cloud prefix (`groq/`, `anthropic/`) never qualifies, because `api_base` cannot redirect those. Enforced twice: at startup, naming every offending field at once — including zakpick categories you did **not** override, which otherwise fall through to Groq defaults — and again at each request, so no failover or auto-resolution path slips past. Raises `LocalOnlyViolation`, deliberately not a `ProviderError`, so nothing retries it onto another model. **Boundary:** the check is by model prefix, so ANY `api_base` is trusted as local — including a **gateway** that forwards to metered providers. Constrain it with `local_api_bases`. |
| `local_api_bases` | `ZAKCODE_LOCAL_API_BASES` | *(empty)* | Comma/space-separated `api_base` values that count as genuinely local under `local_only`. Empty (the default) trusts any base, so nothing that works today starts refusing. Set it and an `api_base` outside the list is treated as **metered** and refused. Needed because a self-hosted pod and an LLM gateway both speak the generic OpenAI protocol — `openai/<model>` + `api_base` cannot distinguish "my hardware" from "a proxy that bills me", and only you know which is which. Matching ignores trailing slashes and case. |
| `provider_max_retries` | `ZAKCODE_PROVIDER_MAX_RETRIES` | `3` | Retries (with `retry_after`-aware backoff) after a rate-limited model call; `0` disables. Only 429s retry. |

### Recipe: running against a self-hosted inference pod

A self-hosted OpenAI-compatible server (llama.cpp, vLLM, or a routing proxy in front of
several of them) is reached with the `openai/` prefix plus `api_base`. The prefix selects
litellm's generic OpenAI wire protocol; `api_base` decides which host actually receives the
request. The model name is whatever alias your server advertises at `/v1/models`.

**One model, guaranteed no spend** — the many-agents case:

```bash
ZAKCODE_DEFAULT_MODEL=openai/zds-qwen3.8-27b
ZAKCODE_API_BASE=http://zakpod1:9090/v1
ZAKCODE_API_KEY=sk-noop          # most local servers ignore it, litellm wants one present
ZAKCODE_LOCAL_ONLY=true          # refuse anything that would bill
ZAKCODE_LOCAL_API_BASES=http://zakpod1:9090/v1   # ...and only THIS base counts as local
```

With `local_only`, a metered call is refused rather than made — including one reached by
runtime failover. Leave `fallback_model` unset, or point it at another local model.

Add `local_api_bases` whenever an LLM **gateway** is reachable on your network. `local_only`
classifies by model prefix, so `openai/<model>` pointed at a gateway passes the guard even
though the gateway may forward to deepinfra/groq/OpenAI and bill you. Listing your real
endpoints closes that: an unlisted base raises `LocalOnlyViolation` before the call.

**Mixed: local for the heavy work, cloud fallback allowed** — omit `local_only`. Both lanes
coexist under one `api_base`, because the base is forwarded only to generic-OpenAI models
and never to a named cloud prefix:

```bash
ZAKCODE_DEFAULT_MODEL=openai/zds-qwen3.8-27b
ZAKCODE_API_BASE=http://zakpod1:9090/v1
ZAKCODE_FALLBACK_MODEL=groq/qwen/qwen3-32b   # reached only if the pod call fails
```

**Per-category routing with per-category thinking** — `zakpick` sends each task category to
the model you assign it, and `thinking` controls reasoning per category. Reasoning tokens
are billed against `max_tokens`, so the bounded categories are much faster with it off:

```bash
ZAKCODE_DEFAULT_MODEL=zakpick
ZAKCODE_API_BASE=http://zakpod1:9090/v1
ZAKCODE_ZAKPICK_MODELS={"deep_code":{"model":"zds-qwen3.8-27b","source":"openai","thinking":true},"quick_code":{"model":"zds-qwen3.8-27b","source":"openai","thinking":false},"classify":{"model":"zds-qwen3.8-27b","source":"openai","thinking":false},"summarize":{"model":"zds-qwen3.8-27b","source":"openai","thinking":false},"plan":{"model":"zds-qwen3.8-27b","source":"openai","thinking":true},"delegate":{"model":"zds-qwen3.8-27b","source":"openai","thinking":true}}
```

Set **every** category when combining `zakpick` with `local_only`: an unset category falls
through to a built-in Groq/OpenAI default rather than to your pod. The startup check names
each one it finds, so a partial config fails immediately instead of billing later.

Billing is not the only cost of a missed category, and the other one slips past `local_only`
entirely. The `delegate` default is `openai/gpt-4o-mini` — a *generic-OpenAI* model, so with
`api_base` set it is routed to your pod, `local_only` classifies it as local (correctly: no
money is spent), and the check passes. Your pod then does not host `gpt-4o-mini`, and a
lenient proxy answers with whatever it does host. Observed on `zakpod1` 2026-08-17:
`SUBSTITUTED requested model='openai/gpt-4o-mini' -> serving 'zds-qwen3.8-27b'`. The two
guards are orthogonal and you want both — `local_only` promises nothing gets billed,
`ZDS_STRICT_MODEL=1` promises you get the model you asked for. Neither implies the other.

`api_base` needs a host this process can actually resolve. An SSH-config `Host` alias is not
DNS — `ssh zakpod1` working says nothing about `http://zakpod1:9090/v1`, which fails with a
bare `Connection error` that reads like the server being down. Use the IP, or add a real
hosts/DNS entry.

Two failure modes worth recognising, both of which look like something else:

- **A confident answer from the wrong model.** If you ask for a model your server does not
  host, a lenient proxy may substitute its default and answer normally. Check `/v1/models`
  for the exact alias; on `zds-inference-server`, `ZDS_STRICT_MODEL=1` turns the
  substitution into a 404 and every substitution is logged either way.
- **An empty reply that took minutes.** Thinking tokens count against `max_tokens`, so a
  reasoning model can spend the whole budget thinking and return nothing — `finish_reason`
  is `length` and the content is empty. Raise `max_tokens` or set `thinking: false`
  (measured on Qwen3.8-27B: a 3000-token budget was fully consumed by reasoning, 0 answer).

### Recipe: Vertex AI (Gemini) on Google Cloud

Install the optional extra first — litellm's vertex path imports Google's auth/SDK stack,
which base zakcode deliberately does not carry (`~30` google-cloud packages non-Vertex
users never need). Without it the first call fails with an install hint (not a raw
`ModuleNotFoundError` — see `providers/litellm_provider._map_error`):

```bash
uv tool install 'zakcode[google]'        # or: uv add 'zakcode[google]' in a project
```

Three settings, no key file:

```bash
ZAKCODE_DEFAULT_MODEL=vertex_ai/gemini-2.5-flash
VERTEXAI_PROJECT=my-gcp-project          # litellm's own env names, not ZAKCODE_*
VERTEXAI_LOCATION=us-central1
```

Auth is **ADC** (Application Default Credentials): on a GCE VM the attached service
account authenticates via the metadata server, so there is no credential on disk at all —
`zakcode info` correctly shows every `*_API_KEY` as *not set*. Two gotchas measured live
(2026-08-18/19): model availability is **per-region and per-model** (a newer model can be
`global`-endpoint-only and 404 on every regional endpoint while an older one serves fine —
enumerate with `gcloud ai model-garden models list --project=<p>` rather than guessing),
and a 404 `publisher model not found` means *reached and authenticated, wrong model id* —
not a permissions or networking problem.

## Agent behavior & permissions

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `max_iterations` | `ZAKCODE_MAX_ITERATIONS` | `50` | Hard cap on agent-loop iterations per turn. |
| `skill_invocation_budget` | `ZAKCODE_SKILL_INVOCATION_BUDGET` | `0` | Max model-driven `use_skill` invocations per turn, shared across the whole sub-agent tree (one counter, reset each top-level turn) — a tight bound on a runaway or cyclic skill chain. A human `/<name>` invocation is operator-controlled and never throttled. `0` = unlimited (off). |
| `context_include_readme` | `ZAKCODE_CONTEXT_INCLUDE_README` | `true` | Fold the workspace `README.md` into the agent's project-context block, alongside the discovered `AGENTS.md` / `CLAUDE.md` / `ZAK.md` agent guides (which are always loaded). README is human-facing and can be large, so it is read only at the workspace root and bounded by the per-file/total context caps; set `false` to load only the agent guides. |
| `lean_rules` | `ZAKCODE_LEAN_RULES` | `false` | Render mind rules as a compact **index** (titles only) instead of full bodies, so the model pulls a rule's body on demand via `read_file` when a summary is relevant. Default off (full render) for parity; set true for token-constrained deployments (e.g. the Vinheim research runtime). Only takes effect when the agent is constructed with `enable_rules=True`. |
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

## Quality engine

The small-model fan-out engine (`src/zakcode/quality/`) wired into the loop (increment 6). **All OFF by default** — the default path is byte-identical; each seam is bounded (by `best_of_attempts` and the per-turn budget) and fail-safe. Quality calls route to `model_roles['judge']`.

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `quality_gate` | `ZAKCODE_QUALITY_GATE` | `false` | Seam A: after the verifier passes, score the turn result on a rubric and refine if it falls short — runs ALONGSIDE the binary completion critic (two independent quality checks). Off = today's behavior. |
| `quality_gate_threshold` | `ZAKCODE_QUALITY_GATE_THRESHOLD` | `0.8` | Seam A ship threshold (overall rubric score, 0–1). |
| `quality_gate_dimensions` | `ZAKCODE_QUALITY_GATE_DIMENSIONS` | _(unset)_ | Seam A rubric (JSON, dimension → what to assess); unset uses a built-in code rubric. |
| `best_of_attempts` | `ZAKCODE_BEST_OF_ATTEMPTS` | `1` | Seam B: fan out N attempts at a stalled step and select the best by the verifier (`1` = off). |

## Web tools & egress

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `search_backend` | `ZAKCODE_SEARCH_BACKEND` | `ddgs` | `ddgs` (free, no key) \| `tavily` (needs `TAVILY_API_KEY`) \| `searxng`. |
| `searxng_url` | `ZAKCODE_SEARXNG_URL` | unset | Self-hosted SearXNG base URL (when `search_backend=searxng`). |
| `web_allowed_domains` | `ZAKCODE_WEB_ALLOWED_DOMAINS` | `[]` | When non-empty, `web_fetch` may only reach these domains (+ subdomains), enforced per redirect hop. |
| `web_fetch_confirm` | `ZAKCODE_WEB_FETCH_CONFIRM` | `false` | Escalate every `web_fetch` to a confirmation prompt (denied outright in `deny`/`autonomous`). |
| `egress_proxy` | `ZAKCODE_EGRESS_PROXY` | `false` | Route bash/powershell egress through a localhost domain-allowlisting proxy. |
| `egress_allowed_domains` | `ZAKCODE_EGRESS_ALLOWED_DOMAINS` | `[]` | Domains the egress proxy permits; empty + proxy on = deny all subprocess egress. |

## Settings ingestion (Claude Code `.claude/settings.json`)

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `settings_hooks` | `ZAKCODE_SETTINGS_HOOKS` | *unset* | Load shell hooks from `<workspace>/.claude/settings.json` + `.claude/settings.local.json` + `.zakcode/settings.json` (event names mapped; `Stop` → `TurnEnd`; dangerous commands hard-denied in autonomous mode; provider keys scrubbed from hook env). **Tri-state.** An explicit `true`/`false` is the operator's global answer and is honored silently everywhere. *Unset* means: **off** for the library/server (a workspace carrying another runtime's hooks never half-fires here by surprise), while the **interactive CLI asks once per workspace and remembers** — Claude Code folder-trust semantics. The question fires only when the workspace actually declares loadable hooks; the answer (`always`/`never`) persists in `~/.zakcode/workspace-trust.json` (`session` does not persist). Headless runs (`chat -p`, no tty) never prompt: they print a one-line pointer and stay off, so scripted behavior is deterministic. Policy + persistence live in `zakcode.workspace_trust` (core); the CLI only renders the question. Per-Agent override: `Agent(enable_settings_hooks=…)`. |
| `settings_permissions` | `ZAKCODE_SETTINGS_PERMISSIONS` | `false` | Translate Claude Code's `permissions.{allow,deny,ask}` `Tool(pattern)` gestures from `<workspace>/.claude/settings.json` + `settings.local.json` into the deny-first permission policy (a `deny Bash(glob)` → a command-deny pattern; a `deny Read/Edit/Write(path-glob)` → a protected path; a bare-tool `deny` → an unconditional, tier-independent deny of that tool; a bare-tool `allow` → a per-tool allow mode). **The safety floor always holds:** the always-on catastrophic + protected-path floor runs *before* any ingested `allow`, so a CC `allow: ["Bash(*)"]` can never auto-run `rm -rf /` or write `.env` (a bare-tool `allow` raises that tool's mode — like an operator `tool_trust_overrides` allow, it can permit benign calls even in a `deny`-mode session, but never past the catastrophic/protected floor). Unmappable gestures (e.g. `AskUserQuestion(*)`, per-pattern `allow`) are logged and skipped, never mis-mapped. Off by default so a workspace carrying another runtime's permission config doesn't silently reshape the posture. Per-Agent override: `Agent(enable_settings_permissions=…)`. |
| `status_line` | `ZAKCODE_STATUS_LINE` | `false` | Render a Claude Code `statusLine` (the `{type: "command", command: …}` object in `<workspace>/.claude/settings.json` + `settings.local.json`) after each turn in the CLI. The command runs **after** the turn with a CC-shaped status JSON on stdin (top-level `hook_event_name`/`session_id`/`cwd`/`transcript_path`/`version`, nested `model`/`workspace`/`cost`) and its first stdout line prints as a dim status line. **Purely cosmetic and fully fail-safe:** any error/timeout/non-zero exit just prints no line and never affects the turn; the command is danger-scanned (hard-denied in autonomous) and provider-key-scrubbed like every settings.json shell command. The local override wins over the shared file. Off by default; per-Agent override: `Agent(enable_status_line=…)`. |
| `output_style` | `ZAKCODE_OUTPUT_STYLE` | `false` | Inject the active Claude Code **output style** into the system prompt. The selected style name comes from `outputStyle` in `<workspace>/.claude/settings.json` + `settings.local.json` (local overrides), and the body from `.claude/output-styles/<name>.md` (an optional `---` frontmatter block is stripped). The body is folded into the **stable, cacheable tier** — the same seam always-on rules use, just after them — so it shapes how the assistant writes while staying prompt-cache safe. The body is capped (`MAX_OUTPUT_STYLE_CHARS`, ~16 KB) so a huge style file can't blow the context window. **Fully defensive:** a missing/unknown style name or file injects nothing and never raises, and when off (or unconfigured) the system prompt is byte-identical to the no-style prompt. Off by default so a workspace carrying another runtime's output-style config doesn't silently reshape this engine's voice; per-Agent override: `Agent(enable_output_style=…)`. |

> Cross-session **memory** is not a harness concern — it is claude-mind's. A Mind attaches its own
> recall/store through the generic hook/tool seams (see [`docs/PERSISTENCE-BOUNDARY.md`](PERSISTENCE-BOUNDARY.md)
> and [`docs/INTEGRATIONS.md`](INTEGRATIONS.md)); the harness ships no memory config.

## HTTP server (`zakcode serve`)

| Field | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `auth_token` | `ZAKCODE_AUTH_TOKEN` | unset | Bearer token required on every route except `/health` when set; unset = loopback-only dev (non-loopback bind needs `--insecure`). Excluded from every `model_dump()`. |
| `allowed_models` | `ZAKCODE_ALLOWED_MODELS` | `[]` | When non-empty, the only model strings a request may override to. |

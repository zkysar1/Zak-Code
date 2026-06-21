<h1 align="center">Zak Code</h1>

<p align="center"><strong>A clean-room, vendor-agnostic, API-first agentic coding tool.</strong></p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-green.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-2057%20passing-brightgreen.svg">
</p>

---

> **Status: alpha — feature-complete against the roadmap (M0–M10) plus a
> learning-substrate layer and an opt-in small-model quality engine, validated live on
> OpenAI _and_ local models.** The core
> engine, CLI, and HTTP API server are built and tested (2,057 passing tests; `ruff` +
> `mypy` clean). It's a young project — expect rough edges — but it really runs: it
> reads/writes files, runs commands, searches code, and drives multi-step tasks to
> completion against a real model.

## What is Zak Code?

Zak Code is a coding agent in the spirit of Claude Code, Hermes, and goose — but
**built from scratch, owned by you, and tied to no single model vendor.** It reads and
writes files, runs shell commands, searches a codebase, and drives multi-step
engineering tasks in a tool-use loop.

The design goal is a single **core engine** that any interface can drive:

```
                ┌─────────────────────────────────────────────┐
                │            zakcode  (core engine)            │
                │  agent loop · tools · sessions · compaction  │
                │  providers (litellm) · permissions · hooks   │
                │  MCP · plugins · skills · sub-agents · evals │
                └───────────────┬───────────────┬─────────────┘
                                │  in-process    │  over HTTP/SSE/WS
                ┌───────────────▼──────┐  ┌──────▼─────────────────────┐
                │   zakcode CLI        │  │   zakcode serve  (FastAPI) │
                │   (typer + rich)     │  └──────┬─────────────────────┘
                └──────────────────────┘         │
                                          ┌───────▼──────┐   ┌──────────────┐
                                          │  Web client  │   │  Your app /  │
                                          │  (bundled)   │   │  automation  │
                                          └──────────────┘   └──────────────┘
```

This is the same shape as the Claude Code SDK: the CLI is just the *first* client of a
reusable core. The bundled web UI, an IDE plugin, or your own automation all drive the
**exact same engine** and consume the **same** typed event stream — no duplicated agent
logic.

## Why "vendor-agnostic"?

Zak Code talks to models through [**litellm**](https://github.com/BerriAI/litellm), so
the same agent runs on ~100 providers. Switching providers is a **config change, never a
code change** — proven end-to-end against two independent providers:

| Provider | Use | Example model string | Status |
| --- | --- | --- | --- |
| **Ollama** (local) | zero-cost, offline, private | `ollama_chat/qwen2.5:3b`, `ollama_chat/llama3.1` | ✅ tested live (GPU) |
| **OpenAI** | cloud quality | `openai/gpt-4o-mini`, `openai/gpt-4o` | ✅ tested live |
| _…and ~100 more_ | one config value | `anthropic/…`, `gemini/…`, `bedrock/…` | via litellm |

> Each was driven through a real agentic task (model → `write_file` → a working Python
> file) with only the model string changed. See [`docs/SHAKEDOWN.md`](docs/SHAKEDOWN.md).

## Install

Requires [`uv`](https://docs.astral.sh/uv/) (it manages the Python toolchain for you).

```bash
git clone https://github.com/zkysar1/Zak-Code.git
cd Zak-Code
uv sync                    # creates the venv, fetches Python 3.11, installs deps + dev tools
uv run zakcode --help      # list commands
uv run zakcode info        # show resolved config + which provider keys are present
```

> Plain `uv sync` installs the core **and** the dev tools (the `dev` dependency group is
> on by default). Add the server with `uv sync --extra server` — the dev tools stay put.

### Point it at a model

**Local (free, private) — recommended for first run:**

```bash
# 1. Install Ollama (https://ollama.com) and pull a tool-capable model:
ollama pull qwen2.5:3b           # small + fast; needs a tool-calling template

# 2. Tell Zak Code to use it (Ollama serves on http://localhost:11434 by default):
uv run zakcode chat --model ollama_chat/qwen2.5:3b
```

**OpenAI (cloud):**

```bash
export OPENAI_API_KEY=sk-...                       # PowerShell: $env:OPENAI_API_KEY="sk-..."
uv run zakcode chat --model openai/gpt-4o-mini
```

> **Pick a tool-capable model.** The agent always offers tools, so the model needs a
> tool-calling template. `qwen2.5`, `llama3.1`, `gpt-4o*`, etc. work. Very small or
> template-less GGUFs are chat-only and Ollama will report `does not support tools`.

### Install once, run anywhere

Install `zakcode` as a per-user command (like any CLI tool) instead of running it
from the source checkout, and put your keys in the per-user config home — then any
directory is a workspace:

```bash
uv tool install "zakcode[server] @ git+https://github.com/zkysar1/Zak-Code.git"
mkdir -p ~/.zakcode && $EDITOR ~/.zakcode/.env   # GROQ_API_KEY=..., ZAKCODE_DEFAULT_MODEL=...
cd any/project/dir
zakcode chat                                     # workspace = the current directory
```

A workspace `.env` still overrides the user file per-project, and real environment
variables override both — see [`docs/CONFIG.md`](docs/CONFIG.md) for the precedence
table. Update with `uv tool upgrade zakcode`.

### Use it as a library (the API-first core)

The CLI is one client; the core is importable with **zero CLI/HTTP dependencies**:

```python
from zakcode import Agent

agent = Agent(default_model="openai/gpt-4o-mini")      # or ollama_chat/qwen2.5:3b
result = agent.run_turn("read pyproject.toml and summarize the dependencies")
print(result.assistant_messages[-1].text)

# Streaming (what the CLI and web client render):
async for event in agent.astream_turn("add type hints to utils.py"):
    ...
```

### Run the HTTP server + web client

```bash
uv sync --extra server
uv run zakcode serve            # FastAPI on http://127.0.0.1:8000 (loopback-only)
# then open http://127.0.0.1:8000/ for the bundled web client
```

REST + SSE + WebSocket all wrap the same core; the WebSocket carries the live event
stream **and** the permission-approval prompts. The web client is a pure renderer with
no agent logic.

## Features (M0–M10)

- **Agent loop** — ReAct-style tool-use loop with layered stop conditions (completion,
  iteration cap, shared budget, a **doom-loop guard** that halts identical repeated calls,
  and a broader **multi-signal stuck detector** that first tries to recover — nudge, then
  narrow to read-only tools — before stopping as `stuck`). Buffered and streaming paths.
- **Quality engine (opt-in, off by default)** — small-model *fan-out for quality*:
  LLM-as-judge (binary / pairwise / N-judge vote), best-of-N generation, oracle-grounded
  selection, rubric scoring + a ship/iterate cost-gate, and a bounded refine loop. Two
  off-by-default seams wire it into the agent — a **quality gate** (scores the written diff)
  and **best-of-N retry** (on a *stalled* turn, fan out N isolated attempts and adopt the
  first that verifies — by diff, never overwriting). The bet: ~10 cheap calls + selection
  beat one big call. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (`quality/`).
- **Tools** — `read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`, `bash`,
  and **`powershell`** (Windows-first; uses `pwsh`/`powershell.exe`) — all scoped to the
  workspace, with path-escape protection — plus **`web_search`** and **`web_fetch`**: a
  vendor-agnostic search backend (DuckDuckGo by default — free, no key; Tavily/SearXNG opt-in
  via `ZAKCODE_SEARCH_BACKEND`) and an SSRF-guarded page fetcher (install with `[web]` extra).
- **Vendor-agnostic providers** — litellm; Ollama + OpenAI are first-class and live-tested.
- **Streaming + rich TUI** — token-by-token output, tool calls, and usage in the terminal.
- **Local-model tool-calling** — a composable text-protocol fallback so models without
  native function-calling (small local GGUFs) still call tools; `tool_calling_mode` =
  `auto` (default) / `native` / `text`.
- **Permissions** — deny-first gate enforced **in the core** (4 modes × 3 tiers) plus a
  catastrophic-command blocklist; the gate is unreachable by the model. Operator deny
  regexes (`denied_commands`) append to the baseline (tighten-only).
- **Hooks** — `PreToolUse` / `PostToolUse` (exit-code protocol + in-process callbacks),
  a cache-safe **`PreLLMCall`** context-injection seam, and **session-lifecycle** hooks
  (`SessionStart` / `PreCompact` / `SessionEnd`).
- **Read-only concurrency** — a wholly read-only tool batch runs in parallel
  (order-preserving), guarded so only side-effect-free, prompt-free tools qualify.
- **Rules** — always-on `.md` guidance (`.zakcode/rules` + `.claude/rules`) in the
  cacheable prompt tier; sub-agents inherit them.
- **Bring-your-own memory** — cross-session memory is claude-mind's job, not the harness's
  (see [`docs/PERSISTENCE-BOUNDARY.md`](docs/PERSISTENCE-BOUNDARY.md)). The harness records the
  transcript (`/resume`) and exposes generic recall/lifecycle/tool seams a Mind attaches its own
  store to; it ships no memory store or `remember`/`recall` tools.
- **Learning substrate** — runtime skill authoring (`save_skill`) + the seams a
  self-learning framework folds into; see [`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md).
- **HTTP server** — FastAPI: REST, SSE, WebSocket; one `AgentEvent` stream across all clients.
- **Sub-agents + Plan Mode** — isolated child agents via a `task` tool; a read-only planner
  whose registry has no write tools (schema-enforced).
- **MCP** — a clean-room (no-SDK) Model Context Protocol client; external tools register
  into the same registry under `mcp__<server>__<tool>`, with lazy discovery + a tool budget.
- **Plugins** — `register(ctx)` entrypoint, trust-gated (untrusted plugin code is **not
  imported** until trusted), error-isolated.
- **Skills** — `SKILL.md` with progressive disclosure (cheap catalog → body on demand);
  invokable by a human (`/<name>`) **or by the model** (the `use_skill` tool), so skills **chain**
  (and branch). Available to sub-agents too; a per-turn `skill_invocation_budget` bounds runaway chains.
- **Context compaction** — real-token-count threshold (not char heuristics); summarizes
  older turns, keeps recent ones, tool-pair-safe and idempotent. `/compact`.
- **Evals** — a behavioral probe suite + `zakcode eval`, runnable as a CI gate.
- **Web client** — a dependency-free single-page UI served by the server.

## CLI commands

| Command | What it does |
| --- | --- |
| `zakcode chat` | interactive agent session (slash commands: `/help`, `/model`, `/permissions`, `/plan`, `/agents`, `/mcp`, `/plugins`, `/skills`, `/compact`, `/cost`, `/clear`) |
| `zakcode serve` | run the FastAPI server (+ bundled web client) |
| `zakcode eval` | run the behavioral eval suite (offline; exits non-zero on failure) |
| `zakcode info` | show resolved config + detected providers (never prints secrets) |
| `zakcode version` | print the version |

Key `chat` flags: `--provider` / `--model` (override the model), `--server <url>` (drive a
remote server), `--skill-dir <dir>` (load an external skill directory), and `--extra-root
<dir>` (grant the file tools an **additional trusted root to read _and write_ under** — a
sandbox-widening flag; see [`docs/GUARDRAILS.md`](docs/GUARDRAILS.md) §4).

## Small-model reliability (the Recipe Cursor)

Zak Code aims to stay useful on **small local models** (e.g. `qwen2.5:3b` via Ollama),
where a weak model can hallucinate a successful write or finish a turn over code it never
ran. The create-and-run loop is hardened by scaffolding that is **always on and steers
itself** — there are no reliability feature flags to set (one way of doing things). It
self-arms from what actually happened this turn, so it costs nothing on turns that don't
write code, and capable cloud models are unaffected:

- **One tool call per turn (text protocol).** When tools reach the model over the text
  protocol, the agent emits/parses exactly one tool call per turn and stops, so a weak
  model can't fabricate tool results or leak chat-template tokens.
- **Write-grounding.** After a successful `write_file`/`edit_file`, the file is read back
  from disk and the real content + a syntax check are injected, so the model can't
  hallucinate that a write did what it intended. (No-ops when nothing was written.)
- **The Recipe Cursor — a verify-before-finish gate.** Once the model writes a **runnable
  script** this turn (`.py`/`.js`/`.ts`/`.sh`/`.rb`/`.ps1`/…), the turn can't end until
  that file has been *run successfully*. The gate self-arms from the observed write; an
  unfixable file ends as `recipe_stalled` (never a false success) after a bounded number of
  attempts.
- **Harness-issued run.** When the model writes a file but doesn't run it, the harness runs
  it *itself* (picking the interpreter from the extension) to verify — but **only when that
  run would auto-allow without a prompt** (allow mode or a prior `bash` grant); otherwise it
  falls back to nudging the model. It never auto-runs shell behind an uninitiated prompt.
- **Acceptance compare.** When the request clearly states an expected output literal, the
  run's output must also contain it (catches "runs but prints the wrong thing"). Extraction
  is high-precision — any ambiguity falls back to exit-0-only, so it can never over-gate.

The only related knob is how tools reach the model, which already self-resolves by provider:

| Setting | Default | What it does |
| --- | --- | --- |
| `ZAKCODE_TOOL_CALLING_MODE` | `auto` | How tools reach the model: `auto` (native when supported; **text protocol for Ollama**, whose native path is unreliable) / `native` / `text`. `native`/`text` are a debug override; `auto` is correct for almost everyone. |

A good starting `.env` for a 3B local model is just the model itself — the reliability
scaffolding needs no configuration:

```dotenv
ZAKCODE_DEFAULT_MODEL=ollama_chat/qwen2.5:3b
```

### Web search & fetch

`web_search` and `web_fetch` are built in; install their (optional) deps with the `web` extra —
`uv sync --extra web` (or `pip install 'zakcode[web]'`). Without them the tools still register and
return a clean "install the web extra" message rather than failing.

`web_search` runs over a swappable, vendor-agnostic backend selected by `ZAKCODE_SEARCH_BACKEND`:

| Backend | Free? | Setup |
| --- | --- | --- |
| `ddgs` *(default)* | yes, no key | nothing — DuckDuckGo via the `ddgs` library |
| `tavily` | 1,000 searches/mo free | `export TAVILY_API_KEY=...` (cleaner, LLM-optimized results) |
| `searxng` | yes (self-hosted) | `ZAKCODE_SEARXNG_URL=http://localhost:8080` (enable the JSON format) |

`web_fetch` needs no backend — it GETs a public `http(s)` URL and returns readable text. It
**refuses** localhost / private / cloud-metadata addresses (an SSRF guard, re-checked across
redirects) and size-caps the output; fetched content is treated as untrusted. Two opt-in egress
controls lock it down: `ZAKCODE_WEB_ALLOWED_DOMAINS=example.com,docs.python.org` confines
`web_fetch` to those domains (and their subdomains), or `ZAKCODE_WEB_FETCH_CONFIRM=true` prompts
for confirmation before each fetch. Unset (the default), any public host is allowed.

## Platform support

Pure-Python; runs anywhere `uv` + Python 3.11+ run — **Windows, macOS, Linux.** One
caveat to know:

> The `bash` tool runs commands through the **platform shell** (`subprocess(shell=True)`):
> `/bin/sh` on macOS/Linux, **`cmd.exe` on Windows**. For PowerShell cmdlets and syntax,
> use the dedicated **`powershell`** tool (prefers `pwsh`, falls back to `powershell.exe`;
> returns a clean error on a host with neither). The agent is told the host OS and picks
> the right shell tool. Both shells go through the same deny-first permission gate and
> catastrophic-command blocklist (which covers PowerShell idioms like
> `Remove-Item -Recurse -Force` and `Format-Volume`).

## How it compares to Claude Code

Zak Code matches Claude Code's **core architecture and capability set** — three-layer
(core/server/clients) design, deny-first permissions, MCP, plugins, skills, sub-agents,
hooks, and real-token compaction — and in a few areas (auto-compaction wired into the
loop, a built-in eval harness) goes a bit further than the studied reference. See
[`docs/PARITY.md`](docs/PARITY.md) for the full matrix.

**Honest gaps vs. Claude Code (deferred, not hidden):** no git-checkpoint/`/undo`; cross-session **memory** is deliberately NOT in the harness — it is claude-mind's (see [`docs/PERSISTENCE-BOUNDARY.md`](docs/PERSISTENCE-BOUNDARY.md)). Runtime skill authoring exists as a **substrate**, but Zak Code ships no autonomous learning *policy* of its own —
that is meant to be supplied by an external self-learning framework folded in through the
documented seams ([`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md)); the autonomous
"never-terminate" loop is explicitly out of scope. Remaining gaps are tracked in
[`docs/ROADMAP.md`](docs/ROADMAP.md) as opt-in follow-ons; none affect the core loop.

## Documentation

| Doc | Purpose |
| --- | --- |
| [`docs/CHARTER.md`](docs/CHARTER.md) | Vision, goals, non-goals, principles |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design & module map |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestones, exit criteria, deferred work |
| [`docs/PARITY.md`](docs/PARITY.md) | Feature parity matrix vs. Claude Code |
| [`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md) | Seams for folding in a self-learning framework |
| [`docs/SHAKEDOWN.md`](docs/SHAKEDOWN.md) | Live-model validation (OpenAI + local) |
| [`docs/GUARDRAILS.md`](docs/GUARDRAILS.md) | Safety, security & clean-room rules |
| [`docs/RISKS.md`](docs/RISKS.md) | Risk register |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architecture Decision Records (ADRs) |
| [`docs/WORKFLOW.md`](docs/WORKFLOW.md) | How the build is orchestrated |

## Repository layout

```
Zak-Code/
├─ src/zakcode/         # the core engine (importable library) + CLI + server
│  ├─ providers/        # vendor-agnostic LLM layer (litellm)
│  ├─ agent/            # the agent loop, prompt assembly, context compaction
│  ├─ quality/          # small-model quality engine (judge, best-of-N, score, gate)
│  ├─ tools/            # tool registry + built-in tools
│  ├─ session/          # conversation state & persistence
│  ├─ permissions.py    # the deny-first permission gate (+ deny-rule grammar)
│  ├─ secrets.py        # secret redaction at persistence boundaries
│  ├─ commands/ hooks/ plugins/ skills/ rules/          # extension surfaces
│  ├─ mcp/              # clean-room Model Context Protocol client
│  ├─ evals/            # behavioral eval harness + probes
│  ├─ server/           # FastAPI app + bundled web client (optional extra)
│  └─ cli/              # the terminal client
├─ docs/                # living project documentation
└─ tests/               # 2,066-test suite (incl. gated live-provider smoke tests)
```

## Acknowledgements & clean-room note

Zak Code is an independent, **clean-room** implementation. We study the *architecture and
public documentation* of prior art — Claude Code, [Hermes](https://github.com/NousResearch/hermes-agent),
[goose](https://github.com/block/goose), and community reverse-engineering efforts — to
learn patterns. **We do not copy proprietary or leaked source code.** See
[`docs/GUARDRAILS.md`](docs/GUARDRAILS.md). Zak Code is not affiliated with or endorsed by
Anthropic or any other vendor.

## License

[MIT](LICENSE) © 2026 Zachary Kysar (Zak Data Solutions)

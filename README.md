<h1 align="center">Zak Code</h1>

<p align="center"><strong>A clean-room, vendor-agnostic, API-first agentic coding tool.</strong></p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-green.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-653%20passing-brightgreen.svg">
</p>

---

> **Status: alpha — feature-complete against the roadmap (M0–M10), validated live on
> OpenAI _and_ local models.** The core engine, CLI, and HTTP API server are built and
> tested (653 passing tests; `ruff` + `mypy` clean). It's a young project — expect rough
> edges — but it really runs: it reads/writes files, runs commands, searches code, and
> drives multi-step tasks to completion against a real model.

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
uv sync --extra dev        # creates the venv, fetches Python 3.11, installs everything
uv run zakcode --help      # list commands
uv run zakcode info        # show resolved config + which provider keys are present
```

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
  iteration cap, shared budget, and a **doom-loop guard** that halts identical repeated
  calls). Buffered and streaming paths.
- **Tools** — `read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`, `bash`
  — all scoped to the workspace, with path-escape protection.
- **Vendor-agnostic providers** — litellm; Ollama + OpenAI are first-class and live-tested.
- **Streaming + rich TUI** — token-by-token output, tool calls, and usage in the terminal.
- **Permissions** — deny-first gate enforced **in the core** (4 modes × 3 tiers) plus a
  catastrophic-command blocklist; the gate is unreachable by the model.
- **Hooks** — `PreToolUse` / `PostToolUse` with an exit-code protocol + in-process callbacks.
- **HTTP server** — FastAPI: REST, SSE, WebSocket; one `AgentEvent` stream across all clients.
- **Sub-agents + Plan Mode** — isolated child agents via a `task` tool; a read-only planner
  whose registry has no write tools (schema-enforced).
- **MCP** — a clean-room (no-SDK) Model Context Protocol client; external tools register
  into the same registry under `mcp__<server>__<tool>`, with lazy discovery + a tool budget.
- **Plugins** — `register(ctx)` entrypoint, trust-gated (untrusted plugin code is **not
  imported** until trusted), error-isolated.
- **Skills** — `SKILL.md` with progressive disclosure (cheap catalog → body on demand).
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

## Platform support

Pure-Python; runs anywhere `uv` + Python 3.11+ run — **Windows, macOS, Linux.** One
caveat to know:

> The `bash` tool runs commands through the **platform shell** (`subprocess(shell=True)`):
> `/bin/sh` on macOS/Linux, **`cmd.exe` on Windows**. The agent sees the host OS and adapts
> its commands, but there is **no dedicated PowerShell tool yet** (planned). On Windows the
> `bash` tool is really "run via `cmd.exe`."

## How it compares to Claude Code

Zak Code matches Claude Code's **core architecture and capability set** — three-layer
(core/server/clients) design, deny-first permissions, MCP, plugins, skills, sub-agents,
hooks, and real-token compaction — and in a few areas (auto-compaction wired into the
loop, a built-in eval harness) goes a bit further than the studied reference. See
[`docs/PARITY.md`](docs/PARITY.md) for the full matrix.

**Honest gaps vs. Claude Code (deferred, not hidden):** no `WebFetch`/`WebSearch` tools,
no dedicated PowerShell tool, no git-checkpoint/`/undo`, and no autonomous
skill-extraction/curator. These are tracked in [`docs/ROADMAP.md`](docs/ROADMAP.md) (M10+)
as opt-in follow-ons; none affect the core loop.

## Documentation

| Doc | Purpose |
| --- | --- |
| [`docs/CHARTER.md`](docs/CHARTER.md) | Vision, goals, non-goals, principles |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design & module map |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestones, exit criteria, deferred work |
| [`docs/PARITY.md`](docs/PARITY.md) | Feature parity matrix vs. Claude Code |
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
│  ├─ tools/            # tool registry + built-in tools
│  ├─ session/          # conversation state & persistence
│  ├─ permissions.py    # the deny-first permission gate
│  ├─ commands/ hooks/ plugins/ skills/   # extension surfaces
│  ├─ mcp/              # clean-room Model Context Protocol client
│  ├─ evals/            # behavioral eval harness + probes
│  ├─ server/           # FastAPI app + bundled web client (optional extra)
│  └─ cli/              # the terminal client
├─ docs/                # living project documentation
└─ tests/               # 653-test suite (incl. gated live-provider smoke tests)
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

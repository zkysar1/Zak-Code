<h1 align="center">Zak Code</h1>

<p align="center"><strong>A clean-room, vendor-agnostic, API-first agentic coding tool.</strong></p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-pre--alpha-orange.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
</p>

---

> **Status: being born.** 🐣 Zak Code is under active construction. The core engine,
> CLI, and API server are being built milestone-by-milestone (see
> [`docs/ROADMAP.md`](docs/ROADMAP.md)). Expect rapid change.

## What is Zak Code?

Zak Code is a coding agent in the spirit of Claude Code, Hermes, and goose — but
**built from scratch, owned by us, and tied to no single model vendor.** It can read
and write files, run commands, search a codebase, and drive multi-step engineering
tasks in a loop.

The design goal is a single **core engine** that any interface can drive:

```
                ┌─────────────────────────────────────────────┐
                │            zakcode  (core engine)           │
                │  agent loop · tools · sessions · context    │
                │  providers (litellm) · hooks/plugins/skills │
                └───────────────┬───────────────┬─────────────┘
                                │  in-process   │  over HTTP/SSE/WS
                ┌───────────────▼──────┐  ┌─────▼─────────────────────┐
                │   zakcode CLI        │  │   zakcode-server (FastAPI) │
                │   (typer + rich)     │  └─────┬─────────────────────┘
                └──────────────────────┘        │
                                          ┌──────▼───────┐   ┌──────────────┐
                                          │  Web client  │   │  Your app /  │
                                          │  (future)    │   │  automation  │
                                          └──────────────┘   └──────────────┘
```

This is the same shape as the Claude Code SDK: the CLI is just the *first* client of
a reusable API. A web app, an IDE plugin, or an automation can drive the exact same
engine.

## Why "vendor-agnostic"?

Zak Code talks to models through [**litellm**](https://github.com/BerriAI/litellm), so
the same agent runs on ~100 providers. First-class, tested targets:

| Provider | Use | Example model string |
| --- | --- | --- |
| **Ollama** (local) | zero-cost dev iteration, offline, private | `ollama_chat/llama3.1`, `ollama_chat/qwen2.5-coder` |
| **OpenAI** | cloud quality | `openai/gpt-4o` |
| _…and ~100 more_ | swap with one config value | `anthropic/...`, `gemini/...`, `bedrock/...` |

Switching providers is a config change, **never** a code change.

## Quickstart (dev)

> Requires [`uv`](https://docs.astral.sh/uv/) (which manages the Python toolchain for you)
> and, for local models, [Ollama](https://ollama.com).

```bash
# from the repo root — uv creates the venv, fetches Python 3.11, and installs deps:
uv sync --extra dev

uv run zakcode --help     # see available commands
uv run zakcode info       # show resolved config & detected providers
```

As of **M0** the agent loop runs: `zakcode chat` drives an interactive session, and the core
is usable as a library:

```python
from zakcode import Agent
result = Agent().run_turn("read pyproject.toml and summarize the dependencies")
print(result.assistant_messages[-1].text)
```

It ships with a small, sharp tool set — `read_file`, `write_file`, `list_dir`, `glob`,
`grep`, `bash` — all scoped to the workspace. Streaming + a richer TUI land in **M1**.

## Documentation

Zak Code is documentation-driven. The team maintains these living docs:

| Doc | Purpose |
| --- | --- |
| [`docs/CHARTER.md`](docs/CHARTER.md) | Vision, goals, non-goals, principles |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design & module map |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestones & exit criteria |
| [`docs/PARITY.md`](docs/PARITY.md) | Feature parity matrix vs. Claude Code |
| [`docs/GUARDRAILS.md`](docs/GUARDRAILS.md) | Safety, security & clean-room rules |
| [`docs/RISKS.md`](docs/RISKS.md) | Risk register |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architecture Decision Records (ADRs) |
| [`docs/WORKFLOW.md`](docs/WORKFLOW.md) | How the build is orchestrated |
| [`docs/references/`](docs/references/) | Clean-room study notes mined from prior art |
| [`docs/references/`](docs/references/) | Clean-room study notes mined from prior art |

## Repository layout

```
Zak-Code/
├─ src/zakcode/         # the core engine (importable library) + CLI + server
│  ├─ providers/        # vendor-agnostic LLM layer (litellm)
│  ├─ agent/            # the agent loop, prompt assembly, context compaction
│  ├─ tools/            # tool registry + built-in tools
│  ├─ session/          # conversation state & persistence
│  ├─ commands/         # slash commands
│  ├─ hooks/ plugins/ skills/   # extension surfaces
│  ├─ server/           # FastAPI app (optional extra)
│  └─ cli/              # the terminal client
├─ docs/                # living project documentation
└─ tests/               # test suite
```

## Acknowledgements & clean-room note

Zak Code is an independent, **clean-room** implementation. We study the *architecture
and public documentation* of prior art — Claude Code, [Hermes](https://github.com/NousResearch/hermes-agent),
[goose](https://github.com/block/goose), and community reverse-engineering efforts — to
learn patterns. **We do not copy proprietary or leaked source code.** See
[`docs/GUARDRAILS.md`](docs/GUARDRAILS.md). Zak Code is not affiliated with or endorsed by
Anthropic or any other vendor.

## License

[MIT](LICENSE) © 2026 Zachary Kysar (Zak Data Solutions)

# CLAUDE.md

Guidance for Claude Code (and any AI agent) working in the **Zak Code** repository.
This is the canonical agent guide; `AGENTS.md` points here.

## What this project is

Zak Code is a **clean-room, vendor-agnostic, API-first agentic coding tool** — our own
implementation in the spirit of Claude Code / Hermes / goose. The architecture is three
layers:

1. **`src/zakcode/` core engine** — an importable Python library: the agent loop, tool
   registry, sessions, context management, providers, the small-model **quality engine**
   (`quality/`), and extension surfaces. No UI here.
2. **`src/zakcode/server/`** — a FastAPI app exposing the core over HTTP/SSE/WebSocket.
3. **`src/zakcode/cli/`** — the terminal client (typer + rich). The first of many clients.

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) before making structural changes.

## Non-negotiable rules

1. **Clean-room.** Study architecture and *public* docs only. **Never copy leaked or
   proprietary source code** (e.g. the reverse-engineered Claude Code material kept under
   `C:\ZakNoCloud\_zakcode_research\` — that directory is read-only study material and must
   **never** be committed or pasted into this repo). Re-express ideas in our own design.
2. **Vendor-agnostic.** All model access goes through the provider layer
   (`zakcode/providers/`, built on litellm). Never hard-code an Anthropic/OpenAI-specific
   request shape in the agent loop. Switching providers must be a config change.
3. **Core/interface separation.** Business logic lives in the core engine. The CLI and
   server are thin clients. Don't leak terminal rendering or HTTP concerns into the core.
4. **Secrets.** Never log, print, or commit API keys. `.env` is gitignored; provider keys
   come from the environment (e.g. `OPENAI_API_KEY`). `zakcode info` may report whether a
   key is *present*, never its value.
5. **Docs travel with code.** When behavior changes, update the relevant doc in `docs/` in
   the same change. The docs are the source of truth the team is held to.

## Conventions

- **Python 3.11+**, `src/` layout, package name `zakcode`.
- **Env / deps:** [`uv`](https://docs.astral.sh/uv/). `uv sync --extra dev` to set up
  (uv fetches the right Python and installs everything). Run tools via `uv run …`.
- **Lint/format:** `ruff` (`uv run ruff check . && uv run ruff format .`).
- **Types:** `mypy` (`uv run mypy`). Prefer typed, `pydantic`-modeled boundaries.
- **Tests:** `pytest` (`uv run pytest`). New behavior ships with tests. Prefer **deterministic,
  offline** behavior tests (scripted providers + the `evals/` probe suite, no API) — see
  [`docs/TESTING.md`](docs/TESTING.md) for the two-tier (fast control-flow vs real-model quality)
  strategy.
- **Style:** match surrounding code; small reviewable changes; clear names over comments.

## Quick verification (run before declaring work done)

```bash
uv run poe check     # lint + format-check + types + tests, in one command
```

Or the individual steps:

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

## Where things live

| Need to… | Look in |
| --- | --- |
| Change how the agent loops / calls the model | `src/zakcode/agent/` |
| Add/change a small-model quality primitive (judge, best-of-N, score, gate) | `src/zakcode/quality/` |
| Add or change a tool | `src/zakcode/tools/` (+ `tools/builtins/`) |
| Add a model provider behavior | `src/zakcode/providers/` |
| Change config / settings | `src/zakcode/config.py` |
| Add a slash command | `src/zakcode/commands/` |
| Add an extension surface (hook/plugin/skill) | `src/zakcode/{hooks,plugins,skills}/` |
| Change the HTTP API | `src/zakcode/server/` |
| Change the terminal UX | `src/zakcode/cli/` |

## How the build is run

This project is built via orchestrated multi-agent **workflows** (see
[`docs/WORKFLOW.md`](docs/WORKFLOW.md)). If you are a build-team subagent, follow the
milestone scope in [`docs/ROADMAP.md`](docs/ROADMAP.md) and keep your work inside the
assigned module boundary.

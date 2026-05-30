# Contributing to Zak Code

Thanks for your interest! Zak Code is a clean-room, vendor-agnostic, API-first agentic
coding tool. This guide covers the essentials; the deeper context lives in
[`docs/`](docs/) (start with [`CHARTER.md`](docs/CHARTER.md) and
[`ARCHITECTURE.md`](docs/ARCHITECTURE.md)).

## Ground rules (non-negotiable)

These mirror [`CLAUDE.md`](CLAUDE.md) / [`docs/GUARDRAILS.md`](docs/GUARDRAILS.md):

1. **Clean-room.** Study architecture and *public* docs of prior art only. Never copy
   leaked or proprietary source. Re-express ideas in our own design.
2. **Vendor-agnostic.** All model access goes through `zakcode/providers/` (LiteLLM). No
   provider-specific request shapes leak into the agent loop.
3. **Core/interface separation.** Logic lives in the core engine; the CLI and server are
   thin clients. Dependencies point inward.
4. **Secrets.** Never log, print, or commit API keys. `.env` is gitignored.
5. **Docs travel with code.** Behavior change ⇒ update the relevant `docs/` file in the
   same change.

## Dev setup

Requires [`uv`](https://docs.astral.sh/uv/) (manages the Python toolchain for you).

```bash
uv sync --extra dev
uv run zakcode info        # sanity check
```

## Before you push

CI runs these on Python 3.11–3.13; run them locally first:

```bash
uv run ruff check .          # lint
uv run ruff format .         # format (CI checks formatting)
uv run mypy                  # type check
uv run pytest                # tests
```

New behavior ships with tests. Prefer small, reviewable changes with clear names.

## Commit & PR style

- Write imperative, descriptive commit subjects (e.g. "Add LiteLLM provider with streaming").
- Reference the milestone (see [`docs/ROADMAP.md`](docs/ROADMAP.md)) where relevant.
- Keep each PR scoped to one logical change; update docs and the parity matrix
  ([`docs/PARITY.md`](docs/PARITY.md)) when you land a tool/command/subsystem.

## Where things live

See the table in [`CLAUDE.md`](CLAUDE.md#where-things-live).

# AGENTS.md

This file exists for agent tooling that looks for `AGENTS.md`.

**The canonical agent guide for this repository is [`CLAUDE.md`](CLAUDE.md).** Read it.

The five non-negotiable rules, in brief:

1. **Clean-room** — study patterns, never copy leaked/proprietary source.
2. **Vendor-agnostic** — all model access through `zakcode/providers/` (litellm).
3. **Core/interface separation** — logic in the core engine; CLI & server stay thin.
4. **Secrets** — never log/print/commit API keys.
5. **Docs travel with code** — update `docs/` in the same change as behavior.

Setup: `uv sync --extra dev`. Verify: `uv run ruff check . && uv run mypy && uv run pytest`.

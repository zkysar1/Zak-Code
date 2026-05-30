# Architecture

> **This document is being authored by the `zakcode-foundation` workflow**
> (run `wf_efd14b18-b4c`, launched 2026-05-30) from a deep study of prior art, and will be
> replaced with the full system design once that research lands. Until then, the
> authoritative summary lives in [`DECISIONS.md`](DECISIONS.md) (ADR-0004, ADR-0005) and
> the diagram in the [root README](../README.md).

## Summary (pending full draft)

Three layers:

1. **`zakcode` core engine** — importable library: agent loop, tools, sessions, context,
   providers (litellm), extension surfaces.
2. **`zakcode-server`** — FastAPI exposing the core over HTTP / SSE / WebSocket.
3. **Clients** — CLI first; web later.

The module map under `src/zakcode/` (`providers/`, `agent/`, `tools/`, `session/`,
`commands/`, `hooks/`, `plugins/`, `skills/`, `server/`, `cli/`) is scaffolded and
described in each package's docstring.

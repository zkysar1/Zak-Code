# Roadmap

> **This document is being authored by the `zakcode-foundation` workflow**
> (run `wf_efd14b18-b4c`, launched 2026-05-30) and will be replaced with the full,
> phased milestone plan once the research lands.

## M0 (target, pending full draft)

A runnable minimal agent loop driven via litellm against **both** Ollama (local, default)
and OpenAI, with a small sharp tool set (`read_file`, `write_file`, `list_dir`, `bash`,
`glob`, `grep`), a working `zakcode chat` CLI, session persistence, and tests.

Later milestones progressively add streaming/TUI, permissions & hooks, the FastAPI server,
sub-agents, MCP extensions, plugins, skills, a web client, advanced context compaction, and
an evaluation harness — sequenced against [`PARITY.md`](PARITY.md).

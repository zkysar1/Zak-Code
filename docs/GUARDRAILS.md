# Guardrails

> **The full version of this document is being authored by the `zakcode-foundation`
> workflow** (run `wf_efd14b18-b4c`, launched 2026-05-30). The rules below are in force now.

## In force now

1. **Clean-room / legal.** Study architecture and *public* docs only. **Never** copy
   leaked or proprietary source (incl. the reverse-engineered Claude Code material under
   `C:\ZakNoCloud\_zakcode_research\`, which is read-only study material and must never be
   committed). Re-express ideas in our own design.
2. **Secrets.** Never log, print, or commit API keys. `.env` is gitignored; provider keys
   come from the environment. Reports may state whether a key is *present*, never its value.
3. **Safe by default.** Tools that write files or run commands must respect the permission
   model and confirm destructive operations. Default to the least dangerous behavior.
4. **Vendor-agnostic.** No provider-specific assumptions leak into the core agent loop.
5. **Dependency hygiene.** Vet new dependencies; pin sensibly; prefer well-maintained libs.

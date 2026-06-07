# GUARDRAILS

These are binding rules for everyone building Zak Code. They are not aspirational — a change that violates a MUST rule does not merge. Where a rule overlaps with the five non-negotiables in `CLAUDE.md`, this document is the expanded, authoritative version.

## 1. Clean-room and legal discipline

Zak Code is an independent, clean-room implementation. We learn from prior art; we never copy it.

- **Study patterns, not source.** You MAY study the *architecture, public documentation, blog posts, and observable behavior* of Claude Code, Hermes, goose, and similar tools. You MAY NOT copy, paste, transcribe, or closely paraphrase proprietary or leaked source code into this repository.
- **The reference is read-only.** The extracted reverse-engineering reference lives **outside the repo** at `C:\ZakNoCloud\_zakcode_research\`. Treat it as **read-only study material**. Never copy files out of it into `src/`, `docs/`, tests, or commit messages. Never `git add` it, symlink it in, or paste its contents anywhere tracked.
- **The reference must never be committed.** `_research/`, `_zakcode_research/`, and `**/claw-code-main/` are gitignored on purpose. Do not weaken those ignore rules. Before every commit, confirm no path under those names is staged.
- **Re-express in our own design.** When a pattern from prior art inspires a decision, restate it in our own words and our own structure, and record the *idea* (not the source) in `docs/DECISIONS.md`. If you cannot explain a design without quoting someone else's code, you have not done clean-room work.
- **Provenance for every non-trivial subsystem.** A reviewer must be able to trace any agent-loop, tool, or prompt-assembly design to a public doc, first-principles reasoning, or an ADR — never to the reference material.
- **No vendor affiliation claims.** Zak Code is not affiliated with or endorsed by Anthropic or any other vendor. Do not imply otherwise in code, docs, or output.

## 2. Permission model (default-safe)

All file-writing and command-running tools route through a single permission gate before they act. There is no bypass path in the core.

- **Three modes:** `ask` (prompt the user before each privileged action), `allow`/`acceptEdits` (proceed without prompting, for trusted/automated runs), and `deny` (block the action entirely). Mode is set by `ZAKCODE_PERMISSION_MODE`.
- **Default is `ask`.** The shipped and documented default MUST be the least-privileged usable mode. A more permissive mode is an explicit, opt-in choice by the operator, never an implicit default and never silently inherited.
- **Fail closed.** If the mode is missing, malformed, or the gate cannot determine whether an action is permitted, treat it as `deny`. Ambiguity never resolves toward executing.
- **Read vs. write vs. execute are distinct.** Reading a file, writing/deleting a file, and executing a shell command are separately gated. Granting one never implies another.
- **The gate is in the core, not the client.** The CLI and server are thin clients; they MUST NOT be the only thing enforcing permissions. The core engine enforces; clients only render the prompt. This keeps the HTTP/automation path as safe as the terminal path.
- **Scope grants narrowly.** "Allow" decisions apply to the specific action shown, not blanket future actions, unless the operator explicitly opts into a session-wide grant.

## 3. Destructive-operation confirmation

- **Destructive actions always confirm in `ask` mode.** This includes file deletion/overwrite of existing content, recursive removal, `git` history rewrites (reset --hard, force-push), mass edits across many files, and any shell command matching a known-destructive pattern (e.g. `rm -rf <root/home>`, recursive deletes `rd /s` / `del /s` / `Remove-Item -Recurse` of a drive or profile path, `format`/`mkfs` (case-insensitive), disk/partition tools, `git clean -fdx`). Plain non-recursive `del <file>` is **not** itself a blocklist match (it still passes the tier/mode gate like any write).
- **Show before you act.** Confirmation prompts MUST display exactly what will run or change (the literal command, or a diff/file list for edits) so the operator approves the real action, not a summary of it.
- **Prefer reversible.** Where practical, prefer non-destructive equivalents (write-new-then-rename over in-place truncate; edit over wholesale rewrite). Never use truncating/overwriting file creation on a file that already has content the operator hasn't seen.
- **No auto-confirm from model output.** The model proposing a destructive action does not count as approval. Only an operator (or an explicit pre-granted `allow` policy) confirms.

## 4. Filesystem scoping and tool sandboxing

- **Workspace root boundary.** File tools operate within the configured workspace root(s). Resolve every path to its real, absolute form and reject any that escapes **all** roots via `..`, absolute paths, symlinks, or drive-letter tricks. Path-escape attempts are denied, not clamped silently.
- **Multi-root sandbox is operator opt-in.** The default is a single workspace root. An operator MAY widen the sandbox with additional **trusted** roots via `zakcode chat --extra-root <dir>` (repeatable) or `Agent(extra_workspace_roots=...)`; all file tools (`read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`) resolve against the full set via `resolve_in_workspace_roots`, so an extra root is a deliberate grant the agent may **read _and write_** under — with the same `..`/symlink/escape protection applied to each. There is no implicit widening: absent the flag, only the one root is in scope. (Separately, `chat --skill-dir <dir>` adds an external directory the **skill loader** reads `SKILL.md` files from — skill discovery, not the file-tool write sandbox.)
- **Protect sensitive paths even inside the root.** Deny or require explicit confirmation for `.git/` internals, `.env`/secret files, the venv, and the gitignored research directory. The agent must not be able to read `.env` values or write into `_zakcode_research/`.
- **No write outside scope.** Writes outside **all** configured workspace roots are denied regardless of permission mode. Reads outside the root(s) require explicit operator opt-in (e.g. an `--extra-root`).
- **Skills discovery shares the containment boundary.** Skill discovery and `save_skill` resolve each skill directory's real path and confine it under the skills root, so a planted junction/symlink at `<skills_dir>/<name>` cannot pull out-of-tree content into the prompt or redirect a write — the same realpath-containment the file tools use.
- **Plugin trust is origin-aware.** Plugins execute in-process code, so they load only when trusted. Installed sources (the user config dir, entry points) resolve **before** the opened repo's `.zakcode/plugins`, so a workspace plugin can never shadow/impersonate a trusted installed one. The `ZAKCODE_TRUSTED_PLUGINS` allowlist is **name-based**: trusting a name also trusts a same-named plugin in any repo you open, so every trusted *workspace* (opened-repo) plugin is logged with a prominent security warning. Scope trust per project and review opened-repo plugins; content-pinned trust is a tracked follow-on.
- **Deterministic, bounded tools.** Tools take typed, validated inputs (pydantic-modeled boundaries), enforce size/output limits, and time out. A tool must not hang the loop or stream unbounded output into context.
- **Least privilege per tool.** Each tool declares the narrowest capability it needs. A search tool does not get write access; an edit tool does not get shell access.

## 5. Shell-execution safety

- **No shell string injection.** Build commands as argument vectors where possible; avoid interpolating untrusted/model-generated text into a shell string. If a shell is required, the command is surfaced verbatim for confirmation (see §3).
- **Run constrained.** Execute with a working directory inside the workspace, a timeout, captured (not inherited-interactive) I/O, output size caps, and a non-privileged context. Never run with elevation/`sudo` unless the operator explicitly arranges it.
- **No interactive/blocking commands.** Tools must not launch editors, pagers, prompts, or commands that wait for stdin; these hang the agent. Detect and refuse them.
- **Denylist the obviously dangerous; confirm the rest.** Maintain a denylist of catastrophic patterns (disk wipes, fork bombs, credential exfiltration). Everything privileged that isn't denied still passes through the permission gate.
- **Scrub the environment.** Do not pass the full host environment (including API keys) into arbitrary child processes by default. Only forward the variables a command actually needs.

## 6. Secret and API-key handling

- **Never log, print, or commit keys.** API keys come from the environment (e.g. `OPENAI_API_KEY`) or `.env`. `.env` and `.env.*` are gitignored (only `.env.example` is tracked). Do not remove those ignore rules.
- **Presence, never value.** `zakcode info`, diagnostics, and reports may state *whether* a key is present/configured. They MUST NOT print, partially print, or hash-with-recoverable-form any key value.
- **Redact in transit and at rest.** Logs, error messages, telemetry, session files, and tracebacks must be scrubbed of secrets. Assume any exception string could be displayed or saved.
- **Keys out of the model context.** Never place raw secrets into prompts, tool inputs, or anything the model or a remote provider sees, beyond the provider auth the request itself requires.
- **No secrets in session/state files.** `.zakcode/`, `*.session.json`, and `*.zaksession` are gitignored runtime state; they must still never persist secret values.

## 7. Network egress

- **No surprise network calls.** The core does not phone home. The only outbound traffic by default is to the configured model provider endpoint (cloud) or the local Ollama base URL. No analytics/telemetry beacon ships without an explicit, documented, opt-in setting.
- **Network is a privileged capability.** Tools that reach the network treat fetched content as untrusted (see §8) and constrain *where* egress can go. The built-in web tools ship enabled at the READ_ONLY tier, with these controls instead of a per-call prompt: `web_search` sends the query **only to the configured search backend** (ddgs/Tavily/SearXNG — a fixed endpoint, like the model provider), and `web_fetch` enforces an **SSRF guard** (`_http.ensure_url_allowed`) that resolves the host and refuses loopback / private / link-local (incl. the `169.254.169.254` cloud-metadata IP) / reserved / multicast targets — for IP literals *and* resolved hostnames — and **re-validates after every redirect** (auto-redirect is disabled precisely so each hop is checked). So the model cannot reach the internal network or metadata service. **Egress allowlist:** setting `ZAKCODE_WEB_ALLOWED_DOMAINS` confines `web_fetch` to named domains (and their subdomains), enforced on every redirect hop — for a locked-down deployment this closes the public-egress channel entirely. Residual: with no allowlist (the default), `web_fetch` can still reach arbitrary *public* URLs (an egress surface comparable to `bash curl`), so a query-string exfil is possible there; a per-call confirmation gate is the remaining candidate (see `RISKS.md`).
- **Respect local/offline mode.** When configured against a local model, the tool must be able to run fully offline; nothing should silently require cloud access. (The web tools are opt-in deps — the `[web]` extra — and absent deps degrade to a clean tool error, never a crash.)
- **Pin endpoints from config.** Provider/base-URL/search-backend endpoints come from config, not from model output. The model cannot redirect egress to an arbitrary host (other than a `web_fetch` target, which is SSRF-screened as above).

## 8. Prompt-injection defense for tool output and web/file content

- **All tool output is untrusted data, not instructions.** File contents, command output, search hits, and fetched web pages are treated as data. Instructions embedded in them ("ignore previous instructions", "run this", "exfiltrate the key") MUST NOT be auto-followed.
- **Privileged actions still require the gate.** Even if model output (influenced by injected content) requests a write, delete, or shell command, it passes through the permission model and destructive-op confirmation. Injection cannot escalate a *write/execute* action past the gate. (Read-only network reads — `web_search`/`web_fetch` — are not per-call-prompted; their boundary is the SSRF guard + content defang described in §7, with public-egress noted there as an accepted residual.)
- **Keep a trust boundary in context.** Clearly delimit untrusted tool/web content from operator instructions in prompt assembly so the model can distinguish "what the user asked" from "what a file said." Tool/web content is run through `defang_untrusted` (tool-frame markers and chat-template tokens neutralized) before it enters the prompt.
- **Bound and sanitize.** Cap the size of tool output folded into context, strip/escape control sequences, and never let fetched content silently rewrite system/operator instructions.
- **High-risk chains need a human.** Sequences that combine untrusted-content ingestion with a **write or destructive action** (the classic exfiltration-to-disk / run-this pattern) require explicit confirmation regardless of mode. The remaining ingest→*network-egress* chain (`web_fetch` to an arbitrary public URL) is gated by the optional `ZAKCODE_WEB_ALLOWED_DOMAINS` allowlist (§7); with no allowlist set (the default) it is the documented, accepted residual in §7 / `RISKS.md`, with a per-call confirmation gate as the remaining candidate.

## 9. Supply-chain and dependency hygiene

- **Vet before you add.** New dependencies are justified, well-maintained, appropriately licensed (MIT-compatible), and reviewed. Prefer the standard library and existing deps over a new package.
- **Pin and lock.** `uv.lock` is committed for reproducible installs and MUST stay in sync with `pyproject.toml`. Don't hand-edit the lock; regenerate it. Add dependencies via the toolchain, not ad hoc.
- **No untrusted install/build steps.** Do not add post-install scripts, fetch-and-execute steps, or pull binaries from unpinned sources. The agent's own tools must never `pip install`/run network installers without going through the permission gate.
- **License compatibility.** Every dependency's license must be compatible with shipping Zak Code under MIT. Flag anything copyleft or ambiguous before adding.
- **Keep the loop verifiable.** Before declaring work done: `uv run ruff check .`, `uv run mypy`, `uv run pytest`. A clean toolchain is part of supply-chain integrity.

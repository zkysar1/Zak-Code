# Integrations: folding in a self-learning framework

Zak Code ships **no learning policy of its own** — it does not decide what to
remember, when to forge a skill, or how to consolidate a session. Instead it
exposes a set of **substrate seams** so an external self-learning framework (for
example, a [Claude-Mind](https://github.com/zkysar1/Claude-Mind)-style "mind") can
provide that policy *on top of* Zak Code's storage and event mechanism.

This document is the contract: what each seam is, how to plug into it, and what is
deliberately **deferred**. The division of labor is intentional —

> **Zak Code = mechanism (storage, retrieval, events, gating). The framework =
> policy (what/when/how to learn).**

so the two never fight over the same files, hooks, or learning loop.

---

## The seams

### 1. Lifecycle hooks (the automation backbone)

`zakcode.hooks.HookManager` fires session-lifecycle events that a framework hooks
its prime / encode / serialize automation onto. All are **observe-only**: a hook
runs for side effects and can never block or rewrite a turn (every failure is
isolated).

| Event | Fires | Typical framework use |
| --- | --- | --- |
| `SessionStart` | once, on the first turn of a session | "prime" — load durable state into the framework's working memory |
| `PreCompact` | before the transcript is compacted (auto or `/compact`) | serialize learning state before context is dropped |
| `SessionEnd` | on `Agent.aclose()` | "encode" — consolidate the just-finished session |

Register in-process:

```python
agent.hook_manager.register_lifecycle(HookEvent.SESSION_START, my_prime_fn)
```

…or as a shell hook (the framework's scripts), which receives a JSON
`LifecyclePayload` (`event`, `session_id`, `cwd`, `data`) on **stdin**:

```python
HookSpec(event=HookEvent.SESSION_START, command=["bash", "core/scripts/prime.sh"])
```

The tool-gate pair (`PreToolUse` / `PostToolUse`) is also available and can **veto**
or **rewrite** a tool call (exit code `2` = block; stdout JSON may rewrite
arguments) — the seam for runtime guardrails.

### 2. Per-turn context injection (`PreLLMCall`)

A **context hook** contributes background text before each model call. The loop
folds it into an *ephemeral* tail message — appended after all real history, never
persisted, and **fenced + defanged as untrusted** (`<injected_context>…`) — so the
cached system+history prefix is untouched (prompt-cache safe) and recalled content
is never treated as instructions.

```python
agent.hook_manager.register_context(lambda payload: retrieve_relevant(payload.user_text))
```

Shell context hooks get the `LLMContextPayload` (`user_text`, `cwd`, `iteration`) on
stdin and return the text to inject on stdout (plain, or `{"context": "..."}`) — a
direct home for a `retrieve.sh`-style retrieval script.

### 3. Skills (`.claude/skills/<name>/SKILL.md`)

The skills loader discovers Agent-Skills-format `SKILL.md` files from, in increasing
precedence: bundled → `~/.config/zakcode/skills` → `<workspace>/.zakcode/skills` →
**`<workspace>/.claude/skills`** (the last for Claude-Code / Claude-Mind
compatibility). The frontmatter parser reads `name`/`description`/`version`/
`allowed-tools` and **tolerates any extra keys** (`user-invocable`, `triggers`,
`forged`, …). The `name` + `description` catalog goes into the cacheable prompt tier;
the body loads on demand.

A framework can **author** skills two ways: write `SKILL.md` files directly (they
are discovered next session), or call `zakcode.skills.save_skill(...)` (validates a
kebab-case name so the write can never escape the skills dir). Zak Code's own
`save_skill` tool (enabled with `enable_skills`) lets the *model* author skills the
same way.

### 4. Rules (`.claude/rules/*.md`)

Always-on, operator-authored Markdown guidance discovered from bundled → user →
`.zakcode/rules` → **`.claude/rules`** (last wins) and rendered into the **stable,
cacheable** prompt tier — the Claude-Code `CLAUDE.md` model. Bounded by a total char
budget so a large rules dir can never blow the context window. Sub-agents inherit the
parent's rules. (For *on-demand* rules a framework loads selectively, keep them as
files its skills read via the file tools rather than relying on always-on injection.)

### 5. Memory (`MemoryProvider`)

`zakcode.memory.MemoryProvider` is the storage + retrieval contract (`add` / `search`
/ `recent` / `update` / `delete` / `count`); `update` lets a learner policy edit a fact's
text/kind/tags in place (surgical edits, not duplicate appends). The default
`SqliteMemoryProvider` is a local
SQLite/FTS5 store whose path is configurable (`ZAKCODE_MEMORY_DB_PATH`) so a framework
can relocate or per-agent it. Recalled text is **secret-redacted** at both the write
(`remember`) and recall boundaries (`docs/GUARDRAILS.md` §6). Inject your own store
via `Agent(memory_provider=...)`; the recall hook surfaces relevant memories each
turn through seam #2.

### 6. Permissions & deny rules

The deny-first gate (`PermissionPolicy`) enforces every privileged call in the core.
Operator/project deny **regexes** (`ZAKCODE_DENIED_COMMANDS`, compiled via
`compile_deny_patterns`) are *appended* to the built-in blocklist — they can only
ever **tighten** the verdict, never remove a baseline footgun. A `PreToolUse` hook
can additionally veto any call.

### 7. Local-model robustness (`tool_calling_mode`)

`TextToolCallingProvider` gives models without native function-calling tool use via a
text protocol (auto / native / text). A framework targeting self-hosted Llama/Qwen/
Mistral can rely on tools working regardless of the model's native capability.

---

## Mapping a Claude-Mind-style framework onto the seams

| Framework need | Zak Code seam |
| --- | --- |
| prime context at session start | `SessionStart` lifecycle hook |
| retrieve relevant memory per turn | `PreLLMCall` context hook (shell `retrieve.sh`) |
| encode / consolidate after a session | `SessionEnd` lifecycle hook |
| serialize before compaction | `PreCompact` lifecycle hook |
| runtime guardrail that can block a tool | `PreToolUse` hook (exit 2) |
| skills (read + author) | `.claude/skills` loader + `save_skill` |
| rules / conventions | `.claude/rules` loader |
| durable cross-session store | `MemoryProvider` (relocatable) |
| path-scoped / command denies | `denied_commands` grammar + `PreToolUse` veto |
| run on self-hosted models | `tool_calling_mode` text fallback |

---

## Deliberately deferred

These were scoped out (the map of Claude-Mind flagged them as the
autonomous-perpetual-loop pieces that don't fit a coding agent whose mission is
shipping code). The safe initial fold-in is **reader/assistant mode + the encode
pass**, not the self-directed loop.

- **Stop-hook continuation / "never terminate".** Zak Code's `SessionEnd` is
  observe-only; it cannot veto turn-end and inject a `LOOP_CONTINUE` to keep an
  autonomous agent running. Adding a blocking, continuation-capable Stop hook is a
  future item, gated on proving it on this runtime (and on Windows).
- **Full `settings.json` ingestion.** Hooks are wired in code / via `HookSpec`;
  parsing a Claude-Code `settings.json` hook block verbatim is not yet implemented.
- **The `mind_api` daemon.** Zak Code does not spawn or manage a framework's
  background HTTP daemon; the framework owns its own process lifecycle.

---

## Clean-room note

Zak Code's skills/rules formats are re-expressed from the **public** Anthropic Agent
Skills spec, not copied from any framework's skill bodies. A framework's own content
(skill bodies, scripts, prompts) stays in *that* framework's repo and is loaded at
runtime through these seams — never vendored into Zak Code core. See
[`docs/GUARDRAILS.md`](GUARDRAILS.md).

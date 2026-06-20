# Persistence boundary: harness substrate vs. claude-mind memory

**Status:** Accepted — implemented (2026-06-20). The cross-session "memory" subsystem was **removed**
from the harness; claude-mind owns memory and attaches its own via the generic seams below.

**One line:** *The harness RECORDS what happened and EXPOSES seams. It does not REMEMBER, RECALL,
or LEARN. "Memory" is claude-mind's word, and claude-mind's job.*

---

## Why

Zak Code (the harness) had accreted a cross-session **memory** subsystem (`src/zakcode/memory/`): a
searchable SQLite store, a recall hook that injected "relevant" past notes, model `remember`/`recall`
tools, and a `LessonWriter` that auto-derived heuristics. Two problems:

1. **It used the word "memory."** That word belongs to **claude-mind** (the cognitive layer). A
   harness that "has memory" blurs the substrate/Mind boundary the whole project rests on.
2. **It did cognition by default** (the CLI turned it on) — deciding what is worth keeping, what is
   relevant now, and what was learned. That is the Mind's job, not the substrate's.

This generalizes the skills pattern: `ON_SKILL_SELECTED` is *"the seam a learning mind records from —
the substrate emits the signal; choosing/learning is the mind's job."*

## What already existed (the two layers)

| Layer | Where | What it does | Verdict |
| --- | --- | --- | --- |
| **Session/transcript store** | `session/store.py` (`Session`, `SessionStore`) | Persists the full conversation + tool calls + usage + resume metadata. Powers `/resume`. | **Mechanical. Kept, unchanged.** |
| **Cross-session "memory"** | `memory/` + `tools/builtins/memory.py` + `agent/lessons.py` | SQLite store, relevance recall hook, `remember`/`recall` tools, auto-lessons. | **Cognitive. REMOVED.** |

The first is the honest "cross-session chat / tool-calling store" — and is *already* not called
memory. The second was the part that crossed into Mind territory, and is gone.

## The boundary

> **Substrate (harness)** = the *record of what happened* (the always-on transcript) + *generic
> seams*. **Cognition (claude-mind)** = *memory* — recall, consolidation, learning — built on top.

### Decisions

1. **`SessionStore` persists ALWAYS, in every mode.** The transcript *is* `/resume`; it is mechanical
   state, not memory, so it is exempt from the "no memory while exploring" concern.
2. **The cognitive pieces are REMOVED, not kept as inert seams.** claude-mind has its own framework;
   it does not need a harness-provided store contract or memory hook. It attaches its own memory
   through the **generic** seams that already exist and are *not* memory-specific (next section).
3. **The `enable_memory` flag is gone.** With no harness memory to switch on, there is nothing to
   gate — "reader mode saves no memory" falls out for free.
4. **"Memory" is reserved for claude-mind.** The harness uses mechanical names (transcript / session
   store).

### The generic seams claude-mind attaches memory to (kept, unchanged)

None of these are memory-specific; the removed subsystem was merely *one consumer* of them:

- **`hook_manager.register_context(hook)`** — a `PRE_LLM_CALL` context hook returns text folded into
  the turn (fenced untrusted). This is the **recall** seam: a Mind injects its relevant hits here.
- **`hook_manager.register_lifecycle(event, hook)`** — `SESSION_START` / `SESSION_END` / `PRE_COMPACT`
  for prime / encode / serialize.
- **The tool registry** — a Mind registers its own `remember`/`recall` tools.
- **Dependency injection + `SessionStore`** — the Mind reads the transcript and wires its own
  providers through the same constructor seams every other capability uses.

`docs/INTEGRATIONS.md` §5 documents exactly how a Mind wires memory onto these.

## What was removed

| Piece | Fate |
| --- | --- |
| `memory/` package (`MemoryProvider`, `SqliteMemoryProvider`, `MemoryRecord`, `MemoryRecallHook`) | **Deleted** |
| `tools/builtins/memory.py` (`remember` / `recall` tools) | **Deleted** |
| `agent/lessons.py` (`LessonWriter`) | **Deleted** (see deferred below) |
| `enable_memory` flag, `memory_provider` injection, `memory_*` config fields, `--no-memory` CLI flag, server shared store | **Removed** |
| `SessionStore` + every generic seam (hooks, tool registry, DI) | **Kept, unchanged** |

## Deferred

The `LessonWriter` derived a recovery lesson from harness-internal `StuckTracker`/`RecipeCursor`
signals. It was removed (clean removal, not new machinery). **If claude-mind later wants
recovery-lesson capability**, the clean way is a future generic `ON_TURN_RECOVERED` lifecycle hook
carrying the recovery payload — a Mind subscribes and decides what to persist. Not built here.

---

*Settled. If we add the `ON_TURN_RECOVERED` seam later, record it as its own decision.*

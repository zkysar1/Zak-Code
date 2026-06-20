# Persistence boundary: harness substrate vs. claude-mind memory

**Status:** Draft / research-in-progress (started 2026-06-20). Living document — we are still
working out the edges; nothing here is locked until it lands in code with an ADR.

**One line:** *The harness RECORDS what happened and EXPOSES seams. It does not REMEMBER, RECALL,
or LEARN. "Memory" is claude-mind's word, and claude-mind's job.*

---

## Why this doc exists

Zak Code (the harness) accreted a cross-session **memory** subsystem (`src/zakcode/memory/`): a
searchable SQLite store, a recall hook that injects "relevant" past notes into the prompt, model
`remember`/`recall` tools, and a `LessonWriter` that auto-derives failure-recovery heuristics. Two
problems surfaced:

1. **It uses the word "memory."** That word belongs to **claude-mind** (the cognitive layer). A
   harness that "has memory" blurs the substrate/Mind boundary the whole project is built on.
2. **It does cognition by default.** When on (the CLI turns it on), the harness *decides what is
   worth keeping*, *what is relevant now*, and *what was learned*. That is the Mind's job, not the
   substrate's — and it should not happen when you are just exploring a repo in a read-only posture.

This is the same boundary the skills work already drew: `ON_SKILL_SELECTED` is *"the seam a learning
mind records from — the substrate emits the signal; choosing/learning is the mind's job."* This doc
generalizes that principle to persistence.

## What already exists (the two layers)

The codebase already has **two distinct persistence layers** — which is what makes the cut clean:

| Layer | Where | What it does | Verdict |
| --- | --- | --- | --- |
| **Session/transcript store** | `session/store.py` (`Session`, `SessionStore`) | Persists the full conversation + tool calls + usage + resume metadata, one versioned JSON per session id. Powers `/resume`. | **Mechanical. Keep. Not "memory."** |
| **Cross-session "memory"** | `memory/` + `tools/builtins/memory.py` + `agent/lessons.py` | SQLite note store, relevance **recall hook**, `remember`/`recall` tools, auto **lesson** derivation+write. | **Cognitive. → inert seams (this doc).** |

The first layer is the honest "cross-session chat / tool-calling store" — and it is *already* not
called memory. The second is the part that crosses into Mind territory.

## The boundary

> **Substrate (harness)** = the *record of what happened* + *inert seams*.
> **Cognition (claude-mind)** = *memory* — recall, consolidation, learning, preference — built on
> top of that record and those seams.

### Decisions (agreed)

1. **`SessionStore` persists ALWAYS, in every mode.** The transcript *is* `/resume`; gating it would
   break resume. It is mechanical state, not memory, so it is exempt from the "no memory in reader
   mode" concern — it is a log of what happened, not a cognitive recall.
2. **The cognitive pieces become INERT SEAMS.** The harness ships the *hook points* and the *store
   contract*, wired to **do nothing by default**. claude-mind hooks into them later and supplies the
   behavior. No default SQLite store, no recall injection, no auto-lessons, no `remember`/`recall`
   tools until a Mind wires them.
3. **Remove the `enable_memory` feature flag.** There is no "harness memory" to switch on. The seams
   are always *present* (so a Mind can attach) and always *inert* (so a bare harness does nothing).
   Because the cognitive layer is inert by default, there is **no mode-gating to build** — "reader
   mode saves no memory" falls out for free (there is no active memory to save).
4. **"Memory" is reserved for claude-mind.** The harness uses mechanical names (transcript / session
   store / note store / recall *seam*). See *Naming* below.

### Why not just delete it?

Because the Mind needs somewhere to plug in. Deleting the seams would force claude-mind to re-add the
hook points and the provider contract from scratch. Keeping them *inert* gives the Mind a clean
attach surface while a bare harness stays pure. (And it answers *"why maintain it if the Mind turns
it off?"* — we are **not** maintaining active cognition; we are maintaining an empty socket.)

## Per-piece plan

| Piece | Today | Boundary (inert seam) |
| --- | --- | --- |
| **Recall hook** (`MemoryRecallHook`, a `PRE_LLM_CALL` context hook) | Agent registers it when `enable_memory`; it searches the store and injects hits. | The seam is the **existing `hook_manager.register_context(...)`** point. The harness registers **no** default recaller. claude-mind registers its own. (Reference impl may stay as an importable utility or move to the Mind.) |
| **Lesson-writer** (`agent/lessons.py`) | The loop derives a recovery lesson and **writes** it to the store. | The loop **emits a lifecycle signal** at the recovery point (e.g. `ON_TURN_RECOVERED`, carrying the failed-then-succeeded command / stuck call) and **writes nothing**. A Mind subscribes and decides whether to persist. Deterministic derivation may stay as a payload helper. |
| **`remember` / `recall` tools** (`tools/builtins/memory.py`) | Registered when `enable_memory`. | **Not registered by default.** The seam is the tool registry — a Mind (or an explicit opt-in) registers them. Classes stay available. |
| **Store** (`MemoryProvider` / `SqliteMemoryProvider`) | Instantiated when `enable_memory`. | **Not instantiated by default.** The abstract contract is the seam (the Agent already accepts an injected provider). `SqliteMemoryProvider` stays an available implementation a Mind can choose. |
| **`SessionStore`** | Persists the transcript. | **Unchanged. Always on.** Powers `/resume`. |
| **`enable_memory` flag + config** | Gates the block above. | **Removed.** |

Net code shape: gut the `if enable_memory:` block in `src/zakcode/__init__.py` (it is the only thing
that *activates* cognition), keep the injection/registration seams, and convert the `LessonWriter`
auto-write into a hook emission. `SessionStore` is untouched.

## Naming (open)

"Memory" leaves the harness. Settled: the transcript layer stays `Session` / `SessionStore`. Open:
what to call the (now-inert) store *contract*, since it is no longer the harness's memory but the
socket a Mind plugs memory into. Candidates:

- `NoteStore` / `Note` — neutral; "a place to put notes," no cognitive claim.
- `RecallStore` — names the seam by what the Mind uses it for (recall), without claiming the harness recalls.
- `CrossSessionStore` — literal.
- Keep `MemoryProvider`, but *only* as "the contract claude-mind's memory implements." (Weakest — keeps the word.)

Leaning `NoteStore`, but parking it until the refactor.

## Open research questions

- **Recovery signal shape.** New `ON_TURN_RECOVERED` event vs. extending `TURN_END` with recovery
  data? What exactly is in the payload (the derived lesson text, or the raw recipe/stuck signals)?
- **Where does the deterministic derivation live?** Harness (build the payload) vs. claude-mind
  (derive from raw signals). It reads harness-internal recipe/stuck state, which argues for staying.
- **Do the reference impls (`SqliteMemoryProvider`, `MemoryRecallHook`) stay in the harness as
  importable utilities, or move into claude-mind?** Keeping them is convenient for a no-Mind user who
  *explicitly* opts into a note store; moving them is more boundary-pure.
- **CLI/server defaults.** With the flag gone, the CLI/server simply stop wiring cognition. Confirm
  no surface still assumes a store exists.
- **Coordination.** The memory subsystem may be part of omni's "Mind-seam" domain — agree the cut
  with omni before deleting/relocating, even though the *harness-side* changes (inert wiring, flag
  removal, rename) are the dev's.

## Implementation sketch (when we proceed — NOT yet)

1. Convert `LessonWriter` auto-write → a fired lifecycle hook; no harness write.
2. Remove the `if enable_memory:` activation block; keep the injected-provider + `register_context`
   + tool-registry seams. Remove `enable_memory` from core/CLI/server + the config fields.
3. Rename the store contract away from "memory" (see *Naming*).
4. Tests: a bare harness wires **no** store / recall / tools / lesson-write; the seams still accept a
   Mind-supplied provider/hook; `/resume` still works in every mode.
5. Docs: ADR for the boundary; update ARCHITECTURE / CONFIG / INTEGRATIONS; this doc → "accepted."

---

*This is a living research doc. Edit it as the boundary sharpens; promote the settled parts to an ADR
when we implement.*

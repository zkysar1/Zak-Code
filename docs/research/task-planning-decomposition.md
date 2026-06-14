# Task Planning & Decomposition in Agent Harnesses — Research & Recommendations for Zak Code

**Date:** 2026-06-14 · **Scope:** how best-in-class agent harnesses and frameworks do near-term
task planning and decomposition, and what Zak Code should change. Produced by a multi-agent
deep-research sweep (~25 systems, ~40 papers/articles). Claims are cited; where a primary page
was unreachable (many vendor docs 403 automated fetchers) the finding rests on search-engine
extracts of the same canonical URL and is flagged. Two decision-critical claims were verified
by direct primary-source fetch (Claude Code Task tools; Codex `update_plan`).

---

## TL;DR — verdict on our design

Our first-layer planning substrate (a model-driven `update_plan` tool, single `in_progress`
focus, live re-injection each iteration, a bounded self-arming completion gate, per-session
persistence) is **squarely on the best-in-class path** and, in several respects, matches what
the strongest harnesses converged on independently. We got the *fundamentals* right.

The evidence points to **four upgrades**, in priority order:

1. **P0 — Generalize the completion gate from "run the script" to the project's real verifier
   (tests/lint/typecheck) when one is discoverable.** This is the single most evidence-backed
   lever in the entire literature: *external, sound verification beats LLM self-critique by a
   wide margin*, and "did the tests pass?" is exactly the sound verifier that coding affords.
2. **P0 — Add optional task *dependencies* (`blocked_by`) to the network.** The frontier moved
   from flat lists to dependency-aware plans (Claude Code's new `TaskUpdate` `addBlocks`/
   `addBlockedBy`, LlamaIndex `SubTask.dependencies`, LLMCompiler/Devin DAGs). Dependencies — not
   parent/child nesting — are what enable correct ordering and safe parallelism.
3. **P1 — Tie decomposition to *capability/failure*, not up-front guessing (ADaPT), and add an
   anti-over-planning floor.** Decompose a step when the executor actually stalls on it; skip
   planning for trivial tasks (Codex: skip the easiest ~25%, "no single-step plans").
4. **P1 — Route decomposition through a cheaper "planner" model when configured.** Strong-planner
   + weak-executor is a proven cost/quality win and fits our local-model-first stance.

A crucial framing caveat runs through all of this (§4): on SWE-bench, the *simplest* scaffolds
(Agentless, mini-SWE-agent, Anthropic's 2-tool ReAct) match or beat heavy planning machinery. So
our planning layer must justify itself by **weak-model support, multi-turn coherence, and UX —
not by chasing benchmark points with complexity.** Keep it sharp and self-effacing.

---

## 1. The landscape (what real systems actually do)

### 1a. Interactive coding harnesses — the directly comparable cohort

| System | Plan structure | Tool | Update semantics | One `in_progress`? | Re-injection to model |
|---|---|---|---|---|---|
| **Claude Code (classic)** | flat list `{content, activeForm, status}` | `TodoWrite` | full-replace | **enforced** ("exactly ONE", stated 3×) | tool_use stream + CLI panel |
| **Claude Code (≥v2.1.142, default)** | **DAG** (`addBlocks`/`addBlockedBy`, `owner`, `metadata`) | `TaskCreate`/`TaskUpdate`/`TaskGet`/`TaskList` | **incremental patch by `taskId`** | lifecycle | `TaskList` snapshots |
| **OpenAI Codex CLI** | flat `{step, status}` | **`update_plan`** | full-replace | **enforced** by tool | TUI |
| **Cursor** | flat `{id, content, status(+cancelled)}` | `todo_write` | **merge flag** (add vs patch) | **enforced** | UI panel, silent |
| **Roo Code** | checklist, 3 visual states | `update_todo_list` | (full rewrite) | — | **"REMINDERS" table in `environment_details`** every turn |
| **Cline** | markdown checklist (2-state) | Focus Chain | regenerated wholesale | no `in_progress` | **re-injected every 6 messages**, survives summarization |
| **Amp (Sourcegraph)** | flat `{id, content, status, priority}` | `todo_write`+`todo_read` | full-replace | prescribed | `todo_read` on demand |
| **GitHub Copilot (VS Code)** | structured list | `manageTodoList` | — | implied | **maintained by a cheap background model** |
| **GitHub Copilot (cloud)** | markdown checklist in PR body | — | check-off + commits | — | the PR body |
| **Windsurf/Cascade** | `plan.md` + in-chat todo | — | background planner refines | — | `plan.md` persists |
| **Aider / Continue / Zed** | **none** (one-shot pass / read-only mode / subagents) | — | — | — | — |

Reading: a **model-driven plan tool + single `in_progress` + re-injection** is the dominant,
near-universal convention (and clearly shares a Claude Code lineage). Our `update_plan` is in this
exact family — same name as Codex's. *Sources:* code.claude.com/docs/en/agent-sdk/todo-tracking
(verified); github.com/openai/codex `plan_spec.rs` (verified); CL4R1T4S Cursor 2.0 prompt;
docs.roocode.com; docs.cline.bot + cline.bot/blog context-engineering; gist gregce/9ae20efc…
(Amp prompt); code.visualstudio.com v1.119; docs.windsurf.com; aider.chat/docs/usage/modes.

### 1b. Autonomous SWE agents — and the "simple beats agentic" result

- **SWE-agent** (NeurIPS'24): single-agent **ReAct, no planner, no decomposition, no subagents**.
  Its thesis is the **Agent-Computer Interface** — a compact command set + a file editor whose
  **linter rejects syntactically-invalid edits**. Ablation: the ACI is worth **+10.7 points** vs a
  raw shell — *interface mattered far more than any planning module.* (arxiv 2405.15793)
- **Agentless** (FSE'25): deliberately **no agent / no LLM-chosen actions** — a fixed
  localize→repair→validate pipeline (localization itself hierarchical). **32% SWE-bench Lite at
  $0.70**, beating all open agents then; **adopted by OpenAI** for GPT-4o/o1 numbers. (arxiv 2407.01489)
- **mini-SWE-agent**: ~100 lines, bash only, no planner/tools/retriever — **>74% SWE-bench Verified.**
- **Anthropic's harness**: deliberately **minimal — two tools (bash+edit)**, explicitly *rejected*
  retries/best-of-N/MCTS. (anthropic.com/research/swe-bench-sonnet)
- **TRAE** (ByteDance, #1 Verified ~75%): wins via an **ensemble (generation/pruning/selection) +
  dual-verification selector + test-time scaling**, not deeper single-agent planning. (arxiv 2507.23370)
- **Devin**: explicit **user-gated interactive plan**, represented as a **DAG**, with
  plan-execute-observe-**replan**; "managed Devins" run in **isolated VMs and validate their own
  changes before reporting back**. (docs.devin.ai/work-with-devin/interactive-planning)
- **OpenHands/CodeAct**: default is interleaved ReAct over a **code action space**; an explicit
  `planner_agent` (add/modify tasks, `verified` state) exists but is **optional, not the default**;
  delegation via `AgentDelegateAction`, but the result **summarized back to the parent was a crude
  key:value string concat** (open TODO for an AI summary) — a cautionary data point on lossy
  hand-offs. "Done" is **self-declared**; the harness only enforces loop/resource safety (a Stuck
  Detector, max-iterations/budget). (All-Hands-AI/OpenHands source, v0.11.0)

### 1c. Planner frameworks — and what they *abandoned*

- **LangGraph plan-and-execute**: plan = **flat list of NL strings**; an explicit **replan node**
  re-derives the remaining plan after each step; "ReAct only plans one sub-problem at a time…
  isn't forced to reason about the whole task." Benefits cited: explicit long-horizon planning +
  **use a small model to execute, big model only to plan**. LLMCompiler variant uses a **task DAG**
  (3.7× speedup). (blog.langchain.com/planning-agents)
- **CrewAI**: `planning=True` is a **single up-front pass** that writes a step plan string and
  **appends it to each task's `description` by position** — no replan loop. Hierarchical work =
  manager agent + `DelegateWork`/`AskQuestion` coworker tools (documented delegation bugs). (crewAI source)
- **LlamaIndex**: the one framework with a **typed, refinable plan** — `StructuredPlannerAgent` →
  `Plan{sub_tasks:[SubTask{name,input,expected_output,dependencies}]}` (a **DAG**), `create_plan`
  up-front + `refine_plan` after. Known to **collapse to a single-task plan** in practice (issue #16625).
- **AutoGen Magentic-One**: a rich model worth borrowing from — a **Task Ledger** (verified facts /
  facts-to-look-up / derived facts / **guesses** / plan) + a **Progress Ledger**
  (`is_request_satisfied`, `is_progress_being_made`, `is_in_loop`, next speaker). A **stall counter**
  (`n_stalls ≥ max_stalls`) **triggers replanning.** (microsoft research; autogen source)
- **Semantic Kernel — the cautionary tale**: shipped *five* planners (Sequential=XML, Action,
  Stepwise=ReAct, Handlebars=template program, FunctionCallingStepwise) and **deprecated/removed all
  of them**. Stated rationale, verbatim: native cross-model **function calling** is "more powerful
  and easier… uses fewer tokens"; bespoke planners "required multiple LLM calls" and "can reduce the
  speed, cost, and accuracy of a plan." (learn.microsoft.com/semantic-kernel/concepts/planning) →
  **Lesson: never invent a plan DSL the model wasn't trained on; lean on native tool-calling + a
  simple schema.** (Our `update_plan` already does exactly this.)
- **OpenAI Swarm/Agents SDK**: deliberately **no plan object** — handoffs + agents-as-tools; "code
  orchestration is more deterministic… LLM orchestration for open-ended."

### 1d. The lineage (where the task-list idea came from)

**BabyAGI** (task-creation + prioritization agents, a re-prioritized task *list*, no done-gate →
loops) and **AutoGPT** (reactive self-prompting loop, `thoughts/reasoning/plan/criticism`, no
verification gate → derails) established the pattern *and* its failure mode: **a mutable task list
with no verifier never converges.** Every serious successor added either explicit planning, a
verifier, or both.

---

## 2. What the evidence says (the cross-cutting questions)

1. **Flat vs hierarchical vs DAG?** The base is a flat list; the *frontier* is a **dependency DAG**
   (Claude Code new Task tools, LlamaIndex `SubTask.dependencies`, LLMCompiler, Devin, Hermes).
   Pure parent/child *nesting* within one todo tree (our compound/primitive model) is **uncommon** at
   this layer — most "hierarchy" in the wild is expressed via **subagent nesting**, not within the
   todo structure. Our hierarchy is defensible, but **dependencies are the higher-value axis.**
2. **Model-driven vs harness-enforced?** Default loops are **model-driven (strongly prompted)**:
   Claude Code/Amp instruct "you must"/"it is unacceptable not to," backed by soft nudges.
   **Harness-enforced** planning shows up as opt-in *plan modes* that gate execution on an approved
   plan (Claude Code Plan Mode, Cursor, Devin, goose `/plan`). Measured: explicit planning improves
   long-horizon completion (Plan-and-Act; LangChain) **but hurts simple tasks** (overthinking). Our
   *hybrid* (model-driven tool + self-arming gate) is the validated middle.
3. **Just-in-time vs up-front?** Evidence favors **as-needed/recursive (ADaPT)**: decompose *only
   when the executor fails*; depth then tracks the task's intrinsic "recipe depth." Always-plan is
   expensive and hurts long horizons; never-plan caps performance (Learning-When-to-Plan, 2509.03581).
   Hand-authored decomposition templates **don't generalize** past their authored size (Chain of
   Thoughtlessness, 2405.04776) → prefer dynamic, failure-triggered decomposition.
4. **Keeping the plan salient (cache-safe)?** Universal practice is **re-injection / "recitation"**:
   Manus pioneered it (constantly rewrite `todo.md` so the goal is "recited into the end of the
   context," after models forget goals ~50 tool calls in); Anthropic frames it identically
   ("rewriting the todo list pushes the global plan into the model's recent attention span, avoiding
   lost-in-the-middle"); Cline re-injects every 6 messages (survives summarization), Roo puts a
   REMINDERS table in `environment_details`, Claude Code re-reads via `TaskList`. **The cache
   interaction has a precise, documented answer: re-injection is cache-safe iff you APPEND at the
   tail, never mutate the prefix.** Caching is exact-prefix ("a change anywhere in the prefix
   recomputes everything after it"), so the rule is *freeze the system prompt + tool defs, put the
   volatile plan in the newest message*; Claude Code literally appends a `<system-reminder>` rather
   than editing earlier context. Recitation (wants the plan at the tail for recent-attention) and
   caching (wants the prefix frozen) are therefore **aligned, not in tension** — and our ephemeral
   tail injection is exactly this pattern. The failure it fights — **plan/goal drift** — is severe
   and quantified: agents "fail to revise stale steps, forget earlier steps (context rot), and
   prematurely conclude," and **even Claude Opus 4.5 abandons its own plan >50% of the time**
   (2601.17915; drift mechanism 2505.02709, 2509.03581). *This >50% abandonment rate is the single
   strongest justification for harness-enforced salience + a completion gate — i.e. our design.*
5. **Decomposition depth / single vs parallel focus?** Stop when the executor *can* do the step;
   recurse when it can't (ADaPT). Right-size granularity (actionable but not bloated). **Single
   `in_progress` focus is the near-universal convention** (Claude Code/Cursor/Codex/Amp) — we match it.
6. **Planning vs delegation boundary?** Converged rule: **read-parallel, write-sequential** (Cognition,
   LangChain); delegate an **isolated** subtask and **summarize back — but beware the lossy boundary**
   (OpenHands' crude concat; Cognition's edit-apply anti-pattern). Long-horizon is handled by
   *context compression* or a *separate planning layer*, not subagent fan-out (Cognition).
7. **Failure modes + mitigations:** plan/goal drift → recitation + goal re-anchoring; stale plans →
   reset on completion; loops/deadlock → **cap consecutive errors (~3) then escalate** (12-factor) and
   prefer a reset over "try to break out" (instructions alone are weak; 2512.04307); over-planning →
   complexity-gated planning; lossy delegation → richer/structured summaries.

The **strongest single result**: *LLMs are poor self-verifiers; external sound verification is what
delivers the gains* (Kambhampati group — 2402.08115, 2310.12397; corroborated 2310.01798). In coding
the sound verifiers are **tests, compilers, linters** — which is why SWE-bench gates on
`FAIL_TO_PASS`+`PASS_TO_PASS`, SWE-agent lints every edit, and TRAE runs a dual-verification selector.

---

## 3. Zak Code scorecard

**Validated — keep as-is (these match or lead best-in-class):**
- Native model-driven `update_plan` tool with a **simple JSON schema** (not a DSL) — Semantic Kernel's
  hard-won lesson; same family as Codex `update_plan`, Claude `TodoWrite`, Amp `todo_write`.
- **Single `in_progress` focus** — near-universal convention.
- **Re-injecting the live plan each iteration as an ephemeral, cache-safe tail** — the recitation
  pattern, the documented anti-drift mitigation. We do this correctly (prefix untouched).
- **Self-arming gate (fires only on multi-step work)** — the validated "don't over-plan" posture.
- **Bounded gate → completes `degraded`, never deadlocks** — matches 12-factor's bounded-error stance.
- **Resetting completed plans across turns** — directly addresses the stale-plan failure mode.
- **External verification via the recipe gate (run the written script)** — the right *kind* of gate
  (external, sound), just narrow today.
- **Full-replace authoring** — robust for weak local models (our priority); matches Codex/Amp. The
  patch-by-id alternative (Cursor/Claude-new) is a scaling option, not a correctness need.

**Gaps / changes:** see §4.

**Honest caveat:** the "simple beats agentic" SWE-bench evidence means our planning layer should never
grow into heavy orchestration. Its justification is **weak-model support** (ADaPT: externalize the plan
to help a weak executor; small models can't absorb long internal chains — 2502.12143), **multi-turn
coherence**, and **UX visibility** — not benchmark-chasing.

---

## 4. Prioritized recommendations

### P0 — highest leverage  ✅ IMPLEMENTED 2026-06-14 (both R1 and R2)

**R1. Generalize the verification gate from "run the script" to the project's real verifier.**
**[IMPLEMENTED]** — `agent/verify.py` `VerificationGate` + `Settings.verify_command`; wired into both
loop paths between the recipe gate and the plan gate, ending `verification_failed` (degraded) after
a bounded number of attempts. Domain-agnostic (operator/mind/skill provides the command; inert when
unset). Original recommendation:
Today `RecipeCursor` gates completion on running a written *script*. The evidence says the biggest
reliability lever is gating "done" on a **sound external verifier**. Extend the gate so that, when a
verifier is *discoverable*, a turn that changed code can't finish `completed` until it passes — e.g.
a tests/lint/typecheck command. **Keep it domain-agnostic:** do not hardcode `pytest`/`poe`; let the
verifier be **provided by a skill or detected** (and fall back to today's behavior when none is
known). This is the coding analogue of SWE-bench's test gate and SWE-agent's per-edit linter, and it
operationalizes the report's strongest finding. Note even Claude Code only *instructs the model*
("ONLY mark a task completed when you have FULLY accomplished it… not if tests fail") — that is
model-driven self-checking, which a model abandons >50% of the time; making it **harness-enforced**
is precisely the upgrade. *Risk:* over-gating/slow turns — make it bounded (like the recipe gate),
skippable, and only-when-a-verifier-is-known.

**R2. Add optional task dependencies (`blocked_by`) to `TaskNetwork`.**
**[IMPLEMENTED]** — `Task.blocked_by`; `normalize()` sanitizes edges into a DAG (drops self/unknown/
cyclic, fail-open); `current()` respects dependencies with a fail-open fallback; `update_plan`
exposes `blocked_by`; `render()` annotates "(after …)". Original recommendation:
The frontier is dependency-aware plans (Claude Code `addBlocks`/`addBlockedBy`, LlamaIndex
`SubTask.dependencies`, LLMCompiler/Devin DAGs). A `blocked_by: list[task_id]` field lets `current()`/
`actionable_remaining()` respect real ordering and unlocks safe parallel execution of independent
leaves later. This is additive to our compound/primitive model and arguably **more useful than deeper
nesting**. Keep it optional (empty = today's behavior).

### P1 — solid, evidence-backed  ✅ IMPLEMENTED 2026-06-14 (R3 + R4)

**R3. Capability-triggered decomposition + an anti-over-planning floor (ADaPT).** **[IMPLEMENTED]** —
`AgentLoop._decompose_hint()` appends a "break this step into sub-steps" suggestion to the stuck-ladder
nudge when the current step is a primitive; the prompt's planning guidance now sets a ~3-step
complexity floor and warns against over-decomposition. Original recommendation:
- Wire our existing **`StuckTracker`** to decomposition: when a *primitive* step trips the stuck
  ladder, nudge the model to **decompose that step into sub-steps** (decompose-on-failure), rather
  than only narrowing tools. This is ADaPT's core mechanism, reusing machinery we already have.
- Tighten the planning trigger in `prompt.py` to a **complexity floor**: plan for ~3+ distinct
  actions (Claude Code's threshold), explicitly **skip trivial tasks and forbid single-step plans**
  (Codex), and warn against over-decomposition (overthinking inverse-U). Optionally add a soft nudge
  if the model over-plans a trivial request.

**R4. Route decomposition through the `planner` role model when configured.** **[IMPLEMENTED]** —
the planner-role routing already existed (`plan_def = PLAN.model_copy(update={"model": roles.get("planner")})`);
made concrete by adding `update_plan` to the planner's toolset so it produces a *structured* plan
(not just prose) on the planner model. We kept the single-threaded inline design (no planner/executor
split) per the "keep it sharp" caveat. Original recommendation:
We already have `Settings.model_roles['planner']` for the PLAN sub-agent. Strong-planner + weak-executor
is a proven win (ADaPT; LangChain "big model plans, small executes"; goose `GOOSE_PLANNER_MODEL`). Let
the *decomposition* path optionally use the planner model — a cheap, local-model-friendly lever.

### P2 — optional / future  ✅ R5 + R6 IMPLEMENTED 2026-06-14; R7 deferred (per its own recommendation)

**R5. Optional "review plan before executing" gate (Plan Mode for the main loop).** **[IMPLEMENTED]**
as an opt-in, off-by-default `Settings.require_plan` harness gate: the first *mutating* tool is
withheld until a plan exists (read-only work is never gated), bounded → fails open, never deadlocks.
(Shipped as harness-enforced "plan before you act" rather than interactive approval, which fits the
headless/autonomous posture.) Original recommendation: Claude Code, Cursor,
Devin, and goose all gate execution on an approved plan. We have a read-only PLAN *sub-agent* but no
"approve the plan, then execute" checkpoint in the main loop. Lower priority given our near-term
auto-decomposition altitude, but a natural extension and a real human-in-the-loop safety feature.

**R6. Richer delegation summaries (guard the lossy boundary).** **[IMPLEMENTED]** — every sub-agent's
prompt now carries a shared structured-handoff instruction (`subagent._HANDOFF`): end with a
self-contained summary (what you did, key files/decisions, what remains), since the final message is
the only thing passed back. (We already serialize writes by path — read-parallel/write-sequential.)
Original recommendation: Our `task` tool returns child summaries;
OpenHands' crude concat and Cognition's edit-apply anti-pattern warn this boundary loses implicit
context. Keep **read-parallel/write-sequential** (we already serialize writes by path) and consider a
structured handoff (objective + outputs + key decisions) over a bare summary.

**R7. (Watch, don't build yet — DEFERRED, per this recommendation) A facts/assumptions ledger.** Magentic-One's Task Ledger separates
*verified facts vs guesses* from the plan — powerful for long horizons but likely over-engineering for
our near-term layer. Note for the higher-level "mind," not the core.

---

## 5. Key sources

**Verified by direct fetch:** code.claude.com/docs/en/agent-sdk/todo-tracking ·
raw.githubusercontent.com/openai/codex …/plan_spec.rs · github.com/12-factor-agents (factors 3/8/9/10)
· raw MicrosoftDocs/semantic-kernel-docs concepts/planning.md · autogen `_magentic_one_orchestrator.py`
· All-Hands-AI/OpenHands `agenthub` source · github.com/openai/swarm.

**Harness docs/prompts (via extract):** CL4R1T4S Cursor 2.0 prompt · docs.roocode.com ·
docs.cline.bot + cline.bot/blog context-engineering · gist gregce/9ae20efc (Amp) · cursor.com/docs/agent/plan-mode
· docs.devin.ai/work-with-devin/interactive-planning · cognition.ai/blog/{dont-build-multi-agents,
multi-agents-working,swe-bench-technical-report} · anthropic.com/{engineering/building-effective-agents,
engineering/built-multi-agent-research-system, research/swe-bench-sonnet} · block.github.io/goose ·
crewAIInc/crewAI source · run-llama issue #16625 · openai.github.io/openai-agents-python.

**Papers:** ReAct 2210.03629 · Least-to-Most 2205.10625 · Decomposed Prompting 2210.02406 ·
Plan-and-Solve 2305.04091 · ToT 2305.10601 · GoT 2308.09687 · Self-Discover 2402.03620 ·
Reflexion 2303.11366 · Self-Refine 2303.17651 · LLMs-Cant-Self-Correct-Yet 2310.01798 ·
**ADaPT 2311.05772** · PlanBench 2206.10498 · LLM-can't-plan ranking 2305.15771 ·
**LLM-Modulo 2402.01817** · GPT-4-doesnt-know-its-wrong 2310.12397 · **Self-Verification Limits 2402.08115** ·
LLM-Modulo TravelPlanner 2405.20625 · formal-verification planning 2404.11891 ·
Chain of Thoughtlessness 2405.04776 · LLMCompiler 2312.04511 · HuggingGPT 2303.17580 ·
CodeAct 2402.01030 · SWE-agent 2405.15793 · SWE-bench 2310.06770 · Agentless 2407.01489 ·
OpenHands 2407.16741 · TRAE 2507.23370 · Overthinking 2502.08235 · Small-Models-Struggle 2502.12143 ·
Learning-When-to-Plan 2509.03581 · Goal-Drift 2505.02709 · HTN+LLM roadmap 2501.08068.

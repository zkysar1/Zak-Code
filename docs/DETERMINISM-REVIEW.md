# Determinism review — where zakcode decides by code, where it asks the model, and what to change first

**Status:** living review, opened 2026-09-12 (ADR-0163). **Scope:** the agent loop as of `main@b0ee395`
(ADR-0162 merged). **Directive it serves:** *"make zakcode start to get better over Claude Code by
making it more deterministic and working well with smaller models."* Line references are to that
commit; they drift, the mechanism names do not.

## 1. What "deterministic" means here

Three different properties get called determinism, and this review keeps them apart:

1. **Model-free steps** — a behaviour the harness performs by code, so it happens the same way for a
   27B and a frontier model. Folding `CONTRIBUTING.md` into the prompt (ADR-0162) is one; "read the
   conventions before you write" as a prompt line is not.
2. **Reproducible outputs** — the same input produces the same bytes. zakcode has a byte-deterministic
   configuration (temperature 0 + stable prompt identity + pinned workspace, ADR-0157) that Claude Code
   as it ships does not. Reproducibility is what makes an A/B arm readable (ADR-0160: three repeats at
   temperature 0 are a reproducibility check, not three samples).
3. **Enforced rules** — a constraint the tool layer refuses to violate, as opposed to one the prompt
   asks for. The edit tool's exact-match refusal is enforced; "read a file before you edit it" is asked.

A lever can improve one without the others. The ranking in §5 says which property each moves.

## 2. Method

* A read-only inventory of `src/zakcode` (agent loop, prompt builder, tools, routing, rules/skills,
  compaction, hooks, bench coverage), then verification of every load-bearing claim by reading the
  cited code. The inventory is §3.
* Measured evidence only from pre-registered arms already in `docs/DECISIONS.md` (ADR-0147–0162) or
  from arms this review pre-registers (`bench/results/*-preregistration.log`, box-clock stamped
  before launch). A finding without a measurement is marked *unmeasured*.
* The head-to-head instrument (ADR-0161): six tasks, Claude Code 2.1.267 on Fable 5.1 vs zakcode on
  `zds-qwen3.6-35b` and `zds-qwen3.8-27b`; with ADR-0162 it reads 12/12 PARITY, so it can no longer
  discriminate. New tasks are part of the work, not a precondition for it.

## 3. Inventory — decision points

| Decision point | Today | Mechanism | Evidence |
|---|---|---|---|
| Iteration bound, cost/token caps, 13 stop reasons | code | `AgentLoop._grant_iteration`, `budget_exhausted` (`loop.py`), `config.py:350-355` | — |
| Provider retries / 429 | code | `_complete_with_retry`, `_retry_delay` (900 s horizon, equal jitter) | — |
| Undecodable tool-call JSON | code diagnoses, model retries | `loop.py:4292` (ADR-0081): cut-off vs escaping, remedy text, `_turn_struggle` latch | ADR-0081 |
| Provider-rejected tool call | code retries **at raised temperature** | `_rejection_retry_temperature` floor 0.5, step 0.3 | unmeasured trade-off (see F5) |
| Degenerate arguments / completion text | code | `degeneration.py`, vetoed before the permission gate | ADR-0024, ADR-0033 |
| Doom loop, stuck ladder | code | `_MAX_DOOM_RECOVERIES=1`; `stuck.py` NUDGE→NARROW→STEP_BACK→STOP, NARROW enforced at the execution seam | — |
| ~15 conclusion gates (claim, blocker, missing, silenced, attribution, identity, figure, intent, verdict, deferral, section, scope) | code (regex), once per turn, `gate_cascade` cap | `loop.py:6363-6602` | m01–m05 pass 3/3 on both local models (ADR-0161) |
| Completion critic, plan critique, quality gate | **model**, fresh context, fail-open; quality gate off by default | `_completion_critic`, `_judged_plan_critique`, `_quality_gate` | ADR-0011 |
| Verify after write — written script | code, always on | `RecipeCursor` (`recipe.py`), `_try_harness_verify` | — |
| Verify after write — project tests/lint | code **only if the operator sets** `verify_command` (default `None`) | `VerificationGate` (`verify.py`), `_try_project_verify` | **inert on every default install** (F4) |
| Write grounding (read-back + `compile()`) | code, always on | `grounding.py` | — |
| Read-before-edit | **prompt only** | `prompt.py:105`; `edit.py` edits a never-read file | m05 passes 3/3 on both models, so no measured failure (F3) |
| Edit exact-match, write content firewall, path sandbox, permissions, dependency gate | code | `edit.py:157-162`, `_safety.py`, `permissions.py`, `deps_gate.py` | — |
| Prompt tier order, tool cheat-sheet, environment section | code | `_build_stable`, `_summarize_tools`, `_environment_section` | ADR-0149, ADR-0053 |
| Project context: guides, `CONTRIBUTING.md`, README | code | `discover_context` (`prompt.py`): `AGENT_GUIDE_FILENAMES`, `CONVENTION_FILENAMES`, `README_FILENAME`, caps 8,192 / 32,768 | ADR-0161 (gap), ADR-0162 (closed) |
| Workspace file survey, test discovery, other convention files (`pyproject`, `Makefile`, `.editorconfig`) | **left to the model** | — | ADR-0161 dumps: both local models `list_dir` the root and still miss a listed file (F2) |
| Prompt cache key | code, toggleable | `_prompt_cache_key`: per-session, per-workspace under `stable_prompt_identity` | ADR-0157 (4th addendum) |
| Temperature | config, default *unset* | `config.py:162` | ADR-0018: harness-wide 0.0 was "fake determinism" |
| Routing quick vs deep | **hybrid**: pure classifier, `difficulty_hint` from a model side-call, fails up to `deep_code` | `routing.py:268-457` | ADR-0009, ADR-0035 |
| Skill selection | **model**, deterministically anchored | `implied_skill_anchored`, `_name_is_generic` | ADR-0158 (fidelity), ADR-0160 (catalogue cost) |
| Rules injection | code | `render` / `render_index`, `lean_rules`, `alwaysApply` | ADR-0105 |
| Compaction trigger | code (0.8 of window, 0.25 tail) | `compact.py` | ADR-0148 |
| Compaction summary | **model**; the position note is code | `_summarize_for_compaction`, `_compaction_position_note` | ADR-0148 |
| Hook seams | code dispatch; PreToolUse is the only vetoing seam; no pre-LLM-call seam | `hooks/__init__.py` | — |
| **Turn driver** | two implementations: `_run_turn` (buffered, 1,376 lines) and `astream_turn` (streaming, 1,738 lines) | `loop.py:5575`, `loop.py:6967` | **the bench calls `run_turn`; the CLI, server and client call `astream_turn`** (F1) |

## 4. Findings, ranked by what they put at risk

**F1 — The bench measures a driver production does not run.** Every number in ADR-0147–0162 came from
the buffered driver; the coach runs the streaming one. A static census finds the same 36 intervention
kinds in both, but the streaming driver reaches the provider through different helpers
(`_prompt_cache_key`, `_rejection_retry_temperature`, `_retry_delay`, `_busy_lease`, `_execute_tool_call`
vs `_call_provider` / `_execute_batch`). Whether outputs agree is empirical. **ARM D** (pre-registered
23:22, `driver-parity-preregistration.log`): `ZBENCH_DRIVER=stream` drives `astream_turn` from
`bench/run_task.py`; seven tasks on the 35B, N=3, byte identity against the buffered cells on the same
build. **Result (ADR-0163): D1 holds on all seven tasks** — 21 streaming runs byte-identical to the buffered
cells, same outcomes, same turn counts, same stop reasons — so every prior bench number transfers to
production's driver. Two drivers still means every lever below must be written twice or it applies to
one surface only; unification stays a determinism lever, now without an instrument-validity emergency
behind it. Wall time differed (streaming 2–5× slower per run in the chain); the latency control is in
the same log.

**F2 — Discovery is the measured small-model gap.** ADR-0161's only GAP was a rule in a file both local
models listed and never opened; ADR-0162 closed it for `CONTRIBUTING.md` by folding the file (model-free,
13 turns instead of a wrong 17–22). The shape generalises: nothing hands a small model the workspace's
file survey, its test layout, or conventions living under other names. Claude Code covers this by
exploring (its first tool call on 06 was `find`); a small model does not explore enough. **Measured
beyond 06 (ADR-0165):** the split is by filename, not by "convention". On `10-rule-in-pyproject` the 35B
opened `pyproject.toml` as its first file in 3/3 runs and wrote the house style unprompted; on
`09-contract-in-tests` it read the failing test file before its first write in 3/3 runs. A canonically
named config or test file is read; a prose convention file (`CONTRIBUTING.md` on 06) was not. Folding
(ADR-0162) is the lever for the prose class; the config and test classes need no fold on this model.

**F3 — Read-before-edit is asked, not enforced.** `edit_file` will edit a file the session never read.
On this instrument the prompt line suffices (m05 3/3 on both models), so this is a hardening lever:
turning a tendency into a guarantee, at the cost of one `read_file` when a model skips it.

**F4 — Project verification is inert by default.** `verify_command` defaults to `None`, so "run the
tests after editing" is model judgment on every default install; the written-script recipe is the only
always-on verifier. On 06 both local models ran pytest themselves, and pytest passed a wrong emitter
(pyyaml was installed) — so a test gate would not have caught that failure, but it is the obvious
model-free check for tasks whose tests encode the contract. *Unmeasured.*

**F5 — Rejection retries raise the temperature.** A provider-rejected tool call is retried at
temperature ≥0.5 to break a temperature-0 re-emit. That is a deliberate robustness/reproducibility
trade; under the byte-deterministic configuration it is the one path that samples. *Frequency
unmeasured on the pod*: the determinism-arm JSONs carry no retry counter (their per-run keys are
`cli_rc`, `digests`, `elapsed_s`, `num_turns`, `verify_rc`, `verify_out`, `stop_reason`, `sources`, …).

**F6 — Compaction summaries are model judgment.** Trigger and tail budget are code; the summary is a
model call with a code-generated position note appended. ADR-0148 measured the cost (25% more
iterations, 10× the variance). Not a small-model-specific gap; recorded for completeness.

**F7 — Routing is a hybrid with a model side-call.** The classifier is pure and fails up; only
`difficulty_hint` comes from a model. Under `ZBENCH_POD_MODEL` every category maps to one model, which
is why routing never entered the head-to-head; on the coach (zakpick, classify on the 35B) it does.
*Measured only indirectly.*

**F8 — The catalogue and skills.** ADR-0158/0160 already measured this: description fidelity drives
selection, catalogue size costs 1.4× wall time and reproducibility on a 27B, shortlisting is a cost
lever only. Nothing new here; the review just declines to re-open it.

**F9 — A model-free prompt lever is a basin move, not a free win (ADR-0164).** The workspace survey
(L2) is deterministic code and saved turns on every task it touched; it also flipped 06 on the 35B from
3/3 to 0/3, byte-identical both times, between two builds whose main-turn system prompts differ in
**one listing line** (a zakcode lease marker hidden, 11,067 → 11,061 chars). With the line the model
opened `CONTRIBUTING.md` as a fifth read and wrote a `- ` list emitter; without it, it skipped the read
and wrote a flat `key: value` emitter whose round-trip keeps only the last row. The read carried no
information (the rule text is already in the prompt since ADR-0162; the file has no format example): the
emitter design is a near-tie decided by context shape. Consequence for the smaller-model goal: every
prompt-side lever, including the ones ranked model-free in §5, must be scored on **outcomes** at N≥3 on
the near-tie tasks (06 on the 35B is the sentinel), never on turns or bytes alone; and the durable fix for
a near-tie is a check the model cannot skip (L4/L6: a round-trip or project test the loop runs), not a
better prompt. The pod adds its own residual on byte-identical prompts (02 7 vs 6 turns, m04's `count.md`),
which the no-survey control in the same log measured directly: without the survey, 06 flipped within its own
cell too (56 / 9 / 9 turns, a third emitter where ADR-0162 had 13 / 13 / 13) while m04 reproduced ADR-0162's
bytes exactly — the pod's instability is confined to the near-tie task, and E4 was not evaluable that night. **Second measured
instance (arm J, ADR-0164 second addendum):** the survey, default-on since #421, moved the 35B on task 10 from
3/3 in 9 turns to 1/3 in 16–17 by changing its reading order (config first → module first, docstring rule
missed), while removing every opening listing call on both models and both tasks and cutting the 27B's 10
from 15 turns to 8–11. The default is reopened: arm K scores it as a rate over sampled basins, and L4 (#429)
is the check the model cannot skip.
The measurement that settled it (ADR-0164 addendum): three identical runs of a deterministic pod are one
sample of one basin, so a basin move is scored as a **rate over sampled basins** — `--no-pin` gives every run a
fresh workspace path, six runs per arm; sampled that way the survey passed 6/6 against 5/6 without it and
halved the median turns, so it shipped default-on. The rule stands: score prompt levers on outcomes, over
sampled basins, on the near-tie tasks.

**F10 — The harness can manufacture the defect the model then chases (ADR-0166).** The post-green holes
that dominated every long 06 run on the 35B (25, 56, 25, 40, 23 turns across ADR-0164's arms E, E3 and G)
were not model wandering: after the first green `pytest` the model summarized, the recipe gate found a
written runnable with no run credited, and the harness verified it as `python -m plugins.yaml_out`.
`plugins/__init__.py` imports its renderers, so runpy warned `'plugins.yaml_out' found in sys.modules
after import of package 'plugins'` at exit 0, and the harness folded that into a trusted user message —
`[harness] I ran the file to verify it:` + warning + `[exit code: 0]`. The model read a warning the
harness produced as a defect in its own work and spent 15–45 calls on `-W error` probes, `filterwarnings`
and ImportError stubs; the injection fired on 12 of 12 runs measured (arm H), and the calls spent after
it — 0 to 31 — are the basin variable. Two defects sat under it, and the second was found only by
measuring the fix for the first. (1) A module with no `__main__` block is a library; `-m` is the wrong
verification for it and the import form (L8, #422) removes the warning — in arm H the calls after the
injection fell to 2 in 6 of 6 runs. (2) The harness should never have run at all after a green suite:
`extract_acceptance` read task 06's *"outputs YAML, registered under the name `yaml`"* as an
expected-stdout literal `yaml` (the 40-character gap after the cue verb crossed the comma), and with a
literal set a green suite is never credited while a run counts only if its output contains the literal —
which the runpy warning text did, by accident, and an import's empty output never can, so #422 alone
stalled every run at the attempt cap. #423 stops the literal at clause punctuation and rejects a naming
lead. Three properties matter for the smaller-model goal: a harness-injected message is a channel the
model cannot tell from a real defect report, so whatever the harness puts there must be actionable — a
warning at exit 0 is not; a gate whose credit depends on the *text* of a run can be satisfied by the wrong
text and starved by the right one; and a pre-registered rule set for a harness change must score the stop
reason and the harness runs per turn, not only passes and turns — arm H passed on its letter while every
run ended `recipe_stalled`. Arm I (both fixes, six unpinned basins per arm): harness runs per turn 1 → **0** in 6 of 6, `completed` 6 of 6 against `recipe_stalled` 6 of 6 on #422 alone, passes 5/6 vs 6/6 (the one failure a write-time design choice), median turns 11 → 10.

**F11 — A tool that reports success on a no-op feeds the small-model doom loop (ADR-0168).** Arm K's three
`doom_loop` runs on 10/35B share one shape: the model finishes a step, resends the plan byte-for-byte (statuses
untouched, sometimes an `outcome` on the finished step), and `update_plan` — a full-replace that compared
nothing — answers "Plan updated: 0/4 steps done · current: 1 …" with the rail "now do the step marked current".
The model reads that something moved and resends; the third identical batch trips the exact-repeat guard, whose
nudge is generic ("READ it now… take a DIFFERENT approach"), and the model writes a meta-plan and ends
`doom_loop` — two of the three before the export the task needs. The trigger is rare (3 of 163 dumped runs,
`bench/plan_resends.py`) and entirely a receipt defect: the fix is the tool saying "Plan unchanged … nothing was
updated" with the step-specific way forward, on the first resend, one iteration before the guard. Lever M
(#431), default-on; arm M measured it — and it FAILS to break the loop (M3): both fired NEW runs still
ended `doom_loop` (2/2, = OLD's 7/7), because the loop's root is not the receipt but that the 35B will not
flip a step's status to `done` even after doing the work and being told to (b3/run1: wrote the module,
exported it, wrote tests, 31 passing, then resent an all-`pending` plan six times). The true receipt stands
as a correctness/census fix; the real fix is a deterministic harness ADVANCE (lever N) — mark the step done
when it has evidence of completion and the model resends unchanged. A follow-up (#432) fires the rail on the
FIRST resend (compare the model's submission, not the non-idempotent network state).

## 5. Levers, ranked

| # | Lever | Property (§1) | Seam | How it is measured | Status |
|---|---|---|---|---|---|
| L1 | **One turn driver** (or a standing driver-parity cell in the bench) | reproducible, enforced | `loop.py` `_run_turn` / `astream_turn` | ARM D byte identity; then the full suite + bench byte identity as the refactor's control | ARM D held D1 on all 7 tasks (ADR-0163): the bench measures production's driver, so a one-driver refactor has its control; the refactor itself is unmeasured and unscheduled |
| L2 | **Turn-1 workspace survey** — capped, ignore-aware file tree folded into the dynamic tier | model-free | `prompt.py` `_build_context` | turns and wall time on 06/02 (same-trajectory latency arm, ADR-0160's design); outcomes must hold; bytes will change | **shipped default-on (ADR-0164 addendum)** — refused first on one pinned basin (06 3/3→0/3 on a one-line listing change), then measured by basin sampling over six unpinned paths: 6/6 vs 5/6, median turns 18→10, long runs 3→1, `CONTRIBUTING.md` opened 5/6 vs 1/6 (F9) — **reopened by arm J** (ADR-0164 second addendum: 35B on 10 3/3 → 1/3 with the survey, a docstring rule the lint would catch; 0 opening listing calls vs 3 on every pair; 27B on 10 15 → 8–11 turns); arm K decided it (ADR-0164 third addendum): **5/6 vs 5/6** over six unpinned basins per arm, listing calls 3 → 0 in every run — **the default stays on**; the live failure on 10 is F11's doom loop, not the docstring |
| L3 | **`edit_file` refuses a path not read this session** (a write of the same path counts) | enforced | pre-execution veto seam (`loop.py:4292` class) — in both drivers | no regression on m01–m05/06/02; refusal counter in the results JSON | measured (`bench/unread_edits.py`, #427): **0 of 361** `edit_file` calls in 127 pod runs touched a path the run had not read or written (control: with reads discounted, 157 of 361); the 361 known edits failed once — these models read before they edit, so the refusal has no trigger here |
| L4 | **Auto-derived `verify_command`** (detect `pytest`/`pyproject`/`Makefile test`) | model-free | `VerificationGate` construction | tasks whose tests encode the contract; needs at least one new task of that shape | **built opt-in (#429, `verify_auto`)**: a Makefile's lint/check/test recipes (else a pyproject ruff config) become the R1 gate's command; arm J found the trigger (the 35B never runs task 10's lint and ships a docstring the lint rejects under the survey); arm L stays drafted: arm K's twelve basins produced 0 lint-rejected outputs, so the gate would not have fired; launch when a basin census shows a lint-class failure |
| L5 | **More convention filenames** (`pyproject.toml [tool.*]`, `Makefile` targets, `.editorconfig`) | model-free | `CONVENTION_FILENAMES` | needs a task whose rule lives in such a file | needs tasks |
| L6 | **Test-file hint on edit** (append `tests/test_<name>.py` to the edit result when it exists) | model-free | grounding message | same tasks as L4 | needs tasks |
| L7 | **Repair, not bounce, a truncated `write_file`** | model-free | `loop.py:4295` `cut_off=True` | count of `cut_off` bounces on the pod first (unmeasured) | measured (`bench/undecodable_bounces.py`, #426): **0** bounces in 127 pod runs / 1,817 tool calls across 43 cells (27B + 35B, 2026-09-12 → 13; positive control 397 shell results) — the trigger is a long single-call module (ADR-0081's coach case) and no bench task is that size; nothing to repair until a task of that shape exists (joins L4–L6's "needs tasks") |
| L8 | **Verify a library module by import, not `-m`** — a package module without a `__main__` guard is checked with `python -c "import pkg.mod"`; `-m` stays for modules with a guard and for unreadable files | model-free | `recipe.py` `_python_run_command` / `_executed_targets` | long runs (> 20 turns) and median turns on 06/35B by basin sampling, OLD vs NEW build, outcomes must hold (ADR-0166 arm H) | **shipped (#422 + #423, ADR-0166)** — arm H: warning gone 6/6 but `recipe_stalled` 6/6 on a mis-extracted acceptance literal; arm I with the literal fixed: 0 harness runs, `completed` 6/6, 5/6 vs 6/6, median turns 11 → 10 (F10) |
| N | **Deterministic plan advance** — when the model resends the plan unchanged and a worked-on step is left non-terminal, the harness marks it done and walks the frontier | model-free, opt-in | `update_plan` `harness_advance` / `plan_autoadvance` | doom-loop rate among triggered runs (N3), pass rate (verify is ground truth), turns | **built opt-in (#433)** — arm M's M3 failure showed a text rail cannot move a 35B that will not emit `status: done`; lever N does it deterministically and makes the advance stick against the full-replace; arm N (flag OFF vs ON, basin-sampled) decides the default |
| — | Lint/format after write (extend `syntax_note`) | model-free | `grounding.py` | — | deferred: no measured failure |

**What not to do, because it was measured:** temperature 0 harness-wide (ADR-0018); a routing shortlist
for accuracy (ADR-0158: cost lever only); the full catalogue on a 27B (ADR-0160); rewriting skill
descriptions with the 35B (ADR-0158 fourth addendum).

## 6. Instrument notes

* The head-to-head instrument is saturated at 12/12. Task shapes worth building next, each chosen
  because a lever above needs it: a rule that lives in `pyproject.toml` or a `Makefile` (L5), a
  contract that only the existing tests state (L4/L6), a workspace large enough that the survey's cap
  matters (L2). **Built (ADR-0165):** `09-contract-in-tests` (L4/L6) and `10-rule-in-pyproject` (L5),
  each self-tested to fail on its seed and on every plausible wrong solution; Claude Code passes both
  in one attempt. Their baselines say which levers still have an outcome to buy — see the ADR.
* `bench/intervention_coverage.py --census` already lists which deterministic paths the bench never
  exercises; ADR-0154 measured that production leans on exactly those. Any new lever should land with
  a `kind=` intervention so the census can see it fire.
* `determinism_arm.py` stores the verifier's tail under `verify_out` and the stop reason under
  `stop_reason` since ADR-0161; older JSONs carry the stop reason in `verify_out`.
* `determinism_arm.py` captures `sources` for every small text file (`.py`, `.md`, `.txt`, `.csv`,
  `.json`, `.toml`, `.yaml`, `.cfg`, `.ini`, ≤4 KB) since ADR-0164; before that only `*.py`, so ADR-0164's
  m04 digest difference on `count.md` is a difference of unknown content.
* `bench/harness_verify_holes.py <cell-dir>...` reads a cell's wire dumps and prints, per run, the call at
  which the harness verify message first appears, its body kind, the calls the model made after it, and
  its first response — the hole-length instrument behind F10 / ADR-0166. The injection itself fires on
  every 06 run, so "calls after it" is the quantity a fix moves; `harness verify at call N` with no
  message means the model ran its own file before finishing and the harness never had to.
* `bench/plan_resends.py <dumps-root> [cell …]` replays every run's recorded `update_plan` calls through the
  real tool on a fresh network and prints one receipt letter per call (U updated, N unchanged, C cleared) — the
  positive control for ADR-0168's rail (fires in exactly the three arm-K doom loops over 163 runs, 12 receipts,
  and in no other run). Since that ADR the loop notes each unchanged receipt as `kind="plan_unchanged"`.
* Since #425 the loop notes every harness-issued run as an intervention (`kind="harness_verify"` with
  the target basename, the command form — tests / import / module / script — the exit code and the
  error flag; `kind="project_verify"` for the project checks), so `intervention_coverage.py` counts
  the one intervention the loop makes on the model's behalf without reading dumps. The census lists
  both as reachable; they show as recorded once a bench run fires them.
* `bench/undecodable_bounces.py <dump-root>` counts ADR-0081's undecodable-argument bounces over the
  wire dumps and prints its own positive control. Per run it reads the LARGEST dump, not the last:
  the turn-end side requests (structured output, system + user only) are a few KB and often sort
  last, and taking them scored 18 of 127 runs as "no tool calls" — the same wrong-file zero
  ADR-0165's mechanism reader made, caught here by the control.
* `bench/unread_edits.py <dump-root>` scores every `edit_file` call in the dumps as known or unread
  (a prior `read_file` / `write_file` / `edit_file` of the path) and by the result the model saw;
  `--reads-dont-count` is its positive control for the unread branch. Zero unread edits in 127 runs
  (ADR-0167).
* `intervention_coverage.py` reads arm-cell rows (`runs`) since #428; before, only suite files
  (`tasks`) counted, so no `determinism_arm.py` cell — 112 of 161 result files — was in the census.

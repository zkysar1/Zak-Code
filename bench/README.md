# bench/ — Lane-D model-routing benchmark harness

Headless task runner that drives the zak-code agent to completion in an isolated temp
workspace, then grades it with a **held-out oracle** (`verify.py`). Used to compare model
+ provider + tool-calling-mode combinations for the `deep_code` / `delegate` zakpick tier.

This harness is **excluded from the package gate** (`ruff` via `extend-exclude`, `pytest`
via `testpaths = ["tests"]`, `mypy` via `packages = ["zakcode"]`) — it is experiment code,
not shipped library code. The PR gate never runs it; run it manually (`uv run poe bench`) or via
the scheduled, key-gated `agent-bench` CI workflow (see [In CI](#in-ci)).

## Layout

```
bench/
  run_task.py            # run ONE task headless, print a JSON result
  run_suite.py           # run several tasks in parallel, print a summary table
  diag_task.py           # run ONE task and dump the tool-call transcript (why a gate fired)
  probe_tool_use_failed.py  # diagnostic: inspect a provider's raw text vs tool_calls
  run_bestof.py          # best-of-N(small) + judge-select vs 1-big on a task
  run_bestof_suite.py    # best-of-N vs 1-big across the whole suite (generalization)
  run_quality.py         # quality gate OFF vs ON on a task (the activation evidence)
  run_seam_b.py          # seam B live: best-of-N retry rescuing a STALLED turn
  run_skill_chain.py     # skills live: a model invoking 3 skills that daisy-chain (use_skill)
  run_skill_branch.py    # skills live: one skill ROUTES to one of two next-skills (conditional)
  skill_chain/skills/    # the 3 relay skills (relay-start -> relay-middle -> relay-finish)
  skill_chain/branch/    # the triage-start + handle-urgent/handle-normal branch skills
  tasks/
    01-wordfreq/         # task.json + held-out verify.py
    02-median-bug/       # + workspace/ seed files (bugfix task)
    03-lru/
    04-todo-cli/         # multi-file CLI stretch task (store.py + cli.py)
    05-ledger/           # multi-file CLI + atomic transfers (ledger.py + cli.py)
  results/               # run artifacts (.log/.json) — gitignored, regenerable
```

The runner puts ITS OWN interpreter dir first on `PATH` for the agent's subprocess tools
(`_ensure_interpreter_on_path`), so a bare `python -m pytest` the agent issues resolves to
this (pytest-capable) venv — mirroring a user running zakcode inside an activated project
venv. Without it the agent's subprocess hits the *system* python (often no pytest), its
verification fails, and the recipe gate stalls a turn whose code is actually fine — while the
oracle, run with the venv python, passes. (That mismatch was the old `recipe_stalled`-but-
oracle-passes artifact.)

A task dir holds `task.json` (`{id, title, prompt, max_iterations?, max_cost_usd?,
verify_timeout_s?}`), an optional `workspace/` seed copied into the temp run dir, and a
`verify.py` that exits 0 on success (run with `cwd` = the temp workspace).

## Running (from the Zak-Code repo root, with the repo venv)

```bash
# one task
./.venv/Scripts/python.exe bench/run_task.py bench/tasks/01-wordfreq

# cheap API smoke (constructs the agent, exercises cost/permission API, NO LLM call)
./.venv/Scripts/python.exe bench/run_task.py --preflight bench/tasks/01-wordfreq

# the suite (see run_suite.py header for flags)
./.venv/Scripts/python.exe bench/run_suite.py

# why did a gate fire? dump the agent's tool-call transcript (names + commands + is_error)
./.venv/Scripts/python.exe bench/diag_task.py bench/tasks/01-wordfreq
```

## Benchmarking a different model / provider (no code change)

`run_task.py` reads two env vars to swap the `deep_code` + `delegate` category model so
the same tasks run against any litellm-supported supplier. The litellm string is
`<source>/<model>`.

```bash
# default deep tier (gpt-4o-mini, openai) — current main default
./.venv/Scripts/python.exe bench/run_task.py bench/tasks/04-todo-cli

# Groq open model in TEXT tool-calling mode (the Groq-only fork path)
ZBENCH_DEEP_MODEL=llama-3.3-70b-versatile ZBENCH_DEEP_SOURCE=groq \
  ZAKCODE_TOOL_CALLING_MODE=text \
  ./.venv/Scripts/python.exe bench/run_task.py bench/tasks/04-todo-cli

# any other supplier: ZBENCH_DEEP_SOURCE ∈ {openai, gemini, deepseek, fireworks_ai,
#   together_ai, groq, local/ollama, ...}
```

## The small-model bet: best-of-N vs 1-big (`run_bestof.py`)

The quality engine's central wager is that **N cheap small-model tries + a judge to pick beat one
big call**. `run_bestof.py` makes that *falsifiable* on a real task: it runs **N small-model
attempts** (diverse via temperature) and **one big-model attempt**, then uses the quality engine's
pairwise tournament (`zakcode.quality.judge.best_of`) to **judge-select** the best small attempt by
reading its source (judges, not oracles). The held-out `verify.py` then grades everything, so the
report separates the **generation ceiling** (did *any* small attempt pass?), the **best-of-N
result** (did the *judge-selected* one pass?), and **judge quality** (did it pick a winner when one
existed) — and compares pass / $ / wall-clock against the single big run.

```bash
# default: best-of-3 small (qwen3-32b) + judge (qwen3-32b) vs 1 big (gpt-4o-mini)
uv run poe bestof bench/tasks/04-todo-cli
# tune the comparison
ZBENCH_SMALL_MODEL=groq/qwen/qwen3-32b ZBENCH_BIG_MODEL=openai/gpt-4o \
  ZBENCH_JUDGE_MODEL=groq/qwen/qwen3-32b ZBENCH_N=5 ZBENCH_SMALL_TEMP=0.8 \
  ./.venv/Scripts/python.exe bench/run_bestof.py bench/tasks/05-ledger
```

Read the `result` block: `selected_passed` is the product outcome; `bestof_won_where_big_lost` /
`bestof_lost_where_big_won` and `bestof_cheaper_than_big` are the headline. **Measure before
gearing the loop toward it** — if best-of-N doesn't match the big run for the money, the bet is off.

## The activation evidence: quality gate OFF vs ON (`run_quality.py`)

The quality engine ships **off by default** — so the real question is *when an operator should flip
it on*. `run_quality.py` (+ `poe quality <task>`) answers it with data: for one task it runs the
agent **`ZBENCH_RUNS` times per condition on a small model** — gate OFF (today's baseline) vs gate ON
(seam A) — grades each with the held-out `verify.py`, and reports the **pass-rate** delta plus mean
$/time. (Per-condition runs because one sample is too noisy — a single `provider_error` says nothing.)

```bash
uv run poe quality bench/tasks/04-todo-cli
ZBENCH_RUNS=5 ZBENCH_SMALL_MODEL=groq/qwen/qwen3-32b ZBENCH_QUALITY_THRESHOLD=0.85 \
  ./.venv/Scripts/python.exe bench/run_quality.py bench/tasks/05-ledger
```

Read the `result` block: `pass_rate_delta` (ON minus OFF — positive = the gate wins) and
`mean_extra_cost_usd` / `mean_extra_time_s` (the cost per run); each condition's `stop_reasons` list
surfaces noise (e.g. a `provider_error`). Run it across the suite and the pattern **is the small-model
preset** — enable the engine where the pass-rate delta beats the spend.

## Generalization: best-of-N across the suite (`run_bestof_suite.py`)

`run_bestof_suite.py` (`poe bestof-suite [01 03]`) runs the `run_bestof` experiment on every task and
aggregates the **generation ceiling** (did any small attempt pass?), the **best-of-N** pass, the
**1-big** pass, and cost — i.e. whether the bet *generalizes*. Measured 2026-06: best-of-N(small) was
**4/5 vs 1-big 3/5**, the edge concentrated on the hard-but-solvable task — best-of-N pays off where
the task is hard, not on easy tasks (a tie) or too-hard ones (both fail).

## Seam B live: best-of-N rescues a stalled turn (`run_seam_b.py`)

The product form of the bet. When a turn STALLS, the `Agent` fans out `best_of_attempts` fresh
attempts in isolated source copies, verifies each against `verify_command`, and adopts the first that
passes by **DIFF** (never a blind overwrite). `run_seam_b.py` runs one task baseline
(`best_of_attempts=1`) vs seam B on (N), with the task's held-out `verify.py` as the verifier, and the
`[seam B]` log surfaces the retry firing.

```bash
ZBENCH_ATTEMPTS=3 ./.venv/Scripts/python.exe bench/run_seam_b.py bench/tasks/04-todo-cli
```

Read the `result` block: `seam_b_rescued` is the headline (seam B passed where the baseline failed). A
clean rescue needs a genuinely-failing baseline — the `recipe_stalled`-but-oracle-passes quirk (above)
can make 04's baseline pass even when the turn stalled, so the demo also confirms the *safe* path:
seam B fires, finds a verified attempt, and adopts it by diff (e.g. "3 changed, 0 deleted").

## Skills chaining live (`run_skill_chain.py`)

The skills system lets the **model** invoke a skill by name via the `use_skill` tool (not just a human
typing `/<name>`) — so skills can **chain**. `run_skill_chain.py` proves it: a kickoff prompt makes the
model call `use_skill('relay-start')`; that skill's body writes a relay log and then calls
`use_skill('relay-middle')`, which hands off to `use_skill('relay-finish')`. Three skills, one turn,
each handing off to the next — driven by the skill bodies (in `skill_chain/skills/`), not the harness.

```bash
./.venv/Scripts/python.exe bench/run_skill_chain.py   # ZSKILL_MODEL=openai/gpt-4o-mini by default
```

You watch it two ways: the `ON_SKILL_SELECTED` signal prints once per `use_skill` (live, in order), and
each skill appends a marked line to `RELAY.md` so the workspace proves every body actually RAN. `PASS` =
all three fired in order **and** all three markers landed. First live run: chain complete, 3/3 markers,
every invocation `source=tool`, `completed` in ~10 s for ~$0.003.

## Skills branching live (`run_skill_branch.py`)

Where the relay is a FIXED chain, this proves **conditional routing**: `triage-start` reads `INPUT.txt`
and calls `use_skill('handle-urgent')` OR `use_skill('handle-normal')` depending on the content — the
model deciding mid-chain, not following a script. The harness runs both an urgent and a routine input
and checks each time that the RIGHT branch fired and the WRONG one did not.

```bash
./.venv/Scripts/python.exe bench/run_skill_branch.py
```

First live run: both scenarios `PASS` — urgent → `handle-urgent` only, routine → `handle-normal` only,
each writing the matching `RESULT.md` marker. (Skills are also invokable from **sub-agents** now: a
delegated general-purpose agent resolves and chains the same skills; the read-only planner cannot. And
`ZAKCODE_SKILL_INVOCATION_BUDGET=N` caps model-driven invocations per turn — run the relay with `=2` to
watch the third hand-off denied while the agent still finishes gracefully.)

## In CI

The PR gate (`ci.yml`) never runs the bench — quality is **measured, not enforced** (a noisy model
regression must not block a merge). A separate scheduled workflow,
`.github/workflows/bench.yml` (`agent-bench`), runs the whole suite against real models:

- **Nightly**, and on demand via *Run workflow* (optional task filter, concurrency, and a `model`
  override).
- **Key-gated**: it runs only on the canonical repo and only when `OPENAI_API_KEY` / `GROQ_API_KEY`
  are configured as repo secrets — otherwise it no-ops, so it costs nothing until you opt in.
- Publishes the pass-rate / $-per-task / per-model summary to the run's **job summary** and uploads
  `results/*.json` as a build artifact.

To compare configurations — e.g. **N-small fan-out vs one big model** (the small-model bet) — dispatch
the workflow twice with different `model` overrides (or run `poe bench` locally with different
`ZBENCH_*` / `ZAKCODE_*` env) and diff the two `results/*.json`.

**Do not use `results/` as the series.** `run_suite.py` writes `results/suite-<n>tasks.json`, keyed
only on the task COUNT, so the next run of the same size overwrites the previous one in place — and
the directory is gitignored, while the CI path publishes to a run artifact with a retention window.
Nothing outside the box that ran it can read a prior value, which is precisely what a regression
needs to be visible against.

One summary row per run is therefore appended to a durable series kept outside this repo, in the
operator's Mind world: `world/telemetry/bench-runs.jsonl`, written by

```
world/scripts/bench-run-record.sh --from <results.json> [--lane hosted|local] [--label ...]
              # --dry-run prints the row and writes nothing; --tail N shows the series
```

Each row carries the run id (content-derived, so re-ingesting one artifact is legible rather than
silently doubling the series), date, box, models exercised, pass-rate, cost, tokens, iterations, and
the per-task `verify_rc` — the held-out oracle's exit code, so a moving pass-rate says *which* task
moved. That series is the time-series this paragraph used to call a "natural follow-up" (g-306-398).

`ZAKCODE_TOOL_CALLING_MODE` ∈ `auto` | `native` | `text` controls the protocol.

## Baseline results (deep set 01-wordfreq / 03-lru / 04-todo-cli, 2026-06-18)

| model (mode) | pass | $/task | cache-hit | notes |
|---|---|---|---|---|
| **openai/gpt-4o-mini (native)** | **3/3** | **$0.0074** | **85%** | reliable native tools incl. multi-file 04; current `deep_code` default |
| groq/openai/gpt-oss-120b (native) | 0/3 | — | — | malformed native tool calls → Groq `tool_use_failed`; reasoning model → text-mode returns empty content (broken both modes) |
| groq/llama-3.3-70b-versatile (text) | 2/3 | ~$0.03 | 0% | works for focused tasks; **04 stalled** (turn-1 prose, no tool call); Groq does not cache it |
| groq/llama-3.3-70b-versatile (native) | 0/2 | $0.022 | 0% | pseudo-XML rejected (`tools_unreliable`) → provider_error / doom_loop |

Upstream confirmation of the Groq tool-call failure mode: pydantic-ai #4350, OpenHands #10187.

## Open testing backlog

See the handoff brief. Status:

- **(2) DONE** — `groq/openai/gpt-oss-120b` is flagged `tools_unreliable` in `registry.py`; the
  `AvailabilityResolver` groq pick is now `qwen/qwen3-32b`, and the runtime failover crosses
  provider once it is excluded.
- **(3) DONE** — the `recipe_stalled` over-fire is fixed: the recipe gate now credits a green
  test-runner run, `extract_acceptance` no longer false-matches a CLI flag (`--top N`), and the
  runner exposes a pytest-capable interpreter (above). 01-wordfreq now completes, and neither 01
  nor 04 reports `recipe_stalled`. (04 can still `doom_loop` — a genuine gpt-4o-mini limitation:
  it repeatedly emits an invalid f-string that the `write_file` Python-validity firewall refuses.)
- **(5) DONE (first task)** — added `05-ledger` (multi-file CLI + atomic transfers; the held-out
  oracle was validated to PASS against a correct reference implementation).
- **(1) PARTIAL — named suppliers need keys** — only `OPENAI_API_KEY` + `GROQ_API_KEY` are
  present here, so Gemini 2.5 Flash / DeepSeek V3 / Fireworks could not be run. With the
  available providers the comparison re-affirms the current default tier: `openai/gpt-4o-mini`
  is the reliable deep model (01 completes and passes the oracle), while `groq/llama-3.3-70b`
  (text mode) is unreliable on the deep set (in this run 03 passed after a doom-loop; 01 and 04
  failed). Re-run the deep set with `ZBENCH_DEEP_SOURCE`/`ZBENCH_DEEP_MODEL` once a
  Gemini/DeepSeek/Fireworks key exists.
- **(4) DIAGNOSED — a model limitation, not a harness bug** — on the complex 04 prompt
  `groq/llama-3.3-70b` in text mode ignores the `<tool_call>` protocol and "answers" with a
  Markdown ```python code block (no tool call — not even its native `<function=...>` form), so
  the loop sees no tool call and completes in **one** iteration having written nothing. The text
  parser is strict by design (precision over recall — prose must not false-trip it), so there is
  no safe parser change; the gap is the model not following the protocol. A bounded text-mode
  nudge ("you wrote code but emitted no `<tool_call>` — emit one to actually create the file")
  is a possible future improvement, but needs its own validation. See `diag_task.py` output.

## Constraints

`.env` (gitignored) holds the real `OPENAI_API_KEY` + `GROQ_API_KEY`; litellm reads them
directly. **Never commit/print keys.** You need your own keys for whatever providers you
test. Local model option is `llama.cpp`'s `llama-server` (source `local`), not Ollama.

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
  veto_door.py           # after a refused stop: capture, fork, one rollout per candidate BUILD
  veto_door_world.py     # the synthesized loop the veto-door bench runs in (no framework text)
  veto_door_arms/        # one patch per arm; an arm is HEAD plus its patches, as a worktree
  served_rests.py        # a served run's logs: idle windows after a turn the veto fence ended
  served_doors.py        # a served run's traces: each refused stop and what its segment did
  served_stops.py        # ...what each refused stop ANSWERED, and what getting back to work cost
  served_coin.py         # ...finished-plan episodes, and the reading of a build that draws lots
  served_coin_build_proof.py  # offline proof of that build: real Agent + hook, scripted model
  served_coin_mutants.py # mutation proof of the two files above (every mutant dies by name)
  served_ladder.py       # ...what the stuck ladder fired on: the product's own tracker replayed
  served_ladder_mutants.py  # mutation proof of that reader (every mutant dies by name)
  served_builds/         # MEASUREMENT builds as patches on main; never merged into src/
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

## A self-hosted endpoint, with no API key at all (ADR-0142)

Every lane above needs a paid provider key, so the nightly CI bench is `skipped` on a repo that
has none — and the small models this harness exists to study are exactly the ones people self-host.
Any OpenAI-compatible server (llama.cpp, vLLM, SGLang, a local gateway) benchmarks for free:

```bash
export ZAKCODE_ZAKPICK_MODELS='{"classify":{"model":"<id>","source":"openai","thinking":false,"context_window":131072},
  "summarize":{"model":"<id>","source":"openai","thinking":false,"context_window":131072},
  "quick_code":{"model":"<id>","source":"openai","thinking":false,"context_window":131072},
  "delegate":{"model":"<id>","source":"openai","thinking":false,"context_window":131072},
  "deep_code":{"model":"<id>","source":"openai","thinking":true,"context_window":131072},
  "plan":{"model":"<id>","source":"openai","thinking":true,"context_window":131072}}'
export OPENAI_BASE_URL=http://<host>:<port>/v1
export OPENAI_API_KEY=dummy          # required by the client, never checked by the server

./.venv/bin/python bench/run_task.py --preflight bench/tasks/01-wordfreq   # wiring check, no LLM call
./.venv/bin/python bench/run_suite.py --jobs 2
```

Three things bite, none of them obvious, all measured 2026-09-11:

1. **Use `ZAKCODE_ZAKPICK_MODELS`, NOT `ZBENCH_DEEP_MODEL`.** `run_task.py` builds its override as
   `ZakpickModel(model=..., source=...)` with no `context_window`, and it OVERWRITES the `deep_code`
   and `delegate` entries — so setting it discards the window you just supplied and the agent
   refuses to start.
2. **`context_window` is mandatory and belongs in the model's entry.** A self-hosted alias is in no
   capability registry and litellm has no metadata for it, so ADR-0066 refuses to guess. Normally
   the server is asked directly (ADR-0065), but `run_task.py` pins `api_base: None`
   ("belt-and-suspenders: never a local base"), so there is no server to ask. `ZAKCODE_CONTEXT_WINDOW`
   does NOT cover the per-category zakpick entries. Read the real value off the server's own
   `/v1/models` listing and paste it once.
3. **`OPENAI_BASE_URL` still reaches the server**, because litellm reads it from the environment
   directly — it is not routed through the `api_base` setting that (2) pins to `None`. The two
   consumers of "where is the server" are separate, which is why this lane works at all.

A reasoning model starved of `max_tokens` returns **empty content and no error** — it spends the
whole budget thinking. That is the failure `thinking: false` exists to avoid on the cheap
categories; keep it off for `classify`/`summarize`/`quick_code` (see `ZakpickModel.thinking`).

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

## The veto door: capture, fork, and arms that are builds (`veto_door.py`)

What a small model does in the completions right after a Stop hook refuses its stop and the
harness delivers the skill the hook named (ADR-0187), under each candidate BUILD of Zak Code.
The reading rule is fixed in `results/veto-door-preregistration.log` before any scored call.

- **Capture and fork.** A capture is one live run of the real product (the served factory's
  posture) in the synthesized world of `veto_door_world.py`, recorded call by call and stopped
  at the first call after the first delivery: the fork. A rollout rebuilds that world at the
  same path, serves every recorded completion from the tape (checked message by message), and
  goes live from the fork. Every arm starts from the same conversation, plan and fence count.
  `--fork-at lap-end` lets the run go on past the door and forks at the first request that
  carries a finished plan's "answer now" line: a plan that finished later in a turn whose end a
  hook already governs. The second refused stop is then counted from the deliveries the fork had
  already seen, and the cap of live completions is two higher (8). Proven offline, never run.
- **Arms are builds.** `veto_door_arms/<letter>.patch` on a detached worktree of HEAD, run as a
  child process on that tree's `src`. Nothing is switched at run time. Each ledger row carries
  the build it ran and a witness only that build can write, and a row missing its witness (or
  showing another arm's) is refused by the reading.
- **Resuming is a loop script that RAN.** A rollout passes on the first successful call that
  ran one of the loop's scripts, and it ends there, so the scorer errs one way only. The
  command must start the script as a shell would (`bash scripts/x.sh`, behind `cd … &&`,
  `timeout`, `$(…)`, `bash -c`; never `cat scripts/x.sh`, `bash -n`, a quoted mention, a
  comment or a here-document body), and the result must show a line only that script prints
  when it runs, because the text of `false && bash scripts/x.sh; echo done` cannot say it ran
  nothing. Rows count what each test left out (`named_not_run`, `run_unproven`), and the
  per-call flags stay in the local detail so a corrected scorer can be re-applied.
- **Offline first.** `selftest` runs the real Agent, the real shell Stop hook and the real
  plan and skill machinery on a scripted model: the baseline's tree captures, every arm's tree
  replays that capture (`--capture`), and the reading rule is checked on ledgers whose answer
  is known. `preflight` prints what each category would put on the wire. Neither touches the
  network.
- **Private by construction.** Tapes, request snapshots and completion text stay under
  `--out`; ledger rows hold labels, counts, hashes and usage. The wire hook keeps a URL path
  and four body fields, never a header. Workspaces are built under `--base`, which must sit
  outside every repository (the product folds project guides up to the repository root).
- **Known limits, from batch 1 (2026-09-20, read NOT DISCRIMINATING).** This world does not
  reproduce the served door: the unpatched build resumed on 16 of 18 rollouts, so no arm was
  read. A capture whose prefix holds a clock (a `plan_recall` result prints event times) cannot
  be replayed character for character and is refused. A patch that changes text a capture's
  prefix already holds (C, where a run met a pointer before its fork) loses that fork. And the
  C witness asked how the door's answer OPENS, which misses a pointer whose call carried an
  argument (the product frames the argument first): it read false on 137 door answers of 137.
  The results block in the registration log has the detail, and a correction of its account
  of why the offline proof missed this.
- **Repaired since (2026-09-20).** The door's witness reads the flag the product sets on a
  pointer it builds (ADR-0203), not the text. The offline proof had checked each witness's
  ingredients by hand and had run the usability RULE only on hand-written rows, so a flag the
  rule needed and no real row carried went unseen (its scripted call did carry the argument).
  `selftest` now runs that rule on every row the real product writes in the tree under test,
  reads the same row under every other arm's name (refused unless the build is the same), and
  asserts the door's flag. With the old detector put back, that last check fails. The
  permutation test is exact at any number of forks (a convolution over the distinct sums,
  checked against one-at-a-time enumeration). A prefix that holds a clock is still refused,
  not normalised: a refusal costs a capture, a normaliser could hide a real divergence.
- **Two readings for the second registration.** `--mode calibrate` reads the baseline ALONE
  (its own ledger, `--stage calibration`) and says PROCEED only if the unpatched build fails
  to resume often enough for an arm to have something to reduce; those rows are never the
  comparison's. `--mode refusal` is the comparison (`--stage comparison`): each arm against
  the baseline on how often a rollout did NOT resume, thresholds relative to the baseline
  measured in the same batch, the fork as the unit of an exact paired test, and a verdict
  that must keep its size with any one fork left out. `arms --arms A,A2,R` builds only the
  arms a batch names, and `--forks-of calibration` keeps the comparison off a fork no
  calibration rollout could read at all (a prefix with a clock in it never replays).
- **Registration 2, read 2026-09-21: R reads GAIN.** Calibration said PROCEED (the unpatched
  build failed to resume on 11 of 58 rollouts). In the comparison, over 29 forks and 116
  rollouts an arm, the unpatched build failed to resume on 18 (15.5%), its placebo on 20
  (FLAT, p = 0.86) and R on 0 (p = 0.0078 with the fork as the unit; the verdict keeps its
  size with any one fork left out). Both audits found nothing uncounted. $2.27. R, which
  keeps a finished plan's "answer now" reminder silent once a stop on that plan has been
  refused, ships as a product change; from that commit `r.patch` is a record too. The bench
  does not say WHY it works, and this world is not the served door: the results block says
  what the numbers do not show.
- **After R shipped (2026-09-21): arm L and the lap-end fork, built and NOT run.** R is
  ADR-0205 now, so it is every build's behaviour and no longer an arm: `ARMS` holds A, A2 and L,
  and the selftest (which main's had been failing since R shipped: it still asked the baseline
  to lack R's note) proves both fork kinds in the baseline's tree and one kind per run in an
  arm's. A ledger of an earlier registration is read by the instrument at that registration's
  own MANIFEST commit. L is the candidate the served runs' own logs pointed at
  (`served_stops.py`): once a refused stop has named a skill re-entry, no plan that finishes
  later in that turn is sent the "answer now" line. It is NOT measured here, for two reasons
  found while sizing a third registration. A fork that deep costs about three times a door
  fork (a whole further pass of the loop is captured, and every rollout pays for the longer
  prefix): estimated at about $7, against registration 2's $2.27. And this world's loop skill
  closes every cycle with "report, in two sentences". In all 32 of registration 2's captures
  the words that made the door came right after the plan was marked finished, in answer to a
  request that carried the line: the skill's instruction or the line, this world cannot say
  which. At a lap-end fork that instruction would bring words in every arm alike, and an arm
  that silences the line could never show it. A registration that forks there has to change
  the world first, and pilot what it yields. The line is to be tested in the served loop
  itself instead; that registration belongs in `results/served-luna-preregistration.log`.
- **The batch-1 patches are a record, not a kit.** `veto_door_arms/*.patch` are the bytes
  batch 1 ran, against `465b332`. `b.patch` and `c.patch` no longer apply to HEAD: ADR-0203
  changed the line they both patch (the resolver now flags the pointer it builds). An arm that
  is used again is re-cut from HEAD and registered under its new hash.

```bash
PY=./.venv/bin/python; OUT=/somewhere/scratch; ARMS=$OUT/arms
$PY bench/veto_door.py selftest --expect A            # prints "capture kept at: <dir>", per fork kind
$PY bench/veto_door.py arms --arms-dir $ARMS          # one worktree per arm, from HEAD
(cd $ARMS/L && PYTHONPATH=$ARMS/L/src $PY bench/veto_door.py selftest --expect L --capture <dir>)
$PY bench/veto_door.py preflight
$PY bench/veto_door.py capture  --out $OUT --arms-dir $ARMS --forks 10 --runs 14 --budget 1.00
$PY bench/veto_door.py rollouts --out $OUT --arms-dir $ARMS --reps 2 --budget 2.50
$PY bench/veto_door.py report   --ledger $OUT/rollouts.jsonl --arms-dir $ARMS --reps 2

# the second registration: three arms, the baseline alone first, then the comparison
$PY bench/veto_door.py arms --arms-dir $ARMS --arms A,A2,R
$PY bench/veto_door.py capture  --out $OUT --arms-dir $ARMS --forks 32 --runs 40 --budget 1.30
$PY bench/veto_door.py rollouts --out $OUT --arms-dir $ARMS --arms A --reps 2 \
    --stage calibration --budget 0.45
$PY bench/veto_door.py report --ledger $OUT/calibration.jsonl --arms-dir $ARMS --out $OUT \
    --arms A --reps 2 --mode calibrate
$PY bench/veto_door.py rollouts --out $OUT --arms-dir $ARMS --arms A,A2,R --reps 4 \
    --stage comparison --forks-of calibration --budget 2.20
$PY bench/veto_door.py report --ledger $OUT/comparison.jsonl --arms-dir $ARMS --out $OUT \
    --arms A,A2,R --reps 4 --mode refusal

# (the commands above are the record of registrations 1 and 2; from this commit on the arms that
# build are A, A2 and L, and `capture --fork-at lap-end` forks past the door)
```

## A reminder tested inside the served loop (`served_coin.py`, `served_builds/k.patch`)

Some questions are about the served loop itself, and a bench world cannot stand in for it (the
veto-door section above says why, for the "answer now" line of a finished plan). For those the
treatment is randomised INSIDE a served run by a **measurement build**: a patch on main that
draws lots, writes every draw to the trace, and is **never merged** (a product does one
thing). `served_builds/k.patch` is the first. It applies arm L's rule to a random half of the
finished plans L would silence, in pairs of one silent and one sent plan in random order, so
one run holds both treatments at the same kind of moment and the unit is the pair, not the run.

A measurement build is trusted only after three free, offline steps, all committed here:

```bash
PY=./.venv/bin/python; K=/somewhere/outside-any-repo/K     # a worktree of main + the patch
git worktree add --detach $K HEAD && git -C $K apply $PWD/bench/served_builds/k.patch
PYTHONPATH=$K/src $PY bench/served_coin_build_proof.py --base /somewhere/outside-any-repo
$PY bench/served_coin_build_proof.py --base /somewhere/outside-any-repo   # main: MUST fail
$PY bench/served_coin.py --selftest                                       # the reader
$PY bench/served_coin_mutants.py reader                                   # 26 mutants
$PY bench/served_coin_mutants.py build $K/src --base /somewhere/outside-any-repo
$PY bench/served_coin.py <world-dir> [<world-dir> ...]                    # the reading
```

The build proof drives the real `Agent` with a real shell Stop hook and a scripted model in
the veto-door world, with the coin forced each way, and checks the wire as well as the trace
label. The reader refuses a run whose requests contradict a draw, leaves out a run with too
few pairs, and codes the registered rule (`RULE`, `verdict`, `reading`) so the reading can be
re-run by anyone. The registration, the gates of a served run and the results are in
`results/served-luna-preregistration.log` (sample 7).

Sample 7 read **GAIN** over 13 pairs in three of its four runs: a finished plan whose
"answer now" line was sent ended in a stop in words 12 times of 13, and one kept silent 5
times of 13 (7 pairs differ, all the same way, p 0.0156, and no single pair undoes it).
That licenses arm L as a product change (ADR-0208; the registration said 0207, a number
another change took first) and nothing about a whole run.

## What the stuck ladder fired on in a served loop (`served_ladder.py`)

The ladder's repeated-outcome signal (ADR-0038) counts identical observations over a whole
turn, and only a successful workspace-write call opens a new epoch. A served perpetual loop
is one very long turn that changes its world through shell scripts, so whether the rungs it
draws are the loop's own regular repeats or a model circling is a question about the logs,
and since ADR-0206 the transcript holds every call and result.

`served_ladder.py` answers it without a run: it feeds the **product's own**
`StuckTracker` the transcript's calls the way the loop does (a fresh tracker per turn, the
epoch read after the batch ran, reset where the loop resets it) and replays the same calls
under candidate counting rules. Everything that could drift is asked of the product, not
copied: the uncounted tools, the workspace-write tier, the refused-stop delivery's note, and
the selftest's synthetic transcripts are rendered by the product's transcript writer.

A replay is **believed only if it reproduces the product's own record**: the `stuck` notes
on the decision trace (tool and repeat count, at least 80% in order, against the longer
list) and the refused stops the trace noted as delivered (give or take one). A world that
fails is left out by name; on a run whose transcript was a live window the reading is
`REPLAY NOT TRUSTED`, as it must be. A **lap** is what the product itself can know: the
skill a turn-end hook named at a refused stop (ADR-0187), delivered again by the harness or
loaded again by the model with its body.

```bash
PY=./.venv/bin/python
$PY bench/served_ladder.py --selftest                 # 65 known answers
$PY bench/served_ladder_mutants.py                    # 51 mutants, each dies by name
PYTHONPATH=<the build that served the run>/src $PY bench/served_ladder.py <world-dir> ...
```

It prints labels, counts and distances only. A replay under a candidate rule shows where
that rule would have fired on the same calls; it cannot show what the model would have done
without the rungs it really drew. The registered reading is in
`results/served-luna-preregistration.log`.

Its first reading, on sample 7's four worlds, was **LOOP REGULARITY** by a thin margin. The
replay reproduced all 47 of the product's own repeated-outcome notes; the lap rule removes
34 of them (72.3% against a threshold of 70%, and 64.7% with one world left out); every
rung on a tool that looks at the world had a lap boundary between its repeats, and all 13
the lap rule leaves are on `update_plan`'s own receipt.

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

---
name: triage-start
description: Branching demo — read the request, then route to handle-urgent OR handle-normal based on its content.
allowed-tools: [read_file, use_skill]
version: 1.0.0
---
# Triage — pick a branch

You are a router that picks ONE of two next skills based on what the input says. Do exactly this:

1. Read the file `INPUT.txt` in the workspace.
2. Decide the branch:
   - If the contents indicate something URGENT (they contain a word like "urgent", "down",
     "outage", or "critical"), call the `use_skill` tool with `name` = `handle-urgent`.
   - Otherwise, call the `use_skill` tool with `name` = `handle-normal`.

Call EXACTLY ONE of the two skills — never both, never neither. Do not do the handling yourself;
the skill you choose does it.

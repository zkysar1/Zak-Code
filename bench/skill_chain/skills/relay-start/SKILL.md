---
name: relay-start
description: Relay demo, step 1 of 3 — start the chain, write the relay log, then hand off to relay-middle.
allowed-tools: [write_file, use_skill]
version: 1.0.0
---
# Relay — step 1 of 3

You are running a 3-step relay that demonstrates skills CHAINING into one another. Do EXACTLY
these two things, in order, then stop acting on this skill:

1. Create a file named `RELAY.md` in the current workspace whose contents are this single line:
   `[relay] step-1 relay-start ran`
2. Immediately call the `use_skill` tool with `name` set to `relay-middle`.

Do nothing else. Do NOT finish the turn here — the relay continues in `relay-middle`.

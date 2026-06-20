---
name: relay-middle
description: Relay demo, step 2 of 3 — append to the relay log, then hand off to relay-finish.
allowed-tools: [read_file, edit_file, use_skill]
version: 1.0.0
---
# Relay — step 2 of 3

Continue the relay. Do EXACTLY these two things, in order, then stop acting on this skill:

1. Append this line to `RELAY.md`, keeping the existing line above it:
   `[relay] step-2 relay-middle ran`
2. Immediately call the `use_skill` tool with `name` set to `relay-finish`.

Do nothing else. Do NOT finish the turn here — the relay finishes in `relay-finish`.

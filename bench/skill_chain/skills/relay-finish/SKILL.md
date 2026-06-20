---
name: relay-finish
description: Relay demo, step 3 of 3 — append the final line and END the chain (do not chain further).
allowed-tools: [read_file, edit_file]
version: 1.0.0
---
# Relay — step 3 of 3 (final)

Finish the relay. Do EXACTLY this:

1. Append this line to `RELAY.md`, keeping the existing lines above it:
   `[relay] step-3 relay-finish ran`
2. The relay is COMPLETE. Do NOT call `use_skill` again. Reply with a one-line confirmation that
   all three relay steps ran (relay-start → relay-middle → relay-finish), then end the turn.

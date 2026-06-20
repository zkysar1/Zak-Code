---
name: handle-urgent
description: Branching demo — the URGENT branch. Records an urgent-path result and ends.
allowed-tools: [write_file, edit_file]
version: 1.0.0
---
# Handle — URGENT branch

The router routed the URGENT path here. Do this, then end:

1. Append this line to `RESULT.md` (create the file if it does not exist):
   `[branch] handled as URGENT`
2. Confirm in one line that the urgent branch ran, then end the turn. Do NOT call `use_skill` again.

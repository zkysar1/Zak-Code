"""The agent loop.

Responsibilities: assemble the prompt, call the model via the provider layer, parse the
response, extract and execute tool calls, append results, and decide whether to iterate
or stop.

Planned modules: ``loop.py`` (the turn loop), ``prompt.py`` (system prompt + memory/
CLAUDE.md-style discovery), ``compact.py`` (context compaction).

Status: scaffolded; implemented in milestone M0. See ``docs/ARCHITECTURE.md``.
"""

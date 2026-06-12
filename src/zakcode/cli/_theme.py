"""The Zak Code CLI color theme (render-only).

Doctrine, enforced everywhere: terminal-default ink does the talking and all chrome
recedes to ``dim``; the brand azure ``color(38)`` may only ever paint 1–2 character
marks (``✦ ✧ › ●`` and the spinner glyph) — never a run of text. Color is reserved
for meaning (green ok / red err / yellow warn), diffs are painted bands (explicit fg
AND bg, bold so the 16-color downgrade keeps contrast, text extent only), error
detail stays at default fg (errors are shown, not dimmed), and the ``│`` rail under
a failed tool turns red. The console is built with ``highlight=False`` so rich never
auto-colors metadata — color is opted into only here and via Syntax.
"""

from __future__ import annotations

from rich.theme import Theme

ZAK_THEME = Theme(
    {
        # brand marks (1–2 char glyphs only — never a run of text)
        "brand": "color(38)",
        "brand.soft": "color(38) dim",
        # welcome banner
        "banner.border": "dim",
        "banner.title": "bold",
        "banner.label": "dim",
        "banner.value": "default",
        "banner.hint": "dim",
        "tip": "dim",
        # input
        "prompt.marker": "bold color(38)",
        # assistant prose
        "assistant.marker": "color(38)",
        "md.h": "bold",
        # dark_cyan, not cyan: the 16-color downgrade of color(38) is cyan, so inline
        # code must not collide with the brand marks (and stays legible on light bg).
        "md.code": "dark_cyan",
        "md.bullet": "dim",
        # tool call headline
        "tool.marker": "dim",
        "tool.name": "bold",
        "tool.paren": "dim",
        "tool.args": "default",
        # result region rail (red binds a failed tool's body)
        "tool.bar": "dim",
        "tool.bar.err": "red",
        # receipts
        "result.connector": "dim",
        "result.summary": "dim",
        "result.output": "dim",
        "result.more": "dim italic",
        # semantic states
        "ok": "green",
        "err": "bold red",
        "err.body": "default",
        "warn": "yellow",
        # mid-turn status notices
        "status": "dim italic",
        # turn receipt
        "footer": "dim",
        "sep": "dim",
        "spinner": "color(38)",
        # diffs (painted bands: explicit fg + bg, bold for the 16-color tier)
        "diff.meta": "dim",
        "diff.add": "bold grey93 on dark_green",
        "diff.del": "bold grey93 on dark_red",
        "diff.ctx": "dim",
        # permission panel
        "perm.border": "yellow",
        "perm.title": "bold yellow",
        "perm.tool": "bold",
        "perm.reason": "default",
        "perm.key": "bold",
        "perm.option": "default",
        # assistant code blocks
        "code.tag": "dim",
        # Todo results
        "todo.done": "green",
        "todo.open": "dim",
        # Compatibility aliases — consumed by /permissions, /hooks, /plugins, /skills,
        # eval, info, _run_server_chat, and plugin output paths this restyle does not
        # fully rewrite; rich raises on unknown style names.
        "notice.dim": "dim",
        "arg.key": "dim",
        "arg.value": "default",
        "banner.version": "dim",
        "tool.verb": "bold",
        "tool.target": "bold",
        "rule.line": "dim",
        "perm.tier": "yellow",
    }
)

"""The Zak Code CLI color theme (render-only).

Doctrine, enforced everywhere: terminal-default ink does the talking and chrome
recedes to ``dim`` — except the two lines the eye must find first (ADR-0185): the
operator's own line and a tool's call line stay bright, so a day-long transcript scans
as turns and tool blocks rather than as one grey stream. The brand azure ``color(38)``
may only ever paint 1–2 character marks (``✦ ✧ › ●`` and the spinner glyph) — never a
run of text. Color is reserved for meaning (green ok / red err / yellow warn), every
success receipt opens with a green ``✓`` and every failure with a red ``✗``, diffs are
painted bands (explicit fg AND bg, bold so the 16-color downgrade keeps contrast, text
extent only), error detail stays at default fg (errors are shown, not dimmed), and the
``│`` rail under a failed tool turns red. The console is built with ``highlight=False``
so rich never auto-colors metadata — color is opted into only here and via Syntax.
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
        # the operator's message — the turn's bright anchor on every door (ADR-0185)
        "user.marker": "bold color(38)",
        "user.text": "bold",
        "user.meta": "dim",
        # assistant prose
        "assistant.marker": "color(38)",
        "md.h": "bold",
        # dark_cyan, not cyan: the 16-color downgrade of color(38) is cyan, so inline
        # code must not collide with the brand marks (and stays legible on light bg).
        "md.code": "dark_cyan",
        "md.bullet": "dim",
        "md.italic": "italic",
        "md.strike": "strike",
        "md.link": "underline",
        "md.link.url": "dim",
        "md.quote": "dim italic",
        # tool call headline — the loudest line of its block (ADR-0185): a bright marker
        # and a bold name; only the parens recede
        "tool.marker": "bold",
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

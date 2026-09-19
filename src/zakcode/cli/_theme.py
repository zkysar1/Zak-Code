"""The Zak Code CLI color theme (render-only).

Doctrine, enforced everywhere: terminal-default ink does the talking and chrome
recedes to grey — except the lines the eye must find first: the operator's own line
(orange, the one warm colour in the transcript — ADR-0186) and a tool's call line
(bright), so a day-long transcript scans as turns and tool blocks rather than as one
grey stream. Grey is a 256-colour INDEX, never the ``dim`` attribute: dim is honoured
by some terminals and dropped by others (tmux forwards it only when the outer terminal
advertises it), so chrome styled ``dim`` rendered at full contrast in the very cockpit
it was meant to recede in; ``color(245)`` (#8a8a8a, the say box's own grey) renders the
same on every 256-colour terminal, and status/log rows sit one step quieter at
``color(242)``. The brand azure ``color(38)`` may only ever paint 1–2 character marks
(``✦ ✧ ●`` and the spinner glyph) — never a run of text. Color is reserved for meaning
(green ok / red err / yellow warn), every success receipt opens with a green ``✓`` and
every failure with a red ``✗``, diffs are painted bands (explicit fg AND bg, bold so
the 16-color downgrade keeps contrast, text extent only), error detail stays at default
fg (errors are shown, not dimmed), and the ``│`` rail under a failed tool turns red. The
console is built with ``highlight=False`` so rich never auto-colors metadata — color is
opted into only here and via Syntax.
"""

from __future__ import annotations

from rich.theme import Theme

ZAK_THEME = Theme(
    {
        # brand marks (1–2 char glyphs only — never a run of text)
        "brand": "color(38)",
        "brand.soft": "color(38) dim",
        # welcome banner
        "banner.border": "color(245)",
        "banner.title": "bold",
        "banner.label": "color(245)",
        "banner.value": "default",
        "banner.hint": "color(245)",
        "tip": "color(245)",
        # input
        "prompt.marker": "bold color(214)",
        # the operator's message — the turn's root, and the one warm colour in the
        # transcript (ADR-0185/0186): orange by index, so it reads the same everywhere
        "user.marker": "bold color(214)",
        "user.text": "bold color(214)",
        "user.meta": "not bold color(245)",
        # assistant prose
        "assistant.marker": "color(38)",
        "md.h": "bold",
        # dark_cyan, not cyan: the 16-color downgrade of color(38) is cyan, so inline
        # code must not collide with the brand marks (and stays legible on light bg).
        "md.code": "dark_cyan",
        "md.bullet": "color(245)",
        "md.italic": "italic",
        "md.strike": "strike",
        "md.link": "underline",
        "md.link.url": "color(245)",
        "md.quote": "color(245) italic",
        # tool call headline — the loudest line of its block (ADR-0185): a bright marker
        # and a bold name; only the parens recede
        "tool.marker": "bold",
        "tool.name": "bold",
        "tool.paren": "color(245)",
        "tool.args": "default",
        # result region rail (red binds a failed tool's body)
        "tool.bar": "color(245)",
        "tool.bar.err": "red",
        # receipts
        "result.connector": "color(245)",
        "result.summary": "color(245)",
        "result.output": "color(245)",
        "result.more": "color(245) italic",
        # semantic states
        "ok": "green",
        "err": "bold red",
        "err.body": "default",
        "warn": "yellow",
        # mid-turn status notices
        "status": "color(242) italic",
        # a log record rendered as a transcript line (ADR-0186)
        "log": "color(242)",
        # turn receipt
        "footer": "color(245)",
        "sep": "color(245)",
        "spinner": "color(38)",
        # diffs (painted bands: explicit fg + bg, bold for the 16-color tier)
        "diff.meta": "color(245)",
        "diff.add": "bold grey93 on dark_green",
        "diff.del": "bold grey93 on dark_red",
        "diff.ctx": "color(245)",
        # permission panel
        "perm.border": "yellow",
        "perm.title": "bold yellow",
        "perm.tool": "bold",
        "perm.reason": "default",
        "perm.key": "bold",
        "perm.option": "default",
        # assistant code blocks
        "code.tag": "color(245)",
        # Todo results
        "todo.done": "green",
        "todo.open": "color(245)",
        # Compatibility aliases — consumed by /permissions, /hooks, /plugins, /skills,
        # eval, info, _run_server_chat, and plugin output paths this restyle does not
        # fully rewrite; rich raises on unknown style names.
        "notice.dim": "color(245)",
        "arg.key": "color(245)",
        "arg.value": "default",
        "banner.version": "color(245)",
        "tool.verb": "bold",
        "tool.target": "bold",
        "rule.line": "color(245)",
        "perm.tier": "yellow",
    }
)

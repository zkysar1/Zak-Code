"""The Zak Code CLI color theme (render-only).

Three tiers, enforced everywhere: PRIMARY = the assistant's words + the salient tool
target (bright); SECONDARY = ``dim`` chrome (labels, arg keys, footer, markers);
ACCENT = exactly one color per state (cyan active, green ok, red err, yellow warn).
The console is built with ``highlight=False`` so rich never auto-colors metadata —
color is opted into only here and via Syntax/Markdown.
"""

from __future__ import annotations

from rich.theme import Theme

ZAK_THEME = Theme(
    {
        # banner / reference metadata
        "banner.title": "bold cyan",
        "banner.version": "dim",
        "banner.label": "dim",
        # input
        "prompt.marker": "bold cyan",
        # assistant prose
        "assistant.marker": "cyan",
        # tool call
        "tool.marker": "dim",
        "tool.verb": "cyan",
        "tool.target": "bold",
        "tool.connector": "dim",
        "arg.key": "dim",
        "arg.value": "default",
        # semantic states
        "ok": "green",
        "err": "bold red",
        "warn": "yellow",
        # out-of-band / chrome — status notices are loop interventions (rate limits,
        # stuck recovery): an accent glyph + italic body so they read apart from both
        # the assistant's prose and the dim chrome, instead of vanishing into dim.
        "status": "italic",
        "status.glyph": "yellow",
        "footer": "dim",
        "rule.line": "dim",
        "notice.dim": "dim",
        # diffs
        "diff.add": "green",
        "diff.del": "red",
        "diff.ctx": "dim",
        # permission panel
        "perm.border": "yellow",
        "perm.tier": "yellow",
    }
)

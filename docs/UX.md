# UX — the Zak look

One visual language, rendered twice. The terminal client (`src/zakcode/cli/`) and the
bundled web client (`src/zakcode/server/static/index.html`) are both thin renderers of
the same `AgentEvent` stream, and they draw it with the **same spatial grammar** so an
operator can move between them without relearning the screen. This document is the
source of truth for that grammar; change it and both renderers in the same PR.

## The spatial grammar

Structure comes from a thin **gutter of glyphs** in a fixed left column, not from boxes:

| Glyph | Meaning | ASCII fallback |
| --- | --- | --- |
| `›` | user input | `>` |
| `·` | assistant reply (first row of a prose run) | `-` |
| `→` | tool call (`→ verb  target`) | `->` |
| `└` | tool result, attached to its call (`└ ✓ summary`) | `` `- `` |
| `✓` / `✗` | result state | `[ok]` / `[x]` |
| `┄` | loop status notice (rate limit, recovery) | `.` |
| `─` | rules: session banner close, per-turn footer | `-` |

(The terminal resolves glyphs through `cli/_glyphs.py`, which falls back to the ASCII
column on a cp1252 console; the web client can always use the unicode set.)

**Indent scale (terminal):** column 0 is empty, column 2 is the gutter (every marker),
column 4 is content (prose text, tool verbs), column 6+ is nested detail (result
previews). The web client renders the same shape with a fixed-width gutter column.

**Blocks, not lines.** Related output forms one block with no blank lines inside it,
and exactly one blank line between blocks:

- a **tool block** is the call line plus its result connector directly beneath
  (`→ read  a.txt` / `└ ✓ 42 lines`), with any preview indented under the summary;
- a **prose run** is the assistant's streamed text (the `·` marker on its first row);
- a **status notice** is one `┄` line — yellow glyph, italic body — visible, because
  it means the loop intervened (retry, stuck recovery), never buried in dim.

**Every turn closes with a footer rule**: `── 2 iter · 15.3k tok · $0.0234 ──…`. A turn
that did not stop with `completed` prints a yellow `! stopped early: <reason>` line
above the rule — abnormal ends are never silent. The session banner closes with the
same rule, so the screen is a sequence of ruled-off turns.

**Live wait feedback.** While the loop is waiting, the renderer says what it is
waiting *for*: a transient spinner (`thinking…`, `running read…`) on the terminal
(real terminals only — hermetic tests never see it), a pulsing `· thinking…` line on
the web. Anything that takes over the terminal's bottom line (the permission
prompter) calls `zakcode.cli.render.suspend_live(console)` first; the renderer
restarts the spinner on its next event.

## Color: three tiers, four states

- **PRIMARY** (bright): the assistant's words and the salient tool target — the
  brightest things on screen.
- **SECONDARY** (`dim`): chrome — markers, labels, summaries, previews, footers.
- **ACCENT** (one color per state): cyan = active (prompt, assistant marker, tool
  verb), green = ok, red = error, yellow = attention (status notices, permission
  panel, stopped-early). Diffs use green/red for `+`/`-`.

The terminal palette lives in `cli/_theme.py`; the web palette is the CSS `:root`
variables in `index.html`. Map new styles into both.

## Truncation is always marked

Capped output says so: run output shows the first 12 lines then `… (+N more lines)`;
diff previews show 6 then the same marker. The web client instead collapses full tool
output behind the `└` summary line (`<details>` — errors arrive expanded), so nothing
floods and nothing is silently lost.

## Web-only affordances

The web client adds what a scrollback terminal cannot: click-to-expand tool output,
a Stop button (sends the WS `interrupt` frame), and a live session token/cost counter
in the header. It must stay a **pure renderer** — dependency-free, no build step, no
agent logic (`tests/test_webclient_contract.py` enforces the wire contract).

## Where the pieces live

| Piece | Terminal | Web |
| --- | --- | --- |
| Event renderer | `cli/render.py` (`StreamRenderer`) | `renderEvent()` in `index.html` |
| Theme / palette | `cli/_theme.py` | CSS `:root` |
| Glyphs + ASCII fallback | `cli/_glyphs.py` | unicode literals |
| Layout primitives | `cli/_layout.py` (margin, kv tables, notices) | CSS grid blocks |
| Permission prompt | `ConsolePermissionPrompter` (`cli/__init__.py`) | approval bar |
| Tests | `tests/test_render.py`, `tests/test_cli_chat.py` | `tests/test_webclient_contract.py` |

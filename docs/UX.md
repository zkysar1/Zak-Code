# UX — the Zak look

One visual language, rendered twice. The terminal client (`src/zakcode/cli/`) and the
bundled web client (`src/zakcode/server/static/index.html`) are thin renderers of the
same `AgentEvent` stream, drawn with the same grammar so an operator moves between them
without relearning the screen. This document is the binding cross-client contract —
either client can be re-derived from it; change it and both renderers in the same PR.

| Piece | Terminal | Web |
| --- | --- | --- |
| Event renderer | `cli/render.py` (`StreamRenderer`) | `renderEvent()` in `index.html` |
| Theme / palette | `cli/_theme.py` | CSS `:root` tokens |
| Glyphs + ASCII fallback | `cli/_glyphs.py` | unicode literals |
| Layout primitives | `cli/_layout.py` (`block`, `rail`, `panel`) | CSS grid blocks |
| Permission prompt | `ConsolePermissionPrompter` (`cli/__init__.py`) | approval card |
| Tests | `tests/test_render.py`, `tests/test_cli_chat.py` | `tests/test_webclient_contract.py` |

## Design thesis

Calm, confident minimalism with an instrument-panel spine: terminal-default ink does
the talking, all chrome recedes to dim, and exactly one brand mark — the azure spark
`✦` — carries identity through the banner, the wait line, and the prompt. Structure
comes from a two-level marker grammar (`●` block / `└` receipt) with true hanging
indents, a continuous `│` rail binding every result body to its block, and a
differentiated blank-line rhythm (one blank inside a turn, two at the turn seam), so
the transcript reads as a narrative of grouped tool blocks, never a log. Every receipt
carries a duration from an injectable monotonic clock, and every turn ends in a
state-colored receipt — the operator always knows what happened and how long it took.
Color is reserved for meaning (green/red/yellow + painted diff bands), boxes are
reserved for the two ceremonial moments (welcome, permission), and the web client
speaks the identical grammar translated into proportional type, soft surfaces, and
120ms motion.

## Terminal: the column grid

All content sits on this grid; nothing else exists:

| Column | Occupant |
| --- | --- |
| 0–1 | document margin, always blank |
| 2 | block markers: `●` `›` `·` `!` `✦` |
| 4 | block body + hanging-indent continuation; the `└` connector; the `│` rail; the `· lang` code tag |
| 6 | result bodies (right of the rail), receipt summaries (right of the elbow), code block text |

## Terminal: spacing & state rules (binding)

1. **One hanging-indent primitive.** Every marked line is a single-row 2-column
   `Table.grid` (gutter cell width 2, body cell) wrapped in the left margin: prose
   lines and tool calls at indent 2; receipts and rail rows at indent 4. Wrapped text
   lands under the body column, never under the marker — a ragged left edge is
   structurally impossible. **Marker on first line only:** within a prose group the
   marker occupies the gutter only on the first content line; every subsequent line
   prints with an empty 2-wide gutter cell — identical column math, one `●` per group,
   printed line-at-a-time with zero retroactive re-layout.
2. **The rail binds result regions.** Every result body line (search/glob previews,
   run output, diff lines, error detail, the `… +N …` more-line) renders as a rail
   row: `│` in the gutter cell at indent 4, body at col 6. Rail style is `tool.bar`
   (dim) normally and `tool.bar.err` (red) for every row under a failed tool. Blank
   lines *inside* a preserved output block render as a bare `│` row, keeping the
   region continuous. Assistant code blocks are NOT railed (they are the assistant's
   voice, not a tool region): blank above, dim `· lang` tag at col 4 (omitted when no
   language), `Syntax` body padded to col 6 (`ansi_dark`, `background_color="default"`).
3. **Blank-line state machine.** `StreamRenderer` tracks `_at_blank: bool` (was the
   last physical line written blank; initialized `False` at `render()` entry) and
   `_last_block: str | None` ∈ {`None`, `"prose"`, `"code"`, `"tool_call"`, `"tool"`,
   `"status"`}. One private method `_gap()` prints one blank line iff `not _at_blank`,
   then sets `_at_blank = True`. Discipline:
   - No block ever prints a trailing blank. Every block calls `_gap()` before printing
     — including the first block of the turn (that gap is the post-prompt breath).
   - Exception (group binding): a `tool_result` arriving directly after its OWN call
     line (`_last_block == "tool_call"` AND matching `tool_use_id`) skips `_gap()` —
     call line, `└` receipt, and rail rows are vertically contiguous. Any other
     (interleaved/out-of-order) result detaches: it gets a gap and its receipt is
     prefixed `{Tool} · ` so the operator can still pair it.
   - Blank lines arriving in prose never print directly; they just call `_gap()` (the
     `_at_blank` guard collapses runs of model blanks to one).
   - Every content print sets `_at_blank = False`; `_last_block` is set after each block.
   - Net effect, asserted in tests: the rendered transcript never contains two
     consecutive blank lines (`"\n\n\n" not in output`), and there is exactly one
     blank between any two adjacent blocks.
4. **`_assistant_marked` reset.** Resets to `False` at turn start and after **every**
   tool call, code block, and status line — each prose group re-anchors with its own `●`.
5. **Turn boundary = two blanks + the echoed prompt.** `read_prompt` prints **two**
   blank lines, then the `›` line; the renderer's first `_gap()` prints one blank
   after it. The seam reads: receipt / blank / blank / bright `›` line / blank / first
   block — a macro beat visibly larger than the intra-turn single blank. No rules, no
   timestamps. Never three blanks.
6. **No horizontal rules anywhere in the transcript.**
7. **Durations everywhere, via the injectable clock (mandatory).**
   `StreamRenderer(console=None, clock: Callable[[], float] | None = None)`;
   `self._clock = clock or time.monotonic`. `render()` records the turn start;
   `_on_tool_call` records `self._tool_started[event.id] = self._clock()` (keyed by
   id — parallel tool calls exist); `_on_tool_result` pops it and appends `· {dur}` to
   the receipt (omitted when the id is unknown). The footer appends per-turn elapsed.
   `_fmt_duration(s)` → `f"{s:.1f}s"` for `s < 60`, else `f"{int(s // 60)}m {int(s % 60)}s"`.
   Hermetic tests inject a `FakeClock` — wall-clock never reaches test output.
8. **Receipts are synthesized, never raw first lines.** All receipts end `· {dur}`:

   | Display name | Receipt |
   | --- | --- |
   | Read / List / Fetch | `N lines` (no preview) |
   | Run | `N lines` + head/tail preview, or `no output` |
   | Search | `N matches` + preview (≤5 lines, then `… +N more`) |
   | Glob | `N files` + preview (≤5 lines) |
   | Edit | `+a -d` (counts of `+`/`-` diff lines excluding `+++`/`---` headers; `N lines` when output isn't a diff). The diff receipt + painted preview activate only for diff-emitting tool output — the current builtin `edit_file` emits prose, so its receipts read `N lines` until the tool emits a unified diff (flagged as a core follow-up) |
   | Write | `written` |
   | Todo | `N items` |
   | unknown tool | `N lines` |
   | any error | `✗ {first line of error}` |

9. **Truncation is head+tail and direction-aware.** Run output > 12 lines shows the
   first `_RUN_HEAD = 6` + `… +N lines …` (`result.more`, as a rail row) + the last
   `_RUN_TAIL = 4` — terminal summaries like `42 passed` always survive. Diff previews
   cap at 12 lines + `… +N lines`. Tool args middle-truncate at 64 chars with `…`
   (`_squeeze_middle`); paths truncate from the **left** with a leading `…` so the
   filename survives (`_squeeze_path`).
10. **Errors restructure, not just recolor.** `└ ✗ summary · dur` (glyph in `err`) +
    up to 8 detail rail rows at col 6 in `err.body` (default fg — errors are shown,
    not dimmed) behind a red rail, then `… +N lines` (`result.more`) when detail was
    cut — truncation is never silent. (Web: the error receipt's first line is
    middle-squeezed at 64 chars; the full text is in the auto-opened card body.)
11. **Todo results glyph-map.** Lines beginning `[x] ` render as `✓ ` (`todo.done`) +
    text, lines beginning `[ ] ` as `○ ` (`todo.open`) + text; all other lines pass
    through untouched.
12. **Boxes only twice.** Welcome box and permission panel, both `ROUNDED`, width
    `max(24, min(terminal_width − 4, 60))`, padding `(1, 2)`, indented to col 2.
    Inside the welcome box, kv values longer than `panel_width − 20` are
    left-truncated with a leading `…`. Nothing else is ever boxed.
13. **Footer receipt (state-colored).** A `block()` at indent 2: marker `●` styled
    `ok` / `warn` / `err`; body dim: `{label} · {n} iterations · {tokens} · {cost} ·
    {elapsed}`. Label map (from the stop reasons in `agent/loop.py`):
    `completed → "done"`, `max_iterations → "stopped early — max iterations"`,
    `provider_error → "provider error"` (+ ` — ` + first line of `done.error` when
    present), `doom_loop → "stopped — repeating itself"`, `stuck → "stopped — no
    progress"`, `recipe_stalled → "stopped — recipe stalled"`, unknown →
    `stop_reason.replace("_", " ")`. Marker: `ok` for `completed`, `err` for
    `provider_error`, `warn` for everything else. Tokens: `f"{n/1000:.1f}k tokens"`
    at ≥1000 else `f"{n} tokens"`. Cost: `f"{c:.4f}"` with trailing zeros stripped to
    a minimum of 2 decimals, `$`-prefixed — `0.0230 → $0.023`, `0.0004 → $0.0004`,
    `0.0 → $0.00`, `1.5 → $1.50`. `done.usage` wins over the accumulated sum when
    `done.usage.total_tokens` is truthy.
14. **Permission-flow interleaving.** The prompter owns its own spacing: one blank
    before the panel and one blank after the decision line; it never touches renderer
    state. The renderer's `tool_result` still skips `_gap()` (rule 3 — it follows
    `_last_block == "tool_call"`), so the post-approval sequence reads: call line /
    blank / panel / `permit … › a` / blank / `└ ✗ receipt`.
15. **Append-only.** History is never repainted; the only live region is the
    REPL-owned wait line, cleanly replaced (transient) by the next printed block.
16. **`/cost` per-model breakdown.** After the session total line, when the session
    spanned **two or more** models (e.g. zakpick routed easy vs hard turns
    differently), `/cost` prints a dim `by model:` header then one indented
    `{model} · {tokens} tok · ${cost}` line per model (from
    `Session.usage_by_model()`, first-used model first; untagged usage is omitted). A
    single-model session shows only the total (no redundant one-line breakdown). Under
    zakpick a closing dim note flags that compaction/sub-agent costs are not broken out
    here and that a "vs all-deep" savings estimate lands with the cost-metadata seam.
17. **zakpick "deep coder wasn't needed" advisory.** zakpick-only, at most **once per
    session**, never naggy. After `_ZAKPICK_ADVISORY_AFTER` (3) turns that ended cleanly
    on `deep_code` and never tripped the soft latch (`AgentDone.routed_category ==
    "deep_code"`, `routed_escalated` False, `stop_reason == "completed"`), print one `tip`
    line suggesting a cheaper `deep_code` model may keep up, pointing at `/cost`. It states
    an observation and an option — never auto-changes routing (the user owns the choice).

**Wait line (REPL layer, never the renderer):** a transient `rich.live.Live` line —
spark frame (glyph-swap `· ✦ ✶ ✧`, brand azure; ASCII `- \ | /`) + gerund verb
(concrete `Running…` while a tool call is outstanding) + dim
`(ctrl-c to interrupt · {N}s)`, elapsed in whole seconds. Auto-disabled on legacy
conhost and off-tty, and force-disabled anywhere by `ZAKCODE_NO_SPINNER=1` — the
next printed `●` block is the fallback narrative. The permission prompter pauses it
before the panel and resumes after. All randomness (gerund choice) lives in the
REPL layer, never in `StreamRenderer`.

## Terminal theme (`cli/_theme.py`)

All values are ANSI names or numbered colors; rich downgrades automatically on legacy
terminals. Diff styles specify **both** fg and bg and are bold so the 16-color
downgrade (bold white on green/red) keeps contrast. No style assumes a dark background.

| Style name | Rich style string | Used for |
| --- | --- | --- |
| `brand` | `color(38)` | the `✦` spark (banner, /help header) |
| `brand.soft` | `color(38) dim` | the `✧` tip glyph |
| `banner.border` | `dim` | welcome box border |
| `banner.title` | `bold` | "Zak Code" in the box; section headings |
| `banner.label` | `dim` | kv labels (model, workspace…) |
| `banner.value` | `default` | kv values |
| `banner.hint` | `dim` | `/help for commands · /exit to quit` |
| `tip` | `dim` | tip line text |
| `prompt.marker` | `bold color(38)` | the `›` input chevron (incl. `permit … ›`) |
| `assistant.marker` | `color(38)` | the `●` before assistant prose |
| `md.h` | `bold` | headings (blank line forced above) |
| `md.code` | `dark_cyan` | inline code spans — dark_cyan, not cyan, so the 16-color downgrade of `color(38)` (→ cyan) never collides with inline code |
| `md.bullet` | `dim` | list bullet glyph |
| `tool.marker` | `dim` | the `●` before tool calls |
| `tool.name` | `bold` | Read / Edit / Run / Search |
| `tool.paren` | `dim` | the `(` `)` and `$ ` |
| `tool.args` | `default` | condensed argument |
| `tool.bar` | `dim` | the `│` rail beside result bodies |
| `tool.bar.err` | `red` | the `│` rail beside a **failed** tool's body |
| `result.connector` | `dim` | the `└` |
| `result.summary` | `dim` | "134 lines", "+6 -2", "· 0.1s" |
| `result.output` | `dim` | preview body lines |
| `result.more` | `dim italic` | `… +10 lines …` |
| `ok` | `green` | `✓`, success receipts, footer marker on clean done, todo done |
| `err` | `bold red` | `✗`, error labels, footer marker on provider error |
| `err.body` | `default` | error detail lines (full brightness — errors are shown, not dimmed) |
| `warn` | `yellow` | `!` interrupt notice, footer marker on early stop |
| `status` | `dim italic` | mid-turn status notices |
| `footer` | `dim` | the turn receipt body |
| `sep` | `dim` | `·` interpunct separators |
| `spinner` | `color(38)` | wait-line glyph |
| `diff.meta` | `dim` | `@@`, `---`, `+++` lines |
| `diff.add` | `bold grey93 on dark_green` | `+` lines (painted band, text extent) |
| `diff.del` | `bold grey93 on dark_red` | `-` lines (painted band, text extent) |
| `diff.ctx` | `dim` | context lines |
| `perm.border` | `yellow` | permission panel border |
| `perm.title` | `bold yellow` | panel title |
| `perm.tool` | `bold` | tool name in panel |
| `perm.reason` | `default` | humanized tier + reason |
| `perm.key` | `bold` | option numbers and y/a/n keys |
| `perm.option` | `default` | option text |
| `code.tag` | `dim` | the `· python` language tag |
| `todo.done` | `green` | `✓` in Todo results |
| `todo.open` | `dim` | `○` in Todo results |

Compatibility aliases (REQUIRED — consumed by `/permissions`, `/hooks`, `/plugins`,
`/skills`, `eval`, `info`, `_run_server_chat`, and plugin output paths the restyle
does not fully rewrite; rich raises on unknown style names):

| Alias | Maps to |
| --- | --- |
| `notice.dim` | `dim` |
| `arg.key` | `dim` |
| `arg.value` | `default` |
| `banner.version` | `dim` |
| `tool.verb` | `bold` |
| `tool.target` | `bold` |
| `rule.line` | `dim` |
| `perm.tier` | `yellow` |

## Glyphs (`cli/_glyphs.py`)

Every gutter-cell glyph is **exactly one character in both modes**, so the col 2/4/6
grid is identical under `ZAKCODE_ASCII`; multi-char fallbacks (`...`, `--`) appear
only inline, never in a gutter. Resolution: an encode probe over the unicode set
(`ZAKCODE_ASCII=1` forces ASCII; any encode failure falls back). Box borders come
from `rich.box.ROUNDED` (rich substitutes on legacy Windows); tests assert content
substrings, never border characters. The web client always uses the unicode set.

| Name | Unicode | ASCII | Notes |
| --- | --- | --- | --- |
| `spark` | `✦` | `*` | brand mark |
| `spark_soft` | `✧` | `*` | tip glyph |
| `marker` | `●` | `*` | assistant prose marker |
| `marker_tool` | `●` | `o` | tool-call marker — distinct ASCII so a NO_COLOR transcript still distinguishes prose from tools |
| `prompt` | `›` | `>` | |
| `elbow` | `└` | `\` | single char (Windows `tree /A` last-child convention) — receipt summary stays at col 6 in both modes |
| `bar` | `│` | `\|` | the result rail |
| `ok` | `✓` | `+` | single char |
| `fail` | `✗` | `x` | single char |
| `bang` | `!` | `!` | |
| `bullet` | `•` | `-` | |
| `dot` | `·` | `-` | also the status marker |
| `dash` | `—` | `--` | inline only |
| `ellipsis` | `…` | `...` | inline only |
| `hline` | `─` | `-` | kept for future use; no current consumer |
| `add` / `del` | `+` / `-` | `+` / `-` | |
| `todo_done` | `✓` | `+` | not `☒` — checkbox glyphs are a font-coverage gamble on Consolas-era consoles |
| `todo_open` | `○` | `o` | same well-covered block as `●` |
| `spin1`–`spin4` | `·` `✦` `✶` `✧` | `-` `\` `\|` `/` | spinner frames; always rendered via `Text` (never markup-parsed) |

## Web (`server/static/index.html`)

One self-contained file: vanilla HTML/CSS/JS, no build, no CDN, **textContent-only DOM
for all model/server data**. Wire-contract literals (the `EVENT_TYPES` array, one
literal `case` arm per event/frame type plus the `default:` arms surfacing
`[unknown event/frame: …]`, WS verbs `input`/`approval`/`interrupt`, the three
`sendApproval(...)` call sites, `fetch("/sessions", { method: "POST" })`) are enforced
by `tests/test_webclient_contract.py`; forbidden vendor/internal strings stay out of
the file, comments included.

### Tokens (CSS custom properties; every hex lives in `:root`)

Dark is default; light via `prefers-color-scheme: light` (`color-scheme: dark light`).

| Token | Dark | Light |
| --- | --- | --- |
| `--bg` / `--surface` / `--inset` | `#14171d` / `#1b2026` / `#0f1217` | `#faf9f7` / `#ffffff` / `#f2f0ec` |
| `--line` / `--line-soft` | `#2a313a` / `#20262e` | `#e2dfd9` / `#eceae5` |
| `--fg` / `--muted` / `--faint` | `#e9ecef` / `#99a3ae` / `#66707b` | `#33312d` / `#6e6b64` / `#a3a09a` |
| `--brand` | `#38b6df` | `#117ba8` |
| `--ok` / `--err` / `--warn` | `#5fc88a` / `#e5646e` / `#d9a13f` | `#2c8a55` / `#c23d49` / `#9a6b1f` |
| `--add-bg` / `--add-fg` | `#173321` / `#a9d8b4` | `#e3f2e6` / `#1d5e35` |
| `--del-bg` / `--del-fg` | `#391d22` / `#e3a9b0` | `#fae3e5` / `#8f303a` |
| `--shadow` | `0 8px 24px rgba(0,0,0,.35), 0 1px 2px rgba(0,0,0,.4)` | `0 8px 24px rgba(40,35,25,.08), 0 1px 2px rgba(40,35,25,.10)` |

`--brand` paints only: the spark, the `›`/`●` markers, focus rings, links, the
connection dot. It is **not** a button color.

### Type & metrics

| Token / metric | Value |
| --- | --- |
| `--font-prose` | `system-ui, "Segoe UI", "Helvetica Neue", Arial, sans-serif` |
| `--font-mono` | `ui-monospace, "Cascadia Code", "Cascadia Mono", Consolas, "SF Mono", Menlo, monospace` |
| Prose / mono / meta | 15px/1.65 `--fg` · 13px/1.5 · 12px mono `--muted` |
| Headings in model output | size-only scale, weight 600, normal color: h1 1.3em, h2 1.15em, h3 1.0em |
| Composer input | 16px (prevents iOS zoom) |
| Numerics | all mono receipt/meta/chip/summary text gets `font-variant-numeric: tabular-nums` |
| Content column | `max-width: 44rem`, centered, `padding: 2rem 1.25rem 3rem` — do **not** widen |
| Marker gutter | each block `display:grid; grid-template-columns: 1.75rem 1fr` (the web hanging indent) |
| Gaps | turn `margin-top: 2.25rem` on user lines; block `.9rem`; paragraph `.55rem` |
| Radii | 4px inline-code pill · 8px tool cards & code blocks · 10px approval card · 12px composer · 999px pills |
| Borders / elevation | 1px `--line` (cards), 1px `--line-soft` (insets); `--shadow` on composer, approval card, jump pill only |
| Motion | 120–150ms ease (chevron, buttons, approval slide-up, pill/overlay fades); `prefers-reduced-motion: reduce` zeroes durations and pauses the pending pulse |
| Focus | global `:focus-visible { outline: 2px solid var(--brand); outline-offset: 2px; }` |

### Components

- **Header** — 48px `--surface` bar: `✦` + "Zak Code"; right: dim mono chips (model,
  `sessionId.slice(0, 8)`), connection dot (`aria-hidden`; `--brand` pulsing /
  `--ok` open / `--err` closed) + adjacent text state label (the accessible carrier).
  Cost lives in per-turn receipts, never the header.
- **Empty state** — centered `✦` + name + `{model} · session {id8}` + hint, plus three
  sample-prompt pills that prefill the composer (never send); removed on first append.
- **User line** — gutter `›` in `--brand` mono; prose weight 500; no bubble — the
  bright short line *is* the turn separator.
- **Assistant prose** — 8px `--brand` dot per prose group (re-anchors after tool
  cards, mirroring the terminal `_assistant_marked` reset); inline code = mono pill
  on `--inset`; links `--brand` underlined.
- **Code blocks** — `--inset`, radius 8, mono 13, `overflow-x:auto`, dim uppercase
  language tag, hover-revealed Copy button ("Copied" for 2s).
- **Tool cards** — one `<details>` per call, created on `tool_call`, kept in a `Map`
  by `ev.id`. Summary row: status dot + **bold name** + dim middle-truncated `(args)`;
  right: dim receipt + rotating `▸` chevron. Pending: dot pulses, receipt shows `…`;
  on result the receipt fills (`134 lines · 0.1s`, `performance.now()` deltas by id)
  and the body fills (`--inset` `pre`, `max-height: 40vh`). Diff lines = full-width
  painted band divs; error cards add a 2px `--err` left border and full-`--fg` output.
  Orphan results (no card for the id) render as a standalone muted row, never throw;
  on `done`/`error` any still-pending card finalizes as `interrupted`. No-output
  results render without chevron, not clickable.
- **Status events** — gutter `·`, 12px mono italic `--muted`.
- **Stream-status overlay** — one pinned element above the composer,
  `pointer-events: none`, `aria-hidden`, never reflows. States: send → `✦ thinking…`;
  first delta → `✦ writing…`; outstanding tool_call → `✦ using tools…`; last
  tool_result, no new delta → `✦ thinking…`; `action_required` → `✦ waiting on you`;
  decision → `✦ thinking…`; `done`/`error`/close → hidden.
- **Turn receipt** — outcome-colored gutter dot (`--ok` completed / `--err`
  provider_error / `--warn` otherwise), 12px mono: `{label} · {n} iterations ·
  {tokens} · {cost} · {elapsed}s`; labels mirror the terminal (rule 13) via
  `STOP_LABELS`. The usage accumulator resets on every send; on `done`,
  `ev.usage.total_tokens ? ev.usage : accumulated` wins; elapsed =
  `performance.now()` since send. `error` control frames render as an error card and
  also end the turn.
- **Approval card** — above the composer on `action_required`: 3px `--warn` left
  border, slides up. Buttons: **Allow once** (solid `--fg`/`--bg` inversion —
  deliberately not brand), **Allow for session** (outline), **Deny** (outline, `--err`
  on hover), each with a `kbd` chip. The y/a/n `keydown` fires only while the card is
  visible **and** focus is not in the composer or any input/textarea/contenteditable.
  On decision the card collapses into a permanent consent receipt row: `{tool} —
  allowed once / allowed for session / denied` — the consent audit trail.
- **Composer** — floating card: `›` in `--brand`, chromeless auto-growing `<textarea>`
  (1–6 rows, Enter sends, Shift+Enter newline), 34px circular Send (`↑`) that morphs
  in place into Stop while streaming (sends `{ type: "interrupt" }`); the textarea
  stays enabled during turns, only submission is gated.
- **Scroll** — stick-to-bottom only within 40px of the bottom; when detached during
  streaming, a "↓ latest" pill floats above the composer and re-attaches on click.

## Model display grammar (zakpick) — binding, both clients

When `default_model` is the **zakpick** sentinel (task-category model routing, ADR-0009),
the model is no longer one slug — it is a model *per task category*. The display contract:

1. **Friendly per-category listing, never a raw slug.** The info panel and the `/model` command
   render zakpick as a **per-category listing**, one entry per routed category formatted
   `{category label} → {model} ({source})` — e.g. `easy coding → gpt-oss-20b (groq)`,
   `hard coding → gpt-oss-120b (groq)` (the current rendering joins them on one line). The
   user-facing name is the **plain-English category label** (`hard coding`, `easy coding`,
   `summaries`, `planning`, `delegated work`), never the internal key (`deep_code`, …). A raw
   litellm slug (`openai/gpt-oss-120b`) is **never** the headline; the model id appears only as
   the un-prefixed `model` half of the `model (source)` cell.
2. **Banner.** The welcome-box / header model line for a zakpick session reads
   **"zakpick · picks a model per task"** (the spark + label grammar; `banner.label` /
   `banner.value` styles), not a single model string. A concrete or `auto` model still shows
   its resolved model as before.
3. **Only routed categories are shown.** The table lists **only** categories that have a real
   call site (`quick_code`, `deep_code`, `summarize`, `plan`, `delegate`). `classify` is a
   reserved seam with no live caller, so it is **never advertised** — a panel must never claim
   a route the engine does not take.
4. **Web parity.** The web Header model chip and Empty-state `{model}` slot follow the same
   rule: under zakpick they show the `zakpick · picks a model per task` label (chip) and may
   expand the per-category `model (source)` table in the identity card; `textContent`-only,
   no raw slug as the headline, `classify` omitted.

## Discipline (binding)

- **Brand paints 1–2 character marks only** (`✦ ✧ › ●` and the spinner glyph; web:
  spark, markers, focus rings, links, connection dot) — never a run of text, never a
  button.
- **Model identity reads friendly, never as plumbing**: under zakpick the headline is
  `zakpick · picks a model per task` and the per-category table is `model (source)` with
  plain-English category labels — a raw litellm slug is never the headline, and `classify`
  (no live call site) is never shown.
- **Boxes only twice**: the welcome box and the permission panel. Nothing else is
  ever boxed.
- **No horizontal rules** anywhere in the transcript.
- **Diff bands paint to text extent** in the terminal (no full-width padded bands);
  the web paints full-width band divs inside tool cards.
- **No density regressions**: the one-blank intra-turn rhythm, the two-blank turn
  seam, and the 44rem web column are the spaciousness fix — do not tighten, do not
  widen.
- **Append-only transcript** in both clients; the only live surfaces are the REPL
  wait line and the pinned web overlay, neither of which reflows history.

## Shared grammar: terminal ↔ web

| Terminal construct | Web construct |
| --- | --- |
| col-2 marker + col-4 hanging body (`block()` grid) | `1.75rem 1fr` gutter grid |
| `●` assistant marker (azure) | 8px `--brand` dot per prose group |
| `●` tool line + `└ summary · dur` receipt | tool `<details>` card summary row + right receipt cell |
| `│` rail region at col 4–6 (red on failure) | card inset `pre` / `--err` left border |
| `·` status line | status row + pinned stream-status overlay |
| `● done ·` state-colored footer receipt | state-colored turn-receipt row |
| `›` echoed prompt + two-blank seam | user line + 2.25rem turn gap |
| welcome box | empty-state identity card |
| permission panel + `permit ›` | approval card + y/a/n keys |
| consent answer echo | consent receipt row |
| wait line (spinner) | stream-status overlay |

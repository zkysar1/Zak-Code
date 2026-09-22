# UX — the Zak look

One visual language, rendered twice. The terminal client (`src/zakcode/cli/`) and the
bundled web client (`src/zakcode/server/static/index.html`) are thin renderers of the
same `AgentEvent` stream, drawn with the same grammar so an operator moves between them
without relearning the screen — and both wear the Vinheim family's colours (see **The
family**). This document is the binding cross-client contract — either client can be
re-derived from it; change it and both renderers in the same PR.

| Piece | Terminal | Web |
| --- | --- | --- |
| Event renderer | `cli/render.py` (`StreamRenderer`) | `renderEvent()` in `index.html` |
| Theme / palette | `cli/_theme.py` | CSS `:root` tokens |
| Glyphs + ASCII fallback | `cli/_glyphs.py` | unicode literals |
| Layout primitives | `cli/_layout.py` (`block`, `rail`, `panel`) | CSS grid blocks |
| Permission prompt | `ConsolePermissionPrompter` (`cli/__init__.py`) | approval card |
| Tests | `tests/test_render.py`, `tests/test_cli_chat.py` | `tests/test_webclient_contract.py` |
| Family contract (both) | `tests/test_ux_family.py` | `tests/test_ux_family.py` |

## Design thesis

Calm, confident minimalism with an instrument-panel spine: terminal-default ink does
the talking, chrome recedes to dim — except the two lines the eye must find first, the
operator's own line and a tool's call line, which stay bright (ADR-0185) — and exactly
one brand mark — the teal spark `✦`, in the Vinheim family's accent (ADR-0218) — carries
identity through the banner, the wait line, and the prompt. Structure
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

## The family (ADR-0218)

The two clients also share one look with the Vinheim web app (vinheim.com, where agent
worlds are built and watched as they learn), so a member moving between Vinheim, this web
client and this terminal never changes rooms. The family's design system is Vinheim's:
navy paper, warm ink, a sea-teal accent, a gold focus ring, Inter for prose and Fraunces
for display.

- **The web client copies the family tokens verbatim** (the Tokens table names each
  one's family source), in the family's own `oklch()` notation, so a reader can diff them
  against the shipped stylesheet. Radii, pill buttons, the gold focus ring, the display
  face on the two identity moments and the uppercase live-status label come with them.
- **The terminal carries the family in its one colour mark.** It cannot paint a
  background, so the family arrives as the brand: `color(80)` is the 256-colour index
  nearest the family accent, so the spark and the assistant's `●` wear the teal that
  Vinheim's accent and the web client's brand wear. Chrome greys are already the family's muted ink by index
  (`color(245)` sits nearest `ink-3` after `color(246)`), and the semantic states stay
  ANSI names so each terminal's theme tunes them for its own background.
- **What stays Zak's own:** the transcript grammar (column grid, `●` / `└` / `│`,
  receipts) and the operator's orange — the human's line and chevron (ADR-0186). The
  family has no token for "the human", and it must stay the one warm run of text.

The contract is tested, not described: `tests/test_ux_family.py` asserts the web tokens
equal the family values, that every `var(--…)` the page reads is declared (an undeclared
custom property renders silently as nothing), that the page carries no second theme,
and that the terminal brand is still the index nearest the web accent — computed from
the colour values, so changing one side without the other fails.

## Terminal: the column grid

All content sits on this grid; nothing else exists:

| Column | Occupant |
| --- | --- |
| 0–1 | document margin — except the operator's line, the root of the turn: its `›` sits at col 0 and its text at col 2 (ADR-0186) |
| 2 | block markers: `●` `·` `!` `✦`; the operator's message text |
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
5. **Turn boundary = two blanks + the operator's bright line, on every door.** The
   keyboard door's frame (`read_prompt` / `open_input_frame`) draws the seam around
   the typed line. The say and harness doors — the only doors inside a cockpit — echo
   the message through `_layout.user_line` (ADR-0185): the seam's blanks (one when the
   idle wait already printed one, else two), then `› message` at column 0 — the root of
   the turn (ADR-0186) — with the chevron in `user.marker` and the text in `user.text`
   (both `bold color(214)`, the one orange in the transcript), and the door plus the
   wall-clock stamp grey at the end of the first line (`(say · 14:22)`); further lines
   sit at col 2; long messages fold (ADR-0119). Every agent block then nests under it at
   col 2. The renderer's first `_gap()` prints one blank after it. The seam reads:
   receipt / blank / blank / orange `›` line / blank / first block — a macro beat visibly
   larger than the intra-turn single blank. No rules; the only timestamps are the
   operator line's and the footer's. Never three blanks.
6. **No horizontal rules anywhere in the transcript.**
7. **Durations everywhere, via the injectable clock (mandatory).**
   `StreamRenderer(console=None, clock: Callable[[], float] | None = None)`;
   `self._clock = clock or time.monotonic`. `render()` records the turn start;
   `_on_tool_call` records `self._tool_started[event.id] = self._clock()` (keyed by
   id — parallel tool calls exist); `_on_tool_result` pops it and appends `· {dur}` to
   the receipt (omitted when the id is unknown). The footer appends per-turn elapsed.
   `_fmt_duration(s)` → `f"{s:.1f}s"` for `s < 60`, else `f"{int(s // 60)}m {int(s % 60)}s"`.
   Hermetic tests inject a `FakeClock` — wall-clock never reaches test output.
8. **Receipts are synthesized, never raw first lines.** Every receipt opens with its
   outcome mark and reads as a sentence that names its tool (ADR-0185) — `✓` (`ok`) on
   success, `✗` (`err`) on failure — and ends `· {dur}`:

   | Display name | Receipt |
   | --- | --- |
   | Read | `✓ Read N lines` (no preview) |
   | List | `✓ Listed N entries` (no preview) |
   | Fetch | `✓ Fetched N lines` + head/tail preview |
   | Run | `✓ Ran · N lines` + head/tail preview, or `✓ Ran · no output` |
   | Search | `✓ Found N matches` + preview (≤5 lines, then `… +N more`) |
   | Glob | `✓ Found N files` + preview (≤5 lines) |
   | Edit | `✓ Updated +a -d` (counts of `+`/`-` diff lines excluding `+++`/`---` headers; `✓ Edited · N lines` when output isn't a diff). The diff receipt + painted preview activate only for diff-emitting tool output |
   | Write | `✓ Written` |
   | Todo | `✓ Plan · N items`; a finished plan `✓ Plan complete · N steps` (ADR-0108); a partial plan `✓ Plan · F/T steps · current: …` (ADR-0124) |
   | unknown tool | `✓ {Name} · N lines` |
   | any error | `✗ {first line of error}`; detached (not directly under its own call line) `✗ {Name} · {first line}` — a success needs no such prefix, its verb names the tool |

   The web client's `receiptText` carries the same words; its card dot is the outcome mark.

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
    through untouched. **Harness plan changes draw too (ADR-0112):** a `task_update`
    event whose checklist rows differ from the last Todo drawn (from an `update_plan`
    result or a prior update) renders as a detached `└ Plan · N items` receipt with the
    same glyph-mapped rows — a request anchor, a skill skeleton or an investigation splice
    is visible without a tool call. A `task_update` that repeats the plan just drawn stays
    silent, so a model-authored plan is never shown twice. (Web: the plan row is headed
    `Goal: <request>` when the update carries the request the plan serves — ADR-0113.)
12. **Boxes only twice.** Welcome box and permission panel, both `ROUNDED`, width
    `max(24, min(terminal_width − 4, 60))`, padding `(1, 2)`, indented to col 2.
    Inside the welcome box, kv values longer than `panel_width − 20` are
    left-truncated with a leading `…`. Nothing else is ever boxed.
13. **Footer receipt (state-colored, stamped).** A `block()` at indent 2: marker `●`
    styled `ok` / `warn` / `err`; body dim: `{label} · {n} iterations · {tokens} · {cost} ·
    {elapsed} · {HH:MM}` — the wall clock when the turn ended (ADR-0185; injectable
    `wall` like the monotonic `clock`), so with the operator line's stamp every turn shows
    its time bracket. Label map (from the stop reasons in `agent/loop.py`):
    `completed → "done"` (`"done — struggled"` with a `warn` marker when
    `done.degraded` — a clean-looking footer over a turn that engaged failure
    recovery hid real give-ups, 2026-08-26; any label gains ` — N plan step(s) left
    open` when `done.open_steps` is non-zero, because a bare "done — struggled" over a
    plan stopped at 9/14 read as finished, ADR-0115), `max_iterations → "stopped early — max
    iterations"`, `provider_error → "provider error"` (+ ` — ` + first line of
    `done.error` when present), `doom_loop → "stopped — repeating itself"`, `stuck →
    "stopped — no progress"`, `gave_up → "stopped — gave up (no output)"`,
    `recipe_stalled → "stopped — recipe stalled"`, `veto_stall → "stopped — re-entry
    stalled: the stop hook kept asking for a skill that never ran"` (ADR-0187), unknown →
    `stop_reason.replace("_", " ")`. Marker: `ok` for un-degraded `completed`, `err`
    for `provider_error`, `warn` for everything else. Tokens: `f"{n/1000:.1f}k tokens"`
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
18. **Body text wraps at the reading width.** `_layout.READ_WIDTH = 100` (ADR-0185): the
    body cell of every `block()` and `rail()` row is capped at `min(100, console.width −
    indent − 2)`, so the return sweep never exceeds a hundred characters however wide the
    pane (the comfort band for body text is 50–75 characters; WCAG 1.4.8 caps it at 80;
    a monospace column with a hanging indent reads well to about 100). Code blocks
    (`Syntax`) keep the console width — they are not body text.
19. **Inline markdown grammar** (`_inline_md` / `_inline_spans`, ADR-0185): bullets
    `- ` / `* ` → the bullet glyph; ATX headings → `md.h`; `> quote` → `md.quote`; a lone
    `---` / `***` / `___` → a gap (rule 6: no rules drawn); `***x***` → bold italic;
    `**bold**` / `__bold__` → bold; `*italic*` / `_italic_` → `md.italic`; `~~strike~~` →
    `md.strike`. An emphasis span needs a word boundary on both sides (`snake_case`,
    `2*3*4`, `a*b*c` stay literal), no space inside the delimiters (`2 * 3`), a letter or
    digit in its content (`*/*`, `**/*.py`, `f(*args, **kwargs)` stay literal), no file
    extension after the closer (`__init__.py`), never its own delimiter inside (`**/*.py
    and src/**/*.py` cannot pair across the globs), and `__` never wraps a bare identifier
    (`__str__`) — code-shaped text renders as written (fresh-eyes review, 2026-09-17); `[label](url)` → the label in `md.link`
    plus ` (url)` in `md.link.url` when they differ; `` `code` `` → `md.code`. Never
    nested; never markup-parsed.
    The web client's `spanAt` / `inlineSpans` / `renderProse` carry the same grammar,
    token for token (ADR-0186): a change to one is a change to both in the same PR.
20. **Log records are transcript lines, never raw output** (ADR-0186). The interactive
    CLI installs `cli/logsink.install_transcript_logging` before anything prints: the
    entry point's stdout handler goes, the full record goes to
    `~/.zakcode/logs/zakcode.log` (rotating), and `TranscriptLogHandler` prints each
    record as a `·` row at indent 4 in `log` (INFO) / `warn` / `err`. Compact form, one
    line each: litellm's call record is `call <model> · <provider>`; an httpx request
    `POST <model:method> · 200 OK`; zakcode's own records `<logger>: <message>` without
    the `zakcode.` prefix. Third-party INFO chatter that is not the per-call line stays in
    the file; WARNING and above always print. Every record passes `RedactingFilter`
    (`redact_url_credentials` then `redact_secrets`) before any handler — the daemon's
    stdout handler carries the same filter, so `serve.log` never holds a `?key=` either.

**Say box (the cockpit's one input, ADR-0119):** the bottom tmux pane runs ONE
persistent `SayBoxEditor` (`cli/saybox.py`) for the life of the box — never a prompt
rebuilt per message. Its contract:

| Gesture | Behaviour |
| --- | --- |
| paste > 3 lines or > 400 chars | collapses to one token `⟪pasted #N · 120 lines⟫`; Backspace/Delete remove the whole token; the message (and the history entry) carries the real text |
| Enter | sends, always (`Ctrl+J` inserts a newline) |
| `Ctrl+U` / `Ctrl+Z` | clear the whole input / bring it back (undo) |
| `Ctrl+C` | clears the input; a second press within 2s closes the box — never an instant exit (Ctrl+C-to-copy is a habit) |
| `↑` / `↓` at the top/bottom line | recall from `~/.zakcode/say-history` (persists across sessions); ghost text from history, `→` accepts |
| `Esc` | recall a still-pending message into the buffer, else stop the running agent (the half-typed text survives) |
| continuation lines | grey `· ` gutter under the `▸ ` prompt — the `▸` bold in the human's orange (`#ffaf00`, the chat pane's `color(214)`) |

The pane starts at 5 rows and **grows with the text** (wrapped rows + toolbar + one
breath, capped at 16) then shrinks back after each send; a token keeps a paste at one
row. The last send's status (`✓ sent (12 lines) 14:22`, busy, stop sent) lives in a dim
bottom toolbar beside the key help, never in the pane's scrollback — which is cleared
on every prompt so a wheel-up in the box shows nothing stale. In the chat pane a long
message echoes folded: the first 6 lines then `… (+N more lines)` (`fold_lines`), so a
200-line paste never buries the turn it started.

**Wait line (REPL layer, never the renderer):** a transient `rich.live.Live` line —
spark frame (glyph-swap `· ✦ ✶ ✧`, brand teal; ASCII `- \ | /`) + gerund verb
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
| `brand` | `color(80)` | the `✦` spark (banner, /help header) — the index nearest the family accent (ADR-0218) |
| `brand.soft` | `color(73)` | the `✧` tip glyph — the family's pressed accent by index, never `dim` (ADR-0186) |
| `banner.border` | `color(245)` | welcome box border |
| `banner.title` | `bold` | "Zak Code" in the box; section headings |
| `banner.label` | `color(245)` | kv labels (model, workspace…) |
| `banner.value` | `default` | kv values |
| `banner.hint` | `color(245)` | `/help for commands · /exit to quit` |
| `tip` | `color(245)` | tip line text |
| `prompt.marker` | `bold color(214)` | the `›` input chevron (incl. `permit … ›`) — orange means the human (ADR-0186) |
| `user.marker` | `bold color(214)` | the `›` at col 0 before the operator's line (ADR-0185/0186) |
| `user.text` | `bold color(214)` | the operator's message — the one orange run of text in the transcript |
| `user.meta` | `not bold color(245)` | its door + wall-clock stamp `(say · 14:22)` |
| `assistant.marker` | `color(80)` | the `●` before assistant prose |
| `md.h` | `bold` | headings (blank line forced above) |
| `md.code` | `dark_cyan` | inline code spans — on a 16-color terminal `dark_cyan` falls to cyan while the brand's `color(80)` falls to bright cyan, so inline code never collides with the brand marks (the old `color(38)` brand fell to cyan too — ADR-0218) |
| `md.bullet` | `color(245)` | list bullet glyph |
| `md.italic` | `italic` | `*italic*` / `_italic_` spans |
| `md.strike` | `strike` | `~~strike~~` spans |
| `md.link` | `underline` | a link's label |
| `md.link.url` | `color(245)` | a link's ` (url)` |
| `md.quote` | `color(245) italic` | `> quote` lines |
| `tool.marker` | `bold` | the `●` before tool calls — bright: the loudest line of its block (ADR-0185) |
| `tool.name` | `bold` | Read / Edit / Run / Search |
| `tool.paren` | `color(245)` | the `(` `)` and `$ ` |
| `tool.args` | `default` | condensed argument |
| `tool.bar` | `color(245)` | the `│` rail beside result bodies |
| `tool.bar.err` | `red` | the `│` rail beside a **failed** tool's body |
| `result.connector` | `color(245)` | the `└` |
| `result.summary` | `color(245)` | "134 lines", "+6 -2", "· 0.1s" |
| `result.output` | `color(245)` | preview body lines |
| `result.more` | `color(245) italic` | `… +10 lines …` |
| `ok` | `green` | the `✓` that opens every success receipt, footer marker on clean done, todo done |
| `err` | `bold red` | `✗`, error labels, footer marker on provider error |
| `err.body` | `default` | error detail lines (full brightness — errors are shown, not dimmed) |
| `warn` | `yellow` | `!` interrupt notice, footer marker on early stop |
| `status` | `color(242) italic` | mid-turn status notices — one step quieter than the chrome |
| `log` | `color(242)` | a log record rendered as a transcript line (ADR-0186) |
| `footer` | `color(245)` | the turn receipt body |
| `sep` | `color(245)` | `·` interpunct separators |
| `spinner` | `color(80)` | wait-line glyph |
| `diff.meta` | `color(245)` | `@@`, `---`, `+++` lines |
| `diff.add` | `bold grey93 on dark_green` | `+` lines (painted band, text extent) |
| `diff.del` | `bold grey93 on dark_red` | `-` lines (painted band, text extent) |
| `diff.ctx` | `color(245)` | context lines |
| `perm.border` | `yellow` | permission panel border |
| `perm.title` | `bold yellow` | panel title |
| `perm.tool` | `bold` | tool name in panel |
| `perm.reason` | `default` | humanized tier + reason |
| `perm.key` | `bold` | option numbers and y/a/n keys |
| `perm.option` | `default` | option text |
| `code.tag` | `color(245)` | the `· python` language tag |
| `todo.done` | `green` | `✓` in Todo results |
| `todo.open` | `color(245)` | `○` in Todo results |

Compatibility aliases (REQUIRED — consumed by `/permissions`, `/hooks`, `/plugins`,
`/skills`, `eval`, `info`, `_run_server_chat`, and plugin output paths the restyle
does not fully rewrite; rich raises on unknown style names):

| Alias | Maps to |
| --- | --- |
| `notice.dim` | `color(245)` |
| `arg.key` | `color(245)` |
| `arg.value` | `default` |
| `banner.version` | `color(245)` |
| `tool.verb` | `bold` |
| `tool.target` | `bold` |
| `rule.line` | `color(245)` |
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

### Tokens (CSS custom properties; every colour lives in `:root`)

One look, dark, all the time (`color-scheme: dark`): the page wears the Vinheim family's
tokens (see **The family** above), copied verbatim, so there is no light theme to keep in
step — the family has none either.

| Token | Value | Family token |
| --- | --- | --- |
| `--bg` / `--surface` / `--inset` | `oklch(17% .030 258)` / `oklch(21% .032 258)` / `oklch(25% .034 258)` | paper / paper-2 / paper-3 |
| `--line` / `--line-strong` | `oklch(34% .030 258)` / `oklch(42% .034 258)` | line / line-2 |
| `--line-soft` | `oklch(29% .032 258)` | — (between paper-3 and line: insets inside cards) |
| `--fg` / `--muted` | `oklch(96% .010 90)` / `oklch(66% .018 250)` | ink / ink-3 |
| `--faint` | `oklch(54% .018 250)` | — (one step under ink-3: urls, tags, placeholders) |
| `--brand` / `--brand-press` / `--on-brand` | `oklch(80% .110 195)` / `oklch(72% .110 195)` / `oklch(20% .030 258)` | accent / accent-press / on-accent |
| `--focus` | `oklch(86% .140 85)` | focus (gold) |
| `--ok` / `--err` / `--warn` | `oklch(78% .130 150)` / `oklch(70% .180 25)` / `oklch(82% .140 75)` | success / danger / warning |
| `--user` | `#f2a53c` | — the operator's orange (ADR-0186), deliberately not a family token |
| `--add-bg` / `--add-fg` | `color-mix(in oklab, var(--ok) 18%, var(--inset))` / `color-mix(in oklab, var(--ok) 40%, var(--fg))` | derived |
| `--del-bg` / `--del-fg` | `color-mix(in oklab, var(--err) 18%, var(--inset))` / `color-mix(in oklab, var(--err) 40%, var(--fg))` | derived |
| `--shadow` | `0 8px 24px rgba(0,0,0,.35), 0 1px 2px rgba(0,0,0,.4)` | — |

`--brand` paints only: the spark, the assistant's `●` marker, links, the connection dot,
and the page's one primary action — Send (the family's primary button: accent fill,
`--on-brand` glyph, `--brand-press` under the pointer). Hover borders on pills and the
attach button may borrow it as an affordance. It is never a run of text and never another
button's fill: Allow once stays a neutral inversion, because a consent choice is not a call
to action. Focus rings are `--focus` gold, as everywhere in the family.

### Type & metrics

| Token / metric | Value |
| --- | --- |
| `--font-prose` | `Inter, system-ui, -apple-system, "Segoe UI", sans-serif` — the family's body face when installed; no webfont is fetched (no CDN) |
| `--font-display` | `Fraunces, Georgia, "Times New Roman", serif` — the family's display face, for the two identity moments only: the header wordmark and the empty-state name |
| `--font-mono` | `ui-monospace, "Cascadia Code", "Cascadia Mono", Consolas, "SF Mono", Menlo, monospace` |
| Prose / mono / meta | 15px/1.65 `--fg` · 13px/1.5 · 12px mono `--muted` |
| Headings in model output | size-only scale, weight 600, normal color: h1 1.3em, h2 1.15em, h3 1.0em |
| Composer input | 16px (prevents iOS zoom) |
| Numerics | all mono receipt/meta/chip/summary text gets `font-variant-numeric: tabular-nums` |
| Content column | `max-width: 44rem`, centered, `padding: 2rem 1.25rem 3rem` — do **not** widen |
| Marker gutter | each block `display:grid; grid-template-columns: 1.75rem 1fr` (the web hanging indent) |
| Gaps | turn `margin-top: 2.25rem` on user lines; block `.9rem`; paragraph `.55rem` |
| Radii | the family's two: `--radius-sm` 9px (tool/thinking/error cards, code blocks, the attach button) and `--radius` 14px (approval card, composer) · 6px insets nested in a card · 4px inline-code pill · 999px pills and buttons |
| Borders / elevation | 1px `--line` (cards), 1px `--line-soft` (insets); `--shadow` on composer, approval card, jump pill only |
| Motion | 120–150ms ease (chevron, buttons, approval slide-up, pill/overlay fades); `prefers-reduced-motion: reduce` zeroes durations and pauses the pending pulse |
| Focus | global `:focus-visible { outline: 2px solid var(--focus); outline-offset: 3px; }` — the family's gold ring |

### Components

- **Header** — 48px bar on the page's own `--bg` with a `--line` rule beneath (the
  family's nav): `✦` + "Zak Code" in `--font-display` bold; right: dim mono chips (model,
  `sessionId.slice(0, 8)`), connection dot (`aria-hidden`; `--brand` pulsing /
  `--ok` open / `--err` closed) + adjacent text state label (the accessible carrier) in
  the family's live-status grammar — 11px, weight 600, uppercase, `.06em` tracking.
  Cost lives in per-turn receipts, never the header.
- **Empty state** — centered `✦` + name (`--font-display`, 30px) + `{model} · session
  {id8}` + hint, plus three sample-prompt pills that prefill the composer (never send);
  removed on first append.
- **User line** — gutter `›` and text in `--user` (orange, weight 700 / 600, ADR-0186); no bubble — the
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
  border, `--radius`, slides up; "Permission" is a label, not a heading — the family's
  small uppercase eyebrow in `--warn`, beside the bold mono tool name. Buttons are the
  family's pills: **Allow once** (solid `--fg`/`--bg` inversion — deliberately not
  brand), **Allow for session** (outline, `--brand` on hover), **Deny** (outline, `--err`
  on hover), each with a `kbd` chip. The y/a/n `keydown` fires only while the card is
  visible **and** focus is not in the composer or any input/textarea/contenteditable.
  On decision the card collapses into a permanent consent receipt row: `{tool} —
  allowed once / allowed for session / denied` — the consent audit trail.
- **Composer** — floating card dressed as the family's input (`--inset` fill,
  `--line-strong` border, `--radius`, a 2px `--focus` gold ring while focused): `›` in
  `--user` — the human's chevron is orange on every door (ADR-0186) — chromeless
  auto-growing `<textarea>` (1–6 rows, Enter sends, Shift+Enter newline), 34px circular
  Send (`↑`, the family's primary button) that morphs in place into Stop while streaming
  (sends `{ type: "interrupt" }`); the textarea stays enabled during turns, only
  submission is gated.
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

- **Brand paints 1–2 character marks only** (`✦ ✧ ●` and the spinner glyph; web:
  spark, the assistant dot, links, connection dot, and Send — the one primary action) —
  never a run of text, never another button's fill. The human's `›`/`▸` is orange on every
  door, in both clients.
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
  seam (on every door), the 100-column reading width and the 44rem web column are the
  spaciousness fix — do not tighten, do not widen.
- **Two loud lines per turn shape, no more**: the operator's line (orange) and the tool
  call line (bright); receipts, rails and previews stay grey, with the outcome mark the
  only colour in a receipt (ADR-0185/0186).
- **Grey by colour index, never by `dim`**: chrome is `color(245)`, status and log rows
  `color(242)`; the `dim` attribute is dropped by tmux and some terminals, so a `dim`
  style is a full-contrast style in the cockpit (ADR-0186).
- **Append-only transcript** in both clients; the only live surfaces are the REPL
  wait line and the pinned web overlay, neither of which reflows history.

## Shared grammar: terminal ↔ web

| Terminal construct | Web construct |
| --- | --- |
| col-2 marker + col-4 hanging body (`block()` grid) | `1.75rem 1fr` gutter grid |
| `●` assistant marker (teal) | 8px `--brand` dot per prose group |
| `●` tool line + `└ summary · dur` receipt | tool `<details>` card summary row + right receipt cell |
| `│` rail region at col 4–6 (red on failure) | card inset `pre` / `--err` left border |
| `·` status line | status row + pinned stream-status overlay |
| `● done ·` state-colored footer receipt | state-colored turn-receipt row |
| `›` operator line (bold, door + stamp) + two-blank seam on every door | user line + 2.25rem turn gap |
| welcome box | empty-state identity card |
| permission panel + `permit ›` | approval card + y/a/n keys |
| consent answer echo | consent receipt row |
| wait line (spinner) | stream-status overlay |

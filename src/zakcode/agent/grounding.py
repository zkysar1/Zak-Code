"""Write-grounding (Slice 1 of the Recipe Cursor design).

A weak local model hallucinates that a write succeeded or did what it intended ("I
added the Zak logic" when it did not, and it cannot see the real file). After each
successful ``write_file`` / ``edit_file``, the harness reads the file back from disk and
injects the REAL content plus a deterministic syntax check as a user observation, so the
model's next decision is grounded in ground truth it cannot invent.

The syntax check runs on the WHOLE file; the echo is capped. A file longer than the cap is
echoed as a window around the line the edit tool reports it changed (ADR-0252), with the
span it shows and the spans it leaves out named in line numbers. Before that decision the
echo was the file's head, cut at the cap, and the syntax check compiled that cut text: on the
pod's worker Bodies 71% of edits to long files got a frame without the edited lines, and
every ``[syntax: FAIL]`` seen in a week sat on a cut file (0 of 181 whole files failed).

Pure and vendor-agnostic: :func:`build_write_grounding` correlates the iteration's tool
calls with their results, reads the written files, and returns one user
:class:`~zakcode.messages.Message` (or ``None``). The loop appends it once per iteration
after the tool batch — no provider, transport, or vendor knowledge involved.
"""

from __future__ import annotations

from pathlib import Path

from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import ToolCall
from zakcode.providers.text_tools import defang_untrusted

#: Tools whose successful result triggers a read-back.
#: Both spellings: a session resumed from before ADR-0190 carries ``write_file`` blocks.
_WRITE_TOOLS = {"Write", "Edit", "write_file", "edit_file"}

#: Cap on re-injected content per file so a large write cannot blow the context window.
_MAX_GROUNDING_CHARS = 4000


def syntax_note(path: str, content: str) -> str:
    """A deterministic ``[syntax: OK|FAIL ...]`` note for a ``.py`` path (else "")."""
    if not path.endswith(".py") or not content.strip():
        return ""
    try:
        compile(content, path, "exec")
    except SyntaxError as exc:
        where = f" line {exc.lineno}" if exc.lineno else ""
        return f"[syntax: FAIL{where}: {exc.msg}]"
    return "[syntax: OK]"


def _read_back(path: str) -> tuple[str, bool]:
    """Read the WHOLE of ``path``. Returns ``(text, ok)``; ``ok=False`` only on a genuine IO
    failure (deleted/locked/raced). Decodes with ``errors="replace"`` (matching
    ``read_file``) so odd-but-readable bytes still ground rather than silently vanishing.
    Capping happens in :func:`_excerpt`, after the syntax check has seen the whole file."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace"), True
    except Exception:  # noqa: BLE001 — genuine IO failure: surfaced as [unverified] below
        return "", False


def _edited_line(result: ToolResultBlock) -> int | None:
    """The 1-based line the edit tool reports it changed, when its result carries one.

    ``Edit`` names the line of its first replacement as ``line`` (with ``replace_all`` the
    later ones fall where they fall; the window centres on the first and the spans it leaves
    out are named). A whole-file write reports no line — there is no single line to centre
    on — and the echo keeps the head.
    """
    if not isinstance(result.data, dict):
        return None
    line = result.data.get("line")
    return line if isinstance(line, int) and line > 0 else None


def _excerpt(text: str, max_chars: int, *, anchor_line: int | None) -> tuple[str, str]:
    """The part of ``text`` to echo and a locator for the header (``""`` when all of it fits).

    A file longer than ``max_chars`` is shown as whole lines around ``anchor_line`` — grown
    one line at a time below and above the anchor until the cap is reached — or from the head
    when no anchor is known. Every span left out is named by line number, so the model knows
    what it is NOT seeing instead of taking a cut file for the whole one.
    """
    if len(text) <= max_chars:
        return text, ""
    lines = text.splitlines(keepends=True)
    n = len(lines)
    centre = min(max(anchor_line or 1, 1), n) - 1
    if len(lines[centre]) > max_chars:
        # One line alone overflows the cap: show its head and say so; no neighbour fits.
        body = lines[centre][:max_chars].rstrip("\n")
        cut = (
            f"\n... (truncated: line {centre + 1} cut at {max_chars} of {len(lines[centre])} chars)"
        )
        lo = hi = centre
    else:
        lo = hi = centre
        total = len(lines[centre])
        while True:
            grew = False
            if hi + 1 < n and total + len(lines[hi + 1]) <= max_chars:
                hi += 1
                total += len(lines[hi])
                grew = True
            if lo > 0 and total + len(lines[lo - 1]) <= max_chars:
                lo -= 1
                total += len(lines[lo])
                grew = True
            if not grew:
                break
        body = "".join(lines[lo : hi + 1]).rstrip("\n")
        cut = ""
    before = f"... (truncated: lines 1-{lo} of {n} not shown)\n" if lo > 0 else ""
    after = f"\n... (truncated: lines {hi + 2}-{n} of {n} not shown)" if hi < n - 1 else ""
    return f"{before}{body}{cut}{after}", f" (lines {lo + 1}-{hi + 1} of {n})"


def _written_path(call: ToolCall, result: ToolResultBlock) -> str | None:
    """The file path a successful write/edit touched (from result data, else args)."""
    if isinstance(result.data, dict):
        path = result.data.get("path")
        if isinstance(path, str) and path:
            return path
    arg_path = call.arguments.get("path")
    return arg_path if isinstance(arg_path, str) and arg_path else None


def build_write_grounding(
    calls: list[ToolCall],
    results: list[ToolResultBlock],
    *,
    max_chars: int = _MAX_GROUNDING_CHARS,
) -> Message | None:
    """Build one grounding user-message echoing back files written this iteration.

    Correlates each successful ``write_file``/``edit_file`` call with its result by id,
    reads the file from disk, and renders the real content + a syntax note. Returns
    ``None`` when no write happened or nothing could be read back.
    """
    by_id = {r.tool_use_id: r for r in results}
    sections: list[str] = []
    for call in calls:
        if call.name not in _WRITE_TOOLS:
            continue
        result = by_id.get(call.id)
        if result is None or result.is_error:
            continue
        path = _written_path(call, result)
        if path is None:
            continue
        content, ok = _read_back(path)
        if not ok:
            # A successful write whose read-back fails: surface it as [unverified] rather
            # than silently dropping the grounding (which would leave the model believing
            # nothing happened). (audit2 #13)
            sections.append(
                f"[unverified] could not read {defang_untrusted(path)} back from disk to "
                "confirm the write (it may have been moved, locked, or changed)."
            )
            continue
        # The syntax check runs on the REAL on-disk bytes — all of them. Compiling the capped
        # excerpt instead reported a cut file as broken (ADR-0252).
        note = syntax_note(path, content)
        shown, where = _excerpt(content, max_chars, anchor_line=_edited_line(result))
        # This content is model-authored and is re-injected into a TRUSTED user message, so
        # neutralize any protocol/template sentinels it contains before embedding — else a
        # file whose text includes </tool_result> or <|im_start|> could forge a frame in
        # the next text-protocol turn. (audit2 #2; same trust-boundary rule as tool output.)
        header = f"{defang_untrusted(path)} now on disk{where}" + (f" {note}" if note else "") + ":"
        sections.append(f"{header}\n{defang_untrusted(shown)}")

    if not sections:
        return None
    body = (
        "[verified] The system read these files back from disk after your write - this "
        "is the real, current content. Use it; do not assume what the files contain:\n\n"
        + "\n\n".join(sections)
    )
    return Message.user(body)

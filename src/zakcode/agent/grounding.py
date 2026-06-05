"""Write-grounding (Slice 1 of the Recipe Cursor design).

A weak local model hallucinates that a write succeeded or did what it intended ("I
added the Zak logic" when it did not, and it cannot see the real file). After each
successful ``write_file`` / ``edit_file``, the harness reads the file back from disk and
injects the REAL content plus a deterministic syntax check as a user observation, so the
model's next decision is grounded in ground truth it cannot invent.

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
_WRITE_TOOLS = {"write_file", "edit_file"}

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


def _read_back(path: str, *, max_chars: int) -> tuple[str, bool]:
    """Read ``path`` (capped). Returns ``(text, ok)``; ``ok=False`` only on a genuine IO
    failure (deleted/locked/raced). Decodes with ``errors="replace"`` (matching
    ``read_file``) so odd-but-readable bytes still ground rather than silently vanishing."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — genuine IO failure: surfaced as [unverified] below
        return "", False
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text, True


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
        content, ok = _read_back(path, max_chars=max_chars)
        if not ok:
            # A successful write whose read-back fails: surface it as [unverified] rather
            # than silently dropping the grounding (which would leave the model believing
            # nothing happened). (audit2 #13)
            sections.append(
                f"[unverified] could not read {defang_untrusted(path)} back from disk to "
                "confirm the write (it may have been moved, locked, or changed)."
            )
            continue
        note = syntax_note(path, content)  # syntax check runs on the REAL on-disk bytes
        # This content is model-authored and is re-injected into a TRUSTED user message, so
        # neutralize any protocol/template sentinels it contains before embedding — else a
        # file whose text includes </tool_result> or <|im_start|> could forge a frame in
        # the next text-protocol turn. (audit2 #2; same trust-boundary rule as tool output.)
        header = f"{defang_untrusted(path)} now on disk" + (f" {note}" if note else "") + ":"
        sections.append(f"{header}\n{defang_untrusted(content)}")

    if not sections:
        return None
    body = (
        "[verified] The system read these files back from disk after your write - this "
        "is the real, current content. Use it; do not assume what the files contain:\n\n"
        + "\n\n".join(sections)
    )
    return Message.user(body)

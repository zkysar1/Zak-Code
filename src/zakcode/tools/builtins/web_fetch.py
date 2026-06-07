"""``web_fetch`` — fetch a public URL and return its readable text.

Security is the whole job here (the URL often comes from a model, so it is untrusted):

* **SSRF + DNS-rebinding** — :func:`zakcode._http.resolve_pinned_url` validates the host AND
  pins the connection to the checked IP (original host kept for the ``Host`` header + TLS SNI),
  run before the request AND before following each redirect. Redirects are followed MANUALLY
  (httpx auto-redirect is disabled) so every hop is re-validated and re-pinned, and a public URL
  cannot 3xx-redirect — or DNS-rebind — into ``localhost`` / a private range / the metadata IP.
* **Bounded** — compression is disabled (``Accept-Encoding: identity``, so on-wire bytes are the
  real bytes — no decompression bomb), a declared over-cap ``Content-Length`` short-circuits, the
  stream is read in bounded chunks and capped, and the returned text is char-capped.
* **Untrusted** — every model-facing string (the body AND error messages that interpolate an
  attacker-influenced URL/exception) is run through ``defang_untrusted``, so page content can
  never smuggle tool-protocol frames or chat-template tokens into the loop.
"""

from __future__ import annotations

import asyncio
import codecs

from zakcode._http import BlockedUrlError, load_httpx, resolve_pinned_url
from zakcode.config import PermissionTier
from zakcode.providers.text_tools import defang_untrusted
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins._html import html_to_text

_DEFAULT_TIMEOUT = 15
_MAX_BYTES = 2 * 1024 * 1024  # cap bytes downloaded off the wire
_CHUNK_SIZE = 64 * 1024  # bound per-read memory so a huge body can't be buffered whole
_MAX_CHARS = 50_000  # cap chars returned to the model
_MAX_REDIRECTS = 5
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_USER_AGENT = "zakcode-webfetch/0.1 (+https://github.com/zkysar1/Zak-Code)"
_INSTALL_FIX = "install web deps with: pip install 'zakcode[web]'"


def _looks_textual(content_type: str) -> bool:
    """True for content types we can render as text (html / plain / json / xml / similar)."""
    ct = content_type.split(";", 1)[0].strip().lower()
    if ct.startswith("text/"):
        return True
    return ct in {
        "application/json",
        "application/xml",
        "application/xhtml+xml",
        "application/rss+xml",
        "application/atom+xml",
        "application/javascript",
    }


def _charset(content_type: str) -> str:
    """The charset from a Content-Type header, validated to a real codec (else utf-8).

    A server fully controls the charset token, and an unknown codec name makes ``bytes.decode``
    raise ``LookupError`` (which ``errors='replace'`` does NOT cover) — so validate it here.
    """
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.strip().lower() == "charset" and value.strip():
            name = value.strip().strip("\"'")
            try:
                codecs.lookup(name)
            except LookupError:
                return "utf-8"
            return name
    return "utf-8"


class WebFetchTool(Tool):
    """Fetch a public http(s) URL and return its readable text content."""

    spec = ToolSpec(
        name="web_fetch",
        description=(
            "Fetch a public http(s) URL and return its readable text (HTML is converted to "
            "plain text). For reading a web page or doc found via web_search. Output is "
            "size-capped; localhost/private/internal addresses are refused."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The http(s) URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Cap on characters of text returned (default 50000).",
                    "minimum": 1,
                },
            },
            "required": ["url"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return ToolResult.error("'url' is required and must be a non-empty string.")
        url = url.strip()

        max_chars = args.get("max_chars")
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
            max_chars = _MAX_CHARS
        max_chars = min(max_chars, _MAX_CHARS)

        try:
            httpx = load_httpx()
        except ImportError as exc:
            return ToolResult.error(str(exc), fix=_INSTALL_FIX)

        try:
            body, final_url, content_type = await self._fetch(httpx, url)
        except BlockedUrlError as exc:
            return ToolResult.error(
                f"refusing to fetch {defang_untrusted(url)}: {defang_untrusted(str(exc))}",
                fix="fetch a public http(s) URL; loopback/private/internal hosts are blocked",
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(
                f"failed to fetch {defang_untrusted(url)}: {defang_untrusted(str(exc))}"
            )

        if not _looks_textual(content_type):
            return ToolResult.error(
                f"{defang_untrusted(final_url)} returned non-text content (content-type: "
                f"{defang_untrusted(content_type) or 'unknown'}); web_fetch only reads text/HTML.",
                data={"final_url": final_url, "content_type": content_type},
            )

        raw = body.decode(_charset(content_type), errors="replace")
        ct = content_type.split(";", 1)[0].strip().lower()
        text = html_to_text(raw) if ct in ("text/html", "application/xhtml+xml") else raw
        text = defang_untrusted(text)

        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + (
                f"\n\n[... content truncated at {max_chars} chars; "
                "the page is longer than was returned ...]"
            )

        return ToolResult.ok(
            text or "(the page had no readable text)",
            data={
                "final_url": final_url,
                "content_type": content_type,
                "truncated": truncated,
            },
        )

    async def _fetch(self, httpx, url: str, *, timeout: int = _DEFAULT_TIMEOUT):
        """GET ``url``, following redirects MANUALLY so every hop is SSRF-revalidated + IP-pinned.

        Returns ``(body_bytes, final_url, content_type)`` where ``final_url`` is the validated
        hostname URL of the last hop. Raises :class:`BlockedUrlError` if any hop targets a
        disallowed host, or a transport error otherwise.
        """
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,text/plain,*/*",
            "Accept-Encoding": "identity",  # no transparent compression -> no decompression bomb
        }
        current = url
        async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
            for _hop in range(_MAX_REDIRECTS + 1):
                # Validate + pin to the checked IP off the event loop (the guard does DNS).
                connect_url, pin_headers, ext = await asyncio.to_thread(resolve_pinned_url, current)
                req_headers = {**headers, **pin_headers}
                async with client.stream(
                    "GET", connect_url, headers=req_headers, extensions=ext
                ) as resp:
                    if resp.status_code in _REDIRECT_CODES:
                        location = resp.headers.get("location")
                        if not location:
                            raise BlockedUrlError("redirect response without a Location header")
                        # Resolve the next hop against the ORIGINAL hostname URL (not the pinned
                        # IP URL), so relative/protocol-relative redirects resolve correctly and
                        # the next iteration re-validates + re-pins the new host.
                        current = str(httpx.URL(current).join(location))
                        continue
                    # A declared length over the cap: refuse before reading the body.
                    declared = resp.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > _MAX_BYTES:
                        raise BlockedUrlError(
                            f"response too large (Content-Length {declared} > {_MAX_BYTES})"
                        )
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "")
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in resp.aiter_bytes(chunk_size=_CHUNK_SIZE):
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= _MAX_BYTES:
                            break
                    return b"".join(chunks)[:_MAX_BYTES], current, content_type
        raise BlockedUrlError("too many redirects (possible redirect loop)")

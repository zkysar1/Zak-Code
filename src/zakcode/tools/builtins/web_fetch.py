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
* **Named secrets** — ``{{secret:NAME}}`` placeholders in the URL and in ``headers`` values are
  resolved by :class:`zakcode.tools.builtins._secrets.SecretsProvider` at request-build time,
  OUTSIDE the model: the resolved form exists only in the outbound request, every model-facing
  string keeps the placeholder form, and everything returned (body, errors, final_url) is
  scrubbed so an echoing API cannot carry a value back into context. Scrub runs BEFORE
  ``defang_untrusted`` — defang rewrites characters, so the value must be folded back into its
  placeholder while the text still contains it verbatim.
"""

from __future__ import annotations

import asyncio
import codecs
import re
from urllib.parse import urlsplit

from zakcode._http import (
    BlockedUrlError,
    host_allowed,
    install_now_fix,
    load_httpx,
    resolve_pinned_url,
)
from zakcode.config import PermissionTier
from zakcode.providers.text_tools import defang_untrusted
from zakcode.secrets import redact_secrets
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins._html import html_to_text
from zakcode.tools.builtins._secrets import (
    SECRET_PLACEHOLDER_RE,
    SecretsProvider,
    UnknownSecretError,
)

_DEFAULT_TIMEOUT = 15
_MAX_BYTES = 2 * 1024 * 1024  # cap bytes downloaded off the wire
_CHUNK_SIZE = 64 * 1024  # bound per-read memory so a huge body can't be buffered whole
_MAX_CHARS = 50_000  # cap chars returned to the model
_MAX_REDIRECTS = 5
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_USER_AGENT = "zakcode-webfetch/0.1 (+https://github.com/zkysar1/Zak-Code)"
_INSTALL_FIX = install_now_fix("httpx")
_MAX_REQ_HEADERS = 16
_MAX_HEADER_VALUE_CHARS = 2048
_HEADER_NAME_RE_TEXT = r"^[A-Za-z0-9-]+$"
# Model-supplied headers may never override the transport-integrity set: Host carries the
# SSRF pin, identity encoding is the decompression-bomb defense, and the length/framing
# headers belong to the transport.
_FORBIDDEN_REQ_HEADERS = frozenset(
    {"host", "content-length", "transfer-encoding", "accept-encoding", "connection"}
)


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
            "size-capped; localhost/private/internal addresses are refused. To call an API "
            "with a saved secret, write {{secret:NAME}} in the url or a header value — the "
            "real value is substituted outside your context (see the secret_names tool)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The http(s) URL to fetch. May contain {{secret:NAME}}.",
                },
                "headers": {
                    "type": "object",
                    "description": (
                        "Optional request headers, e.g. "
                        '{"Authorization": "Bearer {{secret:MY_API_KEY}}"}. Values may '
                        "contain {{secret:NAME}} placeholders."
                    ),
                    "additionalProperties": {"type": "string"},
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

    def __init__(
        self,
        *,
        allowed_domains: list[str] | None = None,
        secrets: SecretsProvider | None = None,
    ) -> None:
        # Optional egress allowlist (ZAKCODE_WEB_ALLOWED_DOMAINS). Empty = any public host.
        self._allowed = [d for d in (allowed_domains or []) if d and d.strip()]
        # Always hold a provider (an empty one when unconfigured) so no call site branches
        # on "is the feature on" — an empty provider's scrub is the identity and any
        # placeholder resolves to a clean unknown-secret error.
        self._secrets = secrets or SecretsProvider(None)

    def _out(self, text: str) -> str:
        """Model-facing string hygiene, in the mandatory order: scrub THEN defang.

        Scrub folds any secret value back into its placeholder; defang then neutralizes
        protocol/template tokens. Reversed order would let defang rewrite characters
        inside a value so the scrub no longer matches it.
        """
        return defang_untrusted(self._secrets.scrub(text))

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return ToolResult.error("'url' is required and must be a non-empty string.")
        url = url.strip()

        max_chars = args.get("max_chars")
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
            max_chars = _MAX_CHARS
        max_chars = min(max_chars, _MAX_CHARS)

        raw_headers = args.get("headers")
        if raw_headers is None:
            raw_headers = {}
        if not isinstance(raw_headers, dict):
            return ToolResult.error("'headers' must be an object of string values.")
        if len(raw_headers) > _MAX_REQ_HEADERS:
            return ToolResult.error(f"too many headers (max {_MAX_REQ_HEADERS}).")
        for name, value in raw_headers.items():
            if not isinstance(name, str) or not re.fullmatch(_HEADER_NAME_RE_TEXT, name):
                return ToolResult.error(f"invalid header name: {defang_untrusted(str(name))!r}")
            if name.lower() in _FORBIDDEN_REQ_HEADERS:
                return ToolResult.error(
                    f"header {name!r} is transport-controlled and cannot be set here."
                )
            if not isinstance(value, str) or len(value) > _MAX_HEADER_VALUE_CHARS:
                return ToolResult.error(
                    f"header {name!r} value must be a string of at most "
                    f"{_MAX_HEADER_VALUE_CHARS} chars."
                )

        # RAW-secret screens (ADR-0019). A saved secret VALUE pasted directly into the url
        # or a header has already leaked into the transcript before any request is made —
        # the sanctioned form is the {{secret:NAME}} placeholder, resolved outside the
        # model below. Header values additionally get the credential-SHAPE screen (an
        # ``Authorization: Bearer <token>`` paste is the classic leak; the shape screen is
        # NOT applied to the url, whose query strings legitimately match ``token=…``-style
        # assignment shapes).
        if self._secrets.scrub(url) != url or any(
            self._secrets.scrub(v) != v for v in raw_headers.values()
        ):
            return ToolResult.error(
                "a raw saved-secret VALUE appears in the url or a header; never paste "
                "secret values into requests.",
                fix="write {{secret:NAME}} instead — the real value is substituted outside "
                "your context (see the secret_names tool)",
            )
        # Placeholders are stripped first: ``{{secret:NAME}}`` is the SAFE form, but its
        # ``secret:NAME`` spelling matches the redactor's key-value pattern.
        if any(
            redact_secrets(SECRET_PLACEHOLDER_RE.sub("", v))[1] > 0 for v in raw_headers.values()
        ):
            return ToolResult.error(
                "a header value contains credential-shaped text; never paste credentials "
                "into requests.",
                fix="write {{secret:NAME}} instead — the real value is substituted outside "
                "your context (see the secret_names tool)",
            )

        # Resolve {{secret:NAME}} OUTSIDE the model: the resolved forms exist only in the
        # outbound request; `url` (the placeholder form) stays the one used in every
        # model-facing message below. The SSRF guard runs on the RESOLVED url — the form
        # that will actually be fetched is the form that gets validated.
        try:
            resolved_url, used = self._secrets.resolve(url)
            resolved_headers: dict[str, str] = {}
            for name, value in raw_headers.items():
                resolved_value, used_in_header = self._secrets.resolve(value)
                resolved_headers[name] = resolved_value
                used |= used_in_header
        except UnknownSecretError as exc:
            return ToolResult.error(
                str(exc), fix="call secret_names to see which secret names are available"
            )

        try:
            httpx = load_httpx()
        except ImportError as exc:
            return ToolResult.error(str(exc), fix=_INSTALL_FIX)

        # Names-only usage record, written once the values are released into a request
        # attempt (whatever the remote end then answers).
        self._secrets.record_use(used)

        try:
            body, final_url, content_type = await self._fetch(
                httpx, resolved_url, extra_headers=resolved_headers
            )
        except BlockedUrlError as exc:
            # str(exc) can embed the RESOLVED url/host of a later redirect hop — _out folds
            # any value back into its placeholder before the model sees it.
            return ToolResult.error(
                f"refusing to fetch {defang_untrusted(url)}: {self._out(str(exc))}",
                fix="fetch a public http(s) URL; loopback/private/internal hosts are blocked",
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(
                f"failed to fetch {defang_untrusted(url)}: {self._out(str(exc))}"
            )

        if not _looks_textual(content_type):
            return ToolResult.error(
                f"{self._out(final_url)} returned non-text content (content-type: "
                f"{defang_untrusted(content_type) or 'unknown'}); web_fetch only reads text/HTML.",
                data={"final_url": self._secrets.scrub(final_url), "content_type": content_type},
            )

        raw = body.decode(_charset(content_type), errors="replace")
        ct = content_type.split(";", 1)[0].strip().lower()
        text = html_to_text(raw) if ct in ("text/html", "application/xhtml+xml") else raw
        # _out = scrub-then-defang: an API that echoes the request cannot hand the model a
        # secret value back through the page body.
        text = self._out(text)

        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + (
                f"\n\n[... content truncated at {max_chars} chars; "
                "the page is longer than was returned ...]"
            )

        return ToolResult.ok(
            text or "(the page had no readable text)",
            data={
                # Scrubbed: a redirect can land on a URL that carries a substituted
                # query value, and data fields reach the model like output does.
                "final_url": self._secrets.scrub(final_url),
                "content_type": content_type,
                "truncated": truncated,
            },
        )

    async def _fetch(
        self,
        httpx,
        url: str,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
        extra_headers: dict[str, str] | None = None,
    ):
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
        # Caller headers may override the defaults above, but never the pin headers merged
        # last per hop (Host carries the SSRF pin) — and execute() has already refused the
        # transport-controlled names outright.
        headers = {**headers, **(extra_headers or {})}
        current = url
        async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
            for _hop in range(_MAX_REDIRECTS + 1):
                # Egress allowlist (policy): when configured, refuse a host not on the list —
                # checked per hop so a redirect cannot walk off it. (Empty list = allow any
                # public host — that is web_fetch's default; the SSRF guard always runs next.)
                host = urlsplit(current).hostname or ""
                if self._allowed and not host_allowed(host, self._allowed):
                    raise BlockedUrlError(
                        f"host {host!r} is not in the configured web_fetch allowlist "
                        f"(ZAKCODE_WEB_ALLOWED_DOMAINS)"
                    )
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

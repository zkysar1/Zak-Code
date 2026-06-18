"""Small image-format helpers for artifact-producing tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageFormat:
    """A supported raster image format identified by bytes, MIME type, and suffix."""

    suffix: str
    mime_type: str


_PNG = ImageFormat(".png", "image/png")
_JPEG = ImageFormat(".jpg", "image/jpeg")
_GIF = ImageFormat(".gif", "image/gif")
_WEBP = ImageFormat(".webp", "image/webp")
_BMP = ImageFormat(".bmp", "image/bmp")

_FORMAT_BY_SUFFIX = {
    ".png": _PNG,
    ".jpg": _JPEG,
    ".jpeg": _JPEG,
    ".gif": _GIF,
    ".webp": _WEBP,
    ".bmp": _BMP,
}

_FORMAT_BY_MIME = {
    "image/png": _PNG,
    "image/jpeg": _JPEG,
    "image/jpg": _JPEG,
    "image/gif": _GIF,
    "image/webp": _WEBP,
    "image/bmp": _BMP,
    "image/x-ms-bmp": _BMP,
}


def format_from_suffix(path: str | Path) -> ImageFormat | None:
    """Return the supported image format implied by ``path``'s suffix."""

    return _FORMAT_BY_SUFFIX.get(Path(path).suffix.lower())


def format_from_mime(mime_type: str | None) -> ImageFormat | None:
    """Return the supported image format implied by a MIME type."""

    if not mime_type:
        return None
    return _FORMAT_BY_MIME.get(mime_type.split(";", 1)[0].strip().lower())


def sniff_image_format(data: bytes) -> ImageFormat | None:
    """Identify a supported raster image from magic bytes."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _PNG
    if data.startswith(b"\xff\xd8\xff"):
        return _JPEG
    if data.startswith((b"GIF87a", b"GIF89a")):
        return _GIF
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return _WEBP
    if data.startswith(b"BM"):
        return _BMP
    return None


def supported_image_suffixes() -> tuple[str, ...]:
    """Suffixes accepted by image-producing and document-embedding tools."""

    return tuple(sorted(_FORMAT_BY_SUFFIX))


__all__ = [
    "ImageFormat",
    "format_from_mime",
    "format_from_suffix",
    "sniff_image_format",
    "supported_image_suffixes",
]

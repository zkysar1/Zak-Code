"""Save raster image bytes as downloadable workspace artifacts."""

from __future__ import annotations

import base64
import contextlib
import io
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zakcode.artifacts import ArtifactError, artifact_from_path
from zakcode.config import PermissionTier
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec
from zakcode.tools.builtins._image_data import (
    ImageFormat,
    format_from_mime,
    format_from_suffix,
    sniff_image_format,
    supported_image_suffixes,
)
from zakcode.tools.builtins._safety import PathEscapeError, resolve_path

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[-\w.+/]+(?:;[-\w=.+]+)*);base64,(?P<data>.*)$", re.S)
_MAX_IMAGE_BYTES = 25 * 1024 * 1024
_MAX_CHART_POINTS = 100
_CHART_COLORS = [
    "#2563eb",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#65a30d",
]


def _decode_image_data(data: str, mime_type: str | None) -> tuple[bytes, ImageFormat]:
    payload = data.strip()
    mime_hint = format_from_mime(mime_type)
    match = _DATA_URL_RE.match(payload)
    if match:
        payload = match.group("data")
        url_format = format_from_mime(match.group("mime"))
        if url_format is None:
            raise ValueError("data URL MIME type must be a supported raster image type")
        if mime_hint is not None and mime_hint.mime_type != url_format.mime_type:
            raise ValueError("mime_type does not match the data URL MIME type")
        mime_hint = url_format

    compact = "".join(payload.split())
    try:
        image_bytes = base64.b64decode(compact, validate=True)
    except Exception as exc:  # noqa: BLE001 - normalize binascii/ValueError variants
        raise ValueError("'data' must be valid base64 image bytes or a base64 data URL") from exc
    if not image_bytes:
        raise ValueError("'data' decoded to an empty image")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image is too large ({len(image_bytes)} bytes; max {_MAX_IMAGE_BYTES})")

    detected = sniff_image_format(image_bytes)
    if detected is None:
        raise ValueError(
            "image bytes are not a supported raster image "
            f"({', '.join(supported_image_suffixes())})"
        )
    if mime_hint is not None and mime_hint.mime_type != detected.mime_type:
        raise ValueError("declared image MIME type does not match the image bytes")
    return image_bytes, detected


def _target_path(path: str, image_format: ImageFormat) -> str:
    suffix_format = format_from_suffix(path)
    suffix = Path(path).suffix
    if not suffix:
        return f"{path}{image_format.suffix}"
    if suffix_format is None:
        raise ValueError(f"path suffix must be one of {', '.join(supported_image_suffixes())}")
    if suffix_format.mime_type != image_format.mime_type:
        raise ValueError(
            f"path suffix {suffix!r} does not match {image_format.mime_type} image bytes"
        )
    return path


def _ensure_image_parent(resolved: Path, final_path: str) -> str | None:
    if resolved.exists() and resolved.is_dir():
        return f"Path is a directory, not a file: {final_path}"
    parent = resolved.parent
    if parent.exists() and not parent.is_dir():
        return f"Parent path is not a directory: {parent}"
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"Could not create parent directory for {final_path}: {exc}"
    return None


def _write_image_artifact(
    *,
    image_bytes: bytes,
    image_format: ImageFormat,
    final_path: str,
    resolved: Path,
    ctx: ToolContext,
    tool_name: str,
    output: str,
    data: dict[str, Any] | None = None,
) -> ToolResult:
    if (parent_error := _ensure_image_parent(resolved, final_path)) is not None:
        return ToolResult.error(parent_error)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=str(resolved.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(image_bytes)
        os.replace(tmp_name, resolved)
    except Exception:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise

    artifacts = []
    with contextlib.suppress(ArtifactError, OSError):
        artifacts.append(
            artifact_from_path(
                resolved,
                workspace_root=ctx.workspace_root,
                created_by_tool=tool_name,
            )
        )
    result_data: dict[str, Any] = {
        "path": str(resolved),
        "bytes": len(image_bytes),
        "mime_type": image_format.mime_type,
        **(data or {}),
    }
    if artifacts:
        result_data["artifact_id"] = artifacts[0].id
    return ToolResult.ok(output, data=result_data, artifacts=artifacts)


def _chart_dimensions(raw_width: Any, raw_height: Any) -> tuple[int, int]:
    width = 960 if raw_width is None else raw_width
    height = 540 if raw_height is None else raw_height
    if isinstance(width, bool) or not isinstance(width, int) or not 320 <= width <= 2400:
        raise ValueError("'width' must be an integer from 320 to 2400 pixels")
    if isinstance(height, bool) or not isinstance(height, int) or not 240 <= height <= 1600:
        raise ValueError("'height' must be an integer from 240 to 1600 pixels")
    return width, height


def _chart_data(raw: Any) -> list[tuple[str, float]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("'data' must be a non-empty list")
    if len(raw) > _MAX_CHART_POINTS:
        raise ValueError(f"'data' must contain at most {_MAX_CHART_POINTS} points")
    points: list[tuple[str, float]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"data[{index}] must be an object")
        label = item.get("label")
        value = item.get("value")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"data[{index}].label must be a non-empty string")
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"data[{index}].value must be a finite number")
        if value < 0:
            raise ValueError(f"data[{index}].value must be non-negative")
        points.append((label.strip(), float(value)))
    if not any(value > 0 for _label, value in points):
        raise ValueError("at least one chart value must be greater than zero")
    return points


def _text(
    draw: Any, xy: tuple[int, int], text: str, *, fill: str = "#111827", anchor: str | None = None
) -> None:
    kwargs = {"fill": fill}
    if anchor is not None:
        kwargs["anchor"] = anchor
    draw.text(xy, text, **kwargs)


def _short_label(label: str, limit: int = 16) -> str:
    if len(label) <= limit:
        return label
    return label[: limit - 1] + "..."


def _chart_type(args: Mapping[str, Any]) -> str:
    raw_chart_type = args.get("chart_type")
    if raw_chart_type is None:
        raw_type = args.get("type", "bar")
        raw_chart_type = "bar" if raw_type == "chart" else raw_type
    chart_type = str(raw_chart_type).strip().lower()
    if chart_type not in {"bar", "line", "pie"}:
        raise ValueError("'chart_type' must be bar, line, or pie")
    return chart_type


def render_chart_png(args: Mapping[str, Any]) -> bytes:
    """Render a simple bar/line/pie chart to PNG bytes."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ValueError("Pillow is required for chart image rendering; run `uv sync`.") from exc

    chart_type = _chart_type(args)
    width, height = _chart_dimensions(args.get("width"), args.get("height"))
    points = _chart_data(args.get("data"))
    title = args.get("title", "")
    if title is not None and not isinstance(title, str):
        raise ValueError("'title' must be a string when provided")

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    padding = max(28, width // 28)
    top = padding + (28 if title else 0)
    left = padding + 38
    right = width - padding
    bottom = height - padding - 34

    if title:
        _text(draw, (width // 2, padding // 2), title, anchor="ma")

    if chart_type == "pie":
        _draw_pie_chart(draw, points, left, top, right, bottom)
    elif chart_type == "line":
        _draw_xy_chart(draw, points, left, top, right, bottom, line=True)
    else:
        _draw_xy_chart(draw, points, left, top, right, bottom, line=False)

    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _draw_axes(draw: Any, left: int, top: int, right: int, bottom: int, max_value: float) -> None:
    draw.line((left, top, left, bottom, right, bottom), fill="#9ca3af", width=2)
    for i in range(1, 5):
        y = bottom - int((bottom - top) * i / 4)
        draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        _text(draw, (left - 8, y), f"{max_value * i / 4:g}", fill="#6b7280", anchor="rm")


def _draw_xy_chart(
    draw: Any,
    points: list[tuple[str, float]],
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    line: bool,
) -> None:
    max_value = max(value for _label, value in points)
    max_value = max(max_value, 1)
    _draw_axes(draw, left, top, right, bottom, max_value)
    plot_width = max(1, right - left)
    plot_height = max(1, bottom - top)

    if line:
        step = plot_width / max(1, len(points) - 1)
        coords: list[tuple[int, int]] = []
        for idx, (label, value) in enumerate(points):
            x = int(left + idx * step)
            y = int(bottom - (value / max_value) * plot_height)
            coords.append((x, y))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=_CHART_COLORS[0])
            _text(draw, (x, bottom + 8), _short_label(label, 10), fill="#374151", anchor="ma")
        if len(coords) > 1:
            draw.line(coords, fill=_CHART_COLORS[0], width=3)
        return

    slot = plot_width / max(1, len(points))
    bar_width = max(8, int(slot * 0.62))
    for idx, (label, value) in enumerate(points):
        center = left + slot * (idx + 0.5)
        x0 = int(center - bar_width / 2)
        x1 = int(center + bar_width / 2)
        y0 = int(bottom - (value / max_value) * plot_height)
        color = _CHART_COLORS[idx % len(_CHART_COLORS)]
        draw.rounded_rectangle((x0, y0, x1, bottom), radius=3, fill=color)
        _text(draw, (int(center), bottom + 8), _short_label(label, 10), fill="#374151", anchor="ma")


def _draw_pie_chart(
    draw: Any,
    points: list[tuple[str, float]],
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> None:
    legend_width = min(190, max(120, (right - left) // 3))
    pie_right = max(left + 80, right - legend_width)
    size = min(pie_right - left, bottom - top)
    box_left = left
    box_top = top + max(0, (bottom - top - size) // 2)
    box = (box_left, box_top, box_left + size, box_top + size)
    total = sum(value for _label, value in points)
    start = -90.0
    legend_x = box_left + size + 18
    legend_y = box_top
    for idx, (label, value) in enumerate(points):
        end = start + 360.0 * (value / total)
        color = _CHART_COLORS[idx % len(_CHART_COLORS)]
        draw.pieslice(box, start=start, end=end, fill=color)
        draw.rectangle(
            (legend_x, legend_y + idx * 22, legend_x + 12, legend_y + 12 + idx * 22), fill=color
        )
        _text(
            draw,
            (legend_x + 18, legend_y - 1 + idx * 22),
            f"{_short_label(label, 14)} ({value:g})",
            fill="#374151",
        )
        start = end


class SaveImageTool(Tool):
    """Save a base64/data-URL raster image under the workspace."""

    spec = ToolSpec(
        name="save_image",
        description=(
            "Save a PNG/JPEG/GIF/WebP/BMP image within the workspace from base64 bytes or a "
            "base64 data URL. The image signature is validated and the result is returned as "
            "a downloadable artifact."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Destination path. If no suffix is provided, the detected image suffix "
                        "is appended."
                    ),
                },
                "data": {
                    "type": "string",
                    "description": "Base64 image bytes, or a data:image/...;base64,... URL.",
                },
                "mime_type": {
                    "type": "string",
                    "description": "Optional MIME type for raw base64 input.",
                },
            },
            "required": ["path", "data"],
        },
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.PATH_SCOPED,
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = args.get("path")
        data = args.get("data")
        mime_type = args.get("mime_type")
        if not isinstance(path, str) or not path:
            return ToolResult.error("'path' is required and must be a string.")
        if not isinstance(data, str) or not data:
            return ToolResult.error("'data' is required and must be a base64 string.")
        if mime_type is not None and not isinstance(mime_type, str):
            return ToolResult.error("'mime_type' must be a string when provided.")

        try:
            image_bytes, image_format = _decode_image_data(data, mime_type)
            final_path = _target_path(path, image_format)
            resolved = resolve_path(final_path, ctx.workspace_root, ctx.extra_workspace_roots)
        except (PathEscapeError, ValueError) as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to resolve image path {path!r}: {exc}")

        try:
            return _write_image_artifact(
                image_bytes=image_bytes,
                image_format=image_format,
                final_path=final_path,
                resolved=resolved,
                ctx=ctx,
                tool_name=self.spec.name,
                output=f"Saved image {final_path}",
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to save image {path!r}: {exc}")


class CreateChartImageTool(Tool):
    """Create a simple PNG chart image as a downloadable artifact."""

    spec = ToolSpec(
        name="create_chart_image",
        description=(
            "Create a PNG bar, line, or pie chart image within the workspace. Use this when a "
            "Word report needs a graph image or the user wants a standalone chart download."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Destination path. .png is appended when no suffix is given.",
                },
                "chart_type": {"type": "string", "enum": ["bar", "line", "pie"]},
                "title": {"type": "string"},
                "data": {
                    "type": "array",
                    "maxItems": _MAX_CHART_POINTS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "number", "minimum": 0},
                        },
                        "required": ["label", "value"],
                    },
                },
                "width": {"type": "integer", "minimum": 320, "maximum": 2400},
                "height": {"type": "integer", "minimum": 240, "maximum": 1600},
            },
            "required": ["path", "data"],
        },
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.PATH_SCOPED,
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        path = args.get("path")
        if not isinstance(path, str) or not path:
            return ToolResult.error("'path' is required and must be a string.")
        try:
            final_path = path if Path(path).suffix else f"{path}.png"
            if Path(final_path).suffix.lower() != ".png":
                return ToolResult.error("'path' must end with .png for chart images.")
            image_bytes = render_chart_png(args)
            resolved = resolve_path(final_path, ctx.workspace_root, ctx.extra_workspace_roots)
            return _write_image_artifact(
                image_bytes=image_bytes,
                image_format=ImageFormat(".png", "image/png"),
                final_path=final_path,
                resolved=resolved,
                ctx=ctx,
                tool_name=self.spec.name,
                output=f"Created chart image {final_path}",
                data={"chart_type": _chart_type(args)},
            )
        except (PathEscapeError, ValueError) as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to create chart image {path!r}: {exc}")


class InspectImageTool(Tool):
    """Inspect a workspace raster image without loading its bytes into the prompt."""

    spec = ToolSpec(
        name="inspect_image",
        description=(
            "Inspect a PNG/JPEG/GIF/WebP/BMP image in the workspace. Returns format, MIME "
            "type, dimensions, color mode, frame count, and byte size."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace image path to inspect."},
            },
            "required": ["path"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = args.get("path")
        if not isinstance(path, str) or not path:
            return ToolResult.error("'path' is required and must be a string.")
        try:
            resolved = resolve_path(path, ctx.workspace_root, ctx.extra_workspace_roots)
            if not resolved.is_file():
                return ToolResult.error(f"Image file not found: {path}")
            suffix_format = format_from_suffix(resolved)
            if suffix_format is None:
                return ToolResult.error(
                    f"path must end with one of {', '.join(supported_image_suffixes())}"
                )
            with resolved.open("rb") as handle:
                detected = sniff_image_format(handle.read(16))
            if detected is None or detected.mime_type != suffix_format.mime_type:
                return ToolResult.error(f"{path!r} is not a valid {suffix_format.mime_type} image")
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to resolve image path {path!r}: {exc}")

        try:
            from PIL import Image
        except ImportError:
            return ToolResult.error("Pillow is required for inspect_image; run `uv sync`.")

        try:
            with Image.open(resolved) as image:
                data = {
                    "path": str(resolved),
                    "bytes": resolved.stat().st_size,
                    "format": image.format,
                    "mime_type": detected.mime_type,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "frames": getattr(image, "n_frames", 1),
                }
            output = "\n".join(
                [
                    f"Image: {path}",
                    f"Format: {data['format']} ({data['mime_type']})",
                    f"Dimensions: {data['width']} x {data['height']}",
                    f"Mode: {data['mode']}",
                    f"Frames: {data['frames']}",
                    f"Bytes: {data['bytes']}",
                ]
            )
            return ToolResult.ok(output, data=data)
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to inspect image {path!r}: {exc}")


__all__ = ["CreateChartImageTool", "InspectImageTool", "SaveImageTool", "render_chart_png"]

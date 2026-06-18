"""Tests for image artifact production and Word embedding."""

from __future__ import annotations

import io
from pathlib import Path

from docx import Document
from PIL import Image

from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.images import (
    CreateChartImageTool,
    InspectImageTool,
    SaveImageTool,
    render_chart_png,
)
from zakcode.tools.builtins.office import CreateDocxTool

_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


async def test_save_image_data_url_returns_image_artifact(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    result = await SaveImageTool().execute(
        {"path": "images/pixel", "data": f"data:image/png;base64,{_PNG_BASE64}"},
        ctx,
    )

    assert not result.is_error, result.output
    path = tmp_path / "images" / "pixel.png"
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(result.artifacts) == 1
    assert result.artifacts[0].path == "images/pixel.png"
    assert result.artifacts[0].mime_type == "image/png"
    assert result.artifacts[0].kind == "image"
    assert result.artifacts[0].created_by_tool == "save_image"
    assert result.data is not None
    assert result.data["artifact_id"] == result.artifacts[0].id


async def test_save_image_rejects_suffix_that_disagrees_with_bytes(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    result = await SaveImageTool().execute(
        {"path": "images/not-a-jpeg.jpg", "data": f"data:image/png;base64,{_PNG_BASE64}"},
        ctx,
    )

    assert result.is_error
    assert "does not match" in result.output
    assert not (tmp_path / "images" / "not-a-jpeg.jpg").exists()


async def test_inspect_image_reports_dimensions_and_format(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    saved = await SaveImageTool().execute(
        {"path": "images/pixel.png", "data": f"data:image/png;base64,{_PNG_BASE64}"},
        ctx,
    )
    assert not saved.is_error, saved.output

    result = await InspectImageTool().execute({"path": "images/pixel.png"}, ctx)

    assert not result.is_error, result.output
    assert "Dimensions: 1 x 1" in result.output
    assert result.data is not None
    assert result.data["mime_type"] == "image/png"
    assert result.data["width"] == 1
    assert result.data["height"] == 1
    assert result.data["frames"] == 1


async def test_create_chart_image_returns_downloadable_png_artifact(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    result = await CreateChartImageTool().execute(
        {
            "path": "charts/sales",
            "chart_type": "bar",
            "title": "Sales",
            "width": 640,
            "height": 360,
            "data": [
                {"label": "Jan", "value": 120},
                {"label": "Feb", "value": 160},
                {"label": "Mar", "value": 140},
            ],
        },
        ctx,
    )

    assert not result.is_error, result.output
    path = tmp_path / "charts" / "sales.png"
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(path.read_bytes())) as image:
        assert image.format == "PNG"
        assert image.size == (640, 360)
    assert len(result.artifacts) == 1
    assert result.artifacts[0].path == "charts/sales.png"
    assert result.artifacts[0].kind == "image"
    assert result.artifacts[0].mime_type == "image/png"
    assert result.artifacts[0].created_by_tool == "create_chart_image"
    assert result.data is not None
    assert result.data["artifact_id"] == result.artifacts[0].id
    assert result.data["chart_type"] == "bar"


def test_render_chart_png_supports_pie_and_rejects_too_many_points() -> None:
    png = render_chart_png(
        {
            "chart_type": "pie",
            "width": 480,
            "height": 320,
            "data": [{"label": "One", "value": 1}, {"label": "Two", "value": 2}],
        }
    )

    with Image.open(io.BytesIO(png)) as image:
        assert image.format == "PNG"
        assert image.size == (480, 320)

    too_many = [{"label": str(i), "value": 1} for i in range(101)]
    try:
        render_chart_png({"data": too_many})
    except ValueError as exc:
        assert "at most 100" in str(exc)
    else:  # pragma: no cover - makes the assertion failure clearer
        raise AssertionError("expected too many chart points to be rejected")


async def test_create_docx_embeds_workspace_image(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    saved = await SaveImageTool().execute(
        {"path": "images/pixel.png", "data": f"data:image/png;base64,{_PNG_BASE64}"},
        ctx,
    )
    assert not saved.is_error, saved.output

    result = await CreateDocxTool().execute(
        {
            "path": "reports/with-image.docx",
            "title": "Image Report",
            "blocks": [
                {
                    "type": "image",
                    "path": "images/pixel.png",
                    "width_inches": 1,
                    "caption": "A tiny generated image.",
                }
            ],
        },
        ctx,
    )

    assert not result.is_error, result.output
    doc = Document(tmp_path / "reports" / "with-image.docx")
    assert len(doc.inline_shapes) == 1
    assert "A tiny generated image." in [paragraph.text for paragraph in doc.paragraphs]

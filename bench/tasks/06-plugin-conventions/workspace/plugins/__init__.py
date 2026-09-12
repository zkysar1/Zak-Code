"""Tiny renderer plugin registry. Standard library only -- see CONTRIBUTING.md."""

__all__ = ["PluginError", "Renderer", "register", "get", "CsvRenderer", "JsonRenderer"]

_REGISTRY: dict[str, type] = {}


class PluginError(Exception):
    """Raised by a renderer for input it cannot render."""


class Renderer:
    """Base renderer. Subclasses implement render(rows) -> str."""

    def render(self, rows: list[dict]) -> str:
        raise NotImplementedError


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get(name: str) -> type:
    if name not in _REGISTRY:
        raise PluginError(f"no renderer named {name!r}")
    return _REGISTRY[name]


from plugins.csv_out import CsvRenderer  # noqa: E402,F401
from plugins.json_out import JsonRenderer  # noqa: E402,F401

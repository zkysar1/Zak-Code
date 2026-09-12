import json

from plugins import PluginError, Renderer, register


@register("json")
class JsonRenderer(Renderer):
    def render(self, rows: list[dict]) -> str:
        if not isinstance(rows, list) or not rows:
            raise PluginError("json needs a non-empty list of dicts")
        if not all(isinstance(r, dict) for r in rows):
            raise PluginError("json needs a non-empty list of dicts")
        return json.dumps(rows, indent=2, sort_keys=True)

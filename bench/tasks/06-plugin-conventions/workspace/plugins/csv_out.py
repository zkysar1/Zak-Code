import csv
import io

from plugins import PluginError, Renderer, register


@register("csv")
class CsvRenderer(Renderer):
    def render(self, rows: list[dict]) -> str:
        if not isinstance(rows, list) or not rows:
            raise PluginError("csv needs a non-empty list of dicts")
        if not all(isinstance(r, dict) for r in rows):
            raise PluginError("csv needs a non-empty list of dicts")
        keys = list(rows[0])
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=keys)
        w.writeheader()
        for r in rows:
            if list(r) != keys:
                raise PluginError("all rows must share the same keys")
            w.writerow(r)
        return buf.getvalue()

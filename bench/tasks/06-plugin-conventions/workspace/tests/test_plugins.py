import pytest

from plugins import PluginError, get

ROWS = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


@pytest.mark.parametrize("name", ["csv", "json"])
def test_renders_non_empty(name):
    out = get(name)().render(ROWS)
    assert isinstance(out, str) and out.strip()


@pytest.mark.parametrize("name", ["csv", "json"])
def test_rejects_empty(name):
    with pytest.raises(PluginError):
        get(name)().render([])

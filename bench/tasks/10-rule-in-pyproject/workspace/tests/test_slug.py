import pytest

from utils import slugify


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Hello, World!", "hello-world"),
        ("  Crème Brûlée  ", "creme-brulee"),
        ("!!!", ""),
    ],
)
def test_slugify(text, want):
    assert slugify(text) == want


def test_slugify_cuts_without_trailing_hyphen():
    assert slugify("aaaa bbbb", max_length=5) == "aaaa"

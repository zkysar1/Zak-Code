from stats import median


def test_odd():
    assert median([3, 1, 2]) == 2


def test_even():
    assert median([1, 2, 3, 4]) == 2.5


def test_even_two():
    assert median([10, 20]) == 15.0


def test_single():
    assert median([42]) == 42

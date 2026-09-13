from decimal import Decimal

import pytest

from pricing.discounts import UnknownCode, apply_discount, list_codes


def test_no_code_is_the_plain_total():
    assert apply_discount("2.50", 4, None) == Decimal("10.00")


def test_welcome5_takes_five_percent_case_insensitively():
    assert apply_discount("10.00", 1, "welcome5") == Decimal("9.50")


def test_unknown_code_raises():
    with pytest.raises(UnknownCode):
        apply_discount("1.00", 1, "NOPE")


def test_cents_round_half_up():
    # 3 x 0.35 = 1.05; 5% off = 0.9975 -> 1.00 (half-up), not 0.99
    assert apply_discount("0.35", 3, "WELCOME5") == Decimal("1.00")


# BULK10: 10% off, but only for a dozen or more units. Below twelve the code is
# accepted and simply does not apply (the cart shows "add N more to save"), so it
# must NOT raise and must NOT discount.
def test_bulk10_is_listed():
    assert "BULK10" in list_codes()


def test_bulk10_applies_from_a_dozen():
    assert apply_discount("4.00", 12, "BULK10") == Decimal("43.20")


def test_bulk10_below_a_dozen_is_full_price_not_an_error():
    assert apply_discount("4.00", 11, "BULK10") == Decimal("44.00")


def test_bulk10_rounds_half_up():
    # 13 x 0.35 = 4.55; 10% off = 4.095 -> 4.10
    assert apply_discount("0.35", 13, "BULK10") == Decimal("4.10")

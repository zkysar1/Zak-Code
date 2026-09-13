"""Order-total discounts. Every public function returns a Decimal quantized to cents."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

__all__ = ["UnknownCode", "apply_discount", "list_codes"]


class UnknownCode(LookupError):
    """Raised when a discount code is not on the list."""


_CENT = Decimal("0.01")

# code -> percentage off. Codes are matched case-insensitively.
CODES: dict[str, Decimal] = {
    "WELCOME5": Decimal("5"),
}


def _cents(value: Decimal) -> Decimal:
    """Quantize to cents, rounding half up (the accounting rule, not Python's default)."""
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


def list_codes() -> list[str]:
    """Every code the cart accepts, sorted."""
    return sorted(CODES)


def apply_discount(unit_price: str | Decimal, quantity: int, code: str | None = None) -> Decimal:
    """Total for `quantity` units at `unit_price`, after the discount `code` (if any)."""
    total = Decimal(str(unit_price)) * quantity
    if code is None:
        return _cents(total)
    key = code.strip().upper()
    if key not in CODES:
        raise UnknownCode(code)
    pct = CODES[key]
    return _cents(total * (Decimal(100) - pct) / Decimal(100))

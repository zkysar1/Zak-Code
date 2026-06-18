"""Small stats helpers."""


def median(nums):
    """Return the median of a list of numbers.

    For an odd-length list this is the middle element; for an even-length list
    it is the average of the two middle elements.
    """
    s = sorted(nums)
    n = len(s)
    # BUG: this returns the upper-middle element for even-length lists instead of
    # averaging the two middle elements. Correct it.
    return s[n // 2]

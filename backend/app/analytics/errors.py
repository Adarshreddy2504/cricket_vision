"""
errors.py — The analytics failure vocabulary.

Everything exported by `app.analytics` is a pure function over data that was
already produced. There is no model to be unavailable and no video to reopen,
so the only genuine failure is an input that cannot support the calculation:
frames out of order, a missing field, a non-finite coordinate, or pitch
measurements that are not in metres.

`AnalyticsInputError` is also a `ValueError`, so ordinary argument validation
and the domain-specific error can be caught together.
"""

from __future__ import annotations


class AnalyticsError(Exception):
    """Base class for every failure raised by the analytics modules."""


class AnalyticsInputError(AnalyticsError, ValueError):
    """The supplied records are missing, inconsistent, or in the wrong units."""


__all__ = ["AnalyticsError", "AnalyticsInputError"]

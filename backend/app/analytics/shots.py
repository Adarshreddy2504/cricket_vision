"""
shots.py — Aggregate shot labels into a distribution.

WHAT THIS MODULE WILL NOT DO
----------------------------
It will not turn a failed classification into a class. A record whose status is
`model_unavailable`, `invalid_input` or `inference_failed` is counted as
unclassified under that status, and `shares` is computed over classified
records only. If nothing was classified, `shares` is empty rather than ten zeros:
a proportion with a zero denominator is undefined, and filling it with zeros
would read as "we looked and found none of each".

It will not accept an unknown label. The vocabulary is fixed by the checkpoint,
so a label outside it is a data error, not a new class to add to the chart.

INPUT SHAPE
-----------
Accept `ShotClassification` records, plain label strings, or mappings with the
same fields. That keeps the aggregation usable on a stored JSON document without
constructing schemas first.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable

from ..schemas.analytics import ShotDistribution
from ..shots.constants import SHOT_CLASSES
from ._common import read_field
from .errors import AnalyticsInputError


def _record_parts(record: object, index: int) -> tuple[bool, str | None, float | None, str | None]:
    """Classify one input as (classified, label, score, unclassified_reason)."""
    if isinstance(record, str):
        return True, record, None, None
    if record is None:
        return False, None, None, "no record"

    status = read_field(record, "status", default=None)
    label = read_field(record, "label", default=None)
    score = read_field(record, "score", default=None)

    if status is not None and status != "ok":
        return False, None, None, str(status)

    if status == "ok" and label is None:
        raise AnalyticsInputError(f"record {index}: status 'ok' without a label")
    if status is None and label is None:
        return False, None, None, "no status or label"

    return True, label, score, None


def _validate_label(label: object, index: int) -> str:
    if not isinstance(label, str):
        raise AnalyticsInputError(f"record {index}: label must be a string, got {label!r}")
    if label not in SHOT_CLASSES:
        raise AnalyticsInputError(
            f"record {index}: {label!r} is not in the shot vocabulary {SHOT_CLASSES}"
        )
    return label


def _validate_score(score: object, index: int) -> float:
    if isinstance(score, bool) or not isinstance(score, numbers.Real):
        raise AnalyticsInputError(f"record {index}: score must be a number, got {score!r}")
    value = float(score)
    if not 0.0 <= value <= 1.0:
        raise AnalyticsInputError(f"record {index}: score must be in [0, 1], got {value!r}")
    return value


def summarise_shots(records: Iterable[object]) -> ShotDistribution:
    """
    Count labels and summarise the scores that are actually present.

    `None` is rejected: an empty analysis is `[]`, not a missing argument.
    """
    if records is None:
        raise AnalyticsInputError("records is None; pass an empty iterable for no records")

    counts = {label: 0 for label in SHOT_CLASSES}
    scores: list[float] = []
    reasons: dict[str, int] = {}
    classified = 0
    unclassified = 0

    for index, record in enumerate(records):
        is_classified, label, score, reason = _record_parts(record, index)
        if is_classified:
            validated = _validate_label(label, index)
            counts[validated] += 1
            classified += 1
            if score is not None:
                scores.append(_validate_score(score, index))
        else:
            unclassified += 1
            key = reason or "unclassified"
            reasons[key] = reasons.get(key, 0) + 1

    shares = (
        {label: counts[label] / classified for label in SHOT_CLASSES}
        if classified > 0
        else {}
    )

    return ShotDistribution(
        counts=counts,
        classified=classified,
        unclassified=unclassified,
        total=classified + unclassified,
        shares=shares,
        unclassified_reasons=reasons,
        scored=len(scores),
        mean_score=sum(scores) / len(scores) if scores else None,
        min_score=min(scores) if scores else None,
        max_score=max(scores) if scores else None,
    )


__all__ = ["summarise_shots"]

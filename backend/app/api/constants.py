"""The versions and stage bands the browser contract names."""

from __future__ import annotations

SCHEMA_VERSION = "2.1"
PIPELINE_VERSION = "1.1.0"

STAGES = (
    "uploading",
    "tracking",
    "validation",
    "shot_classification",
    "overlay",
    "persisting",
)

#: Each stage owns a band of the 0..100 progress document. Progress inside a
#: band comes from measured units (bytes, frames, clips, deliveries), never an
#: assumed time fraction.
STAGE_BANDS: dict[str, tuple[float, float]] = {
    "uploading": (0.0, 5.0),
    "tracking": (5.0, 62.0),
    "validation": (62.0, 68.0),
    "shot_classification": (68.0, 86.0),
    "overlay": (86.0, 96.0),
    "persisting": (96.0, 100.0),
}

__all__ = ["PIPELINE_VERSION", "SCHEMA_VERSION", "STAGES", "STAGE_BANDS"]

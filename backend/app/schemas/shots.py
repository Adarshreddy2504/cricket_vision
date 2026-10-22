"""
shots.py — Typed records produced by the shot-classification pass.

These dataclasses are PLAIN DATA. Nothing here imports torch, OpenCV or the
architecture, so a classification can be stored, serialised and read back
without dragging a 68 MB checkpoint along with it. The module that runs the
model fills them in.

INVARIANTS ENFORCED HERE RATHER THAN TRUSTED
--------------------------------------------
  * `status` says what happened, and every non-`ok` record carries an `error`.
    A caller never has to infer failure from a missing label.
  * An `ok` record always carries a label that is in the model's vocabulary, a
    score, and the full probability vector. A label cannot exist without the
    vector it was chosen from.
  * `frames_repeated` cannot disagree with the frame counts. If the sampled
    positions repeat, the record says so, because a label produced from a
    5-frame sample repeated to 30 is not the same evidence as one produced
    from 30 distinct frames.
  * No accuracy, precision or recall field exists. The checkpoint stores no
    evaluation metrics, so there is no honest number to put there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..shots.constants import SHOT_CLASSES, SHOT_STATUSES

ShotStatus = Literal["ok", "model_unavailable", "invalid_input", "inference_failed"]
ShotInputSource = Literal["in_memory", "clip_file", "unavailable"]


def _check_probabilities(probabilities: dict[str, float], where: str) -> None:
    missing = sorted(set(SHOT_CLASSES) - set(probabilities))
    extra = sorted(set(probabilities) - set(SHOT_CLASSES))
    if missing or extra:
        raise ValueError(
            f"{where} must cover exactly the model vocabulary "
            f"(missing={missing}, unexpected={extra})"
        )
    for label, value in probabilities.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{where}[{label!r}] is not a number: {value!r}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{where}[{label!r}] is outside [0, 1]: {value!r}")
    total = sum(float(value) for value in probabilities.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"{where} sums to {total!r}, not 1")


@dataclass(frozen=True)
class ShotPrediction:
    """
    The model's answer for one input tensor, before any provenance is attached.

    `score` is the softmax probability of `label`; `class_probabilities` is the
    complete 10-way distribution, rounded to six decimals for stable JSON. Only
    construction checks live here — the caller is responsible for supplying a
    tensor of the documented shape.
    """

    label: str
    score: float
    class_probabilities: dict[str, float]

    def __post_init__(self) -> None:
        if self.label not in SHOT_CLASSES:
            raise ValueError(f"unknown shot label: {self.label!r}")
        _check_probabilities(self.class_probabilities, "class_probabilities")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError(f"score is outside [0, 1]: {self.score!r}")
        highest = max(float(value) for value in self.class_probabilities.values())
        if abs(float(self.score) - highest) > 1e-6:
            raise ValueError(
                f"score {self.score!r} is not the maximum probability {highest!r}; "
                "the label and the distribution disagree"
            )

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "class_probabilities": dict(self.class_probabilities),
        }


@dataclass(frozen=True)
class ShotClassification:
    """
    What happened when one delivery was shown to the shot classifier.

    status
        "ok"                  a label and score are present
        "model_unavailable"   no usable checkpoint was loaded
        "invalid_input"       no usable frames, or frames the model cannot take
        "inference_failed"    the model ran and failed, or returned nonsense
    label, score
        Present only when status is "ok"; otherwise None.
    class_probabilities
        The complete 10-way distribution, present only when status is "ok".
    input_source
        "in_memory", "clip_file", or "unavailable". A label from a clip re-read
        is distinguishable from one produced from the frames already in memory.
    frames_sampled, unique_frames_sampled, frames_repeated
        The evidence behind the label. A short clip is sampled with repeated
        positions, and that fact is recorded rather than hidden.
    sampled_clip_frames
        0-based positions within the delivery clip, in model input order.
    sampled_original_frames
        The same positions translated with `frame_offset`, when one was given.
    model_path, model_size_bytes, model_md5, model_device
        Which weights produced the answer, and on what device.
    error
        Why there is no label. Never empty on a non-"ok" record.
    """

    status: ShotStatus
    label: str | None = None
    score: float | None = None
    class_probabilities: dict[str, float] = field(default_factory=dict)
    input_source: ShotInputSource = "unavailable"
    frames_sampled: int = 0
    unique_frames_sampled: int = 0
    frames_repeated: bool = False
    sampled_clip_frames: list[int] = field(default_factory=list)
    sampled_original_frames: list[int] | None = None
    model_path: str | None = None
    model_size_bytes: int | None = None
    model_md5: str | None = None
    model_device: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in SHOT_STATUSES:
            raise ValueError(f"unknown shot status: {self.status!r}")

        if self.status == "ok":
            if self.label is None or self.score is None:
                raise ValueError("an 'ok' classification requires a label and a score")
            if self.label not in SHOT_CLASSES:
                raise ValueError(f"unknown shot label: {self.label!r}")
            _check_probabilities(self.class_probabilities, "class_probabilities")
            if not 0.0 <= float(self.score) <= 1.0:
                raise ValueError(f"score is outside [0, 1]: {self.score!r}")
            if self.error is not None:
                raise ValueError("an 'ok' classification cannot carry an error")
        else:
            if not self.error:
                raise ValueError("a failed classification must record an error")
            if self.label is not None or self.score is not None:
                raise ValueError("a failed classification cannot carry a label or score")
            if self.class_probabilities:
                raise ValueError("a failed classification cannot carry probabilities")

        if self.frames_sampled < 0:
            raise ValueError("frames_sampled cannot be negative")
        if not 0 <= self.unique_frames_sampled <= self.frames_sampled:
            raise ValueError(
                "unique_frames_sampled must be between 0 and frames_sampled "
                f"({self.unique_frames_sampled} vs {self.frames_sampled})"
            )
        expected_repeated = self.frames_sampled > 0 and (
            self.unique_frames_sampled < self.frames_sampled
        )
        if self.frames_repeated != expected_repeated:
            raise ValueError(
                "frames_repeated disagrees with the sampled frame counts "
                f"({self.unique_frames_sampled} unique of {self.frames_sampled})"
            )
        if self.sampled_clip_frames:
            if len(self.sampled_clip_frames) != self.frames_sampled:
                raise ValueError(
                    "sampled_clip_frames must have one position per sampled frame"
                )
            if any(position < 0 for position in self.sampled_clip_frames):
                raise ValueError("sampled_clip_frames cannot contain negative positions")
        if self.sampled_original_frames is not None:
            if len(self.sampled_original_frames) != len(self.sampled_clip_frames):
                raise ValueError(
                    "sampled_original_frames must align with sampled_clip_frames"
                )

    @property
    def ok(self) -> bool:
        """True only when a label is present."""
        return self.status == "ok"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "ok": self.ok,
            "label": self.label,
            "score": self.score,
            "class_probabilities": dict(self.class_probabilities),
            "input_source": self.input_source,
            "frames_sampled": self.frames_sampled,
            "unique_frames_sampled": self.unique_frames_sampled,
            "frames_repeated": self.frames_repeated,
            "sampled_clip_frames": list(self.sampled_clip_frames),
            "sampled_original_frames": (
                list(self.sampled_original_frames)
                if self.sampled_original_frames is not None
                else None
            ),
            "model_path": self.model_path,
            "model_size_bytes": self.model_size_bytes,
            "model_md5": self.model_md5,
            "model_device": self.model_device,
            "error": self.error,
        }


__all__ = [
    "ShotClassification",
    "ShotInputSource",
    "ShotPrediction",
    "ShotStatus",
]

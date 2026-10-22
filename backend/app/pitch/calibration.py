"""
calibration.py — The pitch-calibration record: four corners, where they came
from, and whether they may be trusted.

A calibration is the answer to "where is the pitch in THIS frame?". It holds

    * the four corners, in source-frame pixels, in the canonical order;
    * the size of the frame they belong to;
    * HOW the corners were obtained (provenance) and WHAT produced them
      (method);
    * a status that says whether the result may be used as a validated
      calibration or only as an approximation.

VALID, APPROXIMATE, INVALID
----------------------------
The status is the point of the whole record.

    VALID        the four corners are a measurement of where this pitch's
                 corners are in this frame. Either a human marked them, or a
                 model reported them as corners.

    APPROXIMATE  the four points are a GUESS about where the pitch is: a
                 detection box turned into four points, or a quad assumed
                 from a template. They are perfectly good for saying where
                 the ball was as a FRACTION of the pitch on screen, and they
                 are not a validated calibration. A client that draws them
                 must show them as an estimate.

    INVALID      the geometry was refused: not a quad, not on this picture,
                 or the wrong size to be a pitch. The reasons say which.

An approximate pitch box must never be represented as a validated
calibration, so `CornerMethod.BOX_ESTIMATE` and `CornerMethod.TEMPLATE`
cannot produce a VALID status however good their geometry looks.

CONFIDENCE IS ONLY RECORDED WHERE IT MEANS SOMETHING
-----------------------------------------------------
A keypoint or segmentation model reports a score for the instance it found.
That score is evidence about the DETECTION, and — for those two methods —
about the corners derived from it. A detection box's score is evidence about
the BOX, not about the four points guessed from it, so a box-derived
calibration carries no confidence at all rather than a borrowed one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .corners import ImageSize, PitchCorners, validate_corners
from .geometry import PitchGeometry


class CalibrationProvenance(str, Enum):
    """
    WHERE the four corners came from.

    This is the question a consumer asks first: was this measured, found by
    a model, or assumed?
    """

    #: Found by a model on this frame.
    DETECTED = "detected"
    #: A human measured these corners on this frame.
    MANUALLY_SUPPLIED = "manually_supplied"
    #: Assumed from a template or a prior; not measured on this frame.
    ASSUMED = "assumed"


class CornerMethod(str, Enum):
    """
    WHAT produced the four corners.

    Finer-grained than the provenance, because two detected sources are not
    equally trustworthy: a model that reports four corner keypoints has
    measured the corners, while a model that reports a box has only said
    where the pitch-shaped region is.
    """

    #: A keypoint model reported the four corners directly.
    KEYPOINTS = "keypoints"
    #: The four corners were simplified out of a segmentation mask boundary.
    SEGMENTATION = "segmentation"
    #: The four corners were GUESSED from a detection box.
    BOX_ESTIMATE = "box_estimate"
    #: A human marked the four corners.
    MANUAL = "manual"
    #: The four corners came from a template, not from this frame.
    TEMPLATE = "template"


class CalibrationStatus(str, Enum):
    """How far the calibration can be trusted."""

    #: Measured corners: usable as a validated calibration.
    VALID = "valid"
    #: Usable as an estimate of where the pitch is; NOT a validated
    #: calibration.
    APPROXIMATE = "approximate"
    #: Refused. The reasons say why.
    INVALID = "invalid"


#: Methods whose corners are a guess about the pitch rather than a
#: measurement of its corners. These can never produce a VALID status.
APPROXIMATE_METHODS = frozenset({CornerMethod.BOX_ESTIMATE, CornerMethod.TEMPLATE})

#: Methods whose model score is evidence about the corners themselves, and
#: not only about the region they were found in.
CONFIDENCE_METHODS = frozenset({CornerMethod.KEYPOINTS, CornerMethod.SEGMENTATION})


@dataclass(frozen=True)
class PitchCalibration:
    """
    One frame's pitch calibration.

    `frame_number` the 1-based frame this calibration describes. Zero means
                        "not tied to a particular frame".
    `image_size`    the (width, height) of the frame the corners are in.
    `corners`       the four corners, in source pixels, canonical order.
    `provenance`    where the corners came from.
    `method`        what produced them.
    `status`        how far they can be trusted.
    `reasons`       why the status is what it is. Empty for a VALID
                    calibration whose geometry simply passed.
    `confidence`    the model's own score, when it is evidence about these
                    corners. None when it would be misleading.
    `geometry`      the pitch's dimensions and their justification.
    """

    frame_number: int
    image_size: ImageSize
    corners: PitchCorners
    provenance: CalibrationProvenance
    method: CornerMethod
    status: CalibrationStatus
    reasons: tuple[str, ...] = ()
    confidence: float | None = None
    geometry: PitchGeometry = field(default_factory=PitchGeometry)

    @property
    def is_valid(self) -> bool:
        """May this calibration be presented as a validated calibration?"""
        return self.status is CalibrationStatus.VALID

    @property
    def is_approximate(self) -> bool:
        """Is this an estimate of the pitch rather than a measurement of it?"""
        return self.status is CalibrationStatus.APPROXIMATE

    @property
    def is_usable(self) -> bool:
        """Is there a quad to work with at all?"""
        return self.status in (CalibrationStatus.VALID, CalibrationStatus.APPROXIMATE)

    @property
    def was_measured_on_this_frame(self) -> bool:
        """
        Were these corners measured ON THIS FRAME, by a human?

        This is the gate on the metric (metres) conversion: a quad that
        nobody measured on this video can say where the pitch is in the
        picture, but not how many metres across it is.
        """
        return self.provenance is CalibrationProvenance.MANUALLY_SUPPLIED

    def as_dict(self) -> dict:
        """
        The persisted form.

        `status` is the first key because it is the one a consumer must not
        skip: an APPROXIMATE calibration drawn as if it were VALID is the
        failure this whole module exists to prevent.
        """
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "reasons": list(self.reasons),
            "frame_number": self.frame_number,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "corners": self.corners.as_dict(),
            "provenance": self.provenance.value,
            "method": self.method.value,
            "confidence": (
                None if self.confidence is None else round(float(self.confidence), 4)
            ),
            "geometry": self.geometry.as_dict(),
        }


def build_calibration(
    corners: PitchCorners,
    *,
    image_size: ImageSize,
    provenance: CalibrationProvenance,
    method: CornerMethod,
    frame_number: int = 0,
    confidence: float | None = None,
    geometry: PitchGeometry | None = None,
) -> PitchCalibration:
    """
    Build a calibration from four corners and say how far it can be trusted.

    The geometry is checked first: a quad that is not a quad is INVALID
    whatever produced it, and a refusal must not be softened by a good
    provenance. Only then does the provenance decide whether the result is
    VALID or APPROXIMATE.

    `confidence` is kept only for the methods whose model score is evidence
    about the corners. For any other method it is dropped rather than
    carried, because a number that looks like a corner confidence but is not
    one is worse than no number.
    """
    geom = geometry if geometry is not None else PitchGeometry()

    reasons: list[str] = []
    ok, reason = validate_corners(corners, image_size)
    if not ok:
        status = CalibrationStatus.INVALID
        reasons.append(reason)
    elif provenance is CalibrationProvenance.ASSUMED:
        # Checked before the method: "these corners were assumed" is the
        # stronger statement about where they came from, and a template quad
        # is assumed by definition, so it must not be reported as merely
        # approximate.
        status = CalibrationStatus.APPROXIMATE
        reasons.append("assumed_corners")
    elif method in APPROXIMATE_METHODS:
        status = CalibrationStatus.APPROXIMATE
        reasons.append("approximate_corners")
    else:
        status = CalibrationStatus.VALID

    kept_confidence: float | None = None
    if status is not CalibrationStatus.INVALID and method in CONFIDENCE_METHODS:
        if confidence is not None and math.isfinite(float(confidence)):
            kept_confidence = float(confidence)

    return PitchCalibration(
        frame_number=int(frame_number),
        image_size=(int(image_size[0]), int(image_size[1])),
        corners=corners,
        provenance=provenance,
        method=method,
        status=status,
        reasons=tuple(reasons),
        confidence=kept_confidence,
        geometry=geom,
    )


def manual_calibration(
    corners: PitchCorners,
    *,
    image_size: ImageSize,
    frame_number: int = 0,
    geometry: PitchGeometry | None = None,
) -> PitchCalibration:
    """
    A calibration from corners a human measured on this frame.

    The only route to a metric (metres) conversion, because a human who
    marked the four corners of the pitch in this picture is the only source
    that can say what those corners MEAN.
    """
    return build_calibration(
        corners,
        image_size=image_size,
        provenance=CalibrationProvenance.MANUALLY_SUPPLIED,
        method=CornerMethod.MANUAL,
        frame_number=frame_number,
        geometry=geometry,
    )


def detected_calibration(
    corners: PitchCorners,
    *,
    image_size: ImageSize,
    method: CornerMethod,
    confidence: float | None = None,
    frame_number: int = 0,
    geometry: PitchGeometry | None = None,
) -> PitchCalibration:
    """
    A calibration from corners a model found in this frame.

    `method` decides how far it can be trusted: a keypoint or segmentation
    quad is a measurement of the corners, while a box-derived quad is a
    guess and comes back APPROXIMATE.
    """
    return build_calibration(
        corners,
        image_size=image_size,
        provenance=CalibrationProvenance.DETECTED,
        method=method,
        frame_number=frame_number,
        confidence=confidence,
        geometry=geometry,
    )


def assumed_calibration(
    corners: PitchCorners,
    *,
    image_size: ImageSize,
    frame_number: int = 0,
    geometry: PitchGeometry | None = None,
) -> PitchCalibration:
    """
    A calibration from corners assumed from a template rather than measured.

    Always APPROXIMATE: nobody has looked at this frame.
    """
    return build_calibration(
        corners,
        image_size=image_size,
        provenance=CalibrationProvenance.ASSUMED,
        method=CornerMethod.TEMPLATE,
        frame_number=frame_number,
        geometry=geometry,
    )


__all__ = [
    "APPROXIMATE_METHODS",
    "CONFIDENCE_METHODS",
    "CalibrationProvenance",
    "CalibrationStatus",
    "CornerMethod",
    "PitchCalibration",
    "assumed_calibration",
    "build_calibration",
    "detected_calibration",
    "manual_calibration",
]

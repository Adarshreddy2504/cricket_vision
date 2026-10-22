"""
test_pitch_calibration.py — provenance, validity status, and geometry.

The status is the point of the calibration record: an approximate pitch box
must never be presented as a validated calibration. These tests pin the rule
for every method the project can produce, and pin the metric geometry
defaults to the Laws of Cricket with their provenance saying so.
"""

from __future__ import annotations

import unittest

from app.pitch.calibration import (
    APPROXIMATE_METHODS,
    CONFIDENCE_METHODS,
    CalibrationProvenance,
    CalibrationStatus,
    CornerMethod,
    assumed_calibration,
    build_calibration,
    detected_calibration,
    manual_calibration,
)
from app.pitch.constants import CREASE_WIDTH_M, PITCH_LENGTH_M
from app.pitch.corners import PitchCorners, order_corners
from app.pitch.errors import CalibrationError
from app.pitch.geometry import GeometryProvenance, LongAxis, PitchGeometry
from tests.pitch_helpers import FRAME_SIZE, trapezoid

# A well-formed, correctly ordered quad that is simply off the picture: valid
# geometry, invalid placement. Used to prove that a refusal cannot be rescued
# by provenance.
OFF_PICTURE = [(-300.0, 100.0), (-100.0, 100.0), (-80.0, 300.0), (-320.0, 300.0)]


class TestStatus(unittest.TestCase):
    def test_manual_corners_are_valid(self):
        calibration = manual_calibration(trapezoid(), image_size=FRAME_SIZE, frame_number=7)
        self.assertIs(calibration.status, CalibrationStatus.VALID)
        self.assertTrue(calibration.is_valid)
        self.assertFalse(calibration.is_approximate)
        self.assertTrue(calibration.is_usable)
        self.assertEqual(calibration.reasons, ())
        self.assertEqual(calibration.frame_number, 7)

    def test_keypoint_corners_are_valid(self):
        calibration = detected_calibration(
            trapezoid(),
            image_size=FRAME_SIZE,
            method=CornerMethod.KEYPOINTS,
            confidence=0.88,
        )
        self.assertIs(calibration.status, CalibrationStatus.VALID)
        self.assertEqual(calibration.confidence, 0.88)

    def test_segmentation_corners_are_valid(self):
        calibration = detected_calibration(
            trapezoid(),
            image_size=FRAME_SIZE,
            method=CornerMethod.SEGMENTATION,
            confidence=0.71,
        )
        self.assertIs(calibration.status, CalibrationStatus.VALID)
        self.assertEqual(calibration.confidence, 0.71)

    def test_a_box_estimate_is_only_approximate(self):
        calibration = detected_calibration(
            trapezoid(),
            image_size=FRAME_SIZE,
            method=CornerMethod.BOX_ESTIMATE,
            confidence=0.93,
        )
        self.assertIs(calibration.status, CalibrationStatus.APPROXIMATE)
        self.assertFalse(calibration.is_valid)
        self.assertTrue(calibration.is_approximate)
        self.assertTrue(calibration.is_usable)
        self.assertEqual(calibration.reasons, ("approximate_corners",))

    def test_a_template_is_only_approximate(self):
        calibration = assumed_calibration(trapezoid(), image_size=FRAME_SIZE)
        self.assertIs(calibration.status, CalibrationStatus.APPROXIMATE)
        self.assertIs(calibration.provenance, CalibrationProvenance.ASSUMED)
        self.assertIs(calibration.method, CornerMethod.TEMPLATE)
        self.assertEqual(calibration.reasons, ("assumed_corners",))

    def test_an_assumed_provenance_is_approximate_whatever_the_method(self):
        calibration = build_calibration(
            trapezoid(),
            image_size=FRAME_SIZE,
            provenance=CalibrationProvenance.ASSUMED,
            method=CornerMethod.KEYPOINTS,
        )
        self.assertIs(calibration.status, CalibrationStatus.APPROXIMATE)
        self.assertEqual(calibration.reasons, ("assumed_corners",))

    def test_bad_geometry_wins_over_good_provenance(self):
        calibration = manual_calibration(
            order_corners(OFF_PICTURE), image_size=FRAME_SIZE
        )
        self.assertIs(calibration.status, CalibrationStatus.INVALID)
        self.assertFalse(calibration.is_valid)
        self.assertFalse(calibration.is_approximate)
        self.assertFalse(calibration.is_usable)
        self.assertEqual(calibration.reasons, ("corners_outside_frame",))

    def test_the_method_tables_are_the_documented_sets(self):
        self.assertEqual(
            APPROXIMATE_METHODS,
            frozenset({CornerMethod.BOX_ESTIMATE, CornerMethod.TEMPLATE}),
        )
        self.assertEqual(
            CONFIDENCE_METHODS,
            frozenset({CornerMethod.KEYPOINTS, CornerMethod.SEGMENTATION}),
        )


class TestConfidence(unittest.TestCase):
    def test_a_box_estimate_carries_no_confidence(self):
        # The box's score is evidence about the box, not about four points
        # guessed from it.
        calibration = detected_calibration(
            trapezoid(),
            image_size=FRAME_SIZE,
            method=CornerMethod.BOX_ESTIMATE,
            confidence=0.93,
        )
        self.assertIsNone(calibration.confidence)

    def test_a_template_carries_no_confidence(self):
        self.assertIsNone(assumed_calibration(trapezoid(), image_size=FRAME_SIZE).confidence)

    def test_a_manual_calibration_carries_no_confidence(self):
        self.assertIsNone(manual_calibration(trapezoid(), image_size=FRAME_SIZE).confidence)

    def test_a_non_finite_score_is_dropped(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                calibration = detected_calibration(
                    trapezoid(),
                    image_size=FRAME_SIZE,
                    method=CornerMethod.SEGMENTATION,
                    confidence=bad,
                )
                self.assertIsNone(calibration.confidence)

    def test_an_invalid_calibration_keeps_no_confidence(self):
        calibration = detected_calibration(
            order_corners(OFF_PICTURE),
            image_size=FRAME_SIZE,
            method=CornerMethod.SEGMENTATION,
            confidence=0.95,
        )
        self.assertIs(calibration.status, CalibrationStatus.INVALID)
        self.assertIsNone(calibration.confidence)


class TestProvenance(unittest.TestCase):
    def test_the_provenance_vocabulary_is_the_documented_one(self):
        self.assertEqual(CalibrationProvenance.DETECTED.value, "detected")
        self.assertEqual(CalibrationProvenance.MANUALLY_SUPPLIED.value, "manually_supplied")
        self.assertEqual(CalibrationProvenance.ASSUMED.value, "assumed")

    def test_only_a_manual_calibration_was_measured_on_this_frame(self):
        self.assertTrue(
            manual_calibration(trapezoid(), image_size=FRAME_SIZE).was_measured_on_this_frame
        )
        self.assertFalse(
            detected_calibration(
                trapezoid(), image_size=FRAME_SIZE, method=CornerMethod.SEGMENTATION
            ).was_measured_on_this_frame
        )
        self.assertFalse(
            assumed_calibration(trapezoid(), image_size=FRAME_SIZE).was_measured_on_this_frame
        )


class TestPersistedForm(unittest.TestCase):
    def test_status_is_the_first_key(self):
        document = manual_calibration(trapezoid(), image_size=FRAME_SIZE).as_dict()
        self.assertEqual(next(iter(document)), "status")
        self.assertEqual(document["status"], "valid")
        self.assertTrue(document["is_valid"])
        self.assertEqual(document["image_size"], [640, 360])
        self.assertEqual(document["frame_number"], 0)
        self.assertEqual(document["provenance"], "manually_supplied")
        self.assertEqual(document["method"], "manual")
        self.assertEqual(document["reasons"], [])
        self.assertEqual(document["corners"]["top_left"], [100.0, 100.0])
        self.assertEqual(document["geometry"]["long_axis"], "vertical")

    def test_an_approximate_document_says_so(self):
        document = detected_calibration(
            trapezoid(), image_size=FRAME_SIZE, method=CornerMethod.BOX_ESTIMATE
        ).as_dict()
        self.assertEqual(document["status"], "approximate")
        self.assertFalse(document["is_valid"])
        self.assertEqual(document["reasons"], ["approximate_corners"])


class TestGeometryRecord(unittest.TestCase):
    def test_the_defaults_are_the_law_6_dimensions_assumed(self):
        geometry = PitchGeometry()
        self.assertEqual(geometry.length_m, 20.12)
        self.assertEqual(geometry.width_m, 2.64)
        self.assertIs(geometry.long_axis, LongAxis.VERTICAL)
        self.assertIs(geometry.provenance, GeometryProvenance.ASSUMED)
        self.assertFalse(geometry.is_measured)
        self.assertIn("Law 6", geometry.justification)

    def test_the_vertical_target_quad(self):
        self.assertEqual(
            PitchGeometry().target_quad(),
            ((-1.32, 0.0), (1.32, 0.0), (1.32, 20.12), (-1.32, 20.12)),
        )

    def test_the_horizontal_target_quad(self):
        self.assertEqual(
            PitchGeometry(long_axis=LongAxis.HORIZONTAL).target_quad(),
            ((-1.32, 0.0), (-1.32, 20.12), (1.32, 20.12), (1.32, 0.0)),
        )

    def test_measured_dimensions_can_be_supplied(self):
        geometry = PitchGeometry(
            length_m=19.5,
            width_m=2.5,
            provenance=GeometryProvenance.MANUALLY_SUPPLIED,
        )
        self.assertTrue(geometry.is_measured)
        self.assertEqual(geometry.target_quad()[3], (-1.25, 19.5))

    def test_unusable_dimensions_are_refused(self):
        cases = [
            ({"length_m": 0.0}, "invalid_pitch_length"),
            ({"length_m": -1.0}, "invalid_pitch_length"),
            ({"length_m": float("nan")}, "invalid_pitch_length"),
            ({"width_m": 0.0}, "invalid_pitch_width"),
            ({"width_m": float("inf")}, "invalid_pitch_width"),
        ]
        for kwargs, reason in cases:
            with self.subTest(kwargs=kwargs):
                geometry = PitchGeometry(**kwargs)
                self.assertEqual(geometry.validate(), (False, reason))
                with self.assertRaises(CalibrationError):
                    geometry.target_quad()

    def test_the_geometry_document_is_configuration_not_a_measurement(self):
        document = PitchGeometry().as_dict()
        self.assertEqual(
            set(document),
            {"length_m", "width_m", "long_axis", "provenance", "justification"},
        )
        self.assertEqual(document["long_axis"], "vertical")
        self.assertEqual(document["provenance"], "assumed")
        self.assertEqual(document["length_m"], PITCH_LENGTH_M)
        self.assertEqual(document["width_m"], CREASE_WIDTH_M)

    def test_a_calibration_carries_its_geometry(self):
        calibration = manual_calibration(trapezoid(), image_size=FRAME_SIZE)
        self.assertIsInstance(calibration.geometry, PitchGeometry)
        self.assertEqual(calibration.geometry.length_m, PITCH_LENGTH_M)
        self.assertEqual(calibration.geometry.width_m, CREASE_WIDTH_M)

    def test_a_calibration_accepts_measured_geometry(self):
        geometry = PitchGeometry(provenance=GeometryProvenance.MANUALLY_SUPPLIED)
        calibration = manual_calibration(
            trapezoid(), image_size=FRAME_SIZE, geometry=geometry
        )
        self.assertIs(calibration.geometry, geometry)


class TestCornersAreKeptAsGiven(unittest.TestCase):
    def test_the_calibration_does_not_reorder_or_repair_the_corners(self):
        # A caller handing over a non-convex quad gets an INVALID calibration
        # that still shows the quad it was given, so the mistake is visible.
        quad = PitchCorners((0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0))
        calibration = manual_calibration(quad, image_size=FRAME_SIZE)
        self.assertIs(calibration.corners, quad)
        self.assertIs(calibration.status, CalibrationStatus.INVALID)
        self.assertEqual(calibration.reasons, ("non_convex_corners",))


if __name__ == "__main__":
    unittest.main()

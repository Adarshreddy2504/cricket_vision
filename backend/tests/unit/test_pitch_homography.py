"""
test_pitch_homography.py — the pixel -> pitch conversion.

Every expected number here is computed BY HAND from the corner coordinates,
not captured from a previous run. The rectangle's homography is affine, so
its interior points map linearly and the arithmetic is checkable on paper;
the trapezoid's is projective, so the tests also pin a property that only a
true homography satisfies (the diagonals meet at the image of the square's
centre) and prove that a bounding-box stretch would NOT pass.
"""

from __future__ import annotations

import unittest

from app.pitch.calibration import (
    CalibrationStatus,
    CornerMethod,
    assumed_calibration,
    detected_calibration,
    manual_calibration,
)
from app.pitch.constants import CREASE_WIDTH_M, PITCH_LENGTH_M
from app.pitch.corners import PitchCorners, order_corners
from app.pitch.geometry import GeometryProvenance, LongAxis, PitchGeometry
from app.pitch.homography import (
    SPACE_AXES,
    SPACE_UNITS,
    PitchProjection,
    PitchSpace,
    build_homography,
    build_projection,
    metric_gate,
    normalized_target,
    pitch_to_pixel,
    pixel_to_pitch,
)
from tests.pitch_helpers import (
    FRAME_SIZE,
    rectangle,
    trapezoid,
)

# The trapezoid's two diagonals meet here: TL->BR is
# (100, 100) + t * (220, 200), TR->BL is (300, 100) + s * (-220, 200), and
# t = s = 5/11. In a projective map that point is the image of the square's
# centre, (0.5, 0.5).
DIAGONAL_INTERSECTION = (200.0, 2100.0 / 11.0)

# The homography the trapezoid implies, derived by hand. It is the reason the
# two expected numbers below are exact fractions rather than recorded floats.
#   u = (x/180 + y/1800 - 11/18) / (y/900 + 1)
#   v = (y/150 - 2/3)            / (y/900 + 1)
TRAPEZOID_CENTRE_PIXEL = (200.0, 200.0)  # -> (0.5, 6/11)


def projection_for(calibration, space: PitchSpace = PitchSpace.NORMALIZED) -> PitchProjection:
    """The projection the calibration justifies, or a test failure."""
    projection, reason = build_projection(calibration, space=space)
    if projection is None:
        raise AssertionError(f"no projection: {reason}")
    return projection


def manual_rectangle():
    return manual_calibration(rectangle(), image_size=FRAME_SIZE)


class TestTransformDirection(unittest.TestCase):
    def test_each_corner_lands_on_its_own_target_corner(self):
        projection = projection_for(manual_rectangle())
        expected = [
            ((100.0, 100.0), (0.0, 0.0)),
            ((300.0, 100.0), (1.0, 0.0)),
            ((300.0, 300.0), (1.0, 1.0)),
            ((100.0, 300.0), (0.0, 1.0)),
        ]
        for pixel, target in expected:
            with self.subTest(pixel=pixel):
                mapped = pixel_to_pitch(*pixel, projection)
                self.assertIsNotNone(mapped)
                self.assertAlmostEqual(mapped[0], target[0], places=6)
                self.assertAlmostEqual(mapped[1], target[1], places=6)

    def test_a_rectangle_s_interior_maps_linearly(self):
        # Affine: (x - 100) / 200 and (y - 100) / 200, by hand.
        projection = projection_for(manual_rectangle())
        for pixel, target in [
            ((200.0, 200.0), (0.5, 0.5)),
            ((250.0, 150.0), (0.75, 0.25)),
            ((150.0, 275.0), (0.25, 0.875)),
        ]:
            with self.subTest(pixel=pixel):
                mapped = pixel_to_pitch(*pixel, projection)
                self.assertAlmostEqual(mapped[0], target[0], places=6)
                self.assertAlmostEqual(mapped[1], target[1], places=6)

    def test_the_direction_is_pixel_to_pitch_and_not_the_reverse(self):
        projection = projection_for(manual_rectangle())
        # The pixel origin is OUTSIDE the quad, so it maps outside the square.
        outside = pixel_to_pitch(0.0, 0.0, projection)
        self.assertIsNotNone(outside)
        self.assertLess(outside[0], 0.0)
        self.assertLess(outside[1], 0.0)
        # The pitch coordinate's origin is the top-left CORNER pixel.
        corner = pitch_to_pixel(0.0, 0.0, projection)
        self.assertIsNotNone(corner)
        self.assertAlmostEqual(corner[0], 100.0, places=6)
        self.assertAlmostEqual(corner[1], 100.0, places=6)
        # A pitch coordinate outside the square unprojects off the quad.
        away = pitch_to_pixel(2.0, 2.0, projection)
        self.assertIsNotNone(away)
        self.assertGreater(away[0], 300.0)
        self.assertGreater(away[1], 300.0)

    def test_the_round_trip_returns_the_original_pixel(self):
        projection = projection_for(
            detected_calibration(
                trapezoid(), image_size=FRAME_SIZE, method=CornerMethod.SEGMENTATION
            )
        )
        for pixel in [(100.0, 100.0), (200.0, 150.0), (300.0, 250.0), (150.0, 280.0)]:
            with self.subTest(pixel=pixel):
                mapped = pixel_to_pitch(*pixel, projection)
                self.assertIsNotNone(mapped)
                back = pitch_to_pixel(*mapped, projection)
                self.assertIsNotNone(back)
                self.assertAlmostEqual(back[0], pixel[0], places=6)
                self.assertAlmostEqual(back[1], pixel[1], places=6)

    def test_a_projective_quad_puts_its_diagonal_intersection_at_the_centre(self):
        projection = projection_for(manual_calibration(trapezoid(), image_size=FRAME_SIZE))
        mapped = pixel_to_pitch(*DIAGONAL_INTERSECTION, projection)
        self.assertIsNotNone(mapped)
        self.assertAlmostEqual(mapped[0], 0.5, places=6)
        self.assertAlmostEqual(mapped[1], 0.5, places=6)

    def test_a_projective_quad_is_not_a_bounding_box_stretch(self):
        projection = projection_for(manual_calibration(trapezoid(), image_size=FRAME_SIZE))
        mapped = pixel_to_pitch(*TRAPEZOID_CENTRE_PIXEL, projection)
        self.assertIsNotNone(mapped)
        # Across is unaffected: the trapezoid is symmetric about x = 200.
        self.assertAlmostEqual(mapped[0], 0.5, places=6)
        # Along is 6/11, not the 0.5 a stretch of the bounding box would give.
        self.assertAlmostEqual(mapped[1], 6.0 / 11.0, places=6)
        self.assertNotAlmostEqual(mapped[1], 0.5, places=3)


class TestSpacesAndUnits(unittest.TestCase):
    def test_the_normalized_space(self):
        projection = projection_for(manual_rectangle())
        self.assertIs(projection.space, PitchSpace.NORMALIZED)
        self.assertEqual(projection.units, "pitch_fraction")
        self.assertEqual(projection.axes, ("across", "along"))
        self.assertIsNone(projection.geometry)

    def test_the_metric_space(self):
        projection = projection_for(manual_rectangle(), space=PitchSpace.METRES)
        self.assertIs(projection.space, PitchSpace.METRES)
        self.assertEqual(projection.units, "metre")
        self.assertIsNotNone(projection.geometry)

    def test_the_units_table_is_the_documented_vocabulary(self):
        self.assertEqual(SPACE_UNITS[PitchSpace.NORMALIZED], "pitch_fraction")
        self.assertEqual(SPACE_UNITS[PitchSpace.METRES], "metre")
        self.assertEqual(SPACE_AXES[PitchSpace.NORMALIZED], ("across", "along"))
        self.assertEqual(SPACE_AXES[PitchSpace.METRES], ("across", "along"))

    def test_the_normalized_target_quad_for_each_long_axis(self):
        self.assertEqual(
            normalized_target(LongAxis.VERTICAL),
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        )
        self.assertEqual(
            normalized_target(LongAxis.HORIZONTAL),
            ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
        )

    def test_the_homography_rows_are_persistable(self):
        rows = projection_for(manual_rectangle()).homography_rows()
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row) == 3 for row in rows))
        self.assertAlmostEqual(rows[2][0], 0.0)
        self.assertAlmostEqual(rows[2][2], 1.0)


class TestMetricCoordinates(unittest.TestCase):
    # The transform is computed from CV_32F correspondences (an OpenCV
    # requirement), so a metre value carries about a micrometre of noise --
    # 20.12 m comes back as 20.1200008. Asserting to 0.1 mm states exactly what
    # the conversion can claim at pitch scale.
    METRE_PLACES = 4

    def test_the_metric_corners_are_the_pitch_corners(self):
        projection = projection_for(manual_rectangle(), space=PitchSpace.METRES)
        half = CREASE_WIDTH_M / 2.0
        expected = [
            ((100.0, 100.0), (-half, 0.0)),
            ((300.0, 100.0), (half, 0.0)),
            ((300.0, 300.0), (half, PITCH_LENGTH_M)),
            ((100.0, 300.0), (-half, PITCH_LENGTH_M)),
        ]
        for pixel, target in expected:
            with self.subTest(pixel=pixel):
                mapped = pixel_to_pitch(*pixel, projection)
                self.assertIsNotNone(mapped)
                self.assertAlmostEqual(mapped[0], target[0], places=self.METRE_PLACES)
                self.assertAlmostEqual(mapped[1], target[1], places=self.METRE_PLACES)

    def test_metric_interior_points(self):
        # across runs -1.32 .. +1.32 and along runs 0 .. 20.12, so
        # (250, 150) is 0.66 m across and 5.03 m along.
        projection = projection_for(manual_rectangle(), space=PitchSpace.METRES)
        for pixel, target in [
            ((200.0, 200.0), (0.0, 10.06)),
            ((250.0, 150.0), (0.66, 5.03)),
        ]:
            with self.subTest(pixel=pixel):
                mapped = pixel_to_pitch(*pixel, projection)
                self.assertIsNotNone(mapped)
                self.assertAlmostEqual(mapped[0], target[0], places=self.METRE_PLACES)
                self.assertAlmostEqual(mapped[1], target[1], places=self.METRE_PLACES)

    def test_metric_contains_uses_the_measured_extent(self):
        projection = projection_for(manual_rectangle(), space=PitchSpace.METRES)
        self.assertTrue(projection.contains(200.0, 200.0))
        self.assertTrue(projection.contains(100.0, 300.0))
        self.assertFalse(projection.contains(100.0, 320.0))

    def test_normalized_contains_is_the_unit_square(self):
        projection = projection_for(manual_rectangle())
        self.assertTrue(projection.contains(200.0, 200.0))
        self.assertFalse(projection.contains(50.0, 200.0))
        self.assertFalse(projection.contains(200.0, 340.0))


class TestMetricGate(unittest.TestCase):
    def test_manually_measured_corners_justify_metres(self):
        calibration = manual_rectangle()
        self.assertEqual(metric_gate(calibration), (True, "ok"))
        self.assertIsNotNone(projection_for(calibration, space=PitchSpace.METRES))

    def test_measured_dimensions_justify_metres(self):
        geometry = PitchGeometry(
            length_m=19.8, width_m=2.5, provenance=GeometryProvenance.MANUALLY_SUPPLIED
        )
        calibration = detected_calibration(
            rectangle(),
            image_size=FRAME_SIZE,
            method=CornerMethod.SEGMENTATION,
            geometry=geometry,
        )
        self.assertEqual(metric_gate(calibration), (True, "ok"))

    def test_a_detected_quad_with_assumed_dimensions_is_refused(self):
        calibration = detected_calibration(
            rectangle(),
            image_size=FRAME_SIZE,
            method=CornerMethod.SEGMENTATION,
            confidence=0.95,
        )
        self.assertEqual(metric_gate(calibration), (False, "metric_geometry_not_justified"))
        projection, reason = build_projection(calibration, space=PitchSpace.METRES)
        self.assertIsNone(projection)
        self.assertEqual(reason, "metric_geometry_not_justified")

    def test_an_approximate_quad_with_assumed_dimensions_is_refused(self):
        calibration = detected_calibration(
            rectangle(),
            image_size=FRAME_SIZE,
            method=CornerMethod.BOX_ESTIMATE,
            confidence=0.93,
        )
        self.assertEqual(metric_gate(calibration), (False, "metric_geometry_not_justified"))

    def test_normalized_coordinates_are_always_available(self):
        usable = [
            manual_calibration(rectangle(), image_size=FRAME_SIZE),
            detected_calibration(
                rectangle(), image_size=FRAME_SIZE, method=CornerMethod.SEGMENTATION
            ),
            detected_calibration(
                rectangle(), image_size=FRAME_SIZE, method=CornerMethod.BOX_ESTIMATE
            ),
            assumed_calibration(rectangle(), image_size=FRAME_SIZE),
        ]
        for calibration in usable:
            with self.subTest(method=calibration.method.value):
                projection, reason = build_projection(calibration, space=PitchSpace.NORMALIZED)
                self.assertIsNotNone(projection, reason)
                self.assertEqual(reason, "")


class TestRefusals(unittest.TestCase):
    def test_an_invalid_calibration_projects_nowhere(self):
        # Off the picture but correctly ordered: the refusal is about
        # placement, not about the corners being a non-quad.
        off_picture = order_corners(
            [(-300.0, 100.0), (-100.0, 100.0), (-80.0, 300.0), (-320.0, 300.0)]
        )
        calibration = manual_calibration(off_picture, image_size=FRAME_SIZE)
        self.assertIs(calibration.status, CalibrationStatus.INVALID)
        for space in (PitchSpace.NORMALIZED, PitchSpace.METRES):
            with self.subTest(space=space.value):
                self.assertEqual(
                    build_projection(calibration, space=space),
                    (None, "invalid_calibration"),
                )
        self.assertEqual(metric_gate(calibration), (False, "invalid_calibration"))

    def test_collinear_corners_have_no_homography(self):
        collinear = PitchCorners((0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0))
        self.assertIsNone(build_homography(collinear, normalized_target(LongAxis.VERTICAL)))

    def test_duplicate_target_points_have_no_homography(self):
        target = ((0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        self.assertIsNone(build_homography(rectangle(), target))

    def test_a_wrong_shaped_target_has_no_homography(self):
        self.assertIsNone(build_homography(rectangle(), ((0.0, 0.0), (1.0, 1.0))))

    def test_a_non_finite_pixel_is_refused(self):
        projection = projection_for(manual_rectangle())
        self.assertIsNone(pixel_to_pitch(float("nan"), 100.0, projection))
        self.assertIsNone(pixel_to_pitch(100.0, float("inf"), projection))

    def test_a_pixel_on_the_horizon_is_refused(self):
        # w = y / 900 + 1 for this quad, so w = 0 at y = -900 and the pixel
        # has no finite image. It must not come back as a huge number.
        projection = projection_for(manual_calibration(trapezoid(), image_size=FRAME_SIZE))
        self.assertIsNone(pixel_to_pitch(0.0, -900.0, projection))


class TestHorizontalLongAxis(unittest.TestCase):
    def test_a_side_on_pitch_remaps_the_corners(self):
        calibration = manual_calibration(
            trapezoid(),
            image_size=FRAME_SIZE,
            geometry=PitchGeometry(long_axis=LongAxis.HORIZONTAL),
        )
        projection = projection_for(calibration)
        expected = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
        for pixel, target in zip(trapezoid().points(), expected, strict=True):
            with self.subTest(pixel=pixel):
                mapped = pixel_to_pitch(*pixel, projection)
                self.assertAlmostEqual(mapped[0], target[0], places=6)
                self.assertAlmostEqual(mapped[1], target[1], places=6)


class TestSyntheticPointSweep(unittest.TestCase):
    def test_a_grid_of_pitch_coordinates_round_trips_through_pixels(self):
        # Sweep the pitch square, not the bounding box: every point of the
        # square has a pixel inside the quad, and every one must come back.
        projection = projection_for(manual_calibration(trapezoid(), image_size=FRAME_SIZE))
        for i in range(11):
            for j in range(11):
                across = i / 10.0
                along = j / 10.0
                pixel = pitch_to_pixel(across, along, projection)
                self.assertIsNotNone(pixel, (across, along))
                back = pixel_to_pitch(*pixel, projection)
                self.assertIsNotNone(back, (across, along))
                self.assertAlmostEqual(back[0], across, places=6)
                self.assertAlmostEqual(back[1], along, places=6)
                self.assertTrue(projection.contains(pixel[0], pixel[1]))


if __name__ == "__main__":
    unittest.main()

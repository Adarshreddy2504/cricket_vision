"""
test_pitch_corners.py — the canonical corner order and the geometry gate.

The corner order is part of the coordinate contract: the homography decides
which corner of the pitch coordinate system each pixel corner lands on, so a
quad in a different order silently rotates or mirrors the whole conversion.
These tests pin the order, the refusals, and the size/position gate.
"""

from __future__ import annotations

import unittest

from app.pitch.corners import (
    CORNER_ORDER,
    PitchCorners,
    corners_from_box,
    corners_in_frame,
    is_convex,
    order_corners,
    polygon_area,
    validate_corners,
)
from app.pitch.errors import CalibrationError
from tests.pitch_helpers import (
    FRAME_SIZE,
    RECTANGLE_POINTS,
    TRAPEZOID_POINTS,
    rectangle,
    trapezoid,
)


class TestCanonicalOrder(unittest.TestCase):
    def test_the_order_is_the_declared_one(self):
        self.assertEqual(
            CORNER_ORDER,
            ("top_left", "top_right", "bottom_right", "bottom_left"),
        )
        self.assertEqual(tuple(PitchCorners.__dataclass_fields__), CORNER_ORDER)

    def test_a_rectangle_is_ordered_by_its_corners(self):
        corners = order_corners(RECTANGLE_POINTS)
        self.assertEqual(corners.top_left, (100.0, 100.0))
        self.assertEqual(corners.top_right, (300.0, 100.0))
        self.assertEqual(corners.bottom_right, (300.0, 300.0))
        self.assertEqual(corners.bottom_left, (100.0, 300.0))

    def test_the_order_does_not_depend_on_the_input_order(self):
        expected = order_corners(TRAPEZOID_POINTS)
        shuffles = [
            [TRAPEZOID_POINTS[2], TRAPEZOID_POINTS[0], TRAPEZOID_POINTS[3], TRAPEZOID_POINTS[1]],
            list(reversed(TRAPEZOID_POINTS)),
            [TRAPEZOID_POINTS[3], TRAPEZOID_POINTS[1], TRAPEZOID_POINTS[0], TRAPEZOID_POINTS[2]],
        ]
        for shuffled in shuffles:
            with self.subTest(shuffled=shuffled):
                self.assertEqual(order_corners(shuffled), expected)

    def test_the_four_corners_are_the_four_extremes(self):
        corners = trapezoid()
        points = corners.points()
        self.assertEqual(corners.top_left, min(points, key=lambda p: (p[0] + p[1], p[0])))
        self.assertEqual(corners.bottom_right, max(points, key=lambda p: (p[0] + p[1], p[0])))
        self.assertEqual(corners.top_right, max(points, key=lambda p: (p[0] - p[1], -p[0])))
        self.assertEqual(corners.bottom_left, min(points, key=lambda p: (p[0] - p[1], -p[0])))

    def test_a_rotated_quad_is_still_ordered(self):
        diamond = [(100.0, 0.0), (200.0, 100.0), (100.0, 200.0), (0.0, 100.0)]
        corners = order_corners(diamond)
        self.assertTrue(is_convex(corners))
        self.assertEqual(set(corners.points()), set(diamond))
        # A diamond's "top-left" is its leftmost point, not its topmost.
        self.assertEqual(corners.top_left, (0.0, 100.0))

    def test_the_persisted_form_is_named_and_rounded(self):
        doc = trapezoid().as_dict()
        self.assertEqual(list(doc), list(CORNER_ORDER))
        self.assertEqual(doc["top_left"], [100.0, 100.0])
        self.assertEqual(trapezoid().as_list()[2], [320.0, 300.0])


class TestOrderingRefusals(unittest.TestCase):
    def test_fewer_than_four_points(self):
        with self.assertRaises(CalibrationError) as ctx:
            order_corners([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
        self.assertIn("exactly four", str(ctx.exception))

    def test_more_than_four_points(self):
        with self.assertRaises(CalibrationError) as ctx:
            order_corners(
                [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (5.0, 5.0)]
            )
        self.assertIn("exactly four", str(ctx.exception))

    def test_a_non_finite_corner(self):
        for bad in [(float("nan"), 0.0), (0.0, float("inf")), (float("-inf"), 0.0)]:
            with self.subTest(bad=bad):
                with self.assertRaises(CalibrationError) as ctx:
                    order_corners([bad, (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
                self.assertIn("not finite", str(ctx.exception))

    def test_a_repeated_corner(self):
        with self.assertRaises(CalibrationError) as ctx:
            order_corners([(0.0, 0.0), (0.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        self.assertIn("four distinct", str(ctx.exception))

    def test_a_point_inside_the_others_triangle(self):
        # (8, 3) is inside the triangle (0,0)-(10,0)-(10,10), so the extremes
        # collide and there is no quad to order.
        with self.assertRaises(CalibrationError) as ctx:
            order_corners([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (8.0, 3.0)])
        self.assertIn("four distinct", str(ctx.exception))

    def test_collinear_points(self):
        with self.assertRaises(CalibrationError):
            order_corners([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)])

    def test_something_that_is_not_a_point(self):
        with self.assertRaises(CalibrationError) as ctx:
            order_corners([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), None])
        self.assertIn("not a point", str(ctx.exception))


class TestIsConvex(unittest.TestCase):
    def test_a_rectangle_is_convex(self):
        self.assertTrue(is_convex(rectangle()))

    def test_a_trapezoid_is_convex(self):
        self.assertTrue(is_convex(trapezoid()))

    def test_a_bow_tie_is_not_convex(self):
        # The canonical NAMES, but the points cross over in space.
        bow_tie = PitchCorners(
            (0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)
        )
        self.assertFalse(is_convex(bow_tie))

    def test_three_collinear_corners_are_not_a_quad(self):
        flat = PitchCorners((0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (100.0, 100.0))
        self.assertFalse(is_convex(flat))


class TestValidateCorners(unittest.TestCase):
    def test_a_good_quad_passes(self):
        self.assertEqual(validate_corners(trapezoid(), FRAME_SIZE), (True, "ok"))

    def test_an_unknown_frame_size_is_refused(self):
        for size in [(0, 360), (640, 0), (-1, -1)]:
            with self.subTest(size=size):
                self.assertEqual(
                    validate_corners(trapezoid(), size), (False, "unknown_frame_size")
                )

    def test_a_non_finite_corner_is_refused(self):
        quad = PitchCorners(
            (float("nan"), 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)
        )
        self.assertEqual(validate_corners(quad, FRAME_SIZE), (False, "non_finite_corner"))

    def test_a_repeated_corner_is_refused(self):
        quad = PitchCorners(
            (100.0, 100.0), (100.0, 100.0), (300.0, 300.0), (100.0, 300.0)
        )
        self.assertEqual(validate_corners(quad, FRAME_SIZE), (False, "duplicate_corners"))

    def test_a_non_convex_quad_is_refused(self):
        quad = PitchCorners(
            (0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)
        )
        self.assertEqual(validate_corners(quad, FRAME_SIZE), (False, "non_convex_corners"))

    def test_a_quad_off_the_picture_is_refused(self):
        quad = order_corners(
            [(-40.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
        )
        self.assertEqual(
            validate_corners(quad, FRAME_SIZE), (False, "corners_outside_frame")
        )

    def test_a_handful_of_pixels_is_refused(self):
        quad = order_corners(
            [(100.0, 100.0), (104.0, 100.0), (104.0, 104.0), (100.0, 104.0)]
        )
        self.assertEqual(
            validate_corners(quad, FRAME_SIZE), (False, "pitch_quad_too_small")
        )

    def test_a_full_frame_quad_is_refused(self):
        # A mask that covers the whole picture is a segmentation failure.
        quad = order_corners(
            [(0.0, 0.0), (640.0, 0.0), (640.0, 360.0), (0.0, 360.0)]
        )
        self.assertEqual(
            validate_corners(quad, FRAME_SIZE), (False, "pitch_quad_too_large")
        )


class TestCornersInFrame(unittest.TestCase):
    def test_the_rectangle_is_inside(self):
        self.assertTrue(corners_in_frame(rectangle(), FRAME_SIZE))

    def test_a_few_pixels_past_the_edge_are_tolerated(self):
        # The slack is 2% of the frame: 12.8 px across, 7.2 px down.
        quad = order_corners(
            [(-5.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
        )
        self.assertTrue(corners_in_frame(quad, FRAME_SIZE))

    def test_well_past_the_edge_is_not(self):
        quad = order_corners(
            [(-100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
        )
        self.assertFalse(corners_in_frame(quad, FRAME_SIZE))

    def test_an_unknown_frame_size_is_not_inside(self):
        self.assertFalse(corners_in_frame(rectangle(), (0, 0)))


class TestPolygonArea(unittest.TestCase):
    def test_the_rectangle(self):
        self.assertAlmostEqual(polygon_area(rectangle()), 40000.0)

    def test_the_trapezoid(self):
        # (200 + 240) / 2 * 200 pixels.
        self.assertAlmostEqual(polygon_area(trapezoid()), 44000.0)

    def test_the_area_does_not_depend_on_the_winding(self):
        reversed_quad = order_corners(list(reversed(RECTANGLE_POINTS)))
        self.assertAlmostEqual(polygon_area(reversed_quad), 40000.0)


class TestCornersFromBox(unittest.TestCase):
    def test_the_far_edge_is_narrower_than_the_near_edge(self):
        corners = corners_from_box(200.0, 60.0, 440.0, 340.0)
        inset = 0.08 * 240.0
        self.assertEqual(corners.top_left, (200.0 + inset, 60.0))
        self.assertEqual(corners.top_right, (440.0 - inset, 60.0))
        self.assertEqual(corners.bottom_right, (440.0, 340.0))
        self.assertEqual(corners.bottom_left, (200.0, 340.0))
        far_width = corners.top_right[0] - corners.top_left[0]
        near_width = corners.bottom_right[0] - corners.bottom_left[0]
        self.assertLess(far_width, near_width)

    def test_the_inset_is_a_fraction_of_the_box_width(self):
        corners = corners_from_box(0.0, 0.0, 100.0, 50.0, top_inset=0.2)
        self.assertEqual(corners.top_left, (20.0, 0.0))
        self.assertEqual(corners.top_right, (80.0, 0.0))

    def test_the_derived_corners_are_in_the_canonical_order(self):
        corners = corners_from_box(200.0, 60.0, 440.0, 340.0)
        self.assertEqual(corners, order_corners(corners.points()))

    def test_a_box_with_no_area_is_refused(self):
        for box in [
            (10.0, 10.0, 10.0, 50.0),
            (10.0, 10.0, 50.0, 10.0),
            (50.0, 10.0, 10.0, 50.0),
        ]:
            with self.subTest(box=box):
                with self.assertRaises(CalibrationError):
                    corners_from_box(*box)

    def test_a_non_finite_box_is_refused(self):
        with self.assertRaises(CalibrationError):
            corners_from_box(0.0, 0.0, float("nan"), 50.0)

    def test_an_inset_that_cannot_be_honoured_is_refused(self):
        for inset in (-0.1, 0.5, 0.9, 1.0):
            with self.subTest(inset=inset):
                with self.assertRaises(CalibrationError) as ctx:
                    corners_from_box(0.0, 0.0, 100.0, 50.0, top_inset=inset)
                self.assertIn("top_inset", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

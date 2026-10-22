"""
test_analytics_trajectory.py — Pixel-space analytics with hand-computed answers.

The expected numbers are arithmetic, not captures: the 3-4-5 triangle makes a
5-pixel step, two such steps make a 10-pixel path, and a 10 fps timebase turns
that into 50 px/s. The tests also pin the distinction the module exists to
preserve: no points is unavailable, one point is a genuine zero, and a jump
across a gap is reported separately instead of inflating the path.
"""

from __future__ import annotations

import math
import unittest

from app.analytics import (
    AnalyticsInputError,
    summarise_trajectory,
)
from app.analytics.trajectory import POINT_SOURCES, TrajectoryLike
from tests.shots_helpers import Point, Trajectory


class TestPathAndTime(unittest.TestCase):
    def test_two_known_steps(self) -> None:
        trajectory = Trajectory(
            [Point(1, 0, 0), Point(2, 3, 4), Point(3, 6, 8)]
        )
        summary = summarise_trajectory(trajectory, fps=10)

        self.assertEqual(summary.point_count, 3)
        self.assertEqual(summary.detected_count, 3)
        self.assertEqual(summary.predicted_count, 0)
        self.assertEqual(summary.frame_start, 1)
        self.assertEqual(summary.frame_end, 3)
        self.assertAlmostEqual(summary.duration_s or 0.0, 0.2)
        self.assertAlmostEqual(summary.path_length_px or 0.0, 10.0)
        self.assertAlmostEqual(summary.detected_path_length_px or 0.0, 10.0)
        self.assertAlmostEqual(summary.net_displacement_px or 0.0, 10.0)
        self.assertAlmostEqual(summary.mean_speed_px_s or 0.0, 50.0)
        self.assertAlmostEqual(summary.detected_mean_speed_px_s or 0.0, 50.0)
        self.assertEqual(summary.segments, 2)
        self.assertEqual(summary.adjacent_segments, 2)
        self.assertEqual(summary.unobserved_segments, 0)
        self.assertIsNone(summary.unobserved_span_px)
        self.assertEqual(summary.gaps, ())
        self.assertTrue(summary.continuous)
        self.assertTrue(summary.time_available)
        self.assertEqual(summary.units, "pixel")

    def test_no_fps_leaves_time_metrics_unavailable(self) -> None:
        trajectory = Trajectory([Point(1, 0, 0), Point(2, 3, 4)])
        summary = summarise_trajectory(trajectory)

        self.assertAlmostEqual(summary.path_length_px or 0.0, 5.0)
        self.assertIsNone(summary.duration_s)
        self.assertIsNone(summary.mean_speed_px_s)
        self.assertFalse(summary.time_available)

    def test_one_point_is_a_genuine_zero_not_missing(self) -> None:
        summary = summarise_trajectory(Trajectory([Point(7, 10, 20)]), fps=10)

        self.assertEqual(summary.point_count, 1)
        self.assertEqual(summary.path_length_px, 0.0)
        self.assertEqual(summary.detected_path_length_px, 0.0)
        self.assertEqual(summary.net_displacement_px, 0.0)
        self.assertEqual(summary.duration_s, 0.0)
        self.assertIsNone(summary.mean_speed_px_s)
        self.assertIsNone(summary.detected_mean_speed_px_s)
        self.assertTrue(summary.time_available)
        self.assertTrue(summary.continuous)

    def test_identical_consecutive_positions_have_zero_speed(self) -> None:
        summary = summarise_trajectory(
            Trajectory([Point(1, 10, 10), Point(2, 10, 10)]),
            fps=25,
        )
        self.assertEqual(summary.path_length_px, 0.0)
        self.assertEqual(summary.mean_speed_px_s, 0.0)
        self.assertEqual(summary.net_displacement_px, 0.0)

    def test_empty_trajectory_is_unavailable(self) -> None:
        summary = summarise_trajectory(Trajectory([]), fps=25)

        self.assertEqual(summary.point_count, 0)
        self.assertFalse(summary.available)
        self.assertIsNone(summary.path_length_px)
        self.assertIsNone(summary.detected_path_length_px)
        self.assertIsNone(summary.net_displacement_px)
        self.assertIsNone(summary.duration_s)
        self.assertIsNone(summary.mean_speed_px_s)
        self.assertEqual(summary.gaps, ())


class TestGaps(unittest.TestCase):
    def test_jump_across_a_gap_is_not_path_length(self) -> None:
        summary = summarise_trajectory(
            Trajectory([Point(1, 0, 0), Point(5, 3, 4)]),
            fps=10,
        )

        self.assertIsNone(summary.path_length_px)
        self.assertIsNone(summary.detected_path_length_px)
        self.assertIsNone(summary.mean_speed_px_s)
        self.assertEqual(summary.unobserved_segments, 1)
        self.assertAlmostEqual(summary.unobserved_span_px or 0.0, 5.0)
        self.assertEqual(summary.segments, 1)
        self.assertEqual(summary.adjacent_segments, 0)
        self.assertEqual(summary.gaps, ((2, 4),))
        self.assertFalse(summary.continuous)
        self.assertAlmostEqual(summary.duration_s or 0.0, 0.4)
        # Displacement is still the first-to-last straight line.
        self.assertAlmostEqual(summary.net_displacement_px or 0.0, 5.0)

    def test_adjacent_steps_survive_alongside_a_gap(self) -> None:
        summary = summarise_trajectory(
            Trajectory([Point(1, 0, 0), Point(2, 3, 4), Point(5, 3, 4)]),
            fps=10,
        )

        self.assertAlmostEqual(summary.path_length_px or 0.0, 5.0)
        self.assertEqual(summary.adjacent_segments, 1)
        self.assertEqual(summary.unobserved_segments, 1)
        # The two points across the gap are at the same place: a genuine zero,
        # not a missing measurement.
        self.assertEqual(summary.unobserved_span_px, 0.0)
        # The gap does not dilute the observed 5 px in 1/10 s.
        self.assertAlmostEqual(summary.mean_speed_px_s or 0.0, 50.0)

    def test_multiple_gaps_are_reported_in_order(self) -> None:
        summary = summarise_trajectory(
            Trajectory([Point(1, 0, 0), Point(4, 1, 0), Point(9, 2, 0)]),
            fps=10,
        )
        self.assertEqual(summary.gaps, ((2, 3), (5, 8)))
        self.assertEqual(summary.unobserved_segments, 2)


class TestProvenance(unittest.TestCase):
    def test_predicted_points_do_not_count_as_detected(self) -> None:
        summary = summarise_trajectory(
            Trajectory(
                [
                    Point(1, 0, 0, "predicted"),
                    Point(2, 3, 4, "predicted"),
                    Point(3, 6, 8, "predicted"),
                ]
            ),
            fps=10,
        )
        self.assertEqual(summary.detected_count, 0)
        self.assertEqual(summary.predicted_count, 3)
        self.assertIsNone(summary.detected_path_length_px)
        self.assertIsNone(summary.detected_mean_speed_px_s)
        self.assertAlmostEqual(summary.path_length_px or 0.0, 10.0)

    def test_detected_distance_ignores_predicted_points(self) -> None:
        # Detected at frames 1 and 2 (a 5 px adjacent step), predicted at 3,
        # detected again at 4. The detected subset is frames 1, 2, 4: the step
        # from 2 to 4 spans a gap and must not be counted as observed motion.
        summary = summarise_trajectory(
            Trajectory(
                [
                    Point(1, 0, 0, "detected"),
                    Point(2, 3, 4, "detected"),
                    Point(3, 9, 9, "predicted"),
                    Point(4, 6, 8, "detected"),
                ]
            ),
            fps=10,
        )

        self.assertAlmostEqual(summary.detected_path_length_px or 0.0, 5.0)
        self.assertAlmostEqual(summary.detected_mean_speed_px_s or 0.0, 50.0)
        self.assertEqual(summary.detected_count, 3)
        self.assertEqual(summary.predicted_count, 1)
        self.assertAlmostEqual(
            summary.path_length_px or 0.0,
            5.0 + math.hypot(6, 5) + math.hypot(3, 1),
        )


class TestAcceptedShapes(unittest.TestCase):
    def test_a_bare_sequence_of_points(self) -> None:
        summary = summarise_trajectory([Point(1, 0, 0), Point(2, 3, 4)], fps=10)
        self.assertAlmostEqual(summary.path_length_px or 0.0, 5.0)

    def test_mapping_documents_round_trip(self) -> None:
        document = {
            "points": [
                {"frame": 1, "x": 0, "y": 0, "source": "detected"},
                {"frame": 2, "x": 3, "y": 4, "source": "detected"},
            ]
        }
        summary = summarise_trajectory(document, fps=10)
        self.assertAlmostEqual(summary.path_length_px or 0.0, 5.0)
        self.assertAlmostEqual(summary.mean_speed_px_s or 0.0, 50.0)

    def test_structural_protocol_matches_a_part1_shaped_object(self) -> None:
        trajectory = Trajectory([Point(1, 0, 0), Point(2, 3, 4)])
        self.assertIsInstance(trajectory, TrajectoryLike)
        self.assertEqual(POINT_SOURCES, ("detected", "predicted"))

    def test_as_dict_serialises_gaps_as_lists(self) -> None:
        summary = summarise_trajectory(
            Trajectory([Point(1, 0, 0), Point(4, 3, 4)]),
            fps=10,
        )
        document = summary.as_dict()
        self.assertEqual(document["gaps"], [[2, 3]])
        self.assertEqual(document["units"], "pixel")
        self.assertFalse(document["continuous"])


class TestRejectedInputs(unittest.TestCase):
    def assert_rejected(self, trajectory: object, fps: object = 10) -> None:
        with self.assertRaises(AnalyticsInputError):
            summarise_trajectory(trajectory, fps=fps)  # type: ignore[arg-type]

    def test_duplicate_frames(self) -> None:
        self.assert_rejected(Trajectory([Point(1, 0, 0), Point(1, 2, 2)]))

    def test_decreasing_frames(self) -> None:
        self.assert_rejected(Trajectory([Point(2, 0, 0), Point(1, 2, 2)]))

    def test_frame_zero(self) -> None:
        self.assert_rejected(Trajectory([Point(0, 0, 0)]))

    def test_non_integer_frame(self) -> None:
        self.assert_rejected(Trajectory([{"frame": 1.5, "x": 0, "y": 0, "source": "detected"}]))

    def test_unknown_source(self) -> None:
        self.assert_rejected(Trajectory([Point(1, 0, 0, "guessed")]))

    def test_missing_field(self) -> None:
        self.assert_rejected(Trajectory([{"frame": 1, "x": 0, "source": "detected"}]))

    def test_non_finite_coordinates(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assert_rejected(Trajectory([Point(1, value, 0)]))

    def test_non_numeric_coordinates(self) -> None:
        for value in ("0", None, True):
            with self.subTest(value=value):
                self.assert_rejected(Trajectory([Point(1, value, 0)]))  # type: ignore[arg-type]

    def test_invalid_fps(self) -> None:
        trajectory = Trajectory([Point(1, 0, 0)])
        for value in (0, -1, float("nan"), float("inf"), "10", True):
            with self.subTest(value=value):
                self.assert_rejected(trajectory, fps=value)

    def test_none_and_string_are_not_trajectories(self) -> None:
        self.assert_rejected(None)
        self.assert_rejected("points")

    def test_mapping_without_points_key(self) -> None:
        self.assert_rejected({})


if __name__ == "__main__":
    unittest.main()

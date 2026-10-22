"""
test_analytics_pitch.py — Metre-space analytics, and the units gate.

The numbers are synthetic and deliberately simple: 10 m in 0.2 s is 50 m/s,
which is 180 km/h. That is fast for a cricket ball, but this test is about the
arithmetic and the units, not about whether the delivery was plausible.

The important assertions are the refusals. A pitch-fraction trajectory, a
missing units field, or a point that disagrees with its trajectory must raise,
because a "speed" derived from a fraction of an unknown pitch is not a slower
measurement — it is a different quantity wearing the same name.
"""

from __future__ import annotations

import unittest

from app.analytics import (
    AnalyticsInputError,
    PitchPointLike,
    PitchTrajectoryLike,
    summarise_pitch_trajectory,
)
from app.analytics.pitch_plane import MPS_TO_KMH, PITCH_UNITS
from tests.shots_helpers import PitchPoint, PitchTrajectory


class TestMetreArithmetic(unittest.TestCase):
    def test_known_ten_metre_step(self) -> None:
        trajectory = PitchTrajectory(
            [PitchPoint(1, 0, 0), PitchPoint(2, 0, 10), PitchPoint(3, 0, 20)]
        )
        summary = summarise_pitch_trajectory(trajectory, fps=10)

        self.assertEqual(summary.point_count, 3)
        self.assertEqual(summary.detected_count, 3)
        self.assertAlmostEqual(summary.duration_s or 0.0, 0.2)
        self.assertAlmostEqual(summary.path_length_m or 0.0, 20.0)
        self.assertAlmostEqual(summary.detected_path_length_m or 0.0, 20.0)
        self.assertAlmostEqual(summary.net_displacement_m or 0.0, 20.0)
        self.assertAlmostEqual(summary.mean_speed_mps or 0.0, 100.0)
        self.assertAlmostEqual(summary.mean_speed_kmh or 0.0, 360.0)
        self.assertAlmostEqual(summary.detected_mean_speed_kmh or 0.0, 360.0)
        self.assertAlmostEqual(summary.mean_speed_kmh or 0.0, 100.0 * MPS_TO_KMH)
        self.assertEqual(summary.segments, 2)
        self.assertEqual(summary.adjacent_segments, 2)
        self.assertEqual(summary.unobserved_segments, 0)
        self.assertIsNone(summary.unobserved_span_m)
        self.assertTrue(summary.continuous)
        self.assertTrue(summary.time_available)
        self.assertEqual(summary.units, PITCH_UNITS)

    def test_two_dimensional_distance(self) -> None:
        summary = summarise_pitch_trajectory(
            PitchTrajectory([PitchPoint(1, 0, 0), PitchPoint(2, 3, 4)]),
            fps=10,
        )
        self.assertAlmostEqual(summary.path_length_m or 0.0, 5.0)
        self.assertAlmostEqual(summary.net_displacement_m or 0.0, 5.0)

    def test_no_fps_keeps_distances_and_drops_speeds(self) -> None:
        summary = summarise_pitch_trajectory(
            PitchTrajectory([PitchPoint(1, 0, 0), PitchPoint(2, 0, 10)])
        )
        self.assertAlmostEqual(summary.path_length_m or 0.0, 10.0)
        self.assertIsNone(summary.duration_s)
        self.assertIsNone(summary.mean_speed_mps)
        self.assertIsNone(summary.mean_speed_kmh)
        self.assertFalse(summary.time_available)

    def test_one_point_is_a_genuine_zero(self) -> None:
        summary = summarise_pitch_trajectory(PitchTrajectory([PitchPoint(4, 1, 2)]), fps=25)
        self.assertEqual(summary.path_length_m, 0.0)
        self.assertEqual(summary.detected_path_length_m, 0.0)
        self.assertEqual(summary.net_displacement_m, 0.0)
        self.assertEqual(summary.duration_s, 0.0)
        self.assertIsNone(summary.mean_speed_mps)
        self.assertTrue(summary.time_available)

    def test_empty_trajectory_is_unavailable(self) -> None:
        summary = summarise_pitch_trajectory(PitchTrajectory([]), fps=25)
        self.assertEqual(summary.point_count, 0)
        self.assertFalse(summary.available)
        self.assertIsNone(summary.path_length_m)
        self.assertIsNone(summary.net_displacement_m)
        self.assertIsNone(summary.mean_speed_kmh)

    def test_identical_positions_have_zero_speed(self) -> None:
        summary = summarise_pitch_trajectory(
            PitchTrajectory([PitchPoint(1, 2, 3), PitchPoint(2, 2, 3)]),
            fps=25,
        )
        self.assertEqual(summary.path_length_m, 0.0)
        self.assertEqual(summary.mean_speed_mps, 0.0)
        self.assertEqual(summary.mean_speed_kmh, 0.0)


class TestGaps(unittest.TestCase):
    def test_gap_span_is_reported_not_measured(self) -> None:
        summary = summarise_pitch_trajectory(
            PitchTrajectory([PitchPoint(1, 0, 0), PitchPoint(5, 0, 5)]),
            fps=10,
        )
        self.assertIsNone(summary.path_length_m)
        self.assertIsNone(summary.mean_speed_mps)
        self.assertAlmostEqual(summary.unobserved_span_m or 0.0, 5.0)
        self.assertEqual(summary.unobserved_segments, 1)
        self.assertEqual(summary.gaps, ((2, 4),))
        self.assertFalse(summary.continuous)
        self.assertAlmostEqual(summary.duration_s or 0.0, 0.4)

    def test_observed_steps_survive_a_gap(self) -> None:
        summary = summarise_pitch_trajectory(
            PitchTrajectory(
                [PitchPoint(1, 0, 0), PitchPoint(2, 0, 1), PitchPoint(6, 0, 9)]
            ),
            fps=10,
        )
        self.assertAlmostEqual(summary.path_length_m or 0.0, 1.0)
        self.assertEqual(summary.adjacent_segments, 1)
        self.assertEqual(summary.unobserved_segments, 1)
        self.assertAlmostEqual(summary.unobserved_span_m or 0.0, 8.0)
        self.assertAlmostEqual(summary.mean_speed_mps or 0.0, 10.0)


class TestUnitsGate(unittest.TestCase):
    def test_pitch_fraction_is_rejected(self) -> None:
        trajectory = PitchTrajectory(
            [PitchPoint(1, 0.1, 0.2), PitchPoint(2, 0.2, 0.3)],
            units="pitch_fraction",
        )
        with self.assertRaises(AnalyticsInputError) as caught:
            summarise_pitch_trajectory(trajectory, fps=25)
        self.assertIn("pitch_fraction", str(caught.exception))

    def test_missing_units_is_rejected(self) -> None:
        points = [
            {"frame": 1, "across": 0.0, "along": 0.0, "source": "detected"},
            {"frame": 2, "across": 1.0, "along": 1.0, "source": "detected"},
        ]
        with self.assertRaises(AnalyticsInputError) as caught:
            summarise_pitch_trajectory(points, fps=25)
        self.assertIn("does not declare units", str(caught.exception))

    def test_point_units_must_agree_with_the_trajectory(self) -> None:
        trajectory = PitchTrajectory(
            [PitchPoint(1, 0, 0), PitchPoint(2, 0, 1, units="pitch_fraction")],
            units="metre",
        )
        with self.assertRaises(AnalyticsInputError) as caught:
            summarise_pitch_trajectory(trajectory, fps=25)
        self.assertIn("point 1", str(caught.exception))

    def test_units_on_each_point_are_enough(self) -> None:
        points = [
            {"frame": 1, "across": 0.0, "along": 0.0, "source": "detected", "units": "metre"},
            {"frame": 2, "across": 0.0, "along": 2.0, "source": "detected", "units": "metre"},
        ]
        summary = summarise_pitch_trajectory(points, fps=10)
        self.assertAlmostEqual(summary.path_length_m or 0.0, 2.0)

    def test_trajectory_units_cover_points_that_omit_them(self) -> None:
        trajectory = {
            "units": "metre",
            "points": [
                {"frame": 1, "across": 0.0, "along": 0.0, "source": "detected"},
                {"frame": 2, "across": 0.0, "along": 2.0, "source": "detected"},
            ],
        }
        summary = summarise_pitch_trajectory(trajectory, fps=10)
        self.assertAlmostEqual(summary.path_length_m or 0.0, 2.0)

    def test_units_are_case_sensitive(self) -> None:
        trajectory = PitchTrajectory(
            [PitchPoint(1, 0, 0), PitchPoint(2, 1, 1)],
            units="Metre",
        )
        with self.assertRaises(AnalyticsInputError):
            summarise_pitch_trajectory(trajectory, fps=25)


class TestProvenance(unittest.TestCase):
    def test_outside_pitch_points_are_counted_and_kept(self) -> None:
        trajectory = PitchTrajectory(
            [
                PitchPoint(1, 0, 0, inside_pitch=True),
                PitchPoint(2, 0, 5, inside_pitch=False),
                PitchPoint(3, 0, 10, inside_pitch=False),
            ]
        )
        summary = summarise_pitch_trajectory(trajectory, fps=10)
        self.assertEqual(summary.outside_pitch_count, 2)
        self.assertAlmostEqual(summary.path_length_m or 0.0, 10.0)

    def test_predicted_points_are_excluded_from_detected_distance(self) -> None:
        trajectory = PitchTrajectory(
            [
                PitchPoint(1, 0, 0, "detected"),
                PitchPoint(2, 0, 1, "detected"),
                PitchPoint(3, 0, 6, "predicted"),
            ]
        )
        summary = summarise_pitch_trajectory(trajectory, fps=10)
        self.assertEqual(summary.detected_count, 2)
        self.assertEqual(summary.predicted_count, 1)
        self.assertAlmostEqual(summary.detected_path_length_m or 0.0, 1.0)
        self.assertAlmostEqual(summary.path_length_m or 0.0, 6.0)

    def test_structural_protocol_matches_a_part2_shaped_object(self) -> None:
        trajectory = PitchTrajectory([PitchPoint(1, 0, 0), PitchPoint(2, 1, 1)])
        self.assertIsInstance(trajectory, PitchTrajectoryLike)
        self.assertIsInstance(trajectory.points[0], PitchPointLike)

    def test_as_dict_declares_estimates(self) -> None:
        summary = summarise_pitch_trajectory(
            PitchTrajectory([PitchPoint(1, 0, 0), PitchPoint(2, 0, 1)]),
            fps=10,
        )
        document = summary.as_dict()
        self.assertTrue(document["speed_is_estimate"])
        self.assertFalse(document["speed_is_calibrated"])
        self.assertEqual(document["units"], "metre")
        self.assertEqual(document["outside_pitch_count"], 0)


class TestRejectedInputs(unittest.TestCase):
    def assert_rejected(self, trajectory: object, fps: object = 25) -> None:
        with self.assertRaises(AnalyticsInputError):
            summarise_pitch_trajectory(trajectory, fps=fps)  # type: ignore[arg-type]

    def test_duplicate_frames(self) -> None:
        self.assert_rejected(
            PitchTrajectory([PitchPoint(1, 0, 0), PitchPoint(1, 1, 1)])
        )

    def test_decreasing_frames(self) -> None:
        self.assert_rejected(
            PitchTrajectory([PitchPoint(2, 0, 0), PitchPoint(1, 1, 1)])
        )

    def test_frame_zero(self) -> None:
        self.assert_rejected(PitchTrajectory([PitchPoint(0, 0, 0)]))

    def test_unknown_source(self) -> None:
        self.assert_rejected(PitchTrajectory([PitchPoint(1, 0, 0, source="guessed")]))

    def test_non_finite_coordinates(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assert_rejected(
                    PitchTrajectory([PitchPoint(1, value, 0)])
                )

    def test_non_numeric_coordinates(self) -> None:
        for value in ("0", None, True):
            with self.subTest(value=value):
                self.assert_rejected(
                    PitchTrajectory([PitchPoint(1, value, 0)])  # type: ignore[arg-type]
                )

    def test_non_boolean_inside_pitch(self) -> None:
        self.assert_rejected(
            PitchTrajectory([PitchPoint(1, 0, 0, inside_pitch=1)])  # type: ignore[arg-type]
        )

    def test_invalid_fps(self) -> None:
        trajectory = PitchTrajectory([PitchPoint(1, 0, 0)])
        for value in (0, -1, float("nan"), float("inf"), "25", True):
            with self.subTest(value=value):
                self.assert_rejected(trajectory, fps=value)

    def test_missing_points(self) -> None:
        self.assert_rejected({"units": "metre"})
        self.assert_rejected(None)

    def test_non_iterable_points(self) -> None:
        self.assert_rejected({"units": "metre", "points": 3})


if __name__ == "__main__":
    unittest.main()

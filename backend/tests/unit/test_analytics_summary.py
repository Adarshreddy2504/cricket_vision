"""
test_analytics_summary.py — The roll-up, and the map of what is missing.

The roll-up is the easiest place to lose honesty: a missing shot becomes an
absent key, a missing fps silently drops a speed, and the document looks
complete because every field it has is filled. These tests therefore assert on
`unavailable` as carefully as on the numbers, and on the difference between a
delivery that was never given a trajectory and one whose trajectory was a
genuine zero-length observation.
"""

from __future__ import annotations

import unittest

from app.analytics import (
    AnalyticsInputError,
    summarise_analysis,
    summarise_delivery,
)
from app.schemas.analytics import DeliveryAnalytics
from tests.shots_helpers import (
    PitchPoint,
    PitchTrajectory,
    Point,
    Trajectory,
    shot_failed,
    shot_ok,
)


def straight_trajectory() -> Trajectory:
    return Trajectory([Point(1, 0, 0), Point(2, 3, 4)])


def straight_pitch() -> PitchTrajectory:
    return PitchTrajectory([PitchPoint(1, 0, 0), PitchPoint(2, 0, 1)])


class TestSummariseDelivery(unittest.TestCase):
    def test_everything_present_leaves_nothing_unavailable(self) -> None:
        delivery = summarise_delivery(
            "d-1",
            shot=shot_ok("Cover", 0.8),
            trajectory=straight_trajectory(),
            fps=10,
            pitch_trajectory=straight_pitch(),
        )

        self.assertEqual(delivery.delivery_id, "d-1")
        self.assertEqual(delivery.unavailable, {})
        self.assertTrue(delivery.has_shot)
        self.assertTrue(delivery.has_trajectory)
        self.assertTrue(delivery.has_pitch)
        self.assertAlmostEqual(delivery.speed_kmh or 0.0, 36.0)
        self.assertAlmostEqual(delivery.trajectory.path_length_px or 0.0, 5.0)
        self.assertAlmostEqual(delivery.pitch.path_length_m or 0.0, 1.0)

    def test_nothing_supplied_explains_every_absent_field(self) -> None:
        delivery = summarise_delivery(1)

        self.assertEqual(
            set(delivery.unavailable),
            {"shot", "trajectory", "pitch", "time"},
        )
        self.assertIsNone(delivery.shot)
        self.assertIsNone(delivery.trajectory)
        self.assertIsNone(delivery.pitch)
        self.assertFalse(delivery.has_shot)
        self.assertFalse(delivery.has_trajectory)
        self.assertFalse(delivery.has_pitch)

    def test_failed_shot_is_kept_with_its_reason(self) -> None:
        delivery = summarise_delivery(
            1,
            shot=shot_failed("model_unavailable", "weights missing"),
        )
        self.assertIsNotNone(delivery.shot)
        self.assertFalse(delivery.has_shot)
        self.assertIn("model_unavailable", delivery.unavailable["shot"])
        self.assertIn("weights missing", delivery.unavailable["shot"])

    def test_invalid_trajectory_is_recorded_not_hidden(self) -> None:
        delivery = summarise_delivery(
            1,
            trajectory=Trajectory([Point(2, 0, 0), Point(1, 1, 1)]),
            fps=10,
        )
        self.assertIsNone(delivery.trajectory)
        self.assertIn("out of order", delivery.unavailable["trajectory"])

    def test_fractional_pitch_trajectory_is_refused(self) -> None:
        delivery = summarise_delivery(
            1,
            pitch_trajectory=PitchTrajectory(
                [PitchPoint(1, 0, 0), PitchPoint(2, 0.1, 0.2)],
                units="pitch_fraction",
            ),
            fps=10,
        )
        self.assertIsNone(delivery.pitch)
        self.assertIn("physical distances require", delivery.unavailable["pitch"])

    def test_missing_fps_keeps_distances_and_drops_speeds(self) -> None:
        delivery = summarise_delivery(
            1,
            trajectory=straight_trajectory(),
            pitch_trajectory=straight_pitch(),
        )

        self.assertIn("time", delivery.unavailable)
        self.assertTrue(delivery.has_trajectory)
        self.assertTrue(delivery.has_pitch)
        self.assertAlmostEqual(delivery.trajectory.path_length_px or 0.0, 5.0)
        self.assertIsNone(delivery.trajectory.mean_speed_px_s)
        self.assertIsNone(delivery.pitch.mean_speed_kmh)
        self.assertIsNone(delivery.speed_kmh)

    def test_invalid_fps_removes_the_timed_fields_with_a_reason(self) -> None:
        delivery = summarise_delivery(
            1,
            trajectory=straight_trajectory(),
            pitch_trajectory=straight_pitch(),
            fps=0,
        )

        self.assertIsNone(delivery.trajectory)
        self.assertIsNone(delivery.pitch)
        self.assertIn("positive", delivery.unavailable["trajectory"])
        self.assertIn("positive", delivery.unavailable["pitch"])
        # fps was supplied, so "time" is not reported as missing: it was
        # rejected, which the field-level reasons already say.
        self.assertNotIn("time", delivery.unavailable)

    def test_empty_but_supplied_trajectory_is_not_unavailable(self) -> None:
        delivery = summarise_delivery(1, trajectory=Trajectory([]), fps=10)

        self.assertIsNotNone(delivery.trajectory)
        self.assertNotIn("trajectory", delivery.unavailable)
        self.assertFalse(delivery.has_trajectory)
        self.assertIsNone(delivery.trajectory.path_length_px)

    def test_one_point_is_a_zero_not_an_absence(self) -> None:
        delivery = summarise_delivery(
            1,
            trajectory=Trajectory([Point(1, 2, 3)]),
            fps=10,
        )

        self.assertNotIn("trajectory", delivery.unavailable)
        self.assertTrue(delivery.has_trajectory)
        self.assertEqual(delivery.trajectory.path_length_px, 0.0)
        self.assertEqual(delivery.trajectory.duration_s, 0.0)

    def test_as_dict_keeps_the_unavailable_map(self) -> None:
        document = summarise_delivery(1, shot=shot_failed()).as_dict()
        self.assertIsNone(document["trajectory"])
        self.assertIn("shot", document["unavailable"])


class TestSummariseAnalysis(unittest.TestCase):
    def _delivery(self, delivery_id: int, **kwargs: object) -> DeliveryAnalytics:
        return summarise_delivery(delivery_id, **kwargs)

    def test_rolls_up_deliveries_without_counting_absent_shots(self) -> None:
        deliveries = [
            self._delivery(
                1,
                shot=shot_ok("Cover", 0.9),
                trajectory=straight_trajectory(),
                pitch_trajectory=straight_pitch(),
                fps=10,
            ),
            self._delivery(
                2,
                shot=shot_ok("Cover", 0.7),
                trajectory=straight_trajectory(),
                fps=10,
            ),
            self._delivery(3),
        ]
        summary = summarise_analysis(deliveries)

        self.assertEqual(summary.delivery_count, 3)
        self.assertEqual(summary.with_shot, 2)
        self.assertEqual(summary.with_trajectory, 2)
        self.assertEqual(summary.with_pitch, 1)
        self.assertEqual(summary.with_speed_kmh, 1)

        self.assertEqual(summary.shot_distribution.total, 2)
        self.assertEqual(summary.shot_distribution.classified, 2)
        self.assertEqual(summary.shot_distribution.counts["Cover"], 2)
        self.assertAlmostEqual(summary.shot_distribution.shares["Cover"], 1.0)

    def test_failed_shot_is_counted_as_unclassified(self) -> None:
        summary = summarise_analysis(
            [self._delivery(1, shot=shot_failed("model_unavailable", "missing"))]
        )

        self.assertEqual(summary.delivery_count, 1)
        self.assertEqual(summary.with_shot, 0)
        self.assertEqual(summary.shot_distribution.unclassified, 1)
        self.assertEqual(
            summary.shot_distribution.unclassified_reasons["model_unavailable"],
            1,
        )

    def test_empty_analysis_is_zeroed_not_missing(self) -> None:
        summary = summarise_analysis([])
        self.assertEqual(summary.delivery_count, 0)
        self.assertEqual(summary.with_shot, 0)
        self.assertEqual(summary.shot_distribution.total, 0)
        self.assertFalse(summary.shot_distribution.available)
        self.assertEqual(summary.deliveries, ())

    def test_generator_input(self) -> None:
        summary = summarise_analysis(
            self._delivery(index, shot=shot_ok("Pull", 0.5)) for index in range(3)
        )
        self.assertEqual(summary.delivery_count, 3)
        self.assertEqual(summary.shot_distribution.counts["Pull"], 3)

    def test_none_is_rejected(self) -> None:
        with self.assertRaises(AnalyticsInputError):
            summarise_analysis(None)  # type: ignore[arg-type]

    def test_as_dict_is_a_complete_document(self) -> None:
        summary = summarise_analysis(
            [
                self._delivery(
                    1,
                    shot=shot_ok("Hook", 0.6),
                    trajectory=straight_trajectory(),
                    fps=10,
                )
            ]
        )
        document = summary.as_dict()

        self.assertEqual(document["delivery_count"], 1)
        self.assertEqual(document["shot_distribution"]["counts"]["Hook"], 1)
        self.assertEqual(document["deliveries"][0]["delivery_id"], 1)
        self.assertEqual(
            document["deliveries"][0]["trajectory"]["available"],
            True,
        )
        self.assertNotIn("time", document["deliveries"][0]["unavailable"])
        self.assertNotIn("trajectory", document["deliveries"][0]["unavailable"])


if __name__ == "__main__":
    unittest.main()

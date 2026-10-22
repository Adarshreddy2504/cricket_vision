"""
test_shot_schemas.py — The invariants the typed records enforce on construction.

These are deliberately narrow tests. They do not check arithmetic (the analytics
tests do that); they check that a record which would mislead a consumer cannot
exist in the first place: a failed classification without a reason, a label
without its distribution, a score that is not the maximum, or a "repeated
frames" flag that disagrees with the counts.
"""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from app.schemas.analytics import (
    AnalysisSummary,
    DeliveryAnalytics,
    PitchPlaneSummary,
    ShotDistribution,
    TrajectorySummary,
)
from app.schemas.shots import ShotClassification, ShotPrediction
from app.shots.constants import SHOT_CLASSES
from tests.shots_helpers import shot_failed, shot_ok


def distribution(label: str = "Cover", score: float = 0.7) -> dict[str, float]:
    others = [name for name in SHOT_CLASSES if name != label]
    share = (1.0 - score) / len(others)
    probabilities = dict.fromkeys(others, share)
    probabilities[label] = score
    return probabilities


class TestShotPrediction(unittest.TestCase):
    def test_valid_prediction(self) -> None:
        prediction = ShotPrediction("Cover", 0.7, distribution("Cover", 0.7))
        self.assertEqual(prediction.label, "Cover")
        self.assertEqual(prediction.score, 0.7)
        self.assertAlmostEqual(sum(prediction.class_probabilities.values()), 1.0)

    def test_unknown_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ShotPrediction("Helicopter", 0.7, distribution())

    def test_score_must_be_the_maximum_probability(self) -> None:
        with self.assertRaises(ValueError):
            ShotPrediction("Cover", 0.2, distribution("Cover", 0.7))

    def test_missing_class_is_rejected(self) -> None:
        probabilities = distribution()
        del probabilities["Sweep"]
        with self.assertRaises(ValueError):
            ShotPrediction("Cover", 0.7, probabilities)

    def test_extra_class_is_rejected(self) -> None:
        probabilities = distribution()
        probabilities["Helicopter"] = 0.0
        with self.assertRaises(ValueError):
            ShotPrediction("Cover", 0.7, probabilities)

    def test_distribution_must_sum_to_one(self) -> None:
        probabilities = distribution()
        probabilities["Cover"] = 0.1
        with self.assertRaises(ValueError):
            ShotPrediction("Cover", 0.1, probabilities)

    def test_negative_probability_is_rejected(self) -> None:
        probabilities = distribution()
        probabilities["Sweep"] = -0.1
        probabilities["Cover"] = 1.1
        with self.assertRaises(ValueError):
            ShotPrediction("Cover", 1.1, probabilities)

    def test_score_outside_the_unit_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ShotPrediction("Cover", 1.5, distribution("Cover", 1.5))

    def test_non_numeric_probability_is_rejected(self) -> None:
        probabilities = distribution()
        probabilities["Sweep"] = "0.1"  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            ShotPrediction("Cover", 0.7, probabilities)

    def test_prediction_is_frozen(self) -> None:
        prediction = ShotPrediction("Cover", 0.7, distribution())
        with self.assertRaises(FrozenInstanceError):
            prediction.label = "Pull"  # type: ignore[misc]

    def test_as_dict_copies_the_distribution(self) -> None:
        prediction = ShotPrediction("Cover", 0.7, distribution())
        document = prediction.as_dict()
        document["class_probabilities"]["Cover"] = 0.0
        self.assertEqual(prediction.class_probabilities["Cover"], 0.7)


class TestShotClassification(unittest.TestCase):
    def test_valid_success(self) -> None:
        record = shot_ok()
        self.assertTrue(record.ok)
        self.assertEqual(record.status, "ok")
        self.assertEqual(record.as_dict()["ok"], True)

    def test_failed_status_requires_an_error(self) -> None:
        with self.assertRaises(ValueError):
            ShotClassification(status="model_unavailable")

    def test_failed_status_forbids_label_and_score(self) -> None:
        with self.assertRaises(ValueError):
            ShotClassification(status="invalid_input", error="bad", label="Cover")
        with self.assertRaises(ValueError):
            ShotClassification(status="invalid_input", error="bad", score=0.5)

    def test_failed_status_forbids_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            ShotClassification(
                status="inference_failed",
                error="boom",
                class_probabilities=distribution(),
            )

    def test_success_requires_label_and_score(self) -> None:
        with self.assertRaises(ValueError):
            ShotClassification(status="ok", score=0.5, class_probabilities=distribution())
        with self.assertRaises(ValueError):
            ShotClassification(status="ok", label="Cover", class_probabilities=distribution())

    def test_success_cannot_carry_an_error(self) -> None:
        with self.assertRaises(ValueError):
            shot_ok(error="should not be here")

    def test_unknown_status_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ShotClassification(status="great", error="no")

    def test_label_outside_vocabulary_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            shot_ok(label="Helicopter")

    def test_incomplete_distribution_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            shot_ok(class_probabilities={"Cover": 1.0})

    def test_frames_repeated_must_match_the_counts(self) -> None:
        with self.assertRaises(ValueError):
            shot_ok(unique_frames_sampled=10, frames_repeated=False)
        with self.assertRaises(ValueError):
            shot_ok(unique_frames_sampled=30, frames_repeated=True)

    def test_unique_frames_cannot_exceed_sampled_frames(self) -> None:
        with self.assertRaises(ValueError):
            shot_ok(frames_sampled=10, unique_frames_sampled=11, frames_repeated=False)

    def test_sampled_positions_must_align_with_the_frame_count(self) -> None:
        with self.assertRaises(ValueError):
            shot_ok(sampled_clip_frames=[0, 1, 2])

    def test_negative_sampled_positions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            shot_ok(sampled_clip_frames=[-1] * 30)

    def test_original_frames_must_align(self) -> None:
        with self.assertRaises(ValueError):
            shot_ok(sampled_original_frames=[1, 2, 3])

    def test_as_dict_covers_every_field(self) -> None:
        document = shot_failed().as_dict()
        self.assertEqual(document["status"], "model_unavailable")
        self.assertFalse(document["ok"])
        self.assertIsNone(document["label"])
        self.assertEqual(document["error"], "test reason")


class TestAnalyticsRecords(unittest.TestCase):
    def test_empty_trajectory_is_unavailable_not_zero(self) -> None:
        summary = TrajectorySummary()
        self.assertFalse(summary.available)
        self.assertIsNone(summary.path_length_px)
        self.assertTrue(summary.continuous)
        self.assertFalse(summary.time_available)

    def test_one_point_can_have_a_genuine_zero_duration(self) -> None:
        summary = TrajectorySummary(point_count=1, duration_s=0.0)
        self.assertTrue(summary.time_available)
        self.assertEqual(summary.duration_s, 0.0)

    def test_gaps_make_a_trajectory_non_continuous(self) -> None:
        summary = TrajectorySummary(point_count=3, gaps=((2, 4),))
        self.assertFalse(summary.continuous)

    def test_pitch_summary_is_always_an_estimate(self) -> None:
        summary = PitchPlaneSummary()
        self.assertTrue(summary.speed_is_estimate)
        self.assertFalse(summary.speed_is_calibrated)
        self.assertEqual(summary.units, "metre")

    def test_shot_distribution_availability(self) -> None:
        self.assertFalse(ShotDistribution().available)
        self.assertTrue(ShotDistribution(total=1, counts={"Cover": 1}).available)

    def test_delivery_properties(self) -> None:
        successful = DeliveryAnalytics(1, shot=shot_ok())
        self.assertTrue(successful.has_shot)
        self.assertFalse(successful.has_trajectory)
        self.assertFalse(successful.has_pitch)
        self.assertIsNone(successful.speed_kmh)

        failed = DeliveryAnalytics(2, shot=shot_failed())
        self.assertFalse(failed.has_shot)

        with_pitch = DeliveryAnalytics(
            3,
            pitch=PitchPlaneSummary(point_count=2, mean_speed_kmh=90.0),
        )
        self.assertTrue(with_pitch.has_pitch)
        self.assertEqual(with_pitch.speed_kmh, 90.0)

    def test_delivery_as_dict_handles_absent_parts(self) -> None:
        document = DeliveryAnalytics(7).as_dict()
        self.assertIsNone(document["shot"])
        self.assertIsNone(document["trajectory"])
        self.assertIsNone(document["pitch"])
        self.assertEqual(document["unavailable"], {})

    def test_analysis_summary_serialises_nested_documents(self) -> None:
        delivery = DeliveryAnalytics(
            1,
            shot=shot_ok(),
            trajectory=TrajectorySummary(point_count=1, path_length_px=0.0),
        )
        summary = AnalysisSummary(
            deliveries=(delivery,),
            shot_distribution=ShotDistribution(
                counts={"Cover": 1},
                classified=1,
                total=1,
                shares={"Cover": 1.0},
            ),
            delivery_count=1,
            with_shot=1,
            with_trajectory=1,
        )
        document = summary.as_dict()
        self.assertEqual(document["delivery_count"], 1)
        self.assertEqual(document["deliveries"][0]["shot"]["label"], "Cover")
        self.assertEqual(document["shot_distribution"]["counts"], {"Cover": 1})


if __name__ == "__main__":
    unittest.main()

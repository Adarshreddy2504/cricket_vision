"""
test_analytics_shots.py — Counting labels without inventing proportions.

The arithmetic is simple, so the tests concentrate on the places where an
aggregation can lie: a failed classification becoming a class, a share
computed over a zero denominator, a score borrowed from an unclassified
record, or an unknown label quietly joining the chart.
"""

from __future__ import annotations

import unittest

from app.analytics import AnalyticsInputError, summarise_shots
from app.shots.constants import SHOT_CLASSES
from tests.shots_helpers import shot_failed, shot_ok


class TestEmptyInput(unittest.TestCase):
    def test_no_records_is_available_but_empty(self) -> None:
        distribution = summarise_shots([])

        self.assertEqual(distribution.total, 0)
        self.assertEqual(distribution.classified, 0)
        self.assertEqual(distribution.unclassified, 0)
        self.assertFalse(distribution.available)
        self.assertEqual(distribution.shares, {})
        self.assertEqual(distribution.scored, 0)
        self.assertIsNone(distribution.mean_score)
        self.assertIsNone(distribution.min_score)
        self.assertIsNone(distribution.max_score)

    def test_every_class_is_present_with_a_zero_count(self) -> None:
        distribution = summarise_shots([])
        self.assertEqual(set(distribution.counts), set(SHOT_CLASSES))
        self.assertTrue(all(count == 0 for count in distribution.counts.values()))

    def test_none_is_rejected(self) -> None:
        with self.assertRaises(AnalyticsInputError):
            summarise_shots(None)  # type: ignore[arg-type]


class TestMixedRecords(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            shot_ok("Cover", 0.6),
            shot_ok("Cover", 0.8),
            shot_ok("Pull", 0.5),
            shot_failed("model_unavailable", "weights not installed"),
            shot_failed("invalid_input", "bad clip"),
        ]
        self.distribution = summarise_shots(self.records)

    def test_counts_only_classified_records(self) -> None:
        self.assertEqual(self.distribution.total, 5)
        self.assertEqual(self.distribution.classified, 3)
        self.assertEqual(self.distribution.unclassified, 2)
        self.assertEqual(self.distribution.counts["Cover"], 2)
        self.assertEqual(self.distribution.counts["Pull"], 1)
        self.assertEqual(self.distribution.counts["Sweep"], 0)
        self.assertEqual(sum(self.distribution.counts.values()), 3)

    def test_shares_use_the_classified_denominator(self) -> None:
        shares = self.distribution.shares
        self.assertEqual(set(shares), set(SHOT_CLASSES))
        self.assertAlmostEqual(shares["Cover"], 2 / 3)
        self.assertAlmostEqual(shares["Pull"], 1 / 3)
        self.assertEqual(shares["Sweep"], 0.0)
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    def test_unclassified_records_are_grouped_by_status(self) -> None:
        self.assertEqual(
            self.distribution.unclassified_reasons,
            {"model_unavailable": 1, "invalid_input": 1},
        )

    def test_score_statistics_use_only_classified_scores(self) -> None:
        self.assertEqual(self.distribution.scored, 3)
        self.assertAlmostEqual(
            self.distribution.mean_score or 0.0,
            (0.6 + 0.8 + 0.5) / 3,
        )
        self.assertEqual(self.distribution.min_score, 0.5)
        self.assertEqual(self.distribution.max_score, 0.8)

    def test_document_round_trips(self) -> None:
        document = self.distribution.as_dict()
        self.assertEqual(document["total"], 5)
        self.assertEqual(document["classified"], 3)
        self.assertEqual(document["unclassified_reasons"]["model_unavailable"], 1)
        self.assertEqual(document["counts"]["Cover"], 2)


class TestAcceptedShapes(unittest.TestCase):
    def test_plain_labels(self) -> None:
        distribution = summarise_shots(["Cover", "Cover", "Pull"])
        self.assertEqual(distribution.counts["Cover"], 2)
        self.assertEqual(distribution.counts["Pull"], 1)
        self.assertEqual(distribution.scored, 0)
        self.assertIsNone(distribution.mean_score)

    def test_mapping_records(self) -> None:
        distribution = summarise_shots(
            [
                {"status": "ok", "label": "Sweep", "score": 0.9},
                {"status": "ok", "label": "Sweep"},
                {"status": "model_unavailable", "error": "missing"},
            ]
        )
        self.assertEqual(distribution.counts["Sweep"], 2)
        self.assertEqual(distribution.classified, 2)
        self.assertEqual(distribution.scored, 1)
        self.assertAlmostEqual(distribution.mean_score or 0.0, 0.9)

    def test_label_without_status_is_classified(self) -> None:
        distribution = summarise_shots([{"label": "Hook"}])
        self.assertEqual(distribution.counts["Hook"], 1)
        self.assertEqual(distribution.classified, 1)

    def test_none_record_is_unclassified(self) -> None:
        distribution = summarise_shots([None])
        self.assertEqual(distribution.unclassified, 1)
        self.assertEqual(distribution.unclassified_reasons, {"no record": 1})

    def test_record_without_status_or_label_is_unclassified(self) -> None:
        distribution = summarise_shots([{}])
        self.assertEqual(distribution.unclassified_reasons, {"no status or label": 1})

    def test_failed_record_is_never_given_its_label(self) -> None:
        # The schema refuses to build this, but a stored document might contain
        # it; the aggregator must still refuse to count the label.
        distribution = summarise_shots(
            [{"status": "inference_failed", "label": "Cover", "error": "boom"}]
        )
        self.assertEqual(distribution.classified, 0)
        self.assertEqual(distribution.counts["Cover"], 0)
        self.assertEqual(distribution.unclassified_reasons, {"inference_failed": 1})

    def test_generator_is_consumed_once(self) -> None:
        records = (shot_ok("Flick") for _ in range(2))
        distribution = summarise_shots(records)
        self.assertEqual(distribution.counts["Flick"], 2)


class TestRejectedInputs(unittest.TestCase):
    def test_unknown_label_is_rejected(self) -> None:
        for value in ("Helicopter", "", "cover"):
            with self.subTest(value=value):
                with self.assertRaises(AnalyticsInputError) as caught:
                    summarise_shots([value])
                self.assertIn("not in the shot vocabulary", str(caught.exception))

    def test_non_string_label_is_rejected(self) -> None:
        with self.assertRaises(AnalyticsInputError):
            summarise_shots([{"status": "ok", "label": 3, "score": 0.5}])

    def test_ok_status_without_a_label_is_rejected(self) -> None:
        with self.assertRaises(AnalyticsInputError) as caught:
            summarise_shots([{"status": "ok", "score": 0.5}])
        self.assertIn("without a label", str(caught.exception))

    def test_invalid_scores_are_rejected(self) -> None:
        for value in (1.5, -0.1, "0.5", True):
            with self.subTest(value=value):
                with self.assertRaises(AnalyticsInputError):
                    summarise_shots([{"status": "ok", "label": "Cover", "score": value}])

    def test_absent_score_is_allowed(self) -> None:
        distribution = summarise_shots(
            [{"status": "ok", "label": "Cover", "score": None}]
        )
        self.assertEqual(distribution.classified, 1)
        self.assertEqual(distribution.scored, 0)

    def test_scores_on_unclassified_records_are_ignored(self) -> None:
        distribution = summarise_shots(
            [{"status": "invalid_input", "error": "bad", "score": 99}]
        )
        self.assertEqual(distribution.scored, 0)
        self.assertEqual(distribution.unclassified, 1)


if __name__ == "__main__":
    unittest.main()

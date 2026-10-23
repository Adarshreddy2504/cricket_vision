"""
Tracking service: the per-frame contract of the tracking module.

The service is the module's public entry point. It turns a stream of frames
into a trajectory with truthful provenance and stable track identifiers, and it
must fail loudly on bad input rather than returning a plausible-looking but
wrong answer.

What is pinned here:

  * Frames are numbered from 1, in decode order.
  * A detection becomes a "detected" point; a frame with no detection becomes
    a "predicted" point; a frame where the ball is lost produces NO point at
    all — nothing is invented to bridge a gap.
  * The track identifier is stable while the ball is being followed, including
    across a short occlusion, and changes when the ball is genuinely lost and
    re-acquired.
  * A detection refused by the association gate is not reported as a
    measurement.
  * Invalid input raises; it never returns an empty success.
"""

from __future__ import annotations

import unittest

import numpy as np

from app.core.constants import MAX_MISSED
from app.tracking.tracking_service import (
    DetectorUnavailableError,
    InvalidFrameError,
    TrackingService,
    VideoOpenError,
)
from tests.helpers import SequenceDetector, StubDetector, make_frame, make_video


class ConfidenceSequenceDetector:
    """Scripted detections with a per-frame confidence, including low boxes."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = 0
        self.available = True
        self.load_error = None

    def detect_for_association(self, _frame):
        self.calls += 1
        index = min(self.calls, len(self.rows)) - 1
        row = self.rows[index]
        if row is None:
            return None, None, None
        return row[0], row[1], row[2]

    def detect(self, _frame):
        x, y, _confidence = self.detect_for_association(_frame)
        return None if x is None else (x, y)


class TestPerFrameTracking(unittest.TestCase):
    """One frame in, one answer out."""

    def setUp(self):
        self.frame = make_frame(640, 360)

    def test_a_detection_becomes_a_detected_point(self):
        service = TrackingService(detector=StubDetector([(1, 10)]))
        outcome = service.process_frame(self.frame)
        self.assertEqual(outcome.source, "detected")
        self.assertIsNotNone(outcome.position)
        self.assertEqual(outcome.confidence, 0.9)
        self.assertIsNotNone(outcome.track_id)

    def test_frames_are_numbered_from_one(self):
        service = TrackingService(detector=StubDetector([(1, 3)]))
        numbers = [service.process_frame(self.frame).frame for _ in range(3)]
        self.assertEqual(numbers, [1, 2, 3])

    def test_an_explicit_frame_number_is_honoured(self):
        service = TrackingService(detector=StubDetector([(1, 1)]))
        outcome = service.process_frame(self.frame, frame_no=42)
        self.assertEqual(outcome.frame, 42)

    def test_a_frame_without_a_detection_yields_no_point(self):
        service = TrackingService(detector=StubDetector([]))
        outcome = service.process_frame(self.frame)
        self.assertIsNone(outcome.position)
        self.assertIsNone(outcome.source)

    def test_the_raw_detection_is_kept_alongside_the_filtered_position(self):
        service = TrackingService(detector=StubDetector([(1, 1)]))
        outcome = service.process_frame(self.frame)
        self.assertIsNotNone(outcome.detection)
        self.assertIsNotNone(outcome.position)


class TestProvenance(unittest.TestCase):
    """Detected, predicted, and the difference between them."""

    def test_a_missed_detection_becomes_a_predicted_point(self):
        detector = SequenceDetector([(100, 100), None, None])
        service = TrackingService(detector=detector)
        service.process_frame(make_frame(320, 180))
        outcome = service.process_frame(make_frame(320, 180))
        self.assertEqual(outcome.source, "predicted")
        self.assertIsNotNone(outcome.position)

    def test_a_predicted_point_carries_no_confidence(self):
        detector = SequenceDetector([(100, 100), None])
        service = TrackingService(detector=detector)
        service.process_frame(make_frame(320, 180))
        outcome = service.process_frame(make_frame(320, 180))
        self.assertIsNone(outcome.confidence)

    def test_a_refused_detection_is_not_reported_as_a_measurement(self):
        """A detection beyond the gate is a false positive, not a measurement."""
        detector = SequenceDetector([(100, 100), (900, 600)])
        service = TrackingService(detector=detector)
        service.process_frame(make_frame(320, 180))
        outcome = service.process_frame(make_frame(320, 180))
        self.assertTrue(outcome.gated_out)
        self.assertEqual(outcome.source, "predicted")
        self.assertIsNone(outcome.confidence)

    def test_nothing_is_invented_to_bridge_a_gap(self):
        detector = SequenceDetector([(100, 100)] + [None] * 20)
        service = TrackingService(detector=detector)
        for _ in range(21):
            service.process_frame(make_frame(320, 180))
        points = service.result.points
        # One real point, and then only the filter's own extrapolations while
        # it coasts. Once the ball is declared lost the frames carry NO point
        # at all — nothing is interpolated to bridge the hole.
        self.assertEqual(len(points), 1 + MAX_MISSED)
        self.assertEqual(points[0].source, "detected")
        self.assertEqual(sum(1 for p in points if p.source == "predicted"), MAX_MISSED)
        self.assertEqual(max(p.frame for p in points), 1 + MAX_MISSED)


class TestLowConfidenceAssociation(unittest.TestCase):
    """A low-confidence box may extend a flight, never create one."""

    def test_a_low_confidence_box_cannot_start_a_track(self):
        detector = ConfidenceSequenceDetector([(100, 100, 0.20)])
        service = TrackingService(detector=detector)
        outcome = service.process_frame(make_frame(320, 180))
        self.assertIsNone(outcome.position)
        self.assertIsNone(outcome.source)
        self.assertIsNone(outcome.detection)
        self.assertEqual(service.result.tracks, [])

    def test_a_low_confidence_box_extends_a_live_track(self):
        detector = ConfidenceSequenceDetector(
            [(100, 100, 0.90), (105, 101, 0.20)]
        )
        service = TrackingService(detector=detector)
        first = service.process_frame(make_frame(320, 180))
        second = service.process_frame(make_frame(320, 180))

        self.assertEqual(second.track_id, first.track_id)
        self.assertEqual(second.source, "detected")
        self.assertTrue(second.low_confidence)
        self.assertAlmostEqual(second.confidence, 0.20)
        self.assertEqual(service.result.tracks[0].low_confidence_count, 1)

    def test_a_low_confidence_box_outside_the_gate_is_refused(self):
        detector = ConfidenceSequenceDetector(
            [(100, 100, 0.90), (900, 600, 0.20)]
        )
        service = TrackingService(detector=detector)
        service.process_frame(make_frame(320, 180))
        outcome = service.process_frame(make_frame(320, 180))

        self.assertTrue(outcome.gated_out)
        self.assertEqual(outcome.source, "predicted")
        self.assertFalse(outcome.low_confidence)
        self.assertIsNone(outcome.confidence)

    def test_a_low_confidence_box_cannot_re_acquire_after_loss(self):
        detector = ConfidenceSequenceDetector(
            [(100, 100, 0.90)]
            + [None] * (MAX_MISSED + 1)
            + [(110, 100, 0.20)]
        )
        service = TrackingService(detector=detector)
        for _ in range(MAX_MISSED + 2):
            service.process_frame(make_frame(320, 180))
        lost = service.process_frame(make_frame(320, 180))
        self.assertIsNone(lost.position)
        self.assertIsNone(lost.track_id)


class TestTrackIdentity(unittest.TestCase):
    """Stable identifiers across an occlusion; a new one after a real loss."""

    def test_the_track_id_is_stable_while_the_ball_is_tracked(self):
        detector = SequenceDetector([(100, 100), (105, 100), (110, 100)])
        service = TrackingService(detector=detector)
        ids = [service.process_frame(make_frame(320, 180)).track_id for _ in range(3)]
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[1], ids[2])

    def test_the_track_id_survives_a_short_occlusion(self):
        detector = SequenceDetector([(100, 100)] + [None] * 3 + [(115, 100)])
        service = TrackingService(detector=detector)
        ids = []
        for _ in range(5):
            ids.append(service.process_frame(make_frame(320, 180)).track_id)
        self.assertEqual(len(set(ids)), 1)

    def test_a_new_track_id_is_issued_after_the_ball_is_lost(self):
        detector = SequenceDetector([(100, 100)] + [None] * (MAX_MISSED + 1) + [(900, 600)])
        service = TrackingService(detector=detector)
        first = service.process_frame(make_frame(320, 180)).track_id
        for _ in range(MAX_MISSED + 1):
            lost = service.process_frame(make_frame(320, 180))
        self.assertIsNone(lost.position)
        self.assertIsNone(lost.track_id)
        second = service.process_frame(make_frame(320, 180)).track_id
        self.assertNotEqual(first, second)

    def test_track_ids_increase_monotonically(self):
        detector = SequenceDetector(
            [(100, 100)]
            + [None] * (MAX_MISSED + 1)
            + [(900, 600)]
            + [None] * (MAX_MISSED + 1)
            + [(200, 200)]
        )
        service = TrackingService(detector=detector)
        ids = []
        for _ in range(2 * (MAX_MISSED + 1) + 3):
            outcome = service.process_frame(make_frame(320, 180))
            if outcome.track_id is not None:
                ids.append(outcome.track_id)
        self.assertEqual(ids, sorted(ids))
        # Three flights: the first ball, the one after it is lost and
        # re-acquired, and the one after that.
        self.assertEqual(len(set(ids)), 3)


class TestTrackRecords(unittest.TestCase):
    """The per-track summaries the service produces."""

    def test_a_track_records_its_frames_and_points(self):
        detector = SequenceDetector([(100, 100), (105, 100), (110, 100)])
        service = TrackingService(detector=detector)
        result = service.track_frames([make_frame(320, 180) for _ in range(3)])
        self.assertEqual(len(result.tracks), 1)
        track = result.tracks[0]
        self.assertEqual(track.start_frame, 1)
        self.assertEqual(track.end_frame, 3)
        self.assertEqual(len(track.points), 3)
        self.assertEqual(track.detected_count, 3)
        self.assertEqual(track.predicted_count, 0)

    def test_a_one_frame_track_is_not_confirmed(self):
        """A single false positive must not be presented as a flight."""
        detector = SequenceDetector([(100, 100)] + [None] * (MAX_MISSED + 1))
        service = TrackingService(detector=detector)
        result = service.track_frames(
            [make_frame(320, 180) for _ in range(MAX_MISSED + 2)]
        )
        self.assertEqual(len(result.tracks), 1)
        self.assertFalse(result.tracks[0].is_confirmed)
        self.assertEqual(result.confirmed_tracks, [])

    def test_a_repeated_track_is_confirmed(self):
        detector = SequenceDetector([(100 + i, 100) for i in range(5)])
        service = TrackingService(detector=detector)
        result = service.track_frames([make_frame(320, 180) for _ in range(5)])
        self.assertTrue(result.tracks[0].is_confirmed)
        self.assertEqual(len(result.confirmed_tracks), 1)


class TestResultCounts(unittest.TestCase):
    """The counters a consumer reads instead of counting the points."""

    def test_the_counts_match_the_points(self):
        detector = SequenceDetector([(100, 100), None, (105, 100)])
        service = TrackingService(detector=detector)
        result = service.track_frames([make_frame(320, 180) for _ in range(3)])
        self.assertEqual(result.frames_decoded, 3)
        self.assertEqual(result.detection_frames, 2)
        self.assertEqual(result.predicted_frames, 1)
        self.assertEqual(len(result.points), 3)

    def test_a_video_with_no_ball_produces_no_points_and_no_tracks(self):
        detector = SequenceDetector([None, None, None])
        service = TrackingService(detector=detector)
        result = service.track_frames([make_frame(320, 180) for _ in range(3)])
        self.assertEqual(result.points, [])
        self.assertEqual(result.tracks, [])
        self.assertEqual(result.frames_decoded, 3)

    def test_the_gaps_in_the_trajectory_are_recorded(self):
        # The gap has to outlast the coast window, otherwise the filter's own
        # extrapolations cover it and there is no gap to report.
        detector = SequenceDetector([(100, 100)] + [None] * (MAX_MISSED + 3) + [(900, 600)])
        service = TrackingService(detector=detector)
        result = service.track_frames(
            [make_frame(320, 180) for _ in range(MAX_MISSED + 5)]
        )
        self.assertEqual(result.trajectory.gaps, [[MAX_MISSED + 2, MAX_MISSED + 4]])

    def test_the_result_serialises_to_plain_data(self):
        detector = SequenceDetector([(100, 100), (105, 100)])
        service = TrackingService(detector=detector)
        result = service.track_frames([make_frame(320, 180) for _ in range(2)])
        document = result.as_dict()
        self.assertEqual(document["frames_decoded"], 2)
        self.assertEqual(len(document["points"]), 2)
        self.assertEqual(document["points"][0]["source"], "detected")
        self.assertIn("tracks", document)


class TestInputValidation(unittest.TestCase):
    """Bad input fails loudly."""

    def setUp(self):
        self.service = TrackingService(detector=StubDetector([(1, 1)]))

    def test_none_is_rejected(self):
        with self.assertRaises(InvalidFrameError):
            self.service.process_frame(None)

    def test_a_non_array_is_rejected(self):
        with self.assertRaises(InvalidFrameError):
            self.service.process_frame("not a frame")

    def test_a_greyscale_frame_is_rejected(self):
        with self.assertRaises(InvalidFrameError):
            self.service.process_frame(np.zeros((10, 10), dtype=np.uint8))

    def test_an_empty_frame_is_rejected(self):
        with self.assertRaises(InvalidFrameError):
            self.service.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_the_invalid_frame_error_is_also_a_value_error(self):
        with self.assertRaises(ValueError):
            self.service.process_frame(None)

    def test_a_missing_video_raises(self):
        service = TrackingService(detector=StubDetector([(1, 1)]))
        with self.assertRaises(VideoOpenError):
            service.track_video("no-such-video.mp4")


class TestDetectorAvailability(unittest.TestCase):
    """A detector with no usable model must not look like an empty frame."""

    def test_a_detector_without_weights_raises(self):
        from pathlib import Path

        from app.core.config import TrackingConfig

        config = TrackingConfig(model_path=Path("no-such-model.pt"))
        service = TrackingService(detector=None, config=config)
        with self.assertRaises(DetectorUnavailableError):
            service.process_frame(make_frame(320, 180))


class TestVideoTracking(unittest.TestCase):
    """The video path decodes and numbers frames itself."""

    def test_a_video_is_decoded_and_tracked(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = make_video(Path(tmp) / "clip.mp4", n_frames=8)
            detector = SequenceDetector([(100, 100)] * 8)
            service = TrackingService(detector=detector)
            result = service.track_video(path)

        self.assertEqual(result.frames_decoded, 8)
        self.assertEqual(len(result.points), 8)
        self.assertEqual(result.video["width"], 320)
        self.assertEqual(result.video["height"], 180)
        self.assertEqual(result.video["path"], str(path))


if __name__ == "__main__":
    unittest.main()

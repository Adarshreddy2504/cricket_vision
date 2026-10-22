"""
Ball detector: selection, filtering and coordinate scaling.

The detector is the only place a cricket-ball candidate is found. These tests
pin the four decisions it makes, because each of them silently corrupts every
downstream coordinate if it is wrong:

  1. Which box wins  — the highest-confidence one.
  2. Which boxes are refused — wrong class, or a bounding-box diagonal
     outside the plausible size range (sponsor logos and crowd noise).
  3. Which space the answer is in — ALWAYS full-resolution source pixels.
  4. What happens when there is no model, no box, or no valid frame.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from app.core.constants import (
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_MAX_DIAG_PX,
    DETECTOR_MIN_DIAG_PX,
    DETECTOR_RESIZE_H,
    DETECTOR_RESIZE_W,
)
from app.tracking.detector import BallDetector
from tests.helpers import StubYoloModel, make_frame


class TestDetectorSelection(unittest.TestCase):
    """Which of the model's boxes becomes the ball."""

    def test_no_boxes_yields_no_detection(self):
        detector = BallDetector(model=StubYoloModel([[]]))
        self.assertEqual(detector.detect_with_confidence(make_frame()), (None, None, None))

    def test_absent_boxes_attribute_yields_no_detection(self):
        """ultralytics returns `boxes = None` when nothing matched."""
        detector = BallDetector(model=StubYoloModel([None]))
        self.assertEqual(detector.detect_with_confidence(make_frame()), (None, None, None))

    def test_best_box_centre_is_returned(self):
        detector = BallDetector(model=StubYoloModel([[[10, 20, 30, 40, 0.80, 0]]]))
        # 640x360 is the inference size, so the scale-back is the identity and
        # the answer can be compared with the box directly.
        cx, cy, conf = detector.detect_with_confidence(make_frame(640, 360))
        self.assertEqual((cx, cy), (20, 30))
        self.assertEqual(conf, 0.80)

    def test_highest_confidence_box_wins(self):
        detector = BallDetector(
            model=StubYoloModel([[[0, 0, 10, 10, 0.40, 0], [20, 20, 40, 40, 0.90, 0]]])
        )
        cx, cy, conf = detector.detect_with_confidence(make_frame(640, 360))
        self.assertEqual((cx, cy), (30, 30))
        self.assertEqual(conf, 0.90)

    def test_non_ball_classes_are_ignored(self):
        """Only class 0 is the ball; a second class must never be picked."""
        detector = BallDetector(model=StubYoloModel([[[0, 0, 20, 20, 0.95, 1]]]))
        self.assertEqual(detector.detect_with_confidence(make_frame()), (None, None, None))

    def test_ball_class_is_preferred_over_a_higher_scoring_other_class(self):
        detector = BallDetector(
            model=StubYoloModel([[[0, 0, 20, 20, 0.99, 1], [30, 30, 50, 50, 0.60, 0]]])
        )
        cx, cy, conf = detector.detect_with_confidence(make_frame(640, 360))
        self.assertEqual((cx, cy), (40, 40))
        self.assertEqual(conf, 0.60)


class TestDetectorSizeFilter(unittest.TestCase):
    """Bounding boxes whose size cannot be a ball are refused."""

    def test_oversized_box_is_refused(self):
        """A box larger than the plausible maximum is crowd or a graphic."""
        big = [[0, 0, DETECTOR_MAX_DIAG_PX, DETECTOR_MAX_DIAG_PX, 0.95, 0]]
        detector = BallDetector(model=StubYoloModel([big]))
        self.assertEqual(detector.detect_with_confidence(make_frame()), (None, None, None))

    def test_undersized_box_is_refused(self):
        """A box too small to be a ball is noise."""
        # A square whose diagonal falls below the plausible minimum.
        side = max(1, DETECTOR_MIN_DIAG_PX // 2)
        tiny = [[0, 0, side, side, 0.95, 0]]
        detector = BallDetector(model=StubYoloModel([tiny]))
        self.assertEqual(detector.detect_with_confidence(make_frame()), (None, None, None))

    def test_the_size_filter_bounds_come_from_the_constants(self):
        """The filter's bounds are the configured ones, not hard-coded numbers."""
        detector = BallDetector(model=StubYoloModel([[[0, 0, 20, 20, 0.95, 0]]]))
        self.assertEqual(detector.min_diag_px, DETECTOR_MIN_DIAG_PX)
        self.assertEqual(detector.max_diag_px, DETECTOR_MAX_DIAG_PX)

    def test_a_refused_box_does_not_hide_a_valid_one(self):
        rows = [
            [0, 0, DETECTOR_MAX_DIAG_PX, DETECTOR_MAX_DIAG_PX, 0.99, 0],
            [10, 10, 20, 20, 0.50, 0],
        ]
        detector = BallDetector(model=StubYoloModel([rows]))
        cx, cy, conf = detector.detect_with_confidence(make_frame(640, 360))
        self.assertEqual((cx, cy), (15, 15))
        self.assertEqual(conf, 0.50)


class TestDetectorCoordinateSpace(unittest.TestCase):
    """
    The answer is ALWAYS in full-resolution source pixels.

    Every downstream consumer — the trajectory, the overlay, the analytics —
    works in source pixels, so a detector that returned resized coordinates
    would produce a ball that moves a fraction of its real speed.
    """

    def test_coordinates_are_scaled_back_to_the_source_resolution(self):
        detector = BallDetector(model=StubYoloModel([[[10, 20, 30, 40, 0.80, 0]]]))
        cx, cy, _ = detector.detect_with_confidence(make_frame(1280, 720))
        # 1280x720 resized to 640x360 is exactly a factor of two.
        self.assertEqual((cx, cy), (40, 60))

    def test_the_frame_is_resized_before_inference(self):
        model = StubYoloModel([[[10, 20, 30, 40, 0.80, 0]]])
        detector = BallDetector(model=model)
        detector.detect_with_confidence(make_frame(1280, 720))
        source = model.calls[0]["source"]
        self.assertEqual(source.shape[:2], (DETECTOR_RESIZE_H, DETECTOR_RESIZE_W))

    def test_non_square_scaling_is_applied_per_axis(self):
        detector = BallDetector(model=StubYoloModel([[[0, 0, 20, 20, 0.80, 0]]]))
        cx, cy, _ = detector.detect_with_confidence(make_frame(640, 360))
        # 640x360 -> 640x360 is the identity.
        self.assertEqual((cx, cy), (10, 10))

    def test_the_confidence_threshold_is_passed_to_the_model(self):
        model = StubYoloModel([[[10, 20, 30, 40, 0.80, 0]]])
        detector = BallDetector(model=model)
        detector.detect_with_confidence(make_frame())
        self.assertEqual(model.calls[0]["conf"], DETECTOR_CONF_THRESHOLD)

    def test_a_custom_confidence_threshold_is_passed_through(self):
        model = StubYoloModel([[[10, 20, 30, 40, 0.80, 0]]])
        detector = BallDetector(model=model, conf_threshold=0.55)
        detector.detect_with_confidence(make_frame())
        self.assertEqual(model.calls[0]["conf"], 0.55)


class TestDetectorPublicApi(unittest.TestCase):
    """The convenience wrapper and the failure modes."""

    def test_detect_returns_a_position_tuple(self):
        detector = BallDetector(model=StubYoloModel([[[10, 20, 30, 40, 0.80, 0]]]))
        self.assertEqual(detector.detect(make_frame(640, 360)), (20, 30))

    def test_detect_returns_none_when_nothing_was_found(self):
        detector = BallDetector(model=StubYoloModel([[]]))
        self.assertIsNone(detector.detect(make_frame()))

    def test_an_unavailable_detector_finds_nothing_rather_than_raising(self):
        detector = BallDetector(model_path=Path("does-not-exist.pt"))
        self.assertFalse(detector.available)
        self.assertIsNotNone(detector.load_error)
        self.assertEqual(detector.detect_with_confidence(make_frame()), (None, None, None))

    def test_a_loaded_detector_is_available(self):
        detector = BallDetector(model=StubYoloModel([[]]))
        self.assertTrue(detector.available)
        self.assertIsNone(detector.load_error)


class TestDetectorInputValidation(unittest.TestCase):
    """Invalid input fails loudly instead of producing a plausible number."""

    def setUp(self):
        self.detector = BallDetector(model=StubYoloModel([[[10, 20, 30, 40, 0.80, 0]]]))

    def test_none_is_rejected(self):
        with self.assertRaises(ValueError):
            self.detector.detect_with_confidence(None)

    def test_a_non_array_is_rejected(self):
        with self.assertRaises(ValueError):
            self.detector.detect_with_confidence("not a frame")

    def test_a_greyscale_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            self.detector.detect_with_confidence(np.zeros((10, 10), dtype=np.uint8))

    def test_an_empty_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            self.detector.detect_with_confidence(np.zeros((0, 0, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()

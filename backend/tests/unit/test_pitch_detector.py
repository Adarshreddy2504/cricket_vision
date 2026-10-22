"""
test_pitch_detector.py — frame validation, weight handling, and extraction.

The detectors are tested against an INJECTED model that implements the
ultralytics contract (see `tests/pitch_helpers.py`), so the extraction,
validation and selection logic under test is the real one while the
checkpoints stay external and untracked. Nothing here downloads weights.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import app.pitch.config as config
from app.pitch.calibration import CalibrationStatus, CornerMethod
from app.pitch.config import (
    PITCH_BOX_MODEL_FILENAME,
    PITCH_SEG_MODEL_FILENAME,
    PitchConfig,
    describe_weights,
    resolve_pitch_box_model_path,
    resolve_pitch_segmentation_model_path,
)
from app.pitch.detector import (
    DetectionStatus,
    PitchBoxDetector,
    PitchDetector,
    PitchSegmentationDetector,
    detection_to_calibration,
    validate_frame,
)
from app.pitch.errors import InvalidFrameError
from tests.pitch_helpers import (
    FRAME_SIZE,
    StubBoxes,
    StubMasks,
    StubModel,
    StubResult,
    StubSequenceModel,
    make_frame,
    pitch_mask_polygon,
)

# A plausible pitch-shaped detection box in a 640x360 frame.
BOX = (200.0, 60.0, 440.0, 340.0)
BOX_ROW = (*BOX, 0.88, 0)

# The corners the default mask (see pitch_mask_polygon) must produce.
MASK_CORNERS = ((270.0, 60.0), (370.0, 60.0), (400.0, 340.0), (240.0, 340.0))


def boxes_for(polygons, confidences, classes) -> StubBoxes:
    """The confidence/class carrier a segmentation result also reports."""
    rows = []
    for polygon, confidence, class_id in zip(polygons, confidences, classes, strict=True):
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        rows.append((min(xs), min(ys), max(xs), max(ys), confidence, class_id))
    return StubBoxes(rows)


def box_detector(*rows) -> PitchBoxDetector:
    model = StubModel([StubResult(boxes=StubBoxes(list(rows)))])
    return PitchBoxDetector(Path("injected"), model=model)


def mask_detector(polygons, *, confidences=None, classes=None, conf=0.25):
    confidences = confidences if confidences is not None else [0.9] * len(polygons)
    classes = classes if classes is not None else [0] * len(polygons)
    model = StubModel(
        [
            StubResult(
                boxes=boxes_for(polygons, confidences, classes),
                masks=StubMasks(polygons),
            )
        ]
    )
    return PitchSegmentationDetector(Path("injected"), model=model, conf=conf)


class TestFrameValidation(unittest.TestCase):
    def test_a_bgr_uint8_frame_reports_its_size(self):
        self.assertEqual(validate_frame(make_frame(1280, 720)), (1280, 720))
        self.assertEqual(validate_frame(make_frame(7, 3)), (7, 3))

    def test_anything_that_is_not_a_bgr_uint8_frame_is_refused(self):
        cases = {
            "two dimensional": np.zeros((10, 10), dtype=np.uint8),
            "four channel": np.zeros((10, 10, 4), dtype=np.uint8),
            "float pixels": np.zeros((10, 10, 3), dtype=np.float32),
            "sixteen bit": np.zeros((10, 10, 3), dtype=np.uint16),
            "no pixels": np.zeros((0, 10, 3), dtype=np.uint8),
            "not an array": "a frame",
            "a list": [[0, 0, 0]],
        }
        for label, frame in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(InvalidFrameError):
                    validate_frame(frame)

    def test_a_bad_frame_raises_rather_than_returning_a_status(self):
        # A bad frame is a bug in the caller's code, not a fact about the
        # environment, so it must not be swallowed into a "no detection".
        detector = box_detector(BOX_ROW)
        with self.assertRaises(InvalidFrameError):
            detector.detect(np.zeros((10, 10), dtype=np.uint8))
        self.assertEqual(detector.stats()["frames_seen"], 0)


class TestWeightsAvailability(unittest.TestCase):
    def test_a_missing_checkpoint_is_an_unavailable_status(self):
        path = Path("definitely") / "not" / "here" / "pitch_detect.pt"
        detector = PitchBoxDetector(path)
        self.assertFalse(detector.available)
        self.assertIn("weights not found", detector.disabled_reason)

        detection = detector.detect(make_frame(), frame_number=3)
        self.assertIs(detection.status, DetectionStatus.UNAVAILABLE)
        self.assertFalse(detection.ok)
        self.assertIsNone(detection.corners)
        self.assertIsNone(detection.method)
        self.assertEqual(detection.frame_number, 3)
        self.assertEqual(detection.image_size, FRAME_SIZE)
        self.assertIn(str(path), detection.reason)
        self.assertIsNone(detection_to_calibration(detection))

    def test_the_missing_file_is_never_created(self):
        # `ultralytics.YOLO("name.pt")` would fetch the checkpoint; the
        # detector must never give it the chance.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / PITCH_BOX_MODEL_FILENAME
            detector = PitchBoxDetector(path)
            detector.detect(make_frame())
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_the_segmentation_detector_behaves_the_same_way(self):
        path = Path("definitely") / "not" / "here" / "pitch_seg_yolo11s.pt"
        detector = PitchSegmentationDetector(path)
        self.assertFalse(detector.available)
        detection = detector.detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.UNAVAILABLE)
        self.assertIn(str(path), detection.reason)
        self.assertEqual(detector.stats()["task"], "segment")

    def test_every_unavailable_frame_is_counted(self):
        detector = PitchBoxDetector(Path("nowhere") / "pitch_detect.pt")
        for _ in range(3):
            detector.detect(make_frame())
        stats = detector.stats()
        self.assertEqual(stats["frames_seen"], 3)
        self.assertEqual(stats["unavailable"], 3)
        self.assertEqual(stats["detections"], 0)
        self.assertFalse(stats["model_found"])

    def test_an_injected_model_needs_no_weights_on_disk(self):
        detector = box_detector(BOX_ROW)
        self.assertTrue(detector.available)
        self.assertIsNone(detector.disabled_reason)
        self.assertIs(detector.detect(make_frame()).status, DetectionStatus.OK)

    def test_a_corrupt_checkpoint_is_unavailable_with_the_load_error(self):
        # The file EXISTS, so this is not the missing-weights case: it is
        # "unusable". The status must still not become NO_DETECTION, which
        # would claim the model looked at the frame and found nothing.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / PITCH_BOX_MODEL_FILENAME
            path.write_bytes(b"not a checkpoint")
            detector = PitchBoxDetector(path)
            self.assertTrue(detector.available)

            detection = detector.detect(make_frame(), frame_number=2)
            self.assertIs(detection.status, DetectionStatus.UNAVAILABLE)
            self.assertIsNone(detection.corners)
            self.assertIn(str(path), detection.reason)
            self.assertIsNotNone(detector.load_error)
            self.assertIn(str(path), detector.load_error)

            stats = detector.stats()
            self.assertTrue(stats["model_found"])
            self.assertEqual(stats["unavailable"], 1)
            self.assertEqual(stats["no_detections"], 0)
            self.assertEqual(stats["errors"], 0)


class TestBoxDetection(unittest.TestCase):
    def test_a_box_becomes_an_approximate_quad(self):
        detector = box_detector(BOX_ROW)
        detection = detector.detect(make_frame(), frame_number=5)
        self.assertIs(detection.status, DetectionStatus.OK)
        self.assertTrue(detection.ok)
        self.assertIs(detection.method, CornerMethod.BOX_ESTIMATE)
        self.assertTrue(detection.is_approximate)
        self.assertEqual(detection.quad_method, "box_estimate")
        self.assertAlmostEqual(detection.confidence, 0.88, places=5)
        self.assertEqual(detection.frame_number, 5)
        # The far edge is pulled in by 8% of the box width.
        inset = 0.08 * 240.0
        self.assertEqual(detection.corners.top_left, (200.0 + inset, 60.0))
        self.assertEqual(detection.corners.bottom_left, (200.0, 340.0))

    def test_a_box_can_never_become_a_validated_calibration(self):
        detection = box_detector(BOX_ROW).detect(make_frame())
        calibration = detection_to_calibration(detection)
        self.assertIsNotNone(calibration)
        self.assertIs(calibration.status, CalibrationStatus.APPROXIMATE)
        self.assertFalse(calibration.is_valid)
        self.assertEqual(calibration.reasons, ("approximate_corners",))
        self.assertIsNone(calibration.confidence)

    def test_the_frame_reaches_the_model_at_its_own_size(self):
        model = StubModel([StubResult(boxes=StubBoxes([BOX_ROW]))])
        detector = PitchBoxDetector(Path("injected"), model=model, imgsz=1280, conf=0.4)
        frame = make_frame(960, 540)
        detector.detect(frame)
        self.assertEqual(model.calls, 1)
        self.assertIs(model.last_kwargs["source"], frame)
        self.assertEqual(model.last_kwargs["imgsz"], 1280)
        self.assertEqual(model.last_kwargs["conf"], 0.4)
        self.assertFalse(model.last_kwargs["verbose"])
        self.assertIn(model.last_kwargs["device"], {"cpu", "cuda", "cuda:0"})

    def test_no_boxes_is_no_detection(self):
        for result in (StubResult(), StubResult(boxes=None), StubResult(boxes=StubBoxes([]))):
            with self.subTest(result=result):
                detector = PitchBoxDetector(
                    Path("injected"), model=StubModel([result])
                )
                detection = detector.detect(make_frame())
                self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
                self.assertEqual(detection.reason, "no_pitch_box")
                self.assertIsNone(detection.corners)

    def test_no_results_at_all_is_reported_as_such(self):
        detector = PitchBoxDetector(Path("injected"), model=StubModel([]))
        detection = detector.detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
        self.assertEqual(detection.reason, "model returned no results")

    def test_a_box_too_small_to_be_a_pitch_is_refused(self):
        detection = box_detector((100.0, 100.0, 110.0, 110.0, 0.9, 0)).detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
        self.assertEqual(detection.reason, "pitch_box_too_small")

    def test_a_box_that_is_the_whole_picture_is_refused(self):
        detection = box_detector((0.0, 0.0, 640.0, 360.0, 0.9, 0)).detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
        self.assertEqual(detection.reason, "pitch_box_too_large")

    def test_a_detection_of_another_class_is_refused(self):
        detection = box_detector((*BOX, 0.88, 3)).detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
        self.assertEqual(detection.reason, "wrong_class")

    def test_the_most_confident_box_is_used(self):
        detection = box_detector(
            (10.0, 10.0, 60.0, 60.0, 0.30, 0),
            (*BOX, 0.88, 0),
        ).detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.OK)
        self.assertAlmostEqual(detection.confidence, 0.88, places=5)
        self.assertEqual(detection.corners.bottom_left, (200.0, 340.0))

    def test_a_model_failure_is_an_error_status_not_a_crash(self):
        class Boom:
            def predict(self, **kwargs):
                raise RuntimeError("cuda exploded")

        detector = PitchBoxDetector(Path("injected"), model=Boom())
        detection = detector.detect(make_frame(), frame_number=9)
        self.assertIs(detection.status, DetectionStatus.ERROR)
        self.assertFalse(detection.ok)
        self.assertIn("cuda exploded", detection.reason)
        stats = detector.stats()
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["last_error"], "cuda exploded")
        self.assertEqual(detection.frame_number, 9)


class TestSegmentationDetection(unittest.TestCase):
    def test_a_mask_becomes_valid_corners(self):
        polygon = pitch_mask_polygon()
        detection = mask_detector([polygon]).detect(make_frame(), frame_number=11)
        self.assertIs(detection.status, DetectionStatus.OK)
        self.assertIs(detection.method, CornerMethod.SEGMENTATION)
        self.assertFalse(detection.is_approximate)
        self.assertEqual(detection.quad_method, "direct_hull")
        self.assertEqual(detection.corners.points(), MASK_CORNERS)
        self.assertAlmostEqual(detection.confidence, 0.9, places=5)
        self.assertEqual(detection.rejections, {})

    def test_a_mask_can_become_a_validated_calibration(self):
        detection = mask_detector([pitch_mask_polygon()]).detect(make_frame())
        calibration = detection_to_calibration(detection)
        self.assertIsNotNone(calibration)
        self.assertIs(calibration.status, CalibrationStatus.VALID)
        self.assertTrue(calibration.is_valid)
        self.assertAlmostEqual(calibration.confidence, 0.9, places=5)

    def test_a_mask_with_extra_vertices_is_simplified_to_four(self):
        # A convex pentagon: five hull vertices, four corners. The fifth is
        # dropped by simplification, never guessed at.
        pentagon = [
            (220.0, 80.0),
            (420.0, 80.0),
            (420.0, 320.0),
            (320.0, 320.0),
            (220.0, 290.0),
        ]
        detection = mask_detector([pentagon]).detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.OK)
        self.assertEqual(len(set(detection.corners.points())), 4)
        self.assertIn(detection.quad_method, {"approx_polydp", "hull_reduce"})
        self.assertIs(detection_to_calibration(detection).status, CalibrationStatus.VALID)

    def test_no_masks_is_no_detection(self):
        for result in (StubResult(), StubResult(masks=None), StubResult(masks=StubMasks([]))):
            with self.subTest(result=result):
                detector = PitchSegmentationDetector(
                    Path("injected"), model=StubModel([result])
                )
                detection = detector.detect(make_frame())
                self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
                self.assertEqual(detection.reason, "no_pitch_mask")

    def test_a_mask_too_small_to_be_a_pitch_is_refused(self):
        tiny = [(300.0, 170.0), (330.0, 170.0), (330.0, 190.0), (300.0, 190.0)]
        detection = mask_detector([tiny]).detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
        self.assertEqual(detection.rejections, {"mask_area_too_small": 1})
        self.assertIn("mask_area_too_small", detection.reason)

    def test_a_crowd_sized_mask_is_refused(self):
        crowd = [(100.0, 20.0), (540.0, 20.0), (540.0, 340.0), (100.0, 340.0)]
        detection = mask_detector([crowd]).detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
        self.assertEqual(detection.rejections, {"mask_area_too_large": 1})

    def test_a_ribbon_shaped_mask_is_refused(self):
        # Area 0.038 of the frame passes; the 22:1 aspect ratio does not.
        ribbon = [(100.0, 170.0), (540.0, 170.0), (540.0, 190.0), (100.0, 190.0)]
        detection = mask_detector([ribbon]).detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
        self.assertEqual(detection.rejections, {"mask_aspect_too_elongated": 1})

    def test_a_mask_off_the_picture_is_refused(self):
        detection = mask_detector([pitch_mask_polygon(centre_x=-200.0)]).detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
        self.assertEqual(detection.rejections, {"mask_outside_frame": 1})

    def test_a_mask_below_the_confidence_floor_is_refused(self):
        detection = mask_detector([pitch_mask_polygon()], confidences=[0.4], conf=0.5).detect(
            make_frame()
        )
        self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
        self.assertEqual(detection.rejections, {"below_confidence": 1})

    def test_a_mask_of_another_class_is_refused(self):
        detection = mask_detector([pitch_mask_polygon()], classes=[2]).detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
        self.assertEqual(detection.rejections, {"wrong_class": 1})

    def test_a_triangle_mask_cannot_become_a_quad(self):
        # A triangle passes the shape gate (right area, right aspect) but has
        # only three corners: the fourth would have to be invented.
        triangle = [(150.0, 100.0), (450.0, 100.0), (300.0, 340.0)]
        detection = mask_detector([triangle]).detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.NO_DETECTION)
        self.assertEqual(detection.rejections, {"quad_unavailable": 1})

    def test_accepted_and_rejected_instances_are_both_reported(self):
        tiny = [(300.0, 170.0), (330.0, 170.0), (330.0, 190.0), (300.0, 190.0)]
        good = pitch_mask_polygon()
        detection = mask_detector([tiny, good, tiny]).detect(make_frame())
        self.assertIs(detection.status, DetectionStatus.OK)
        self.assertEqual(detection.rejections, {"mask_area_too_small": 2})
        self.assertEqual(detection.corners.points(), MASK_CORNERS)


class TestCandidateSelection(unittest.TestCase):
    def test_a_tie_is_broken_by_continuity(self):
        left = pitch_mask_polygon(centre_x=160.0)
        right = pitch_mask_polygon(centre_x=480.0)
        model = StubSequenceModel(
            [
                [
                    StubResult(
                        boxes=boxes_for([right], [0.9], [0]), masks=StubMasks([right])
                    )
                ],
                [
                    StubResult(
                        boxes=boxes_for([left, right], [0.9, 0.9], [0, 0]),
                        masks=StubMasks([left, right]),
                    )
                ],
            ]
        )
        detector = PitchSegmentationDetector(Path("injected"), model=model)
        first = detector.detect(make_frame())
        second = detector.detect(make_frame())
        self.assertIs(first.status, DetectionStatus.OK)
        self.assertIs(second.status, DetectionStatus.OK)
        # Equal confidence: the estimate that stayed put wins.
        self.assertEqual(second.corners, first.corners)

    def test_a_real_confidence_gap_beats_continuity(self):
        left = pitch_mask_polygon(centre_x=160.0)
        right = pitch_mask_polygon(centre_x=480.0)
        model = StubSequenceModel(
            [
                [
                    StubResult(
                        boxes=boxes_for([right], [0.9], [0]), masks=StubMasks([right])
                    )
                ],
                [
                    StubResult(
                        boxes=boxes_for([left, right], [0.9, 0.55], [0, 0]),
                        masks=StubMasks([left, right]),
                    )
                ],
            ]
        )
        detector = PitchSegmentationDetector(Path("injected"), model=model)
        first = detector.detect(make_frame())
        second = detector.detect(make_frame())
        self.assertIs(first.status, DetectionStatus.OK)
        self.assertEqual(second.corners.top_left, (110.0, 60.0))
        self.assertEqual(second.corners.bottom_right, (240.0, 340.0))
        self.assertNotEqual(second.corners, first.corners)


class TestStats(unittest.TestCase):
    def test_frames_seen_equals_the_sum_of_the_outcomes(self):
        class Flaky:
            """A detection, then a failure, then nothing found."""

            def __init__(self):
                self.calls = 0

            def predict(self, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("boom")
                if self.calls == 3:
                    return [StubResult(boxes=StubBoxes([]))]
                return [StubResult(boxes=StubBoxes([BOX_ROW]))]

        detector = PitchBoxDetector(Path("injected"), model=Flaky())
        frame = make_frame()
        statuses = [detector.detect(frame).status for _ in range(3)]
        self.assertEqual(
            statuses,
            [DetectionStatus.OK, DetectionStatus.ERROR, DetectionStatus.NO_DETECTION],
        )
        stats = detector.stats()
        self.assertEqual(stats["frames_seen"], 3)
        self.assertEqual(stats["detections"], 1)
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["no_detections"], 1)
        self.assertEqual(stats["unavailable"], 0)
        self.assertEqual(
            stats["frames_seen"],
            stats["detections"]
            + stats["no_detections"]
            + stats["unavailable"]
            + stats["errors"],
        )


class TestDetectorInterface(unittest.TestCase):
    def test_both_detectors_satisfy_the_interface(self):
        for cls in (PitchBoxDetector, PitchSegmentationDetector):
            with self.subTest(detector=cls.__name__):
                detector = cls(Path("injected"), model=StubModel([StubResult()]))
                self.assertIsInstance(detector, PitchDetector)
                self.assertIsInstance(detector.detect(make_frame()), object)


class TestWeightsConfiguration(unittest.TestCase):
    def test_the_box_resolver_names_every_path_it_tried(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "elsewhere" / PITCH_BOX_MODEL_FILENAME
            with mock.patch.object(config, "PITCH_BOX_MODEL_PATH", missing):
                with self.assertRaises(FileNotFoundError) as ctx:
                    resolve_pitch_box_model_path(Path(directory))
        message = str(ctx.exception)
        self.assertIn(str(Path(directory) / PITCH_BOX_MODEL_FILENAME), message)
        self.assertIn(str(missing), message)
        self.assertIn("CRICKET_PITCH_BOX_MODEL", message)

    def test_the_segmentation_resolver_names_every_path_it_tried(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "elsewhere" / PITCH_SEG_MODEL_FILENAME
            with mock.patch.object(config, "PITCH_SEG_MODEL_PATH", missing):
                with self.assertRaises(FileNotFoundError) as ctx:
                    resolve_pitch_segmentation_model_path(Path(directory))
        message = str(ctx.exception)
        self.assertIn(str(Path(directory) / PITCH_SEG_MODEL_FILENAME), message)
        self.assertIn("CRICKET_PITCH_SEG_MODEL", message)

    def test_a_resolver_finds_the_weights_in_the_models_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / PITCH_BOX_MODEL_FILENAME
            target.write_bytes(b"stand-in for a checkpoint")
            self.assertEqual(resolve_pitch_box_model_path(Path(directory)), target)

    def test_describe_weights_reports_presence_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "weights.pt"
            self.assertIn("NOT FOUND", describe_weights(target))
            target.write_bytes(b"x" * (2 * 1048576))
            described = describe_weights(target)
            self.assertIn(str(target), described)
            self.assertIn("2.0 MiB", described)

    def test_the_configuration_document_never_claims_a_missing_file(self):
        configuration = PitchConfig(
            box_model_path=Path("nowhere") / "a.pt",
            segmentation_model_path=Path("nowhere") / "b.pt",
        )
        document = configuration.as_metadata()
        self.assertFalse(document["box_model_found"])
        self.assertFalse(document["segmentation_model_found"])
        self.assertFalse(any("token" in str(value).lower() for value in document.values()))


if __name__ == "__main__":
    unittest.main()

"""
test_shot_inference.py — Checkpoint loading, output validation, and the
delivery-level failure contract.

The loader is exercised against tiny synthetic checkpoints so that a missing,
empty, corrupt, wrong-architecture or checksum-mismatched file can each be
provoked on purpose. One gated class runs against the real 68 MB checkpoint when
`CRICKET_SHOT_MODEL` is set; it is skipped otherwise rather than pretending a
synthetic file proves the real architecture loads.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from app.shots import inference, load_shot_model, predict_shot
from app.shots.constants import SHOT_CLASSES, SHOT_NUM_CLASSES
from app.shots.errors import ShotInferenceError, ShotInputError, ShotModelUnavailable
from app.shots.inference import (
    LoadedShotModel,
    ShotClassifier,
    resolve_device,
)
from tests.shots_helpers import (
    StubShotModel,
    TinyModel,
    clip_frames,
    file_md5,
    tiny_factory,
    write_clip,
    write_non_checkpoint,
    write_tiny_checkpoint,
)


class TemporaryDirectoryCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)


def classifier_with(logits: object = None, *, fail: bool = False) -> ShotClassifier:
    loaded = LoadedShotModel(
        module=StubShotModel(logits, fail=fail),
        path=None,
        size_bytes=None,
        md5=None,
        device=torch.device("cpu"),
    )
    return ShotClassifier(loaded)


class TestLoadShotModel(TemporaryDirectoryCase):
    def test_loads_a_tiny_checkpoint_strictly(self) -> None:
        path = write_tiny_checkpoint(self.directory / "tiny.ckpt")
        raw = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        loaded = load_shot_model(path, device="cpu", model_factory=tiny_factory)

        self.assertIsInstance(loaded.module, TinyModel)
        self.assertFalse(loaded.module.training)
        self.assertEqual(loaded.device, torch.device("cpu"))
        self.assertEqual(loaded.md5, file_md5(path))
        self.assertEqual(loaded.size_bytes, path.stat().st_size)
        self.assertEqual(loaded.path, path.resolve())

        for key, expected in raw.items():
            self.assertTrue(
                torch.equal(loaded.module.state_dict()[key], expected),
                f"{key} was not restored from the checkpoint",
            )

    def test_module_prefixes_are_stripped(self) -> None:
        path = write_tiny_checkpoint(self.directory / "tiny.ckpt", prefix="module.")
        raw = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        loaded = load_shot_model(path, device="cpu", model_factory=tiny_factory)
        self.assertTrue(
            torch.equal(
                loaded.module.state_dict()["proj.weight"],
                raw["module.proj.weight"],
            )
        )

    def test_raw_state_dict_is_accepted(self) -> None:
        path = write_tiny_checkpoint(self.directory / "tiny.ckpt", wrapper=False)
        loaded = load_shot_model(path, device="cpu", model_factory=tiny_factory)
        self.assertIsInstance(loaded.module, TinyModel)

    def test_class_weights_is_restored_from_the_fresh_model(self) -> None:
        path = write_tiny_checkpoint(self.directory / "tiny.ckpt")
        loaded = load_shot_model(path, device="cpu", model_factory=tiny_factory)
        self.assertTrue(
            torch.equal(
                loaded.module.state_dict()["class_weights"],
                torch.ones(SHOT_NUM_CLASSES),
            )
        )

    def test_missing_checkpoint_is_unavailable(self) -> None:
        with self.assertRaises(ShotModelUnavailable) as caught:
            load_shot_model(self.directory / "missing.ckpt", device="cpu")
        self.assertIn("not found", str(caught.exception))

    def test_empty_checkpoint_is_unavailable(self) -> None:
        path = self.directory / "empty.ckpt"
        path.write_bytes(b"")
        with self.assertRaises(ShotModelUnavailable) as caught:
            load_shot_model(path, device="cpu", model_factory=tiny_factory)
        self.assertIn("empty", str(caught.exception))

    def test_corrupt_checkpoint_is_unavailable(self) -> None:
        path = write_non_checkpoint(self.directory / "corrupt.ckpt")
        with self.assertRaises(ShotModelUnavailable) as caught:
            load_shot_model(path, device="cpu", model_factory=tiny_factory)
        self.assertIn("could not read", str(caught.exception))

    def test_checksum_mismatch_is_unavailable(self) -> None:
        path = write_tiny_checkpoint(self.directory / "tiny.ckpt")
        with self.assertRaises(ShotModelUnavailable) as caught:
            load_shot_model(
                path,
                device="cpu",
                expected_md5="0" * 32,
                model_factory=tiny_factory,
            )
        self.assertIn("checksum mismatch", str(caught.exception))

    def test_checksum_match_is_case_insensitive(self) -> None:
        path = write_tiny_checkpoint(self.directory / "tiny.ckpt")
        loaded = load_shot_model(
            path,
            device="cpu",
            expected_md5=file_md5(path).upper(),
            model_factory=tiny_factory,
        )
        self.assertEqual(loaded.md5, file_md5(path))

    def test_architecture_mismatch_is_unavailable(self) -> None:
        path = write_tiny_checkpoint(self.directory / "tiny.ckpt", drop=("proj.bias",))
        with self.assertRaises(ShotModelUnavailable) as caught:
            load_shot_model(path, device="cpu", model_factory=tiny_factory)
        self.assertIn("does not match the architecture", str(caught.exception))

    def test_non_dictionary_checkpoint_is_unavailable(self) -> None:
        path = self.directory / "tensor.ckpt"
        torch.save(torch.zeros(3), path)
        with self.assertRaises(ShotModelUnavailable) as caught:
            load_shot_model(path, device="cpu", model_factory=tiny_factory)
        self.assertIn("unsupported checkpoint format", str(caught.exception))

    def test_empty_state_dict_is_unavailable(self) -> None:
        path = self.directory / "empty_state.ckpt"
        torch.save({"state_dict": {}}, path)
        with self.assertRaises(ShotModelUnavailable) as caught:
            load_shot_model(path, device="cpu", model_factory=tiny_factory)
        self.assertIn("no state-dict entries", str(caught.exception))

    def test_factory_failure_is_unavailable(self) -> None:
        path = write_tiny_checkpoint(self.directory / "tiny.ckpt")

        def broken_factory(**kwargs: object) -> TinyModel:
            raise RuntimeError("factory refused")

        with self.assertRaises(ShotModelUnavailable) as caught:
            load_shot_model(path, device="cpu", model_factory=broken_factory)
        self.assertIn("could not construct", str(caught.exception))

    def test_unusable_device_is_unavailable(self) -> None:
        path = write_tiny_checkpoint(self.directory / "tiny.ckpt")
        with self.assertRaises(ShotModelUnavailable):
            load_shot_model(path, device="not-a-device", model_factory=tiny_factory)


class TestResolveDevice(unittest.TestCase):
    def test_cpu(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_auto_picks_an_available_device(self) -> None:
        device = resolve_device("auto")
        self.assertIn(device.type, {"cpu", "cuda"})

    def test_whitespace_and_case_are_tolerated(self) -> None:
        self.assertEqual(resolve_device(" CPU "), torch.device("cpu"))

    def test_invalid_specs_are_rejected(self) -> None:
        for spec in ("", "   ", "gpu", "cuda:x", "cuda:-1", "mps"):
            with self.subTest(spec=spec):
                with self.assertRaises(ShotModelUnavailable):
                    resolve_device(spec)

    def test_non_string_is_rejected(self) -> None:
        with self.assertRaises(ShotModelUnavailable):
            resolve_device(None)  # type: ignore[arg-type]

    def test_cuda_is_unavailable_or_visible(self) -> None:
        if torch.cuda.is_available():
            self.assertEqual(resolve_device("cuda").type, "cuda")
        else:
            with self.assertRaises(ShotModelUnavailable):
                resolve_device("cuda")

    def test_out_of_range_cuda_index_is_rejected(self) -> None:
        with self.assertRaises(ShotModelUnavailable):
            resolve_device("cuda:99")


class TestPredictShot(TemporaryDirectoryCase):
    def test_known_logits_select_the_expected_label(self) -> None:
        logits = [0.0] * SHOT_NUM_CLASSES
        logits[3] = 10.0  # SHOT_CLASSES[3] == "Hook"
        prediction = predict_shot(StubShotModel(logits), torch.zeros(1, 30, 3, 224, 224))

        self.assertEqual(prediction.label, "Hook")
        self.assertGreater(prediction.score, 0.99)
        self.assertEqual(
            prediction.score,
            max(prediction.class_probabilities.values()),
        )
        self.assertAlmostEqual(
            sum(prediction.class_probabilities.values()),
            1.0,
            delta=1e-5,  # probabilities are rounded to six decimals
        )

    def test_uniform_logits_give_a_uniform_distribution(self) -> None:
        prediction = predict_shot(StubShotModel(), torch.zeros(1, 30, 3, 224, 224))
        self.assertEqual(len(prediction.class_probabilities), SHOT_NUM_CLASSES)
        for value in prediction.class_probabilities.values():
            self.assertAlmostEqual(value, 0.1, places=6)
        self.assertEqual(prediction.label, SHOT_CLASSES[0])

    def test_forward_failure_is_an_inference_error(self) -> None:
        with self.assertRaises(ShotInferenceError) as caught:
            predict_shot(
                StubShotModel(fail=True),
                torch.zeros(1, 30, 3, 224, 224),
            )
        self.assertIn("forward pass failed", str(caught.exception))

    def test_wrong_number_of_logits_is_rejected(self) -> None:
        with self.assertRaises(ShotInferenceError):
            predict_shot(
                StubShotModel([0.0] * (SHOT_NUM_CLASSES - 1)),
                torch.zeros(1, 30, 3, 224, 224),
            )

    def test_wrong_batch_dimension_is_rejected(self) -> None:
        with self.assertRaises(ShotInferenceError):
            predict_shot(
                StubShotModel(torch.zeros(2, SHOT_NUM_CLASSES)),
                torch.zeros(1, 30, 3, 224, 224),
            )

    def test_non_finite_logits_are_rejected(self) -> None:
        logits = torch.zeros(1, SHOT_NUM_CLASSES)
        logits[0, 4] = float("nan")
        with self.assertRaises(ShotInferenceError) as caught:
            predict_shot(StubShotModel(logits), torch.zeros(1, 30, 3, 224, 224))
        self.assertIn("non-finite", str(caught.exception))

    def test_integer_logits_are_rejected(self) -> None:
        with self.assertRaises(ShotInferenceError):
            predict_shot(
                StubShotModel(torch.zeros(1, SHOT_NUM_CLASSES, dtype=torch.int64)),
                torch.zeros(1, 30, 3, 224, 224),
            )

    def test_non_tensor_output_is_rejected(self) -> None:
        class BadModel(StubShotModel):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return [0.0] * SHOT_NUM_CLASSES  # type: ignore[return-value]

        with self.assertRaises(ShotInferenceError):
            predict_shot(BadModel(), torch.zeros(1, 30, 3, 224, 224))

    def test_non_tensor_input_is_rejected(self) -> None:
        with self.assertRaises(ShotInputError):
            predict_shot(StubShotModel(), [[0.0]])  # type: ignore[arg-type]

    def test_wrong_input_shape_is_rejected(self) -> None:
        for shape in ((1, 29, 3, 224, 224), (1, 30, 3, 224), (30, 3, 224, 224)):
            with self.subTest(shape=shape):
                with self.assertRaises(ShotInputError):
                    predict_shot(StubShotModel(), torch.zeros(*shape))

    def test_integer_input_is_rejected(self) -> None:
        with self.assertRaises(ShotInputError):
            predict_shot(
                StubShotModel(),
                torch.zeros(1, 30, 3, 224, 224, dtype=torch.uint8),
            )


class TestShotClassifier(TemporaryDirectoryCase):
    def test_in_memory_sequence_is_classified(self) -> None:
        result = classifier_with().classify(
            in_memory_frames=clip_frames(30),
            expected_total_frames=30,
            delivery_id=1,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.input_source, "in_memory")
        self.assertEqual(result.frames_sampled, 30)
        self.assertEqual(result.unique_frames_sampled, 30)
        self.assertFalse(result.frames_repeated)
        self.assertEqual(result.sampled_clip_frames, list(range(30)))
        self.assertIsNone(result.sampled_original_frames)
        self.assertEqual(result.label, "Cover")
        self.assertAlmostEqual(result.score, 0.1, places=6)

    def test_short_sequence_is_repeated_to_thirty_frames(self) -> None:
        result = classifier_with().classify(in_memory_frames=clip_frames(10))
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.frames_repeated)
        self.assertEqual(result.frames_sampled, 30)
        self.assertEqual(result.unique_frames_sampled, 10)
        self.assertEqual(len(result.sampled_clip_frames), 30)
        self.assertEqual(set(result.sampled_clip_frames), set(range(10)))

    def test_sequence_shorter_than_claimed_total_is_inconsistent(self) -> None:
        result = classifier_with().classify(
            in_memory_frames=clip_frames(10),
            expected_total_frames=30,
        )
        self.assertEqual(result.status, "invalid_input")
        self.assertIn("clip position", result.error or "")
        self.assertIsNone(result.label)

    def test_mapping_keyed_by_clip_position(self) -> None:
        frames = {index: frame for index, frame in enumerate(clip_frames(30))}
        result = classifier_with().classify(
            in_memory_frames=frames,
            expected_total_frames=30,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.input_source, "in_memory")

    def test_mapping_with_a_missing_position_falls_back_to_no_source(self) -> None:
        frames = {index: frame for index, frame in enumerate(clip_frames(30))}
        del frames[29]
        result = classifier_with().classify(
            in_memory_frames=frames,
            expected_total_frames=30,
        )
        self.assertEqual(result.status, "invalid_input")
        self.assertIn("first 29", result.error or "")

    def test_mapping_without_total_is_invalid(self) -> None:
        result = classifier_with().classify(in_memory_frames={0: clip_frames(1)[0]})
        self.assertEqual(result.status, "invalid_input")
        self.assertIn("expected_total_frames", result.error or "")

    def test_empty_frame_source_is_invalid(self) -> None:
        for source in ([], {}):
            with self.subTest(source=type(source).__name__):
                result = classifier_with().classify(
                    in_memory_frames=source,
                    expected_total_frames=0 if isinstance(source, dict) else None,
                )
                self.assertEqual(result.status, "invalid_input")

    def test_string_frame_source_is_invalid(self) -> None:
        result = classifier_with().classify(in_memory_frames="frames")  # type: ignore[arg-type]
        self.assertEqual(result.status, "invalid_input")
        self.assertIn("string", result.error or "")

    def test_no_source_is_invalid(self) -> None:
        result = classifier_with().classify()
        self.assertEqual(result.status, "invalid_input")
        self.assertIn("no frame source", result.error or "")
        self.assertEqual(result.frames_sampled, 0)

    def test_malformed_frame_is_invalid_input(self) -> None:
        frames = clip_frames(30)
        frames[5] = frames[5].astype("float32")
        result = classifier_with().classify(
            in_memory_frames=frames,
            expected_total_frames=30,
        )
        self.assertEqual(result.status, "invalid_input")
        self.assertIn("frame 5", result.error or "")
        self.assertEqual(result.frames_sampled, 30)
        self.assertEqual(len(result.sampled_clip_frames), 30)

    def test_short_clip_file_is_read_when_no_frames_are_retained(self) -> None:
        path = write_clip(self.directory, frame_count=10)
        result = classifier_with().classify(clip_path=path)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.input_source, "clip_file")
        self.assertTrue(result.frames_repeated)
        self.assertEqual(result.unique_frames_sampled, 10)

    def test_in_memory_frames_win_over_the_clip(self) -> None:
        path = write_clip(self.directory, frame_count=30)
        result = classifier_with().classify(
            clip_path=path,
            in_memory_frames=clip_frames(30),
            expected_total_frames=30,
        )
        self.assertEqual(result.input_source, "in_memory")

    def test_unreadable_clip_is_reported(self) -> None:
        result = classifier_with().classify(clip_path=self.directory / "missing.avi")
        self.assertEqual(result.status, "invalid_input")
        self.assertIn("clip not found", result.error or "")

    def test_forward_failure_is_a_recorded_inference_failure(self) -> None:
        result = classifier_with(fail=True).classify(
            in_memory_frames=clip_frames(30),
            expected_total_frames=30,
        )
        self.assertEqual(result.status, "inference_failed")
        self.assertIn("stub forward failure", result.error or "")
        self.assertIsNone(result.label)
        self.assertEqual(result.class_probabilities, {})
        self.assertEqual(result.frames_sampled, 30)

    def test_frame_offset_is_recorded_as_original_frames(self) -> None:
        result = classifier_with().classify(
            in_memory_frames=clip_frames(30),
            expected_total_frames=30,
            frame_offset=100,
        )
        self.assertEqual(
            result.sampled_original_frames,
            [position + 100 for position in result.sampled_clip_frames],
        )

    def test_negative_frame_offset_is_a_caller_error(self) -> None:
        with self.assertRaises(ValueError):
            classifier_with().classify(frame_offset=-1)

    def test_constructor_requires_a_model_or_a_reason(self) -> None:
        with self.assertRaises(ValueError):
            ShotClassifier(None)
        with self.assertRaises(ValueError):
            ShotClassifier(
                LoadedShotModel(
                    module=StubShotModel(),
                    path=None,
                    size_bytes=None,
                    md5=None,
                    device=torch.device("cpu"),
                ),
                unavailable_reason="both",
            )

    def test_unavailable_reason_alone_is_a_valid_constructor(self) -> None:
        classifier = ShotClassifier(None, unavailable_reason="nothing installed")
        result = classifier.classify(in_memory_frames=clip_frames(30))
        self.assertEqual(result.status, "model_unavailable")
        self.assertEqual(result.error, "nothing installed")

    def test_unavailable_classifier_records_the_reason(self) -> None:
        result = ShotClassifier.unavailable("weights not installed").classify(
            in_memory_frames=clip_frames(30),
        )
        self.assertEqual(result.status, "model_unavailable")
        self.assertEqual(result.error, "weights not installed")
        self.assertEqual(result.input_source, "unavailable")
        self.assertEqual(result.frames_sampled, 0)
        self.assertIsNone(result.label)
        self.assertIsNone(result.model_path)

    def test_model_metadata_is_attached_to_success(self) -> None:
        path = write_tiny_checkpoint(self.directory / "tiny.ckpt")
        classifier = ShotClassifier.from_checkpoint(
            path,
            device="cpu",
            model_factory=tiny_factory,
        )
        result = classifier.classify(
            in_memory_frames=clip_frames(30),
            expected_total_frames=30,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.model_path, str(path.resolve()))
        self.assertEqual(result.model_size_bytes, path.stat().st_size)
        self.assertEqual(result.model_md5, file_md5(path))
        self.assertEqual(result.model_device, "cpu")

    def test_from_checkpoint_missing_file_is_unavailable(self) -> None:
        with self.assertRaises(ShotModelUnavailable):
            ShotClassifier.from_checkpoint(
                self.directory / "missing.ckpt",
                device="cpu",
                model_factory=tiny_factory,
            )

    def test_from_checkpoint_relative_path_uses_project_root(self) -> None:
        (self.directory / "models").mkdir()
        path = write_tiny_checkpoint(self.directory / "models" / "tiny.ckpt")
        with patch.object(inference, "PROJECT_DIR", self.directory):
            classifier = ShotClassifier.from_checkpoint(
                Path("models/tiny.ckpt"),
                device="cpu",
                model_factory=tiny_factory,
            )
        self.assertEqual(classifier._loaded.path, path.resolve())  # noqa: SLF001

    def test_from_checkpoint_without_a_path_uses_the_environment(self) -> None:
        missing = self.directory / "missing.ckpt"
        with patch.dict(os.environ, {"CRICKET_SHOT_MODEL": str(missing)}, clear=False):
            with self.assertRaises(ShotModelUnavailable):
                ShotClassifier.from_checkpoint(device="cpu", model_factory=tiny_factory)


@unittest.skipUnless(
    os.environ.get("CRICKET_SHOT_MODEL"),
    "set CRICKET_SHOT_MODEL to load the real shot_classifier.ckpt",
)
class TestRealCheckpoint(unittest.TestCase):
    def test_real_checkpoint_loads_and_classifies(self) -> None:
        checkpoint = os.environ["CRICKET_SHOT_MODEL"]
        classifier = ShotClassifier.from_checkpoint(checkpoint, device="cpu")
        result = classifier.classify(
            in_memory_frames=clip_frames(30),
            expected_total_frames=30,
        )

        self.assertEqual(result.status, "ok", result.error)
        self.assertIn(result.label, SHOT_CLASSES)
        self.assertGreaterEqual(result.model_size_bytes or 0, 70_000_000)
        self.assertEqual(len(result.model_md5 or ""), 32)
        self.assertAlmostEqual(sum(result.class_probabilities.values()), 1.0, places=5)
        self.assertEqual(result.model_device, "cpu")


if __name__ == "__main__":
    unittest.main()

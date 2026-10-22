"""
test_shot_preprocessing.py — Sampling and the exact input tensor.

Expected values are computed from the documented transform, not captured from a
run: a uniform pixel of 255 becomes 1.0 before normalisation, and ImageNet
normalisation then gives `(1 - mean) / std` per channel.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from app.shots.constants import IMAGENET_MEAN, IMAGENET_STD, SHOT_IMAGE_SIZE, SHOT_N_FRAMES
from app.shots.errors import ShotInputError
from app.shots.preprocessing import (
    frames_to_tensor,
    sample_clip_frames,
    sample_indices,
)
from tests.shots_helpers import bgr_frame, clip_frames, write_clip


class TestSampleIndices(unittest.TestCase):
    def test_exact_length_clip_samples_every_frame(self) -> None:
        self.assertEqual(sample_indices(30), list(range(30)))

    def test_longer_clip_samples_exactly_thirty_unique_frames(self) -> None:
        indices = sample_indices(31)
        self.assertEqual(len(indices), SHOT_N_FRAMES)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 30)
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(set(indices)), SHOT_N_FRAMES)

    def test_short_clip_repeats_positions(self) -> None:
        indices = sample_indices(10)
        self.assertEqual(len(indices), SHOT_N_FRAMES)
        self.assertEqual(set(indices), set(range(10)))
        self.assertGreater(len(indices), len(set(indices)))

    def test_single_frame_clip_repeats_it_thirty_times(self) -> None:
        self.assertEqual(sample_indices(1), [0] * SHOT_N_FRAMES)

    def test_empty_clip_has_nothing_to_sample(self) -> None:
        self.assertEqual(sample_indices(0), [])

    def test_invalid_frame_counts_are_rejected(self) -> None:
        for value in (-1, 30.0, "30", True):
            with self.subTest(value=value):
                with self.assertRaises(ShotInputError):
                    sample_indices(value)  # type: ignore[arg-type]

    def test_invalid_sample_count_is_rejected(self) -> None:
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ShotInputError):
                    sample_indices(30, value)  # type: ignore[arg-type]


class TestFramesToTensor(unittest.TestCase):
    def test_shape_dtype_and_normalisation(self) -> None:
        frames = [bgr_frame(255) for _ in range(SHOT_N_FRAMES)]
        tensor = frames_to_tensor(frames)

        self.assertEqual(
            tuple(tensor.shape),
            (1, SHOT_N_FRAMES, 3, SHOT_IMAGE_SIZE, SHOT_IMAGE_SIZE),
        )
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(tensor).all()))

        # BGR 255 -> RGB 255 -> ToTensor 1.0 -> (1 - mean) / std.
        for channel in range(3):
            pixel = float(tensor[0, 0, channel, 0, 0])
            expected = (1.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
            self.assertAlmostEqual(pixel, expected, places=5)

    def test_zero_frame_is_normalised_to_negative_mean_over_std(self) -> None:
        frames = [bgr_frame(0) for _ in range(SHOT_N_FRAMES)]
        tensor = frames_to_tensor(frames)
        for channel in range(3):
            expected = (0.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
            self.assertAlmostEqual(float(tensor[0, 0, channel, 0, 0]), expected, places=5)

    def test_wrong_frame_count_is_rejected(self) -> None:
        for count in (0, 1, 29, 31):
            with self.subTest(count=count):
                with self.assertRaises(ShotInputError):
                    frames_to_tensor(clip_frames(count))

    def test_non_uint8_frame_is_rejected(self) -> None:
        frames = clip_frames(30)
        frames[4] = frames[4].astype(np.float32)
        with self.assertRaises(ShotInputError) as caught:
            frames_to_tensor(frames)
        self.assertIn("frame 4", str(caught.exception))
        self.assertIn("uint8", str(caught.exception))

    def test_wrong_channel_count_is_rejected(self) -> None:
        frames = clip_frames(30)
        frames[2] = np.zeros((32, 24, 4), dtype=np.uint8)
        with self.assertRaises(ShotInputError) as caught:
            frames_to_tensor(frames)
        self.assertIn("channels", str(caught.exception))

    def test_two_dimensional_frame_is_rejected(self) -> None:
        frames = clip_frames(30)
        frames[0] = np.zeros((32, 24), dtype=np.uint8)
        with self.assertRaises(ShotInputError) as caught:
            frames_to_tensor(frames)
        self.assertIn("frame 0", str(caught.exception))

    def test_non_array_frame_is_rejected(self) -> None:
        frames = clip_frames(30)
        frames[7] = "not a frame"  # type: ignore[assignment]
        with self.assertRaises(ShotInputError) as caught:
            frames_to_tensor(frames)
        self.assertIn("frame 7", str(caught.exception))


class TestSampleClipFrames(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def test_full_clip_samples_every_frame_once(self) -> None:
        path = write_clip(self.directory, frame_count=30)
        frames, positions = sample_clip_frames(path)

        self.assertEqual(positions, list(range(30)))
        self.assertEqual(len(frames), 30)
        self.assertEqual(frames[0].shape, (48, 64, 3))
        self.assertEqual(frames[0].dtype, np.uint8)

    def test_short_clip_is_repeated_to_thirty_frames(self) -> None:
        path = write_clip(self.directory, frame_count=10)
        frames, positions = sample_clip_frames(path)

        self.assertEqual(len(frames), SHOT_N_FRAMES)
        self.assertEqual(len(positions), SHOT_N_FRAMES)
        self.assertEqual(set(positions), set(range(10)))
        self.assertGreater(len(positions), len(set(positions)))

        # Repeated positions must produce the same frame; deduplicating the
        # output would hand the model fewer than 30 frames.
        by_position = dict(zip(positions, frames, strict=True))
        for position, frame in zip(positions, frames, strict=True):
            self.assertTrue(np.array_equal(frame, by_position[position]))

    def test_missing_clip_is_an_input_error(self) -> None:
        with self.assertRaises(ShotInputError) as caught:
            sample_clip_frames(self.directory / "missing.avi")
        self.assertIn("not found", str(caught.exception))

    def test_non_video_file_is_an_input_error(self) -> None:
        path = self.directory / "not_a_video.avi"
        path.write_bytes(b"garbage bytes")
        with self.assertRaises(ShotInputError):
            sample_clip_frames(path)

    def test_zero_sample_count_is_rejected(self) -> None:
        path = write_clip(self.directory, frame_count=30)
        with self.assertRaises(ShotInputError):
            sample_clip_frames(path, 0)


if __name__ == "__main__":
    unittest.main()

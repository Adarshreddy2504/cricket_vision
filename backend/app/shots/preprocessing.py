"""
preprocessing.py — Turning a delivery clip into the exact tensor the model expects.

THIS IS NOT A TUNING SURFACE
----------------------------
Everything here reproduces the upstream Adarsh pipeline: the `IMAGE_TRANSFORM`
order, the 30-frame sample count, the `np.linspace` sampling, and the rule that
a short clip is handled by REPEATING indices rather than by sending fewer
frames. A different preprocessing pipeline would invalidate the checkpoint's
behaviour while every test still passed, so these functions are reproduced
as-is and must not be "cleaned up".

WHY REPETITION IS REQUIRED, NOT INCREMENTAL
-------------------------------------------
`ImprovedSOTAModel` is a fixed-temporal-length transformer: `PositionalEncoding`
is a `(1, n_frames, d_model)` buffer and `forward()` raises on any
`T != n_frames`. A delivery clip shorter than 30 frames therefore has two
honest options — repeat frames, or report no shot at all. Upstream repeats, and
a shot label is worth more than an exact frame count. It is never silent: the
`ShotClassification` record carries `frames_repeated` and
`unique_frames_sampled`.

VALIDATION IS ADDED, MATHS IS NOT
---------------------------------
The upstream transform is kept byte-for-byte, but malformed frames are rejected
with the frame index and the actual shape/dtype instead of failing deep inside
PIL. The model needs BGR uint8 frames from OpenCV; anything else is an input
error, not something to coerce.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from .constants import IMAGENET_MEAN, IMAGENET_STD, SHOT_IMAGE_SIZE, SHOT_N_FRAMES
from .errors import ShotInputError

#: Identical to upstream's IMAGE_TRANSFORM, in the same order.
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((SHOT_IMAGE_SIZE, SHOT_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

#: A retained frame source is either a mapping of 0-based clip position to frame
#: (a sparse subset of the clip) or a dense sequence indexed by clip position.
FrameSource = Mapping[int, np.ndarray] | Sequence[np.ndarray]


def sample_indices(total_frames: int, n: int = SHOT_N_FRAMES) -> list[int]:
    """
    Which 0-based clip positions to sample from a clip of *total_frames*.

    Always returns exactly `n` positions when the clip has at least one frame,
    because the model has no shorter-input path. A clip below `n` frames gets
    repeated positions; the caller records that fact.

    A clip with zero frames returns `[]`: there is genuinely nothing to sample,
    and no amount of repetition invents a frame that does not exist. A negative
    count is not an empty clip; it is a caller bug and is rejected.
    """
    if isinstance(total_frames, bool) or not isinstance(total_frames, numbers.Integral):
        raise ShotInputError(f"total_frames must be an integer, got {total_frames!r}")
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise ShotInputError(f"n must be an integer, got {n!r}")
    total_frames = int(total_frames)
    n = int(n)
    if n < 1:
        raise ShotInputError(f"n must be at least 1, got {n}")

    if total_frames < 0:
        raise ShotInputError(f"total_frames must be >= 0, got {total_frames}")
    if total_frames == 0:
        return []
    if total_frames == 1:
        return [0] * n
    return [int(i) for i in np.linspace(0, total_frames - 1, n, dtype=np.int64)]


def _validate_frame(frame: np.ndarray, index: int) -> None:
    if not isinstance(frame, np.ndarray):
        raise ShotInputError(
            f"frame {index} is {type(frame).__name__}, not a numpy array"
        )
    if frame.ndim != 3:
        raise ShotInputError(
            f"frame {index} has shape {frame.shape}; expected (H, W, 3)"
        )
    if frame.shape[2] != 3:
        raise ShotInputError(
            f"frame {index} has {frame.shape[2]} channels; expected 3 (BGR)"
        )
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ShotInputError(f"frame {index} is empty: shape {frame.shape}")
    if frame.dtype != np.uint8:
        raise ShotInputError(
            f"frame {index} has dtype {frame.dtype}; expected uint8 BGR from OpenCV"
        )


def frames_to_tensor(frames: Sequence[np.ndarray]) -> torch.Tensor:
    """
    BGR uint8 frames -> `(1, 30, 3, 224, 224)` float tensor.

    Raises `ShotInputError` if the count is not exactly 30, or if any frame is
    not a uint8 BGR image. Padding, truncating or coercing would change what
    the network sees without anything recording it.
    """
    if len(frames) != SHOT_N_FRAMES:
        raise ShotInputError(
            f"expected exactly {SHOT_N_FRAMES} frames, got {len(frames)}"
        )

    tensors = []
    for index, frame in enumerate(frames):
        _validate_frame(frame, index)
        tensors.append(IMAGE_TRANSFORM(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    return torch.stack(tensors, dim=0).unsqueeze(0)


def sample_clip_frames(
    clip_path: str | Path,
    n: int = SHOT_N_FRAMES,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Sample frames from an already-written clip with ONE sequential decode.

    Deliberately a linear read rather than `n` seeks: for a short delivery clip
    a sequential pass is simpler, faster and cannot land on the wrong frame the
    way `CAP_PROP_POS_FRAMES` can.

    `wanted` may contain repeated positions (a clip shorter than `n` frames), so
    each distinct position is decoded once and the result is rebuilt from
    `wanted` in order. Deduplicating the output would hand the model fewer than
    `n` frames and silently reintroduce the defect this path exists to avoid.

    Returns `(frames, positions)`, where positions are 0-based within the clip.
    Raises `ShotInputError` if the file cannot be opened, reports no frames, or
    ends before a sampled position is reached.
    """
    path = Path(clip_path)
    if not path.is_file():
        raise ShotInputError(f"clip not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ShotInputError(f"could not open clip: {path}")

    try:
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported <= 0:
            raise ShotInputError(
                f"clip reports no frame count ({reported}); cannot sample it: {path}"
            )

        wanted = sample_indices(reported, n)
        if not wanted:
            return [], []

        wanted_set = set(wanted)
        collected: dict[int, np.ndarray] = {}
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted_set:
                collected[index] = frame
            index += 1
            if len(collected) == len(wanted_set):
                break

        missing = sorted(position for position in wanted_set if position not in collected)
        if missing:
            raise ShotInputError(
                f"clip ended before {len(missing)} sampled frame(s); first missing "
                f"position {missing[0]} of a reported {reported}: {path}"
            )

        # Rebuilt from `wanted`, not from `collected`, so repeated positions
        # produce repeated frames and the model always receives exactly n.
        return [collected[position] for position in wanted], wanted
    finally:
        cap.release()


__all__ = [
    "IMAGE_TRANSFORM",
    "FrameSource",
    "frames_to_tensor",
    "sample_clip_frames",
    "sample_indices",
]

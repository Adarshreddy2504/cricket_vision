"""
constants.py — Values that define the shot classifier's contract.

THE VOCABULARY BELONGS TO THE MODEL, NOT TO THIS MODULE
-------------------------------------------------------
`shot_classifier.ckpt` stores a 10-way classifier but no class names. The order
below is inherited from the upstream training project and is the only place the
index-to-name mapping is written down. Re-ordering it would keep every test
green while silently relabelling every prediction, so it is treated as frozen.

The input geometry is equally frozen. `SHOT_N_FRAMES`, `SHOT_IMAGE_SIZE`,
`SHOT_TEMPORAL_DIM` and the ImageNet statistics describe the tensor the trained
weights were fitted to; `preprocessing.py` explains why they are reproduced
exactly rather than tuned.
"""

from __future__ import annotations

#: Index → label. The index is the position in this tuple; do not re-order.
SHOT_CLASSES: tuple[str, ...] = (
    "Cover",
    "Defense",
    "Flick",
    "Hook",
    "Late Cut",
    "Lofted",
    "Pull",
    "Square Cut",
    "Straight",
    "Sweep",
)

SHOT_NUM_CLASSES: int = len(SHOT_CLASSES)

#: The fixed temporal length of `ImprovedSOTAModel`.
SHOT_N_FRAMES: int = 30

#: The square resolution every sampled frame is resized to.
SHOT_IMAGE_SIZE: int = 224

#: The transformer width between the backbone and the classifier.
SHOT_TEMPORAL_DIM: int = 256

#: The complete model input, as (B, T, C, H, W).
SHOT_INPUT_SHAPE: tuple[int, int, int, int, int] = (
    1,
    SHOT_N_FRAMES,
    3,
    SHOT_IMAGE_SIZE,
    SHOT_IMAGE_SIZE,
)

#: Default file name inside the models directory.
SHOT_MODEL_FILENAME: str = "shot_classifier.ckpt"

#: Checksum documented for the upstream checkpoint, verified against the copy
#: shipped outside git. It is a reference value, not a default requirement:
#: `CRICKET_SHOT_MODEL_MD5` turns verification on for a given installation.
DOCUMENTED_SHOT_MODEL_MD5: str = "2974efdcf449c8447bef107d94633217"

#: Size of that checkpoint in bytes.
DOCUMENTED_SHOT_MODEL_BYTES: int = 72_010_842

#: ImageNet normalisation, in RGB order, applied after ToTensor().
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: The only statuses a `ShotClassification` may carry.
SHOT_STATUSES: tuple[str, ...] = (
    "ok",
    "model_unavailable",
    "invalid_input",
    "inference_failed",
)

__all__ = [
    "DOCUMENTED_SHOT_MODEL_BYTES",
    "DOCUMENTED_SHOT_MODEL_MD5",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SHOT_CLASSES",
    "SHOT_IMAGE_SIZE",
    "SHOT_INPUT_SHAPE",
    "SHOT_MODEL_FILENAME",
    "SHOT_NUM_CLASSES",
    "SHOT_N_FRAMES",
    "SHOT_STATUSES",
    "SHOT_TEMPORAL_DIM",
]

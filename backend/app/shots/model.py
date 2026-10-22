"""
model.py — The architecture the checkpoint was trained with.

Input:  (B, 30, 3, 224, 224)
Output: (B, 10) logits

    EfficientNet-B0  ->  spatial attention  ->  1280 -> 256
        -> sinusoidal positional encoding  ->  2-layer transformer
        -> temporal attention  ->  classifier  ->  10 shot classes

THIS CLASS IS AN ARCHITECTURE, NOT A TRAINED MODEL
--------------------------------------------------
The backbone is built with `weights=None` deliberately. Upstream constructs
`efficientnet_b0(weights="IMAGENET1K_V1")`, which downloads ~20 MB of ImageNet
weights from the internet the first time it runs. The downloaded values are
then completely overwritten by `strict=True` loading of the shot checkpoint, so
the only thing the download achieves is a silent network request at import
time. This project never downloads model weights: the full state dict is
supplied by `inference.load_shot_model()`.

That means an `ImprovedSOTAModel()` constructed directly contains random
weights and must not be used to classify anything. Load the checkpoint instead.

WHY THE MODULE STRUCTURE IS REPRODUCED EXACTLY
----------------------------------------------
`load_state_dict(strict=True)` matches parameter names and shapes. Renaming
`backbone` or `temporal_attention` would make the checkpoint unloadable, and
changing `nhead`, `dim_feedforward` or the classifier widths would make it
loadable-but-wrong. This file therefore mirrors the reference architecture
field for field; `docs/SHOTS.md` records which values came from the checkpoint's
own `hyper_parameters`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from .constants import SHOT_N_FRAMES, SHOT_NUM_CLASSES, SHOT_TEMPORAL_DIM


class SpatialAttention(nn.Module):
    """Channel attention applied to the backbone's feature maps."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 8),
            nn.ReLU(),
            nn.Linear(feature_dim // 8, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        return x * weights


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Stored as the buffer `pe` so it matches the checkpoint key
    `positional_encoding.pe` with shape `(1, n_frames, d_model)`.
    """

    def __init__(self, d_model: int, max_len: int = SHOT_N_FRAMES):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TemporalAttention(nn.Module):
    """Attention pooling across the temporal frames: 256 -> 64 -> 1."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        attended = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return attended, weights


class ImprovedSOTAModel(nn.Module):
    """The deployment architecture; see the module docstring before using it."""

    def __init__(
        self,
        num_classes: int = SHOT_NUM_CLASSES,
        temporal_dim: int = SHOT_TEMPORAL_DIM,
        n_frames: int = SHOT_N_FRAMES,
    ):
        super().__init__()

        if num_classes < 1:
            raise ValueError(f"num_classes must be at least 1, got {num_classes}")
        if temporal_dim < 4 or temporal_dim % 4 != 0:
            raise ValueError(
                f"temporal_dim must be divisible by the 4 attention heads, got {temporal_dim}"
            )
        if n_frames < 1:
            raise ValueError(f"n_frames must be at least 1, got {n_frames}")

        self.num_classes = num_classes
        self.n_frames = n_frames
        self.feature_dim = 1280
        self.temporal_dim = temporal_dim

        # weights=None: the checkpoint supplies every value, and this project
        # never performs a silent download. See the module docstring.
        self.backbone = efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()

        self.spatial_attention = SpatialAttention(self.feature_dim)
        self.input_proj = nn.Linear(self.feature_dim, temporal_dim)

        self.positional_encoding = PositionalEncoding(
            d_model=temporal_dim,
            max_len=n_frames,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.3,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.temporal_attention = TemporalAttention(temporal_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(temporal_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        # The Lightning checkpoint carries this training artefact but the
        # deployment architecture recreates it, so `load_shot_model` restores it
        # from a fresh model before the strict load.
        self.register_buffer(
            "class_weights",
            torch.ones(num_classes, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected input shape (B,T,C,H,W), got {tuple(x.shape)}")

        batch_size, time_steps, channels, height, width = x.shape
        if time_steps != self.n_frames:
            raise ValueError(f"expected {self.n_frames} frames, got {time_steps}")

        frames = x.reshape(batch_size * time_steps, channels, height, width)

        features = self.backbone.features(frames)
        features = self.spatial_attention(features)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, start_dim=1)
        features = features.reshape(batch_size, time_steps, self.feature_dim)

        features = self.input_proj(features)
        features = self.positional_encoding(features)
        features = self.temporal_encoder(features)

        attended_features, _ = self.temporal_attention(features)
        return self.classifier(attended_features)


__all__ = [
    "ImprovedSOTAModel",
    "PositionalEncoding",
    "SpatialAttention",
    "TemporalAttention",
]

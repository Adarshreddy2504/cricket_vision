"""
constants.py — Every value the pitch module applies, stated once.

Three groups:

  1. PITCH DIMENSIONS. The Laws of Cricket set the size of a pitch. They
     are the ONLY sizes this module knows, and they are used as an explicit,
     labelled ASSUMPTION — never as a fact the camera told us.

  2. GEOMETRIC SANITY LIMITS. The checks that separate "four corners of a
     pitch" from "four arbitrary points". They are expressed as shares of the
     frame, so they mean the same thing on a 640x360 clip and a 4K one.

  3. DETECTOR DEFAULTS. The confidence floor, the inference size and the
     box-inset used to turn a detection box into four approximate corners.

None of these values is fitted to a particular video. The dimensions come
from the Laws; the limits are chosen to reject geometry that cannot be a
pitch in any footage (see the comments on each).
"""

from __future__ import annotations

# ── Pitch dimensions (Law 6 of the Laws of Cricket) ──────────────────────────
# The distance between the two creases: 22 yards.
PITCH_LENGTH_M = 20.12

# The distance between the bowling creases: 8 ft 8 in.
CREASE_WIDTH_M = 2.64

# ── Geometric sanity limits ──────────────────────────────────────────────────
# How far outside the frame a corner may sit before the calibration is
# refused. A mask boundary can bleed a pixel or two past the edge of the
# picture; a corner halfway to the next frame is a different object.
CORNER_OUT_OF_BOUNDS_SLACK = 0.02

# Smallest share of the frame the pitch quad may cover. Below this the quad
# is a handful of pixels — smaller than any pitch seen in broadcast footage,
# where the pitch strip is a visible band of the picture.
MIN_QUAD_AREA_FRACTION = 0.001

# Largest share of the frame the pitch quad may cover. Above this the quad
# is the picture: a full-frame mask is a segmentation failure, not a pitch.
MAX_QUAD_AREA_FRACTION = 0.90

# Share of the frame a detected pitch MASK may cover, and the aspect ratio
# (longest side over shortest) it may have. Both were measured on real pitch
# masks against the crowd false positive in the upstream project.
MASK_MIN_AREA_FRACTION = 0.01
MASK_MAX_AREA_FRACTION = 0.25
MASK_MIN_ASPECT_RATIO = 1.0
MASK_MAX_ASPECT_RATIO = 6.0

# Simplification fractions tried, in order, when a mask boundary has to be
# reduced to exactly four corners.
MASK_APPROX_POLYDP_FRACTIONS = (0.02, 0.03, 0.05, 0.08)

# A hull with more vertices than this has a boundary too wiggly to trust.
# Reducing it to four corners would be fabrication rather than simplification.
MASK_MAX_HULL_VERTICES = 64

# ── Detector defaults ────────────────────────────────────────────────────────
# Minimum confidence a detection must reach before it is considered at all.
DETECTOR_CONF_THRESHOLD = 0.25

# Two masks whose confidences are within this band are a TIE, and the tie is
# broken by consistency with the last accepted estimate rather than by a
# fraction of a percent of confidence. Confidence is the model's evidence;
# position is only ours, so it decides a tie and never a contest.
DETECTOR_CONF_TIE_BAND = 0.05

# The frame is handed to the model at its ORIGINAL resolution and ultralytics
# letterboxes it to this size itself. Pre-resizing the frame is what squashed
# frames and broke the aspect ratio in an earlier version of this project.
DETECTOR_IMGSZ = 640

# How far the far edge of a detected pitch box is pulled IN, as a fraction of
# the box width, to make the derived trapezoid narrower at the far end than
# at the near end — the way a pitch looks in perspective.
BOX_TOP_INSET = 0.08

# Share of the frame the pitch box must cover before its derived corners are
# worth reporting.
MIN_BOX_AREA_FRACTION = 0.002

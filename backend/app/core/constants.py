"""
constants.py — Constants for the ball-tracking module.

Three groups, and every value in them is stated once, here, so it can be
audited or replaced without touching the code that applies it.

  1. BALL DETECTION. What the YOLO detector is allowed to believe: how
     confident a box must be, the size the frame is resized to for inference,
     and the range of bounding-box diagonals that could plausibly be a
     cricket ball. The size range is what keeps sponsor logos and crowd
     movement out of the trajectory.

  2. KALMAN TRACKING. How long the filter is allowed to coast without a
     detection before it declares the ball lost, how far a detection may sit
     from the prediction and still be associated with it, and how many
     positions the trajectory trail keeps.

  3. THE TRACKING SERVICE. How many confirmed detections a track needs before
     it is presented as a flight rather than a stray false positive, and the
     schema version of the document the service produces.

None of these values is fitted to a particular video. They are the values the
upstream CricketVision project uses, kept here so this module behaves the
same way and so a future part can compare against it.
"""

# ── Ball detection ───────────────────────────────────────────────────────────
# Minimum confidence a YOLO box must reach before it is considered at all.
DETECTOR_CONF_THRESHOLD = 0.30

# The frame is downscaled to this size for inference (speed) and the answer is
# scaled back to the source resolution. The aspect ratio is preserved, so
# 640x360 keeps a 16:9 frame 16:9.
DETECTOR_RESIZE_W = 640
DETECTOR_RESIZE_H = 360

# Bounding-box diagonal bounds, in the RESIZED frame's pixels. A cricket ball
# in broadcast footage is a few pixels across; anything much larger is a
# player's boot, a sponsor board or a crowd member, and anything much smaller
# is noise.
DETECTOR_MIN_DIAG_PX = 4
DETECTOR_MAX_DIAG_PX = 80

# ── Kalman tracking ──────────────────────────────────────────────────────────
# Maximum number of trajectory positions kept in the rolling trail.
MAX_TRAIL = 120

# How many consecutive frames the filter may coast on its own prediction
# before it declares the ball lost. At 25 fps this is 0.32 s — long enough to
# cover the ball passing behind a bat, short enough that a genuinely new ball
# is not inherited from the previous one.
MAX_MISSED = 8

# How far, in pixels, a detection may sit from the current prediction and
# still be associated with it. Anything further away is a false positive and
# is refused rather than snapped to.
GATE_RADIUS = 120

# ── Tracking service ─────────────────────────────────────────────────────────
# How many CONFIRMED detections a track needs before it is presented as a
# flight. A single stray box that survived the gate is not a delivery, and
# saying so is cheaper than letting a consumer believe it is.
MIN_TRACK_DETECTIONS = 3

# Schema version of the document produced by `TrackingResult.as_dict()`.
TRACKING_SCHEMA_VERSION = "1.0"

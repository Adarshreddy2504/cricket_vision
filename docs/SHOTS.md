# Shot classification and analytics (Part 3)

Turn one delivery's frames into a shot label, and turn Parts 1–3's records into
numbers.

The part has two halves, and they are deliberately independent:

1. **Classification.** Load the upstream checkpoint, reproduce the exact input
   tensor it was trained on, and return a label — or a record saying precisely
   why there is no label.
2. **Analytics.** Pure calculations over trajectories, pitch-plane points and
   shot records. No model, no video decode, no I/O.

Both halves follow the same rule as Parts 1 and 2: **a label exists only if the
model produced it, and a number exists only if the data measured it.** Nothing
here fills a gap, guesses a class, or reports a speed it cannot justify.

---

## What it does

| Component | File | Purpose |
|---|---|---|
| `ShotConfig` / `resolve_shot_model_path` | `backend/app/shots/config.py` | Where the checkpoint is, from argument, environment or the documented default. Never downloads. |
| `load_shot_model` | `backend/app/shots/inference.py` | Strictly loads a checkpoint, checks its MD5 if asked, resolves the device, and raises `ShotModelUnavailable` with a reason on any failure. |
| `ImprovedSOTAModel` | `backend/app/shots/model.py` | The exact upstream architecture: EfficientNet-B0, spatial attention, a 2-layer transformer over 30 frames, temporal attention and a 10-way head. |
| `sample_indices` / `sample_clip_frames` / `frames_to_tensor` | `backend/app/shots/preprocessing.py` | The 30-frame sample and the exact `(1, 30, 3, 224, 224)` tensor. |
| `predict_shot` | `backend/app/shots/inference.py` | Runs the model, validates its output, and returns a `ShotPrediction`. |
| `ShotClassifier` | `backend/app/shots/inference.py` | One delivery → one `ShotClassification`, including the failure cases. |
| `ShotPrediction` / `ShotClassification` | `backend/app/schemas/shots.py` | The typed records and the invariants they enforce. |
| `summarise_trajectory` | `backend/app/analytics/trajectory.py` | Pixel-space counts, path length, gaps and pixel speed. |
| `summarise_pitch_trajectory` | `backend/app/analytics/pitch_plane.py` | Projected metres, path length and an explicitly estimated km/h. |
| `summarise_shots` | `backend/app/analytics/shots.py` | Label counts, shares and score statistics. |
| `summarise_delivery` / `summarise_analysis` | `backend/app/analytics/summary.py` | One delivery, then an analysis, with a map of what is missing and why. |
| Analytics records | `backend/app/schemas/analytics.py` | `TrajectorySummary`, `PitchPlaneSummary`, `ShotDistribution`, `DeliveryAnalytics`, `AnalysisSummary`. |

---

## The checkpoint

The supplied model weights are tracked in git under `backend/models/`. The
checkpoint used for development is:

| | |
|---|---|
| File | `backend/models/shot_classifier.ckpt` |
| Size | 72,010,842 bytes |
| MD5 | `2974efdcf449c8447bef107d94633217` |
| Format | PyTorch Lightning checkpoint with a `state_dict` |
| Classes | 10 |
| Frames per input | 30 |
| Transformer width | 256 |

Two facts about that file matter more than the rest:

**It stores no class names.** The index-to-name mapping lives in
`backend/app/shots/constants.py`:

```
Cover, Defense, Flick, Hook, Late Cut, Lofted, Pull, Square Cut, Straight, Sweep
```

The order is inherited from the upstream training project. Re-ordering it would
keep every test green while silently relabelling every prediction, so it is
frozen and treated as part of the checkpoint's contract.

**It stores no evaluation metrics.** There is no accuracy, precision or recall
in the checkpoint, and this repository contains no labelled shot footage, so no
accuracy is claimed anywhere — not in the code, not in a document, not in a
test. A number would have to come from an evaluation that was actually run.

### Loading rules

- `torch.load(..., weights_only=True)` is used, so loading a checkpoint cannot
  execute arbitrary pickle code.
- The state dict is unwrapped from `state_dict` or `model_state_dict`, and
  `module.` / `model.` prefixes are stripped.
- The saved `class_weights` tensor is a training artefact and is **not**
  restored; the fresh model's own buffer is kept.
- Loading is `strict=True`. A checkpoint that does not match the architecture
  is reported as unavailable rather than partially loaded.
- The backbone is built with `weights=None`. Nothing is downloaded from
  torchvision; every weight comes from the checkpoint.
- A missing, empty, corrupt, wrong-architecture, checksum-mismatched or
  unusable-device checkpoint all become `ShotModelUnavailable` with a readable
  reason. There is **no fallback to a different checkpoint**, because different
  weights produce confident labels.

---

## The input tensor, and why it is not configurable

`ImprovedSOTAModel` has a fixed temporal length. Its positional encoding is a
`(1, 30, 256)` buffer and its forward pass raises for any other frame count.
The preprocessing therefore reproduces the upstream pipeline exactly:

1. Sample **30 positions** from the clip with `np.linspace`.
   A clip of 30 frames or more gets 30 distinct positions. A shorter clip gets
   **repeated** positions — never fewer than 30 frames, because the model has no
   shorter-input path.
2. Resize each frame to **224 × 224**.
3. Convert BGR (OpenCV's order) to RGB.
4. Scale to `[0, 1]` and apply ImageNet normalisation:
   mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.
5. Stack into a float32 tensor of shape `(1, 30, 3, 224, 224)`.

A different sampling count, image size or normalisation would invalidate the
checkpoint's behaviour while every shape assertion still passed, so these values
are not a tuning surface.

Repetition is never silent. A successful record carries `frames_sampled`,
`unique_frames_sampled`, `frames_repeated`, `sampled_clip_frames` and — when a
`frame_offset` was supplied — `sampled_original_frames`. A label from 10 unique
frames repeated to 30 is distinguishable from one produced from 30 distinct
frames.

Frames must be BGR `uint8` NumPy arrays of shape `(H, W, 3)`. Anything else is
reported as `invalid_input` with the frame index and the actual shape or dtype;
frames are never coerced, resized by hand or silently dropped.

### Choosing a frame source

`ShotClassifier.classify()` accepts:

| Argument | Behaviour |
|---|---|
| `in_memory_frames` | A dense sequence indexed by 0-based clip position, or a mapping from clip position to frame (a sparse subset). Preferred when present. |
| `clip_path` | The written clip, re-read only when no in-memory frames are supplied. |
| `expected_total_frames` | The clip's frame count. Required for a sparse mapping, because a missing position and a short clip are different problems. |
| `frame_offset` | `clip_start_frame - 1`. When given, sampled positions are also recorded as source frame numbers. |
| `delivery_id` | Only used to make log lines attributable. |

---

## The classification record

`status` is the first field a consumer should read.

| `status` | Meaning | Populated fields |
|---|---|---|
| `ok` | The model ran and produced a valid 10-way distribution. | `label`, `score`, `class_probabilities`, frame evidence, model provenance. |
| `model_unavailable` | No usable checkpoint was loaded. | `error` with the reason. No label, score or probabilities. |
| `invalid_input` | No usable frames, or frames the model cannot take. | `error`, `input_source`, frame evidence where available. |
| `inference_failed` | The model ran and failed, or returned nonsense. | `error`, frame evidence, model provenance. |

The schema enforces the table rather than trusting callers:

- an `ok` record always has a label in the vocabulary, a score in `[0, 1]`, the
  complete 10-key distribution, and no `error`;
- a non-`ok` record always has an `error`, and never a label, score or
  probabilities;
- `frames_repeated` cannot disagree with `unique_frames_sampled <
  frames_sampled`;
- `sampled_original_frames` must align with `sampled_clip_frames`.

`score` is the softmax probability of `label` and must equal the maximum of the
distribution within `1e-6`. Probabilities are rounded to six decimals for stable
JSON, so a consumer summing them may see a discrepancy of a few millionths.

Provenance is attached to every record that has a loaded model, so a label can
always be traced to the weights and device that produced it: `model_path`,
`model_size_bytes`, `model_md5`, `model_device`.

`classify()` never raises for delivery data — bad frames, a missing clip and a
forward-pass failure all come back as a status. A malformed `frame_offset` is a
caller bug and does raise `ValueError`.

---

## Analytics: only what the data supports

### Pixel trajectory — `summarise_trajectory`

Accepts a Part 1 `Trajectory`, a sequence of points, or a stored mapping with
`points`. The interface is structural: `app.tracking` is never imported.

- `path_length_px` sums Euclidean distances between points on **consecutive
  frames**. A jump across a gap is not observed motion: it is excluded from the
  path and reported as `unobserved_span_px`, with `gaps` naming the missing
  frames.
- `detected_path_length_px` applies the same rule to detected points only,
  skipping predicted ones.
- `net_displacement_px` is the straight line from the first point to the last,
  gaps included.
- `mean_speed_px_s` is `observed distance / observed time`. It is an average of
  frame-to-frame motion, not an instantaneous or release speed.
- Without an fps, distances and gaps remain; durations and speeds are `None`.

Zero and unavailable are different, everywhere:

| Situation | Result |
|---|---|
| No points | `path_length_px = None`, `available = False` |
| One point | `path_length_px = 0.0` (a real observation of no movement) |
| Two identical positions | speed `0.0` |
| Only gap-spanning steps | path `None`, `unobserved_span_px` set |
| No fps | duration and speed `None`; distances still measured |

### Pitch plane — `summarise_pitch_trajectory`

Accepts a Part 2 `PitchTrajectory` or an equivalent mapping, structurally.

**It requires `units == "metre"`.** A `"pitch_fraction"` trajectory, or one
that declares no units at all, raises `AnalyticsInputError`. A fraction of an
unknown pitch is not a distance, and a speed derived from it would be a fiction.

Every record it produces says:

- `units: "metre"`;
- `speed_is_estimate: True`;
- `speed_is_calibrated: False`.

The ball is airborne; Part 2 projects it onto the ground plane because that is
the only plane the calibration describes. The result is a projected average, not
a radar measurement.

`outside_pitch_count` reports points Part 2 flagged as outside the measured
quad. They are **kept** in the distance — dropping them would create fake gaps —
and exposed separately so a consumer can judge the result.

### Shot distribution — `summarise_shots`

Accepts `ShotClassification` records, plain label strings or stored mappings.

- `counts` always contains all 10 classes, including genuine zeros.
- A failed classification is **never** turned into a class. It is counted in
  `unclassified` and grouped by status in `unclassified_reasons`.
- `shares` is computed over classified records only, and is **empty** when
  nothing was classified: a proportion with a zero denominator is undefined,
  not ten zeros.
- Score statistics (`mean_score`, `min_score`, `max_score`, `scored`) use only
  scores that are actually present on classified records.
- An unknown label raises `AnalyticsInputError`. The vocabulary belongs to the
  checkpoint; a label outside it is a data error, not a new class for the chart.

### Delivery and analysis — `summarise_delivery`, `summarise_analysis`

`DeliveryAnalytics` carries a `shot`, a `trajectory`, a `pitch` summary and an
`unavailable` map keyed by field name (`shot`, `trajectory`, `pitch`, `time`).
The map explains why a field is absent. A genuine zero never appears in it.

The distinction matters in two places:

- A trajectory with no points is supplied but unavailable; the key is absent,
  and `path_length_px` is `None`.
- An invalid fps removes the timed fields and records the reason against the
  affected summariser, rather than pretending the fps was never supplied.

`AnalysisSummary` rolls up already-summarised deliveries. It counts
`delivery_count`, `with_shot`, `with_trajectory`, `with_pitch` and
`with_speed_kmh`. Only deliveries that carry a shot record take part in the
distribution; a delivery with no shot record at all is visible in its own
`unavailable` map, not silently counted as an unclassified shot.

### Invalid data is an error, not something to sort

Both trajectory summarisers raise `AnalyticsInputError` for duplicate or
decreasing frames, a frame below 1, non-finite coordinates, a missing or invalid
`source`, an invalid `inside_pitch` flag, and a non-positive or non-numeric fps.
Inconsistent records are reported, never repaired: sorting points would invent a
timebase, and interpolating across a gap would invent motion.

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CRICKET_MODELS_DIR` | `backend/models/` | Directory holding the checkpoint. |
| `CRICKET_SHOT_MODEL` | — | Explicit checkpoint path; wins over the directory. |
| `CRICKET_SHOT_MODEL_MD5` | unset | Expected MD5. When set, a mismatch is a load failure. |
| `CRICKET_SHOT_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:1`, … |

Relative paths resolve against the repository root, not the current working
directory, so the loader finds the same file however it was started. An
explicitly configured path that is missing is an error; it never falls back to
the default location, and nothing is ever downloaded.

`ShotConfig.as_metadata()` never raises. A missing checkpoint is reported as
`model_available: False` with `model_error`, which is what a status endpoint
needs.

---

## Running the tests

From the repository root:

```bash
pytest backend
python -m ruff check backend
```

The suite uses synthetic checkpoints and tiny written clips, so it does not need
the 68 MB weights. Two groups are skipped by design:

- the tests that import the **real** Part 1 and Part 2 schemas, until those
  branches are merged;
- the real-checkpoint test, unless `CRICKET_SHOT_MODEL` is set.

To run that last test against a local checkpoint:

```powershell
$env:CRICKET_SHOT_MODEL = "backend\models\shot_classifier.ckpt"
pytest backend -q -k RealCheckpoint
```

The test loads the real weights, runs a real forward pass, and asserts that the
label is in the vocabulary, the distribution sums to 1, and the model metadata
is attached. It does not assert an accuracy.

---

## Integration and known limitations

- **No hard dependency on Parts 1 and 2.** The analytics modules import only
  Part 3 schemas and each other. Part 1 and Part 2 records are accepted
  structurally, which is what lets this branch be tested standalone and merged
  without changing their schemas.
- **No API or frontend in this part.** These are library modules and typed
  records only.
- **One label per delivery clip.** The model classifies a fixed 30-frame sample;
  there is no per-frame or per-segment shot labelling.
- **No accuracy claim.** The checkpoint stores no metrics and the repository has
  no labelled shot footage. Any accuracy figure must come from a real evaluation
  on labelled data.
- **Unknown shots collapse to the nearest class.** A helicopter shot or a
  switch hit is still forced into the 10-way softmax; the distribution makes the
  uncertainty visible but does not add a class.
- **Projected speed, not release speed.** The km/h figure is an average over the
  ground-plane projection of an airborne ball, and is always marked as an
  estimate that is not calibrated.
- **Constant frame rate is assumed.** Frame numbers are treated as evenly
  spaced at the supplied fps; variable-frame-rate footage is not modelled.

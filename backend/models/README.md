# Model weights

These four checkpoints are included in this repository so a fresh clone can
analyse footage without a separate weights download. They are the only files
matching the weight patterns that `.gitignore` allows. Docker images do **not**
copy them in; compose mounts this directory read-only at `/models`.

Every loader still checks that the file exists and reports a missing or
incompatible weight clearly. Nothing downloads a checkpoint automatically.

| File | Size | MD5 | Required? | Purpose |
|---|---:|---|---|---|
| `ball_detector.pt` | 6,225,450 B | `ADDE61B5409FCE5EC6583867D00C070C` | **Yes** | YOLO11 single-class ball detector. |
| `shot_classifier.ckpt` | 72,010,842 B | `2974EFDCF449C8447BEF107D94633217` | No | EfficientNet-B0 + temporal transformer, 10 shot classes. |
| `pitch_seg_yolo11s.pt` | 20,521,757 B | `9235C4B3ED956EF56D4C80E2F6BB61A8` | No | YOLO11 segmentation model for the full pitch. |
| `pitch_detect.pt` | 22,517,987 B | `FE980582CB5446062DB5E65C238A8AFB` | No | YOLO11 pitch-box detector; approximate fallback. |

If the ball detector is absent, an analysis fails loudly rather than returning
an empty result. Missing pitch or shot weights produce explicit unavailable
states and diagnostic flags, leaving those values `null`.

## Configuration

The default directory is `backend/models/`. Override it with:

```bash
export CRICKET_MODELS_DIR=/path/to/models
# Or point at individual files:
export CRICKET_BALL_MODEL=/path/to/ball_detector.pt
export CRICKET_PITCH_BOX_MODEL=/path/to/pitch_detect.pt
export CRICKET_PITCH_SEG_MODEL=/path/to/pitch_seg_yolo11s.pt
export CRICKET_SHOT_MODEL=/path/to/shot_classifier.ckpt
```

Relative paths resolve against the repository root, not the current working
directory. In Docker, compose sets `CRICKET_MODELS_DIR=/models` and mounts this
directory there.

`CRICKET_SHOT_MODEL_MD5` is an optional checksum gate. It is not enforced unless
you set it.

## Verifying a copy

```bash
python - <<'PY'
import hashlib
from pathlib import Path

expected = {
    "ball_detector.pt": "adde61b5409fce5ec6583867d00c070c",
    "pitch_detect.pt": "fe980582cb5446062db5e65c238a8afb",
    "pitch_seg_yolo11s.pt": "9235c4b3ed956ef56d4c80e2f6bb61a8",
    "shot_classifier.ckpt": "2974efdcf449c8447bef107d94633217",
}
for name, digest in expected.items():
    path = Path("backend/models") / name
    actual = hashlib.md5(path.read_bytes()).hexdigest()
    print(name, actual, "OK" if actual == digest else "MISMATCH")
PY
```

## Device selection

`CRICKET_BALL_DEVICE`, `CRICKET_PITCH_DEVICE` and `CRICKET_SHOT_DEVICE` default
to `auto`, which chooses CUDA when the installed PyTorch build can see it and
CPU otherwise. Use `cpu`, `cuda` or `cuda:N` to force a device.

The default Docker image installs the CPU PyTorch wheels. `compose.gpu.yaml`
selects the CUDA 12.6 wheel index and requests an NVIDIA device; it requires a
working NVIDIA Container Toolkit setup on the host.

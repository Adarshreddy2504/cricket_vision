# syntax=docker/dockerfile:1.7

# CricketVision backend image.
#
# CPU is the default. The GPU image is produced by compose.gpu.yaml, which
# overrides TORCH_INDEX_URL with the CUDA 12.6 wheel index. Model weights are
# deliberately NOT copied into the image; compose mounts ./backend/models at
# /models read-only so weights can be replaced without rebuilding.

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CRICKET_DATA_DIR=/data \
    CRICKET_MODELS_DIR=/models

# OpenCV needs these runtime libraries even in a headless server. ffmpeg is
# included so video files that rely on system codecs can be decoded and clipped.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the GPU/CPU-specific PyTorch build before the common requirements.
# The pinned torch/torchvision versions in requirements.txt are already
# satisfied by this step, so pip does not replace the selected wheel.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}"

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

COPY ruff.toml ./
COPY backend ./backend

# COPY from a Windows host can mark every file executable. Normalise the modes
# so the Linux lint run matches the native run, then create the runtime
# directories before dropping privileges. A named volume is initialised from
# /data, so the unprivileged user can write analysis state immediately.
RUN chmod -R u=rwX,go=rX backend \
    && mkdir -p /data \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && chown -R app:app /app /data

USER app

EXPOSE 8000

# The health endpoint reports versions and model availability without loading
# torch or touching the network. Use it as the container readiness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)" || exit 1

# One worker is intentional: the job registry and model lifecycle are
# process-local. Multiple workers would split status polling across processes.
CMD ["uvicorn", "main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]

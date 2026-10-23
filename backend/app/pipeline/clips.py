"""Candidate delivery windows and the clip files that carry them.

A confirmed tracking track is not a validated delivery boundary. It is one
continuous flight the Part 1 tracker believes it saw. This module turns each
confirmed track into a *candidate* window with a little context before and
after it, writes the browser-playable clip in a single pass over the source,
and records enough metadata for the diagnostics document to explain every
window that was clipped, truncated or never written.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from ..tracking.tracking_service import Track

if TYPE_CHECKING:
    from ..api.config import Settings

log = logging.getLogger(__name__)

#: Browser playability is the first requirement. VP8/VP9 WebM is preferred;
#: the MP4 fallback is for OpenCV builds without a WebM encoder.
_WEBM_CODECS = ("VP80", "VP90")
_MP4_CODEC = "mp4v"


@dataclass(frozen=True)
class ClipWindow:
    """The source-frame range proposed for one candidate delivery."""

    delivery_index: int
    track_id: int
    start_frame: int
    end_frame: int
    track_start_frame: int
    track_end_frame: int
    truncated_start: bool = False
    truncated_end: bool = False
    limited_by_max_frames: bool = False

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)

    def as_dict(self) -> dict:
        return {
            "delivery_id": self.delivery_index,
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_count": self.frame_count,
            "track_start_frame": self.track_start_frame,
            "track_end_frame": self.track_end_frame,
            "truncated_start": self.truncated_start,
            "truncated_end": self.truncated_end,
            "limited_by_max_frames": self.limited_by_max_frames,
            "candidate_boundary_validated": False,
        }


@dataclass
class ClipArtifact:
    """What was actually written, with a reason when nothing was."""

    window: ClipWindow
    file_name: str | None = None
    path: Path | None = None
    codec: str | None = None
    frame_count: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.path is not None and self.path.is_file() and self.frame_count > 0

    @property
    def duration_seconds(self) -> float | None:
        if self.fps is None or self.fps <= 0 or self.frame_count <= 0:
            return None
        return float(self.frame_count) / float(self.fps)

    def as_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "codec": self.codec,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


def candidate_windows(
    tracks: Iterable[Track],
    settings: Settings,
    *,
    total_frames: int,
    pre_roll_frames: int | None = None,
    post_roll_frames: int | None = None,
) -> list[ClipWindow]:
    """Build one candidate window per confirmed track, in source order."""
    pre = (
        settings.clip_pre_roll_frames
        if pre_roll_frames is None
        else int(pre_roll_frames)
    )
    post = (
        settings.clip_post_roll_frames
        if post_roll_frames is None
        else int(post_roll_frames)
    )
    max_frames = max(1, int(settings.max_clip_frames))
    last_source_frame = max(1, int(total_frames))

    windows: list[ClipWindow] = []
    ordered = sorted(
        (track for track in tracks if track.is_confirmed),
        key=lambda track: (
            track.start_frame if track.start_frame is not None else 0,
            track.track_id,
        ),
    )
    for index, track in enumerate(ordered, start=1):
        if track.start_frame is None or track.end_frame is None:
            continue
        requested_start = int(track.start_frame) - max(0, pre)
        requested_end = int(track.end_frame) + max(0, post)
        start = max(1, requested_start)
        end = min(last_source_frame, requested_end)
        if end < start:
            continue

        limited = False
        if end - start + 1 > max_frames:
            limited = True
            midpoint = (start + end) // 2
            start = max(1, midpoint - max_frames // 2)
            end = min(last_source_frame, start + max_frames - 1)
            start = max(1, end - max_frames + 1)

        windows.append(
            ClipWindow(
                delivery_index=index,
                track_id=int(track.track_id),
                start_frame=start,
                end_frame=end,
                track_start_frame=int(track.start_frame),
                track_end_frame=int(track.end_frame),
                truncated_start=requested_start < 1,
                truncated_end=requested_end > last_source_frame,
                limited_by_max_frames=limited,
            )
        )
    return windows


class ClipExtractor:
    """Write each candidate window once, preferring browser-playable WebM."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract(
        self,
        source_path: Path,
        windows: list[ClipWindow],
        *,
        clips_dir: Path,
        width: int,
        height: int,
        fps: float,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[int, ClipArtifact]:
        clips_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[int, ClipArtifact] = {
            window.delivery_index: ClipArtifact(
                window=window,
                width=width if width > 0 else None,
                height=height if height > 0 else None,
                fps=fps if fps > 0 else None,
            )
            for window in windows
        }
        if not windows:
            return artifacts

        if width <= 0 or height <= 0:
            self._fail_all(artifacts, "invalid_video_dimensions")
            return artifacts
        if fps <= 0:
            self._fail_all(artifacts, "invalid_video_fps")
            return artifacts

        writers: dict[int, cv2.VideoWriter] = {}
        for window in windows:
            artifact = artifacts[window.delivery_index]
            opened = self._open_writer(
                clips_dir,
                window.delivery_index,
                fps=fps,
                width=width,
                height=height,
            )
            if opened is None:
                artifact.error = "clip_encoder_unavailable"
                continue
            writer, path, codec = opened
            writers[window.delivery_index] = writer
            artifact.path = path
            artifact.file_name = path.name
            artifact.codec = codec

        if not writers:
            return artifacts

        last_needed = max(window.end_frame for window in windows)
        current = 0
        capture = cv2.VideoCapture(str(source_path))
        try:
            if not capture.isOpened():
                self._fail_all(artifacts, "source_video_unreadable")
                return artifacts

            frame_no = 0
            while frame_no < last_needed:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame_no += 1
                current = frame_no
                for window in windows:
                    writer = writers.get(window.delivery_index)
                    if writer is None:
                        continue
                    if window.start_frame <= frame_no <= window.end_frame:
                        writer.write(frame)
                        artifacts[window.delivery_index].frame_count += 1
                if on_progress is not None and (
                    frame_no == 1 or frame_no % 10 == 0 or frame_no == last_needed
                ):
                    on_progress(frame_no, last_needed)
        finally:
            capture.release()
            for writer in writers.values():
                writer.release()

        for window in windows:
            artifact = artifacts[window.delivery_index]
            if artifact.window.delivery_index not in writers:
                continue
            if artifact.path is not None and (
                not artifact.path.is_file() or artifact.path.stat().st_size == 0
            ):
                artifact.error = "clip_file_empty"
                artifact.path = None
                artifact.file_name = None
                artifact.frame_count = 0
            elif current < window.end_frame:
                artifact.error = "source_decode_ended_early"
            if on_progress is not None:
                on_progress(current, last_needed)

        return artifacts

    @staticmethod
    def _fail_all(artifacts: dict[int, ClipArtifact], reason: str) -> None:
        for artifact in artifacts.values():
            artifact.error = reason

    @staticmethod
    def _open_writer(
        clips_dir: Path,
        delivery_index: int,
        *,
        fps: float,
        width: int,
        height: int,
    ) -> tuple[cv2.VideoWriter, Path, str] | None:
        size = (int(width), int(height))
        for fourcc in _WEBM_CODECS:
            path = clips_dir / f"delivery_{delivery_index:03d}.webm"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*fourcc),
                float(fps),
                size,
            )
            if writer.isOpened():
                return writer, path, fourcc
            writer.release()
            path.unlink(missing_ok=True)

        path = clips_dir / f"delivery_{delivery_index:03d}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*_MP4_CODEC),
            float(fps),
            size,
        )
        if writer.isOpened():
            return writer, path, _MP4_CODEC
        writer.release()
        path.unlink(missing_ok=True)
        return None


__all__ = [
    "ClipArtifact",
    "ClipExtractor",
    "ClipWindow",
    "candidate_windows",
]

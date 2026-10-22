/**
 * validation.ts — upload validation before a byte leaves the browser.
 *
 * The backend's own door check (docs/API_CONTRACT.md §1) is authoritative:
 * extensions `mp4 avi mov mkv webm`, ceiling 2 GB, and OpenCV must be able to
 * open the file. This module mirrors the cheap half of that check so the user
 * gets an immediate, specific message instead of a 400 after a long upload.
 */

import { ACCEPTED_VIDEO_EXTENSIONS, MAX_UPLOAD_BYTES } from "./api";
import { formatBytes } from "./format";

export interface FileValidation {
  ok: boolean;
  error: string | null;
  extension: string | null;
}

export interface FileLike {
  name: string;
  size: number;
  type?: string;
}

export function fileExtension(name: string): string | null {
  const match = /\.([a-z0-9]+)$/i.exec(name.trim());
  return match ? match[1].toLowerCase() : null;
}

export function validateVideoFile(file: FileLike): FileValidation {
  const extension = fileExtension(file.name);

  if (file.size === 0) {
    return { ok: false, error: "That file is empty. Choose a video with content.", extension };
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      ok: false,
      error: `That video is ${formatBytes(file.size)}; the upload limit is ${formatBytes(
        MAX_UPLOAD_BYTES,
      )}.`,
      extension,
    };
  }

  if (!extension || !(ACCEPTED_VIDEO_EXTENSIONS as readonly string[]).includes(extension)) {
    return {
      ok: false,
      error: `Unsupported file type${extension ? ` ".${extension}"` : ""}. Use MP4, AVI, MOV, MKV or WEBM.`,
      extension,
    };
  }

  // Some browsers leave `type` empty. Only reject a MIME type that is
  // positively not a video (for example a text file renamed to .mp4).
  if (file.type && !file.type.startsWith("video/") && file.type !== "application/octet-stream") {
    return {
      ok: false,
      error: "That file does not look like a video. Choose an MP4, AVI, MOV, MKV or WEBM file.",
      extension,
    };
  }

  return { ok: true, error: null, extension };
}

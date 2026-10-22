import { describe, expect, it } from "vitest";
import { MAX_UPLOAD_BYTES } from "@/lib/api";
import { fileExtension, validateVideoFile } from "@/lib/validation";

function file(name: string, size = 1024, type = "video/mp4") {
  return { name, size, type };
}

describe("fileExtension", () => {
  it("reads a lowercase extension", () => {
    expect(fileExtension("Delivery_01.MP4")).toBe("mp4");
    expect(fileExtension("  spell.mov  ")).toBe("mov");
  });

  it("returns null when there is no extension", () => {
    expect(fileExtension("clip")).toBeNull();
    expect(fileExtension("clip.")).toBeNull();
  });
});

describe("validateVideoFile", () => {
  it("accepts the documented containers", () => {
    for (const extension of ["mp4", "avi", "mov", "mkv", "webm"]) {
      const result = validateVideoFile(file(`clip.${extension}`, 2048, `video/${extension}`));
      expect(result.ok, extension).toBe(true);
      expect(result.error).toBeNull();
      expect(result.extension).toBe(extension);
    }
  });

  it("accepts an empty browser MIME type when the extension is valid", () => {
    expect(validateVideoFile(file("clip.mp4", 2048, "")).ok).toBe(true);
  });

  it("accepts application/octet-stream", () => {
    expect(validateVideoFile(file("clip.mkv", 2048, "application/octet-stream")).ok).toBe(true);
  });

  it("rejects an empty file before uploading it", () => {
    const result = validateVideoFile(file("clip.mp4", 0));
    expect(result.ok).toBe(false);
    expect(result.error).toContain("empty");
  });

  it("rejects a file above the documented 2 GB ceiling", () => {
    const result = validateVideoFile(file("clip.mp4", MAX_UPLOAD_BYTES + 1));
    expect(result.ok).toBe(false);
    expect(result.error).toContain("2.0 GB");
    expect(result.error).toContain("limit");
  });

  it("accepts a file exactly at the ceiling", () => {
    expect(validateVideoFile(file("clip.mp4", MAX_UPLOAD_BYTES)).ok).toBe(true);
  });

  it("rejects an unsupported extension", () => {
    const result = validateVideoFile(file("notes.txt", 2048, "text/plain"));
    expect(result.ok).toBe(false);
    expect(result.error).toContain('".txt"');
    expect(result.error).toContain("MP4");
  });

  it("rejects a file with no extension", () => {
    const result = validateVideoFile(file("clip", 2048, "video/mp4"));
    expect(result.ok).toBe(false);
    expect(result.error).toContain("Unsupported file type");
    expect(result.extension).toBeNull();
  });

  it("rejects a non-video MIME type even with a video extension", () => {
    const result = validateVideoFile(file("renamed.mp4", 2048, "text/plain"));
    expect(result.ok).toBe(false);
    expect(result.error).toContain("does not look like a video");
  });

  it("does not reject a video extension because of a browser-specific video MIME type", () => {
    expect(validateVideoFile(file("clip.mp4", 2048, "video/x-m4v")).ok).toBe(true);
  });
});

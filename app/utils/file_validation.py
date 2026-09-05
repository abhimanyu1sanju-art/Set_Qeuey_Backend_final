"""
SatQuery AI — File Validation Utility (Phase 2)

Responsibilities:
  - Extension validation
  - MIME type validation
  - File size validation
  - Actual image content validation (rejects fake/corrupt files)
  - Decompression-bomb protection
  - Safe filename handling (path-traversal prevention)

All validation functions raise HTTPException with appropriate status codes
so that route handlers stay thin.
"""

from __future__ import annotations

import io
import logging
import mimetypes
import os
import re
from pathlib import Path, PurePosixPath

from fastapi import HTTPException, UploadFile

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})

ALLOWED_MIME_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/geotiff",      # Some clients send this for GeoTIFF
    "application/octet-stream",  # Accepted only when extension+content pass
})

# Pillow PIL.Image.MAX_IMAGE_PIXELS default is 178M pixels; we keep it stricter.
# ~256 MP is generous for remote-sensing but still safe.
MAX_IMAGE_PIXELS: int = 256_000_000  # 256 megapixels

# ─── Safe Filename ────────────────────────────────────────────────────────────

_UNSAFE_CHARS = re.compile(r"[^\w.\-]")  # keep word chars, dots, hyphens only


def sanitise_filename(filename: str) -> str:
    """
    Strip any path components and replace unsafe characters.
    Returns only the basename with safe characters.
    """
    # Extract only the basename — prevents "../../etc/passwd" style attacks.
    safe = Path(filename).name
    # Replace sequences of unsafe characters with underscores.
    safe = _UNSAFE_CHARS.sub("_", safe)
    return safe or "upload"


# ─── Extension Validation ─────────────────────────────────────────────────────

def validate_extension(filename: str) -> str:
    """
    Return the lowercase extension if allowed, raise HTTP 400 otherwise.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file extension '{ext}'. "
                "Allowed formats: JPG, JPEG, PNG, TIFF, GeoTIFF"
            ),
        )
    return ext


# ─── MIME Type Validation ─────────────────────────────────────────────────────

def validate_mime_type(content_type: str | None, filename: str) -> str:
    """
    Check the MIME type supplied by the client.
    Returns the resolved MIME type string.
    """
    # Some browsers report empty content_type; fall back to extension-based guess.
    if not content_type or content_type == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(filename)
        content_type = guessed or content_type or ""

    # Normalise: strip parameters (e.g. "image/jpeg; charset=…")
    base_mime = content_type.split(";")[0].strip().lower()

    if base_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported MIME type '{base_mime}'. "
                "Allowed formats: JPG, JPEG, PNG, TIFF, GeoTIFF"
            ),
        )
    return base_mime


# ─── File Size Validation ─────────────────────────────────────────────────────

async def validate_file_size(file: UploadFile) -> bytes:
    """
    Read the entire file into memory and check it against MAX_FILE_SIZE_MB.
    Returns the raw bytes so callers do not need to re-read the file.
    Raises HTTP 413 if oversized.
    """
    limit = settings.max_file_size_bytes
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await file.read(1024 * 64)  # 64 KB per chunk
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File size exceeds the maximum allowed limit of "
                    f"{settings.max_file_size_mb} MB"
                ),
            )
        chunks.append(chunk)

    return b"".join(chunks)


# ─── Image Content Validation ─────────────────────────────────────────────────

def validate_image_content(data: bytes, filename: str) -> None:
    """
    Confirm that the raw bytes actually represent a valid, non-corrupt image.

    Uses Pillow to open and verify the image. This rejects:
      - Files with a .jpg extension that contain non-image data
      - Truncated/corrupt images
      - Decompression-bomb images (extremely large decompressed size)

    Raises HTTP 400 on any failure.
    """
    # Late import to avoid heavy deps at module load time.
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        logger.error("Pillow not installed: %s", exc)
        raise HTTPException(status_code=500, detail="Image processing library unavailable")

    try:
        # PIL.Image.MAX_IMAGE_PIXELS controls decompression-bomb protection.
        # We set our own stricter limit.
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

        with Image.open(io.BytesIO(data)) as img:
            # verify() catches truncated / corrupt files.
            img.verify()

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file — could not identify image format",
        )
    except Exception as exc:
        exc_str = str(exc).lower()
        if "decompression" in exc_str or "bomb" in exc_str:
            raise HTTPException(
                status_code=400,
                detail="Image rejected: potential decompression bomb (image too large to decompress safely)",
            )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file: {str(exc)}",
        )


# ─── Compound Validator ───────────────────────────────────────────────────────

async def validate_upload(file: UploadFile) -> bytes:
    """
    Run all validation steps in order.
    Returns the raw file bytes for downstream processing.

    Steps:
      1. Extension check
      2. MIME type check
      3. Size check (reads the file)
      4. Content / corruption check
    """
    validate_extension(file.filename or "")
    validate_mime_type(file.content_type, file.filename or "")
    data = await validate_file_size(file)
    validate_image_content(data, file.filename or "")
    return data

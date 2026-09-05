"""
SatQuery AI — Local Storage Service (Phase 2)

Responsibilities:
  - Create and ensure upload directories exist
  - Save original, processed, and thumbnail files
  - Delete image files (all three variants)
  - Return storage paths

Designed for clean replacement with S3-compatible storage in a later phase.
All public functions operate on image_id and bytes — no raw path strings leak
out of this module to callers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.schemas.image import StoragePaths

logger = logging.getLogger(__name__)


# ─── Directory Initialisation ─────────────────────────────────────────────────

def ensure_upload_dirs() -> None:
    """
    Create all required upload sub-directories if they do not exist.
    Safe to call multiple times (exist_ok=True).
    """
    for subdir in ("original", "processed", "thumbnails"):
        path = settings.upload_path / subdir
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Upload directory ensured: %s", path)


# ─── Path Helpers ─────────────────────────────────────────────────────────────

def _safe_path(directory: Path, filename: str) -> Path:
    """
    Construct an absolute path and confirm it stays inside `directory`.
    Raises ValueError on path-traversal attempts.
    """
    resolved = (directory / filename).resolve()
    if not str(resolved).startswith(str(directory.resolve())):
        raise ValueError(f"Path traversal detected: {filename!r}")
    return resolved


def _stored_filename(image_id: str, extension: str) -> str:
    """Generate a safe server-side filename from image_id + extension."""
    ext = extension.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    return f"{image_id}{ext}"


# ─── Save Operations ──────────────────────────────────────────────────────────

def save_original(image_id: str, extension: str, data: bytes) -> str:
    """
    Write the original uploaded bytes to uploads/original/<image_id><ext>.
    Returns the absolute path string.
    Never overwrites an existing file.
    """
    ensure_upload_dirs()
    filename = _stored_filename(image_id, extension)
    path = _safe_path(settings.original_path, filename)

    path.write_bytes(data)
    logger.info("Original saved: %s (%d bytes)", path, len(data))
    return str(path)


def save_processed(image_id: str, extension: str, data: bytes) -> str:
    """
    Write the processed image bytes to uploads/processed/<image_id><ext>.
    Returns the absolute path string.
    """
    ensure_upload_dirs()
    filename = _stored_filename(image_id, extension)
    path = _safe_path(settings.processed_path, filename)

    path.write_bytes(data)
    logger.info("Processed saved: %s (%d bytes)", path, len(data))
    return str(path)


def save_thumbnail(image_id: str, data: bytes) -> str:
    """
    Write thumbnail bytes to uploads/thumbnails/<image_id>.jpg.
    Thumbnails are always stored as JPEG for browser compatibility.
    Returns the absolute path string.
    """
    ensure_upload_dirs()
    filename = f"{image_id}.jpg"
    path = _safe_path(settings.thumbnails_path, filename)

    path.write_bytes(data)
    logger.info("Thumbnail saved: %s (%d bytes)", path, len(data))
    return str(path)


# ─── Delete Operations ────────────────────────────────────────────────────────

def _delete_file(path_str: Optional[str], label: str) -> None:
    """
    Delete a single file. Non-fatal if the file is already missing.
    """
    if not path_str:
        return
    try:
        path = Path(path_str)
        if path.exists():
            path.unlink()
            logger.info("Deleted %s: %s", label, path)
        else:
            logger.debug("%s file already missing: %s", label, path)
    except Exception as exc:
        logger.warning("Could not delete %s (%s): %s", label, path_str, exc)


def delete_image_files(storage: StoragePaths) -> None:
    """
    Delete all stored files for an image (original, processed, thumbnail).
    Continues on individual failures so a partial cleanup is still performed.
    """
    _delete_file(storage.original, "original")
    _delete_file(storage.processed, "processed")
    _delete_file(storage.thumbnail, "thumbnail")


# ─── Build StoragePaths ───────────────────────────────────────────────────────

def build_storage_paths(
    image_id: str,
    extension: str,
    original_path: str,
    processed_path: str,
    thumbnail_path: str,
) -> StoragePaths:
    """Construct a StoragePaths instance from the three saved path strings."""
    return StoragePaths(
        original=original_path,
        processed=processed_path,
        thumbnail=thumbnail_path,
    )

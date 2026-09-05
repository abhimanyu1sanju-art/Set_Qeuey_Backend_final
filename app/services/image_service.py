"""
SatQuery AI — Image Service (Phase 2)

Orchestrates the complete image upload pipeline:

  images.py
      ↓
  image_service.py   ← this file
      ↓
  file_validation.py
      ↓
  image_utils.py
      ↓
  local_storage.py / thumbnails.py
      ↓
  MongoDB

Atomic cleanup: if any step after storage fails, already-written files
are removed to avoid orphan files accumulating on disk.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile

from app.db.mongodb import get_images_collection
from app.schemas.image import ImageDocument, ImageResponse, StoragePaths
from app.storage import local_storage, thumbnails
from app.utils import image_utils
from app.services import session_service

logger = logging.getLogger(__name__)


# ─── Upload Pipeline ──────────────────────────────────────────────────────────

async def process_upload(file: UploadFile, raw_data: bytes) -> ImageResponse:
    """
    Run the full image upload pipeline.

    Steps:
      1.  Generate image_id
      2.  Extract image metadata
      3.  Save original bytes
      4.  Create processed image
      5.  Generate thumbnail
      6.  Save processed + thumbnail
      7.  Insert MongoDB document
      8.  Return ImageResponse

    On failure at steps 3–7, clean up any files already written.
    Never silently swallow a MongoDB write failure.
    """
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    # 1. Generate unique image ID
    image_id = image_utils.generate_image_id()
    logger.info("Processing upload: filename=%r image_id=%s", filename, image_id)

    # Track saved paths for atomic cleanup
    original_path: Optional[str] = None
    processed_path: Optional[str] = None
    thumbnail_path: Optional[str] = None

    try:
        # 2. Extract metadata (width, height, format, MIME, geospatial)
        metadata = image_utils.extract_image_metadata(raw_data, filename)

        # 3. Save original (never overwrite)
        original_path = local_storage.save_original(image_id, ext, raw_data)

        # 4. Create processed image bytes
        processed_data = image_utils.create_processed_image(raw_data, filename)

        # 5. Save processed copy
        processed_path = local_storage.save_processed(image_id, ext, processed_data)

        # 6. Generate and save thumbnail
        try:
            thumb_data = thumbnails.generate_thumbnail(raw_data, filename)
            thumbnail_path = local_storage.save_thumbnail(image_id, thumb_data)
        except Exception as thumb_exc:
            # Non-fatal: thumbnail failure should not abort the whole upload.
            # The image is still usable without a thumbnail.
            logger.warning("Thumbnail generation failed for %s: %s", image_id, thumb_exc)
            thumbnail_path = ""

        # 7. Build the document
        storage = StoragePaths(
            original=original_path,
            processed=processed_path,
            thumbnail=thumbnail_path or "",
        )

        doc = ImageDocument(
            image_id=image_id,
            filename=filename,
            stored_filename=Path(original_path).name,
            mime_type=metadata["mime_type"],
            format=metadata["format"],
            size=len(raw_data),
            width=metadata["width"],
            height=metadata["height"],
            status="ready",
            is_geospatial=metadata["is_geospatial"],
            geospatial=metadata.get("geospatial"),
            storage=storage,
            created_at=datetime.now(timezone.utc),
        )

        # 8. Insert into MongoDB
        _insert_document(doc)

        logger.info(
            "Upload complete: image_id=%s size=%d is_geospatial=%s",
            image_id, len(raw_data), doc.is_geospatial,
        )

        return doc.to_response()

    except HTTPException:
        # Already an HTTP error — clean up and re-raise.
        _cleanup(original_path, processed_path, thumbnail_path)
        raise

    except Exception as exc:
        logger.error("Upload pipeline failed for %s: %s", image_id, exc, exc_info=True)
        _cleanup(original_path, processed_path, thumbnail_path)
        raise HTTPException(
            status_code=500,
            detail="Image processing failed. Please try again.",
        ) from exc


def _insert_document(doc: ImageDocument) -> None:
    """
    Insert an ImageDocument into MongoDB.
    Raises HTTPException 500 on failure — never silently ignores.
    """
    try:
        col = get_images_collection()
        payload = doc.model_dump(mode="json")
        # MongoDB uses _id; we use image_id as the application key.
        col.insert_one(payload)
        logger.debug("MongoDB insert OK: image_id=%s", doc.image_id)
    except Exception as exc:
        logger.error("MongoDB insert failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to save image metadata. Upload rolled back.",
        ) from exc


def _cleanup(
    original_path: Optional[str],
    processed_path: Optional[str],
    thumbnail_path: Optional[str],
) -> None:
    """
    Delete any files that were written during a failed upload.
    Non-fatal — logs warnings but does not raise.
    """
    for path_str, label in [
        (original_path, "original"),
        (processed_path, "processed"),
        (thumbnail_path, "thumbnail"),
    ]:
        if path_str:
            try:
                p = Path(path_str)
                if p.exists():
                    p.unlink()
                    logger.info("Cleanup: removed %s file %s", label, path_str)
            except Exception as exc:
                logger.warning("Cleanup: could not remove %s file %s: %s", label, path_str, exc)


# ─── Get Image ────────────────────────────────────────────────────────────────

def get_image_by_id(image_id: str) -> ImageDocument:
    """
    Fetch an image document from MongoDB by image_id.
    Raises HTTP 404 if not found.
    Never exposes internal filesystem details in errors.
    """
    try:
        col = get_images_collection()
        doc = col.find_one({"image_id": image_id}, {"_id": 0})
    except Exception as exc:
        logger.error("MongoDB query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    if not doc:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        return _doc_to_image_document(doc)
    except Exception as exc:
        logger.error("Failed to parse image document: %s", exc)
        raise HTTPException(status_code=500, detail="Image data corrupted")


# ─── Delete Image ─────────────────────────────────────────────────────────────

def delete_image_by_id(image_id: str) -> dict:
    """
    Delete an image:
      1. Find MongoDB record
      2. Delete all stored files
      3. Delete MongoDB document
    Returns a success dict.
    Raises HTTP 404 if not found.
    """
    img_doc = get_image_by_id(image_id)

    # Delete files first (non-fatal)
    try:
        local_storage.delete_image_files(img_doc.storage)
    except Exception as exc:
        logger.warning("File deletion partially failed for %s: %s", image_id, exc)

    # Delete MongoDB document
    try:
        col = get_images_collection()
        col.delete_one({"image_id": image_id})
        logger.info("Deleted image: image_id=%s", image_id)
    except Exception as exc:
        logger.error("MongoDB delete failed for %s: %s", image_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete image metadata")

    # Phase 3: Remove image_id reference from all sessions (non-fatal)
    session_service.remove_image_from_all_sessions(image_id)

    return {"image_id": image_id, "message": "Image deleted successfully"}


# ─── Document Parser ──────────────────────────────────────────────────────────

def _doc_to_image_document(doc: dict) -> ImageDocument:
    """Parse a raw MongoDB document dict into an ImageDocument."""
    from app.schemas.image import GeospatialMeta

    geo_raw = doc.get("geospatial")
    geospatial = None
    if geo_raw:
        if isinstance(geo_raw, dict):
            geospatial = GeospatialMeta(**geo_raw)
        elif isinstance(geo_raw, GeospatialMeta):
            geospatial = geo_raw

    storage_raw = doc.get("storage", {})
    storage = StoragePaths(
        original=storage_raw.get("original", ""),
        processed=storage_raw.get("processed", ""),
        thumbnail=storage_raw.get("thumbnail", ""),
    )

    created_at = doc.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    elif created_at is None:
        created_at = datetime.now(timezone.utc)

    return ImageDocument(
        image_id=doc["image_id"],
        filename=doc["filename"],
        stored_filename=doc.get("stored_filename", ""),
        mime_type=doc["mime_type"],
        format=doc["format"],
        size=doc["size"],
        width=doc["width"],
        height=doc["height"],
        status=doc.get("status", "ready"),
        is_geospatial=doc.get("is_geospatial", False),
        geospatial=geospatial,
        storage=storage,
        created_at=created_at,
    )

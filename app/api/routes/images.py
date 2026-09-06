"""
SatQuery AI — Images Router (Phase 2)

Endpoints:
  POST   /api/images/upload                  — Upload a new satellite/remote-sensing image
  GET    /api/images/{image_id}              — Retrieve image metadata by ID
  GET    /api/images/{image_id}/thumbnail    — Serve the JPEG thumbnail for an image
  DELETE /api/images/{image_id}              — Delete an image and all its stored files

Route handlers are intentionally thin — all logic lives in image_service.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.schemas.image import DeleteResponse, ImageDetailResponse, ImageResponse
from app.services import image_service
from app.utils.file_validation import validate_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["Images"])


# ─── POST /api/images/upload ──────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=ImageResponse,
    status_code=200,
    summary="Upload a satellite or remote-sensing image",
    description=(
        "Upload a satellite or remote-sensing image for processing and storage.\n\n"
        "**Supported formats:** JPG, JPEG, PNG, TIFF, GeoTIFF\n\n"
        "**Maximum file size:** configurable via `MAX_FILE_SIZE_MB` (default 20 MB)\n\n"
        "The backend will:\n"
        "- Validate extension, MIME type, and actual file content\n"
        "- Extract image metadata (dimensions, format)\n"
        "- Detect GeoTIFF and extract geospatial metadata when present\n"
        "- Store the original file untouched\n"
        "- Create a processed copy\n"
        "- Generate a thumbnail (max 512×512, aspect ratio preserved)\n"
        "- Save all metadata to MongoDB\n"
        "- Return a structured JSON response\n\n"
        "**Never fabricates** satellite sensor, coordinates, or spectral metadata."
    ),
    response_description="Structured image metadata after successful upload",
)
async def upload_image(
    file: UploadFile = File(..., description="Image file (JPG, PNG, TIFF, GeoTIFF)"),
) -> ImageResponse:
    """
    Upload and process a satellite/remote-sensing image.

    Validates the file, extracts metadata, stores original + processed +
    thumbnail, saves to MongoDB, and returns structured metadata.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    logger.info("Received upload: filename=%r content_type=%r", file.filename, file.content_type)

    # Validate: extension → MIME → size → content (raises HTTPException on failure)
    raw_data = await validate_upload(file)

    # Delegate full pipeline to service layer
    return await image_service.process_upload(file, raw_data)


# ─── GET /api/images/{image_id}/thumbnail ────────────────────────────────────
# NOTE: This route MUST be registered before /{image_id} so FastAPI does not
# treat "thumbnail" as a second image_id value.

@router.get(
    "/{image_id}/thumbnail",
    status_code=200,
    summary="Serve the JPEG thumbnail for an image",
    description=(
        "Return the pre-generated JPEG thumbnail for a stored image.\n\n"
        "The thumbnail is a max-512×512 aspect-ratio-preserved JPEG created "
        "during upload. Returns HTTP 404 if the image or its thumbnail does "
        "not exist."
    ),
    response_description="JPEG thumbnail image",
    response_class=FileResponse,
)
async def get_thumbnail(image_id: str) -> FileResponse:
    """
    Serve the thumbnail JPEG for a stored image.

    Returns 404 if the image record or the thumbnail file is missing.
    On Render (ephemeral filesystem), uploaded files are lost after restarts —
    the record exists in MongoDB but the file is gone. This returns a clean 404
    so the frontend can show a placeholder instead of crashing.
    """
    doc = image_service.get_image_by_id(image_id)  # raises 404 if not found

    thumb_path_str = doc.storage.thumbnail
    if not thumb_path_str:
        raise HTTPException(status_code=404, detail="Thumbnail not available for this image")

    thumb_path = Path(thumb_path_str)
    if not thumb_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Thumbnail file not found. The server may have restarted and lost "
                "the uploaded file. Please re-upload the image."
            ),
        )

    return FileResponse(
        path=str(thumb_path),
        media_type="image/jpeg",
        filename=f"{image_id}_thumb.jpg",
    )


# ─── GET /api/images/{image_id} ───────────────────────────────────────────────

@router.get(
    "/{image_id}",
    response_model=ImageResponse,
    status_code=200,
    summary="Get image metadata by ID",
    description=(
        "Retrieve stored metadata for an image by its unique `image_id`.\n\n"
        "Returns HTTP 404 if the image does not exist."
    ),
    response_description="Stored image metadata",
)
async def get_image(image_id: str) -> ImageResponse:
    """
    Return metadata for a previously uploaded image.
    """
    doc = image_service.get_image_by_id(image_id)
    return doc.to_response()


# ─── DELETE /api/images/{image_id} ───────────────────────────────────────────

@router.delete(
    "/{image_id}",
    response_model=DeleteResponse,
    status_code=200,
    summary="Delete an image and all its stored files",
    description=(
        "Delete an image by its `image_id`.\n\n"
        "This will:\n"
        "1. Find the MongoDB record\n"
        "2. Delete the original file\n"
        "3. Delete the processed file\n"
        "4. Delete the thumbnail\n"
        "5. Remove the MongoDB document\n\n"
        "Returns HTTP 404 if the image does not exist.\n"
        "If a storage file is already missing, deletion continues without error."
    ),
    response_description="Deletion confirmation",
)
async def delete_image(image_id: str) -> DeleteResponse:
    """
    Delete an image and all its associated files from storage and MongoDB.
    """
    result = image_service.delete_image_by_id(image_id)
    return DeleteResponse(**result)

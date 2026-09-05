"""
SatQuery AI — Image Pydantic Schemas (Phase 2)

Defines request/response models for the image management system.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Geospatial Metadata ──────────────────────────────────────────────────────

class BoundsSchema(BaseModel):
    """Geographic bounding box extracted from a GeoTIFF."""
    left: float
    bottom: float
    right: float
    top: float


class GeospatialMeta(BaseModel):
    """
    Geospatial metadata extracted from a GeoTIFF file.
    All fields are optional — only populated when actually present in the file.
    Never fabricated.
    """
    is_geospatial: bool = False
    crs: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    bands: Optional[int] = None
    resolution: Optional[list[float]] = None
    bounds: Optional[BoundsSchema] = None
    nodata: Optional[float] = None
    transform: Optional[list[float]] = None   # Affine transform coefficients
    driver: Optional[str] = None              # GDAL driver name (e.g. "GTiff")


# ─── Storage Paths ────────────────────────────────────────────────────────────

class StoragePaths(BaseModel):
    """Filesystem paths for the three stored variants of an image."""
    original: str
    processed: str
    thumbnail: str


# ─── API Response Models ──────────────────────────────────────────────────────

class ImageResponse(BaseModel):
    """
    Response returned to the frontend after a successful image upload.
    Internal filesystem paths are intentionally omitted.
    """
    image_id: str = Field(..., description="Unique collision-safe image identifier (img_<ULID>)")
    filename: str = Field(..., description="Original uploaded filename")
    format: str = Field(..., description="Image format (JPEG, PNG, TIFF, …)")
    mime_type: str = Field(..., description="MIME type of the uploaded file")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    size: int = Field(..., description="File size in bytes")
    status: str = Field(default="ready", description="Processing status")
    is_geospatial: bool = Field(default=False, description="Whether the image contains GeoTIFF metadata")
    geospatial: Optional[GeospatialMeta] = Field(default=None, description="Geospatial metadata (GeoTIFF only)")
    created_at: datetime = Field(..., description="UTC timestamp of upload")


class ImageDetailResponse(ImageResponse):
    """
    Extended response for GET /api/images/{image_id}.
    Includes storage paths for internal tooling if needed.
    """
    stored_filename: str
    storage: StoragePaths


class DeleteResponse(BaseModel):
    """Response returned after a successful image deletion."""
    image_id: str
    message: str = "Image deleted successfully"


# ─── Internal Document Model (for MongoDB) ────────────────────────────────────

class ImageDocument(BaseModel):
    """
    Represents an image document as stored in MongoDB.
    Used internally — not directly serialised to API responses.
    """
    image_id: str
    filename: str
    stored_filename: str
    mime_type: str
    format: str
    size: int
    width: int
    height: int
    status: str = "ready"
    is_geospatial: bool = False
    geospatial: Optional[GeospatialMeta] = None
    storage: StoragePaths
    created_at: datetime

    def to_response(self) -> ImageResponse:
        return ImageResponse(
            image_id=self.image_id,
            filename=self.filename,
            format=self.format,
            mime_type=self.mime_type,
            width=self.width,
            height=self.height,
            size=self.size,
            status=self.status,
            is_geospatial=self.is_geospatial,
            geospatial=self.geospatial,
            created_at=self.created_at,
        )

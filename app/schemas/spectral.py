"""
SatQuery AI — Spectral Analysis Schemas (Phase 2)

Pydantic models for the real spectral index calculation API.

Supported indices:
  - NDVI  = (NIR - RED) / (NIR + RED)   [B08 - B04 / B08 + B04]
  - NDWI  = (GREEN - NIR) / (GREEN + NIR) [B03 - B08 / B03 + B08]
  - NBR   = (NIR - SWIR2) / (NIR + SWIR2) [B08 - B12 / B08 + B12]

All values are calculated from real Sentinel-2 L2A raster bands.
No mock values are returned — if processing fails, a clear error is raised.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Supported indices ────────────────────────────────────────────────────────

SUPPORTED_INDICES = {"NDVI", "NDWI", "NBR"}

# Band requirements per index
INDEX_BANDS: Dict[str, List[str]] = {
    "NDVI": ["B04", "B08"],          # RED, NIR
    "NDWI": ["B03", "B08"],          # GREEN, NIR
    "NBR":  ["B08", "B12"],          # NIR, SWIR2
}

# Nominal GSD per band (Sentinel-2 L2A)
BAND_GSD_METRES: Dict[str, int] = {
    "B02": 10, "B03": 10, "B04": 10, "B08": 10,  # 10m bands
    "B05": 20, "B06": 20, "B07": 20, "B8A": 20,
    "B11": 20, "B12": 20,                          # 20m bands
    "B01": 60, "B09": 60,                          # 60m bands
}


# ─── Request model ────────────────────────────────────────────────────────────

class SpectralAnalysisRequest(BaseModel):
    """
    Request to calculate one or more spectral indices for a Sentinel-2 L2A scene.

    scene_id:   STAC item ID (e.g. S2C_MSIL2A_20250128T053131_…)
    indices:    List of indices to calculate. Supported: NDVI, NDWI, NBR.
    collection: STAC collection ID. Defaults to 'sentinel-2-l2a'.
    force_reprocess: If True, bypass cache and reprocess from raw bands.
    """

    scene_id: str = Field(..., description="STAC item ID of the Sentinel-2 L2A scene.")
    indices: List[str] = Field(
        default=["NDVI", "NDWI", "NBR"],
        description="Spectral indices to calculate. Supported: NDVI, NDWI, NBR.",
        min_length=1,
        max_length=3,
    )
    collection: str = Field(
        default="sentinel-2-l2a",
        description="STAC collection ID. Must be 'sentinel-2-l2a'.",
    )
    force_reprocess: bool = Field(
        default=False,
        description="If True, bypass cache and reprocess from raw bands.",
    )

    @field_validator("indices")
    @classmethod
    def validate_indices(cls, v: List[str]) -> List[str]:
        normalised = [i.upper().strip() for i in v]
        unsupported = [i for i in normalised if i not in SUPPORTED_INDICES]
        if unsupported:
            raise ValueError(
                f"Unsupported indices: {unsupported}. "
                f"Supported: {sorted(SUPPORTED_INDICES)}"
            )
        return normalised

    @field_validator("collection")
    @classmethod
    def validate_collection(cls, v: str) -> str:
        if v != "sentinel-2-l2a":
            raise ValueError(
                "Spectral analysis only supports 'sentinel-2-l2a'. "
                "Sentinel-1 SAR does not have optical bands."
            )
        return v


# ─── Statistics model ─────────────────────────────────────────────────────────

class IndexStats(BaseModel):
    """
    Pixel-level statistics for a calculated spectral index.
    All values are computed from real raster pixels — never estimated or mocked.
    Invalid/nodata pixels are excluded from all statistics.
    """

    minimum: Optional[float] = Field(None, description="Minimum valid pixel value.")
    maximum: Optional[float] = Field(None, description="Maximum valid pixel value.")
    mean: Optional[float] = Field(None, description="Mean of valid pixel values.")
    std: Optional[float] = Field(None, description="Standard deviation of valid pixels.")
    percentile_5: Optional[float] = Field(None, description="5th percentile (low end).")
    percentile_95: Optional[float] = Field(None, description="95th percentile (high end).")
    valid_pixel_count: int = Field(0, description="Count of valid (non-nodata, non-NaN) pixels.")
    nodata_pixel_count: int = Field(0, description="Count of nodata / masked pixels.")
    total_pixel_count: int = Field(0, description="Total pixel count (valid + nodata).")
    valid_pixel_pct: Optional[float] = Field(
        None, description="Percentage of pixels that are valid."
    )


# ─── Per-index result model ───────────────────────────────────────────────────

class IndexResult(BaseModel):
    """
    Result for a single spectral index calculation.
    """

    index: str = Field(..., description="Index name: NDVI, NDWI, or NBR.")
    status: str = Field(..., description="'completed' or 'failed'.")
    formula: str = Field(..., description="The formula used, e.g. (NIR-RED)/(NIR+RED).")
    bands_used: List[str] = Field(..., description="Band names used in calculation.")
    processing_resolution_m: int = Field(
        ..., description="Pixel resolution of the output product in metres."
    )
    resampling_method: Optional[str] = Field(
        None,
        description="Resampling method applied when bands at different resolutions were aligned.",
    )
    crs: Optional[str] = Field(None, description="Coordinate reference system of the output.")
    stats: Optional[IndexStats] = Field(None, description="Pixel statistics.")
    output_geotiff_path: Optional[str] = Field(
        None, description="Relative path to the output GeoTIFF."
    )
    preview_available: bool = Field(
        False, description="True if a PNG preview is available via the preview endpoint."
    )
    processing_time_seconds: Optional[float] = Field(
        None, description="Wall-clock seconds taken to process this index."
    )
    error_message: Optional[str] = Field(
        None, description="Error details if status='failed'."
    )
    cached: bool = Field(False, description="True if result was served from cache.")
    cloud_masking: str = Field(
        "nodata_only",
        description=(
            "Cloud masking approach applied. "
            "'nodata_only' = only nodata/zero pixels masked. "
            "'scl' = Scene Classification Layer applied (when available)."
        ),
    )


# ─── Full response model ──────────────────────────────────────────────────────

class SpectralAnalysisResponse(BaseModel):
    """
    Response from POST /api/spectral/analyze.

    Contains real index results calculated from actual Sentinel-2 L2A bands.
    """

    scene_id: str
    collection: str
    satellite: str = "sentinel-2"
    source: str = "Copernicus Data Space Ecosystem"
    processing_backend: str = "Rasterio + GDAL + NumPy"
    processing: Dict[str, IndexResult] = Field(
        default_factory=dict,
        description="Index results keyed by index name (lowercase): ndvi, ndwi, nbr.",
    )
    overall_status: str = Field(
        ...,
        description="'completed' if all requested indices succeeded, 'partial' if some failed, 'failed' if all failed.",
    )
    requested_indices: List[str]
    completed_indices: List[str] = Field(default_factory=list)
    failed_indices: List[str] = Field(default_factory=list)
    total_processing_time_seconds: Optional[float] = None
    created_at: datetime
    cached: bool = Field(False, description="True if ALL results were served from cache.")
    note: str = Field(
        default=(
            "All spectral values are calculated from real Sentinel-2 L2A "
            "raster bands via Copernicus Data Space. No synthetic values are used."
        )
    )


# ─── MongoDB record model ─────────────────────────────────────────────────────

class SpectralAnalysisRecord(BaseModel):
    """
    MongoDB document for a completed spectral analysis.
    Stored in the 'spectral_analyses' collection.
    Only metadata is stored — raster data lives on disk.
    """

    record_id: str
    scene_id: str
    collection: str
    index_type: str              # 'NDVI', 'NDWI', or 'NBR'
    status: str                  # 'completed' or 'failed'
    source: str = "Copernicus Data Space"
    created_at: datetime
    processing_resolution_m: int
    crs: Optional[str] = None
    stats: Optional[Dict] = None
    output_path: Optional[str] = None    # Relative path to GeoTIFF
    preview_path: Optional[str] = None   # Relative path to PNG preview
    error_message: Optional[str] = None
    cloud_masking: str = "nodata_only"
    resampling_method: Optional[str] = None
    bands_used: List[str] = Field(default_factory=list)

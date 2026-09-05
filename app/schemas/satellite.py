"""
SatQuery AI — Satellite Schemas (Phase 1)

Pydantic models for the satellite search and scene-metadata endpoints.

Supported satellites:
  - sentinel-2  → Sentinel-2 L2A (optical, 10–60m)
  - sentinel-1  → Sentinel-1 GRD (C-band SAR)

No mock data is ever returned. If Copernicus returns nothing, the
response carries an empty `scenes` list.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Supported satellites ─────────────────────────────────────────────────────

SUPPORTED_SATELLITES = {"sentinel-2", "sentinel-1"}

# Map user-facing satellite name → Copernicus STAC collection ID
SATELLITE_TO_COLLECTION: dict[str, str] = {
    "sentinel-2": "sentinel-2-l2a",
    "sentinel-1": "sentinel-1-grd",
}


# ─── Request models ───────────────────────────────────────────────────────────

class SatelliteSearchRequest(BaseModel):
    """
    Search request for Sentinel scenes via Copernicus STAC API.

    bbox:           [west, south, east, north] in WGS84 decimal degrees.
    start_date:     ISO 8601 date string, e.g. '2025-01-01'.
    end_date:       ISO 8601 date string, e.g. '2025-01-31'.
    satellite:      'sentinel-2' or 'sentinel-1'.
    max_cloud_cover: 0–100; ignored for Sentinel-1 (SAR is cloud-penetrating).
    limit:          Maximum number of scenes to return (1–50).
    """

    bbox: List[float] = Field(
        ...,
        description="Bounding box [west, south, east, north] in WGS84.",
        min_length=4,
        max_length=4,
    )
    start_date: str = Field(..., description="Start date, e.g. '2025-01-01'.")
    end_date: str = Field(..., description="End date, e.g. '2025-01-31'.")
    satellite: str = Field(
        default="sentinel-2",
        description="'sentinel-2' or 'sentinel-1'.",
    )
    max_cloud_cover: float = Field(
        default=30.0,
        ge=0,
        le=100,
        description="Max cloud cover % (0–100). Ignored for Sentinel-1.",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum scenes to return (1–50).",
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 values: [west, south, east, north]")
        west, south, east, north = v
        if not (-180 <= west <= 180):
            raise ValueError(f"west longitude {west} out of range [-180, 180]")
        if not (-180 <= east <= 180):
            raise ValueError(f"east longitude {east} out of range [-180, 180]")
        if not (-90 <= south <= 90):
            raise ValueError(f"south latitude {south} out of range [-90, 90]")
        if not (-90 <= north <= 90):
            raise ValueError(f"north latitude {north} out of range [-90, 90]")
        if west >= east:
            raise ValueError("west longitude must be less than east longitude")
        if south >= north:
            raise ValueError("south latitude must be less than north latitude")
        return v

    @field_validator("satellite")
    @classmethod
    def validate_satellite(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in SUPPORTED_SATELLITES:
            raise ValueError(
                f"satellite must be one of: {', '.join(sorted(SUPPORTED_SATELLITES))}"
            )
        return v

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date '{v}'. Use ISO format: YYYY-MM-DD")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Cross-field validation: start_date must be before end_date."""
        start = date.fromisoformat(self.start_date)
        end = date.fromisoformat(self.end_date)
        if start > end:
            raise ValueError("start_date must be on or before end_date")


# ─── Asset summary model ──────────────────────────────────────────────────────

class AssetInfo(BaseModel):
    """Minimal asset descriptor — enough for Phase 2 band retrieval."""
    name: str
    href: Optional[str] = None         # S3 URI (may need auth)
    https_href: Optional[str] = None   # HTTPS URL (needs OIDC auth)
    title: Optional[str] = None
    mime_type: Optional[str] = None
    gsd: Optional[float] = None        # Ground sample distance in metres
    bands: Optional[List[str]] = None  # Band names contained in this asset


# ─── Normalized scene model ───────────────────────────────────────────────────

class SatelliteScene(BaseModel):
    """
    Normalized satellite scene — common model for Sentinel-1 and Sentinel-2.

    This model is returned by the API and also persisted to MongoDB.
    It is deliberately source-agnostic so Phase 2 can add more satellites.
    """

    scene_id: str = Field(..., description="STAC item ID / product ID.")
    platform: str = Field(..., description="e.g. 'sentinel-2a', 'sentinel-1c'.")
    sensor: Optional[str] = Field(None, description="e.g. 'MSI' (S2) or 'SAR' (S1).")
    collection: str = Field(..., description="STAC collection ID.")
    satellite: str = Field(..., description="User-facing: 'sentinel-2' or 'sentinel-1'.")
    acquired_at: datetime = Field(..., description="Acquisition datetime (UTC).")
    cloud_cover: Optional[float] = Field(
        None, description="Cloud cover % (null for SAR)."
    )
    bbox: List[float] = Field(..., description="[west, south, east, north] WGS84.")
    geometry: Optional[Dict[str, Any]] = Field(
        None, description="GeoJSON geometry of the scene footprint."
    )
    preview_url: Optional[str] = Field(
        None, description="Public thumbnail/quicklook URL (may be null)."
    )
    source: str = Field(
        default="Copernicus Data Space",
        description="Data source identifier.",
    )
    bands: List[str] = Field(
        default_factory=list,
        description="Available band names (e.g. B02, B03, B04, B08 for S2).",
    )
    assets: Dict[str, AssetInfo] = Field(
        default_factory=dict,
        description="Key assets keyed by asset name.",
    )

    # Sentinel-1-specific fields
    polarizations: Optional[List[str]] = Field(
        None, description="SAR polarizations, e.g. ['VV', 'VH']."
    )
    orbit_state: Optional[str] = Field(
        None, description="'ascending' or 'descending'."
    )
    instrument_mode: Optional[str] = Field(
        None, description="SAR instrument mode, e.g. 'IW'."
    )

    # Sentinel-2-specific fields
    mgrs_tile: Optional[str] = Field(
        None, description="Sentinel-2 MGRS tile code, e.g. 'T43QCA'."
    )
    processing_level: Optional[str] = Field(
        None, description="e.g. 'L2A' for S2, 'L1' for S1."
    )

    created_at: Optional[datetime] = Field(
        None, description="When this record was saved to MongoDB."
    )


# ─── Response models ──────────────────────────────────────────────────────────

class SatelliteSearchResponse(BaseModel):
    """Response from POST /api/satellite/search."""

    scenes: List[SatelliteScene] = Field(
        default_factory=list,
        description="List of matched satellite scenes (may be empty).",
    )
    total_returned: int = Field(..., description="Number of scenes in this response.")
    satellite: str = Field(..., description="Which satellite was queried.")
    collection: str = Field(..., description="STAC collection searched.")
    bbox: List[float] = Field(..., description="The requested bounding box.")
    start_date: str
    end_date: str
    max_cloud_cover: Optional[float] = Field(
        None, description="Applied cloud cover filter (null for SAR)."
    )
    source: str = Field(default="Copernicus Data Space STAC API")
    stac_url: str = Field(
        default="https://stac.dataspace.copernicus.eu/v1",
        description="The STAC endpoint that was queried.",
    )


class SatelliteSceneDetailResponse(SatelliteScene):
    """
    Full scene detail — same as SatelliteScene but may include richer asset info.
    Returned by GET /api/satellite/{scene_id}.
    """
    pass

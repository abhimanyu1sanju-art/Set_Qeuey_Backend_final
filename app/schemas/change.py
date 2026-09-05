"""
SatQuery AI — Change Detection Schemas (Phase 3)

Pydantic models for the change detection and anomaly detection API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ChangeAnalysisRequest(BaseModel):
    before_scene_id: str = Field(..., description="STAC ID of the older scene")
    after_scene_id: str = Field(..., description="STAC ID of the newer scene")
    collection: str = Field("sentinel-2-l2a", description="STAC collection")
    bbox: Optional[List[float]] = Field(
        None,
        description="Bounding box [minx, miny, maxx, maxy]. If omitted, uses intersection of scenes.",
    )
    method: str = Field("abs_diff", description="Change detection method: abs_diff, ndvi_diff, etc.")
    threshold: float = Field(0.15, description="Threshold for binary change classification")
    force_reprocess: bool = Field(False, description="If true, bypasses the cache")


class ChangeRegion(BaseModel):
    id: int
    area_m2: float
    pixel_count: int
    bbox: List[float] = Field(..., description="[minx, miny, maxx, maxy] of the region")
    centroid: List[float] = Field(..., description="[lon, lat]")
    mean_change_magnitude: float


class ChangeStats(BaseModel):
    valid_pixel_count: int
    changed_pixel_count: int
    changed_area_m2: float
    change_percentage: float
    anomaly_percentage: float
    region_count: int


class ChangeAnalysisResponse(BaseModel):
    analysis_id: str
    before_scene_id: str
    after_scene_id: str
    method: str
    threshold: float
    status: str = Field(..., description="'processing', 'completed', 'failed'")
    stats: Optional[ChangeStats] = None
    regions: Optional[List[ChangeRegion]] = None
    crs: Optional[str] = None
    resolution_m: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None
    cached: bool = False
    created_at: str

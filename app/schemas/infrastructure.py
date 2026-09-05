"""
SatQuery AI — Infrastructure Detection Schemas (Phase 4)

Pydantic models for the real satellite-based infrastructure detection API.
All models use real computed values — no mock data.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Request ──────────────────────────────────────────────────────────────────

class InfrastructureAnalysisRequest(BaseModel):
    scene_id: str = Field(..., description="Sentinel-2 STAC scene ID (must exist in DB)")
    detection_types: List[str] = Field(
        default=["buildings", "roads", "construction"],
        description="Which infrastructure types to detect: buildings, roads, construction",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum per-class confidence threshold (0–1)",
    )
    force_reprocess: bool = Field(
        default=False,
        description="If true, ignore cached result and rerun the full pipeline",
    )


# ─── Detections ───────────────────────────────────────────────────────────────

class InfraDetection(BaseModel):
    label: str = Field(..., description="Class label: building_cluster, road_segment, construction_change_candidate")
    pixel_count: int = Field(..., description="Number of pixels in this detection region")
    area_m2: float = Field(..., description="Estimated ground area in square metres (using raster CRS/transform)")
    confidence: float = Field(..., description="Detection confidence 0–1 (based on spectral index magnitude)")
    bbox: List[float] = Field(..., description="[minx, miny, maxx, maxy] in source CRS units")
    centroid: List[float] = Field(..., description="[x, y] centroid in source CRS units")
    detection_method: str = Field(
        ...,
        description="Method used: ndbi_segmentation, bsi_morphology, change_ndbi_overlap",
    )


class InfraDetections(BaseModel):
    buildings: List[InfraDetection] = Field(default_factory=list)
    roads: List[InfraDetection] = Field(default_factory=list)
    construction_change_candidates: List[InfraDetection] = Field(default_factory=list)


class InfraSummary(BaseModel):
    building_cluster_count: int = 0
    road_segment_count: int = 0
    construction_candidate_count: int = 0
    total_built_up_area_m2: float = 0.0
    total_road_area_m2: float = 0.0
    ndbi_mean: Optional[float] = None
    ndvi_mean: Optional[float] = None


class ModelInfo(BaseModel):
    name: str
    version: str
    description: str


class ProcessingInfo(BaseModel):
    duration_seconds: float
    real_data: bool = True
    bands_used: List[str] = Field(default_factory=list)
    resolution_m: float = 10.0
    resampling_applied: Optional[str] = None


# ─── Response ─────────────────────────────────────────────────────────────────

class InfraAnalysisResponse(BaseModel):
    result_id: str
    scene_id: str
    source: str = "Copernicus Data Space"
    acquired_at: Optional[str] = None
    crs: Optional[str] = None
    resolution_m: float = 10.0
    model: ModelInfo
    detections: InfraDetections = Field(default_factory=InfraDetections)
    summary: InfraSummary = Field(default_factory=InfraSummary)
    processing: ProcessingInfo
    status: str = Field(..., description="'completed' | 'failed'")
    error: Optional[str] = None
    cached: bool = False
    created_at: str

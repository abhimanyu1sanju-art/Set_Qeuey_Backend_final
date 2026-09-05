"""
SatQuery AI — Disaster Detection Schemas (Phase 5)

Pydantic models for the real satellite-based disaster detection API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# ─── Request ──────────────────────────────────────────────────────────────────

class DisasterAnalysisRequest(BaseModel):
    scene_id: str = Field(..., description="Sentinel-2 STAC scene ID for the post-disaster/current image")
    before_scene_id: Optional[str] = Field(None, description="Optional pre-disaster Sentinel-2 scene ID for change analysis")
    disaster_type: str = Field(
        ...,
        description="Type of disaster: flood, fire, drought, landslide, cyclone, infrastructure"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold (0–1)"
    )
    force_reprocess: bool = Field(
        default=False,
        description="If true, ignore cached result and rerun"
    )

# ─── Evidence & Processing ────────────────────────────────────────────────────

class DisasterEvidence(BaseModel):
    spectral_metric: Optional[str] = None
    spectral_mean: Optional[float] = None
    spectral_change: Optional[float] = None
    water_extent_change: Optional[float] = None
    vegetation_loss: Optional[float] = None
    infrastructure_overlap: Optional[float] = None
    notes: Optional[str] = None

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

class DisasterRegion(BaseModel):
    label: str
    area_m2: float
    pixel_count: int
    confidence: float
    bbox: List[float]
    centroid: List[float]
    detection_method: str

# ─── Response ─────────────────────────────────────────────────────────────────

class DisasterAnalysisResponse(BaseModel):
    result_id: str
    scene_id: str
    before_scene_id: Optional[str] = None
    source: str = "Copernicus Data Space"
    crs: Optional[str] = None
    resolution_m: float = 10.0
    
    disaster_type: str
    status: str = Field(..., description="'completed' | 'failed'")
    error: Optional[str] = None
    
    affected_area_m2: float = 0.0
    affected_area_ha: float = 0.0
    pixel_count: int = 0
    region_count: int = 0
    change_percent: Optional[float] = None
    confidence: float = 0.0
    
    severity: Optional[str] = None
    regions: List[DisasterRegion] = Field(default_factory=list)
    evidence: DisasterEvidence = Field(default_factory=DisasterEvidence)
    
    model: ModelInfo
    processing: ProcessingInfo
    cached: bool = False
    created_at: str

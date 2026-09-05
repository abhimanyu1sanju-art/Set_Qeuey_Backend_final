from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class FusionAnalysisRequest(BaseModel):
    sentinel1_scene_id: str = Field(..., description="STAC item ID for the Sentinel-1 GRD scene")
    sentinel2_scene_id: str = Field(..., description="STAC item ID for the Sentinel-2 L2A scene")
    analysis_mode: str = Field(
        default="general", 
        description="Mode of fusion: 'vegetation', 'water', 'burn', or 'general'"
    )
    confidence_threshold: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="Threshold for fusion score to be considered significant"
    )
    force_reprocess: bool = Field(default=False)

class FusionStats(BaseModel):
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    valid_pixel_count: int = 0
    total_pixel_count: int = 0
    valid_pixel_pct: float = 0.0

class FusionAnalysisResponse(BaseModel):
    result_id: str
    status: str
    sentinel1_scene_id: str
    sentinel2_scene_id: str
    analysis_mode: str
    available_polarizations: List[str]
    optical_index_used: str
    crs: Optional[str] = None
    dimensions: Optional[List[int]] = None
    resolution_m: Optional[float] = None
    fusion_score_stats: Optional[FusionStats] = None
    created_at: datetime
    error: Optional[str] = None
    cached: bool = False
    processing: Optional[Dict[str, Any]] = None

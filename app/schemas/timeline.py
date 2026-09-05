"""
SatQuery AI — Timeline Schemas (Phase 7)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class TimelineSearchRequest(BaseModel):
    bbox: List[float] = Field(..., min_length=4, max_length=4, description="[min_lon, min_lat, max_lon, max_lat]")
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    satellite: str = Field(..., description="'sentinel-2' or 'sentinel-1'")
    max_cloud_cover: Optional[float] = Field(100.0, description="Max cloud cover (0-100) for optical")
    limit: Optional[int] = Field(10, description="Maximum number of observations to fetch")

class TimelineChangeMetrics(BaseModel):
    previous_date: str
    current_date: str
    days_between: int
    ndvi_delta: Optional[float] = None
    ndwi_delta: Optional[float] = None
    nbr_delta: Optional[float] = None
    sar_delta: Optional[float] = None
    magnitude: Optional[float] = None
    direction: Optional[str] = None

class TimelineObservation(BaseModel):
    observation_id: str
    scene_id: str
    acquired_at: datetime
    satellite: str
    sensor: str
    collection: str
    cloud_cover: Optional[float] = None
    bbox: List[float]
    metrics: Dict[str, Optional[float]]
    change_from_previous: Optional[TimelineChangeMetrics] = None
    preview_url: Optional[str] = None
    source: str = "Copernicus Data Space"

class TimelineResponse(BaseModel):
    observations: List[TimelineObservation]
    total_observations: int
    satellite: str
    bbox: List[float]

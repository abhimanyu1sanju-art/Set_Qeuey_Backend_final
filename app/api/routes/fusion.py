"""
SatQuery AI — Fusion Router (Phase 6)
"""

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import FileResponse
import logging
from pathlib import Path as FileSysPath

from app.schemas.fusion import FusionAnalysisRequest, FusionAnalysisResponse
from app.services import fusion_service
from app.db.mongodb import get_fusion_collection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fusion", tags=["SAR + Optical Fusion"])

@router.post(
    "/analyze",
    response_model=FusionAnalysisResponse,
    status_code=200,
    summary="Run SAR + Optical Fusion",
)
def run_fusion(body: FusionAnalysisRequest) -> FusionAnalysisResponse:
    """Geospatially fuse a Sentinel-1 and Sentinel-2 scene."""
    result = fusion_service.run_fusion_analysis_safe(
        s1_scene_id=body.sentinel1_scene_id,
        s2_scene_id=body.sentinel2_scene_id,
        analysis_mode=body.analysis_mode,
        confidence_threshold=body.confidence_threshold,
        force_reprocess=body.force_reprocess,
    )
    if result.get("status") == "failed":
        # We can still return 200 with the error field populated,
        # or raise an exception. For the frontend to handle gracefully, we return the object.
        pass
    
    return FusionAnalysisResponse(**result)

@router.get(
    "/results/{analysis_id}",
    response_model=FusionAnalysisResponse,
    status_code=200,
)
def get_fusion_results(analysis_id: str = Path(...)) -> FusionAnalysisResponse:
    col = get_fusion_collection()
    doc = col.find_one({"result_id": analysis_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Fusion result not found")
    return FusionAnalysisResponse(**doc)

@router.get(
    "/preview/{analysis_id}",
    status_code=200,
)
def get_fusion_preview(analysis_id: str = Path(...)):
    img_path = FileSysPath("outputs") / "fusion" / f"{analysis_id}.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Preview image not found")
    return FileResponse(img_path, media_type="image/png")

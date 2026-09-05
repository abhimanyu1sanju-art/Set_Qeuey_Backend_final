"""
SatQuery AI — Change Detection Routes (Phase 3)
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pathlib import Path
import os

from app.schemas.change import ChangeAnalysisRequest, ChangeAnalysisResponse
from app.services import change_service
from app.core.config import settings

router = APIRouter(prefix="/change", tags=["Change Detection"])


@router.post("/analyze", response_model=ChangeAnalysisResponse)
async def analyze_change(request: ChangeAnalysisRequest):
    """
    Run real pixel-level change detection on two Sentinel-2 scenes.
    """
    import traceback
    try:
        result_dict = change_service.run_change_analysis(
            before_scene_id=request.before_scene_id,
            after_scene_id=request.after_scene_id,
            method=request.method,
            threshold=request.threshold,
            force_reprocess=request.force_reprocess
        )
    except Exception as exc:
        err_msg = traceback.format_exc()
        print("CRITICAL ERROR IN CHANGE ANALYSIS:")
        print(err_msg)
        raise HTTPException(status_code=500, detail=str(exc))
    
    if result_dict.get("status") == "failed":
        # We still return 200 with status='failed' so frontend can show error gracefully,
        # but if we want to mimic spectral, we can raise 400.
        # Returning the response model is better.
        pass
        
    return ChangeAnalysisResponse(**result_dict)


@router.get("/results/{analysis_id}", response_model=ChangeAnalysisResponse)
async def get_change_results(analysis_id: str):
    """Retrieve an existing change analysis result by ID."""
    from app.db.mongodb import get_change_analyses_collection
    col = get_change_analyses_collection()
    doc = col.find_one({"analysis_id": analysis_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Analysis result not found")
        
    doc["cached"] = True
    return ChangeAnalysisResponse(**doc)


@router.get("/preview/{analysis_id}")
async def get_change_preview(analysis_id: str):
    """Serve the generated PNG preview for the change map."""
    preview_path = Path(settings.upload_dir) / "change_maps" / f"{analysis_id}.png"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
        
    return FileResponse(
        path=preview_path,
        media_type="image/png",
        filename=f"{analysis_id}.png"
    )

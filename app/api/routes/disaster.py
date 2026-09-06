"""
SatQuery AI — Disaster Detection API Routes (Phase 5)

Endpoints:
  POST /api/disaster/analyze     — Run or retrieve cached disaster detection
  GET  /api/disaster/results/{scene_id}  — Retrieve result by scene_id
  GET  /api/disaster/preview/{result_id} — Serve PNG preview
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas.disaster import (
    DisasterAnalysisRequest,
    DisasterAnalysisResponse,
)
from app.services.disaster_service import run_disaster_analysis_safe
from app.db.mongodb import get_disaster_analyses_collection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/disaster", tags=["disaster"])

DISASTER_TIMEOUT_S = 300  # 5-minute hard cap


@router.post("/analyze", response_model=DisasterAnalysisResponse)
async def analyze_disaster(request: DisasterAnalysisRequest):
    """
    Run real satellite-based disaster detection.
    Hard timeout: 300 seconds.
    """
    logger.info(
        "Disaster analysis request — scene=%s type=%s before=%s threshold=%.2f force=%s",
        request.scene_id, request.disaster_type, request.before_scene_id,
        request.confidence_threshold, request.force_reprocess,
    )

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                run_disaster_analysis_safe,
                request.scene_id,
                request.before_scene_id,
                request.disaster_type,
                request.confidence_threshold,
                request.force_reprocess,
            ),
            timeout=DISASTER_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Disaster analysis timed out after {DISASTER_TIMEOUT_S}s. "
                "The scene may be very large or the Copernicus S3 endpoint is slow. "
                "Try again or use a different scene."
            ),
        )

    if result.get("status") == "failed":
        logger.error("Disaster analysis failed: %s", result.get("error"))

    return result



@router.get("/results/{scene_id}", response_model=DisasterAnalysisResponse)
def get_disaster_results(scene_id: str, disaster_type: str):
    """
    Retrieve a cached disaster analysis result by scene_id and type.
    """
    col = get_disaster_analyses_collection()
    doc = col.find_one({"scene_id": scene_id, "disaster_type": disaster_type}, {"_id": 0})
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"No {disaster_type} analysis found for scene {scene_id}."
        )
    doc["cached"] = True
    return doc


@router.get("/preview/{result_id}")
def get_disaster_preview(result_id: str):
    """
    Serve the colorized disaster detection preview PNG.
    Falls back to MongoDB base64 when local file is missing (Render restart).
    """
    preview_path = Path("outputs") / "disaster" / f"{result_id}.png"

    if preview_path.exists():
        return FileResponse(
            path=str(preview_path),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    col = get_disaster_analyses_collection()
    doc = col.find_one({"result_id": result_id}, {"_id": 0, "preview_image_b64": 1, "result_id": 1})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Preview not found for result_id={result_id}")

    b64 = doc.get("preview_image_b64")
    if not b64:
        raise HTTPException(
            status_code=404,
            detail="Preview image not yet generated. Re-run analysis with force_reprocess=true.",
        )

    import base64
    from fastapi.responses import Response as FastAPIResponse
    image_bytes = base64.b64decode(b64)
    return FastAPIResponse(
        content=image_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )

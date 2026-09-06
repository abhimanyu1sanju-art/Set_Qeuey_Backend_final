"""
SatQuery AI — Infrastructure Detection API Routes (Phase 4)

Endpoints:
  POST /api/infrastructure/analyze     — Run or retrieve cached infrastructure detection
  GET  /api/infrastructure/results/{scene_id}  — Retrieve result by scene_id
  GET  /api/infrastructure/preview/{result_id} — Serve PNG preview
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from app.schemas.infrastructure import (
    InfrastructureAnalysisRequest,
    InfraAnalysisResponse,
)
from app.services.infrastructure_service import run_infrastructure_analysis_safe
from app.db.mongodb import get_infrastructure_results_collection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/infrastructure", tags=["infrastructure"])

INFRA_TIMEOUT_S = 300  # 5-minute hard cap


@router.post("/analyze", response_model=InfraAnalysisResponse)
async def analyze_infrastructure(request: InfrastructureAnalysisRequest):
    """
    Run real satellite-based infrastructure detection on a Sentinel-2 scene.

    Uses NDBI/BSI/NDVI spectral index segmentation — appropriate for 10m resolution data.
    Results are cached in MongoDB per scene_id.
    Hard timeout: 300 seconds.

    - **scene_id**: Must exist in the `satellite_scenes` MongoDB collection
    - **detection_types**: Any combination of ["buildings", "roads", "construction"]
    - **confidence_threshold**: Detections below this are excluded (default 0.5)
    - **force_reprocess**: Re-run even if a cached result exists
    """
    logger.info(
        "Infrastructure analysis request — scene=%s types=%s threshold=%.2f force=%s",
        request.scene_id, request.detection_types, request.confidence_threshold, request.force_reprocess,
    )

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                run_infrastructure_analysis_safe,
                request.scene_id,
                request.detection_types,
                request.confidence_threshold,
                request.force_reprocess,
            ),
            timeout=INFRA_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Infrastructure analysis timed out after {INFRA_TIMEOUT_S}s. "
                "The scene may be very large or the Copernicus S3 endpoint is slow. "
                "Try again or use a different scene."
            ),
        )

    if result.get("status") == "failed":
        logger.error("Infrastructure analysis failed: %s", result.get("error"))
        # Return 200 with status=failed so frontend can show the real error message

    return result



@router.get("/results/{scene_id}", response_model=InfraAnalysisResponse)
def get_infrastructure_results(scene_id: str):
    """
    Retrieve a cached infrastructure analysis result by scene_id.

    Returns 404 if no result exists for this scene yet.
    """
    col = get_infrastructure_results_collection()
    doc = col.find_one({"scene_id": scene_id}, {"_id": 0})
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"No infrastructure analysis found for scene {scene_id}. Run POST /api/infrastructure/analyze first.",
        )
    doc["cached"] = True
    return doc


@router.get("/preview/{result_id}")
def get_infrastructure_preview(result_id: str):
    """
    Serve the colorized infrastructure detection preview PNG.

    First attempts to serve from local filesystem (fast, for local dev).
    Falls back to base64-encoded bytes stored in MongoDB (production-safe,
    survives Render ephemeral filesystem restarts).
    """
    preview_path = Path("outputs") / "infrastructure" / f"{result_id}.png"

    # ── Fast path: serve from local file (local dev / warm Render instance) ──
    if preview_path.exists():
        return FileResponse(
            path=str(preview_path),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # ── Fallback: serve from MongoDB base64 (survives restarts) ──────────────
    col = get_infrastructure_results_collection()
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

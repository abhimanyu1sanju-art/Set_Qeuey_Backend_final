"""
SatQuery AI — Spectral Analysis Router (Phase 2)

Endpoints:
  POST /api/spectral/analyze          — Trigger real spectral index processing
  GET  /api/spectral/results/{scene_id}  — Retrieve cached analysis metadata
  GET  /api/spectral/preview/{scene_id}/{index}  — Serve PNG preview image

All logic lives in spectral_service.py.
No mock data is ever returned.
Gemini AI is NOT involved in spectral calculations.

Authentication:
  - CDSE credentials required for band download.
  - Credentials are read server-side only.
  - Never exposed to the frontend.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse, Response

from app.schemas.spectral import SpectralAnalysisRequest, SpectralAnalysisResponse
from app.services import spectral_service
from app.services.cdse_auth import is_cdse_configured

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spectral", tags=["Spectral Analysis"])


# ─── POST /api/spectral/analyze ───────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=SpectralAnalysisResponse,
    status_code=200,
    summary="Run real spectral analysis on a Sentinel-2 L2A scene",
    description=(
        "Calculate one or more spectral indices from **real** Sentinel-2 L2A "
        "raster bands downloaded from the Copernicus Data Space Ecosystem.\n\n"
        "**Supported indices:**\n"
        "- `NDVI` — Normalized Difference Vegetation Index "
        "  `(B08 - B04) / (B08 + B04)`\n"
        "- `NDWI` — Normalized Difference Water Index (McFeeters 1996) "
        "  `(B03 - B08) / (B03 + B08)`\n"
        "- `NBR` — Normalized Burn Ratio "
        "  `(B08 - B12) / (B08 + B12)` — B12 is resampled from 20m to 10m\n\n"
        "**Required:** `scene_id` must be a valid Sentinel-2 L2A STAC item ID "
        "(search for scenes first via `POST /api/satellite/search`).\n\n"
        "**Processing:** Pure NumPy calculations on real Copernicus raster data. "
        "Gemini AI is NOT used for spectral calculations.\n\n"
        "**Output:** GeoTIFF files written to `outputs/spectral/<scene_id>/`\n\n"
        "**Errors:**\n"
        "- `400` — scene is not Sentinel-2 L2A, or unsupported index requested\n"
        "- `404` — scene not found\n"
        "- `503` — CDSE credentials not configured (set `COPERNICUS_CLIENT_ID` + "
        "`COPERNICUS_CLIENT_SECRET` in your .env file)\n"
    ),
    responses={
        400: {"description": "Invalid request — not a Sentinel-2 L2A scene or bad index"},
        404: {"description": "Scene not found"},
        503: {"description": "CDSE credentials not configured"},
    },
)
def analyze_spectral(body: SpectralAnalysisRequest) -> SpectralAnalysisResponse:
    """
    Run real spectral index calculations from Sentinel-2 L2A bands.
    """
    logger.info(
        "POST /api/spectral/analyze  scene_id=%s indices=%s force=%s",
        body.scene_id, body.indices, body.force_reprocess,
    )

    # Early check: if credentials are not configured, return 503 immediately
    # rather than wasting time fetching scene metadata
    if not is_cdse_configured() and not body.force_reprocess:
        # Check cache — if everything is cached, we don't need credentials
        all_cached = all(
            spectral_service.get_cached_result(body.scene_id, idx) is not None
            for idx in body.indices
        )
        if not all_cached:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Copernicus Data Space credentials are not configured. "
                    "Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in your .env file. "
                    "Register at: https://dataspace.copernicus.eu/ "
                    "No mock spectral data will be returned."
                ),
            )

    return spectral_service.run_spectral_analysis(
        scene_id=body.scene_id,
        indices=body.indices,
        collection=body.collection,
        force_reprocess=body.force_reprocess,
    )


# ─── GET /api/spectral/results/{scene_id} ────────────────────────────────────

@router.get(
    "/results/{scene_id}",
    status_code=200,
    summary="Retrieve cached spectral analysis results",
    description=(
        "Retrieve cached spectral analysis metadata for a scene from MongoDB.\n\n"
        "Returns analysis records for all processed indices (NDVI, NDWI, NBR).\n\n"
        "**Errors:**\n"
        "- `404` — no cached results found for this scene\n"
    ),
    responses={
        404: {"description": "No cached results found"},
    },
)
def get_spectral_results(
    scene_id: str = Path(..., description="STAC scene/item ID"),
    index: Optional[str] = Query(
        default=None,
        description="Filter by index name (NDVI, NDWI, or NBR). Returns all if omitted.",
    ),
) -> dict:
    """Retrieve cached spectral analysis results from MongoDB."""
    logger.info("GET /api/spectral/results/%s index=%s", scene_id, index)

    indices_to_check = [index.upper()] if index else ["NDVI", "NDWI", "NBR"]
    results = {}

    for idx in indices_to_check:
        cached = spectral_service.get_cached_result(scene_id, idx)
        if cached:
            results[idx.lower()] = cached

    if not results:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No cached spectral analysis found for scene '{scene_id}'. "
                "Run POST /api/spectral/analyze first."
            ),
        )

    return {
        "scene_id": scene_id,
        "source": "Copernicus Data Space Ecosystem",
        "results": results,
        "cached": True,
    }


# ─── GET /api/spectral/preview/{scene_id}/{index} ────────────────────────────

@router.get(
    "/preview/{scene_id}/{index}",
    status_code=200,
    summary="Serve PNG preview of a spectral index",
    description=(
        "Serve the PNG preview image for a spectral index result.\n\n"
        "The PNG is a colourmap-normalised visualisation of the GeoTIFF data "
        "suitable for display in the frontend. It is NOT the primary scientific output — "
        "use the GeoTIFF for analysis.\n\n"
        "**Errors:**\n"
        "- `404` — preview not yet generated (run analyze first)\n"
    ),
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG preview image"},
        404: {"description": "Preview not generated"},
    },
)
def get_spectral_preview(
    scene_id: str = Path(..., description="STAC scene/item ID"),
    index: str = Path(..., description="Index name: ndvi, ndwi, or nbr"),
) -> Response:
    """Serve the PNG preview for a spectral index."""
    logger.info("GET /api/spectral/preview/%s/%s", scene_id, index)

    index_lower = index.lower()
    if index_lower not in ("ndvi", "ndwi", "nbr"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid index '{index}'. Must be one of: ndvi, ndwi, nbr.",
        )

    preview_path = spectral_service.get_preview_path(scene_id, index_lower)
    if preview_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Preview for {index.upper()} not found for scene '{scene_id}'. "
                "Run POST /api/spectral/analyze first."
            ),
        )

    return FileResponse(
        path=str(preview_path),
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600",
            "X-Scene-ID": scene_id,
            "X-Index": index.upper(),
            "X-Source": "Copernicus Data Space Ecosystem",
        },
    )


# ─── GET /api/spectral/status ────────────────────────────────────────────────

@router.get(
    "/status",
    status_code=200,
    summary="Check spectral analysis service status",
    description=(
        "Check whether CDSE credentials are configured and the spectral "
        "analysis service is ready.\n\n"
        "Returns `credentials_configured: true/false` — useful for the "
        "frontend to show appropriate UI before the user attempts analysis."
    ),
)
def spectral_status() -> dict:
    """Check if CDSE credentials are configured."""
    configured = is_cdse_configured()
    return {
        "service": "spectral_analysis",
        "credentials_configured": configured,
        "supported_indices": ["NDVI", "NDWI", "NBR"],
        "source": "Copernicus Data Space Ecosystem",
        "processing_backend": "Rasterio + GDAL + NumPy",
        "message": (
            "Ready for spectral analysis."
            if configured
            else (
                "CDSE credentials not configured. "
                "Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in .env. "
                "Register at https://dataspace.copernicus.eu/"
            )
        ),
    }

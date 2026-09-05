"""
SatQuery AI — Satellite Router (Phase 1)

Endpoints:
  POST /api/satellite/search                    — Search Sentinel-2 or Sentinel-1 scenes
  GET  /api/satellite/thumbnail/{scene_id}      — Proxy real thumbnail image (fixes content-type)
  GET  /api/satellite/{scene_id}                — Get full metadata for a scene

Root cause of 'preview unavailable':
  datahub.creodias.eu serves thumbnail JPEGs with Content-Type: application/octet-stream.
  Browsers see this and refuse to render the image. The /thumbnail/{scene_id} endpoint
  proxies the image through FastAPI and re-serves it with the correct Content-Type.

All logic lives in satellite_service.py.
No mock data is ever returned.
Gemini AI is NOT used in any satellite route.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import Response

from app.schemas.satellite import (
    SATELLITE_TO_COLLECTION,
    SatelliteSceneDetailResponse,
    SatelliteSearchRequest,
    SatelliteSearchResponse,
)
from app.services import satellite_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/satellite", tags=["Satellite Data"])


# ─── POST /api/satellite/search ───────────────────────────────────────────────

@router.post(
    "/search",
    response_model=SatelliteSearchResponse,
    status_code=200,
    summary="Search satellite scenes",
    description=(
        "Search for real Sentinel-2 (optical) or Sentinel-1 (SAR) scenes "
        "from the Copernicus Data Space STAC API.\n\n"
        "**Required:** `bbox` [west, south, east, north], `start_date`, `end_date`.\n\n"
        "**Optional:** `satellite` (`sentinel-2` or `sentinel-1`), "
        "`max_cloud_cover` (0-100, Sentinel-2 only), `limit` (1-50).\n\n"
        "Returns real scene metadata — never mock data.\n"
        "Returns an empty `scenes` list when no results match.\n\n"
        "**Data source:** Copernicus Data Space STAC API (public, no auth required "
        "for metadata search).\n\n"
        "**Errors:**\n"
        "- `400` — invalid bbox, dates, satellite, or cloud cover\n"
        "- `502` — Copernicus STAC API returned an error\n"
        "- `503` — Copernicus STAC API unreachable or timed out\n"
    ),
    response_description="List of matched satellite scenes",
    responses={
        400: {"description": "Invalid search parameters"},
        502: {"description": "Copernicus STAC API returned an error"},
        503: {"description": "Copernicus STAC API unreachable"},
    },
)
def search_satellite_scenes(body: SatelliteSearchRequest) -> SatelliteSearchResponse:
    """Search for real Sentinel scenes from Copernicus STAC API."""
    logger.info(
        "POST /api/satellite/search  satellite=%s bbox=%s dates=%s/%s cloud<=%s limit=%d",
        body.satellite, body.bbox, body.start_date, body.end_date,
        body.max_cloud_cover if body.satellite == "sentinel-2" else "N/A",
        body.limit,
    )
    return satellite_service.search_scenes(body)


# ─── GET /api/satellite/thumbnail/{scene_id} ──────────────────────────────────
# NOTE: This route MUST be declared BEFORE /{scene_id} so that FastAPI does not
# match the literal string "thumbnail" as a scene_id path parameter.

@router.get(
    "/thumbnail/{scene_id}",
    status_code=200,
    summary="Proxy scene thumbnail image",
    description=(
        "Fetch and proxy the real thumbnail image for a satellite scene.\n\n"
        "**Why proxy?** datahub.creodias.eu serves thumbnails with "
        "`Content-Type: application/octet-stream` even though the bytes are JPEG or PNG. "
        "Browsers refuse to render images with that content-type. "
        "This endpoint fixes the content-type header so img tags work correctly.\n\n"
        "**Returns:** The actual thumbnail image bytes with the correct Content-Type.\n\n"
        "**No auth required** — thumbnails on datahub.creodias.eu are publicly accessible.\n\n"
        "**Errors:**\n"
        "- `404` — Scene not found or no thumbnail available\n"
        "- `502` — Could not fetch thumbnail from Copernicus\n"
        "- `503` — Copernicus infrastructure unreachable\n"
    ),
    responses={
        200: {"content": {"image/jpeg": {}, "image/png": {}}, "description": "Thumbnail image"},
        404: {"description": "No thumbnail available"},
        502: {"description": "Could not fetch thumbnail from Copernicus"},
        503: {"description": "Copernicus infrastructure unreachable"},
    },
)
def proxy_scene_thumbnail(
    scene_id: str = Path(..., description="STAC scene/item ID"),
    collection: Optional[str] = Query(
        default="sentinel-2-l2a",
        description="STAC collection ID.",
    ),
):
    """
    Proxy the Copernicus thumbnail image with the correct Content-Type.

    datahub.creodias.eu serves thumbnails as application/octet-stream.
    We detect the true format from magic bytes and serve with image/jpeg or image/png.
    """
    logger.info("GET /api/satellite/thumbnail/%s (collection=%s)", scene_id, collection)

    # Resolve the thumbnail URL via service (uses MongoDB cache or STAC fetch)
    preview_url = satellite_service.get_scene_preview_url(
        scene_id, collection or "sentinel-2-l2a"
    )

    if not preview_url:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No thumbnail available for scene '{scene_id}'. "
                "The scene may not have a public preview image."
            ),
        )

    # Fetch the image bytes from Copernicus
    try:
        img_resp = httpx.get(
            preview_url,
            timeout=20.0,
            follow_redirects=True,
            headers={"Accept": "image/jpeg, image/png, image/*, */*"},
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=503, detail="Thumbnail fetch timed out.")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Could not reach Copernicus: {exc}")

    if img_resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Copernicus returned HTTP {img_resp.status_code} for thumbnail.",
        )

    image_bytes = img_resp.content
    if not image_bytes:
        raise HTTPException(status_code=502, detail="Empty thumbnail response from Copernicus.")

    # Detect true image format from magic bytes — do NOT trust the declared Content-Type
    # JPEG: starts with FF D8 FF
    # PNG:  starts with 89 50 4E 47 (ASCII: .PNG)
    if image_bytes[:3] == b"\xff\xd8\xff":
        content_type = "image/jpeg"
    elif image_bytes[:4] == b"\x89PNG":
        content_type = "image/png"
    else:
        declared = img_resp.headers.get("content-type", "")
        content_type = "image/png" if "png" in declared else "image/jpeg"

    logger.info(
        "Proxying thumbnail for %s — %d bytes — %s",
        scene_id, len(image_bytes), content_type,
    )

    return Response(
        content=image_bytes,
        media_type=content_type,
        headers={
            # Cache for 1 hour in browser — thumbnails are stable
            "Cache-Control": "public, max-age=3600",
        },
    )


# ─── GET /api/satellite/{scene_id} ────────────────────────────────────────────

@router.get(
    "/{scene_id}",
    response_model=SatelliteSceneDetailResponse,
    status_code=200,
    summary="Get scene metadata",
    description=(
        "Retrieve full metadata for a satellite scene by its STAC item ID.\n\n"
        "**Required:** `scene_id` — the STAC item ID "
        "(e.g. `S2C_MSIL2A_20250128T053131_N0511_R105_T43QCA_20250128T084454`).\n\n"
        "**Optional:** `collection` — STAC collection ID "
        "(`sentinel-2-l2a` or `sentinel-1-grd`). Defaults to `sentinel-2-l2a`.\n\n"
        "Returns cached result from MongoDB if available, "
        "otherwise fetches live from Copernicus STAC API.\n\n"
        "**Errors:**\n"
        "- `404` — Scene not found in STAC or MongoDB\n"
        "- `503` — Copernicus STAC API unreachable\n"
    ),
    response_description="Full scene metadata",
    responses={
        404: {"description": "Scene not found"},
        503: {"description": "Copernicus STAC API unreachable"},
    },
)
def get_scene(
    scene_id: str = Path(..., description="STAC scene/item ID"),
    collection: Optional[str] = Query(
        default="sentinel-2-l2a",
        description="STAC collection ID. Defaults to 'sentinel-2-l2a'.",
    ),
) -> SatelliteSceneDetailResponse:
    """Retrieve full metadata for a satellite scene."""
    logger.info("GET /api/satellite/%s (collection=%s)", scene_id, collection)

    valid_collections = set(SATELLITE_TO_COLLECTION.values())
    if collection not in valid_collections:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid collection '{collection}'. Valid: {', '.join(sorted(valid_collections))}",
        )

    scene = satellite_service.get_scene_metadata(scene_id, collection)
    if scene is None:
        raise HTTPException(
            status_code=404,
            detail=f"Scene '{scene_id}' not found in collection '{collection}'.",
        )
    return scene

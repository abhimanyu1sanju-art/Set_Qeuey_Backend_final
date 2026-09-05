"""
SatQuery AI — Timeline Router (Phase 7)

Uses asyncio.wait_for + run_in_executor so the long-running synchronous
timeline processing does not block the FastAPI event loop.
Returns 504 on overall timeout (300s).
"""

import asyncio
import logging
from fastapi import APIRouter, HTTPException
from app.schemas.timeline import TimelineSearchRequest, TimelineResponse
from app.services.timeline_service import generate_timeline

logger = logging.getLogger(__name__)

router = APIRouter()

TIMELINE_TIMEOUT_S = 300  # 5-minute hard cap

@router.post(
    "/search",
    response_model=TimelineResponse,
    summary="Generate Historical Timeline",
    description=(
        "Generate a historical timeline for the given bounding box, pulling real "
        "Copernicus STAC scenes and computing windowed raster spectral indices "
        "(NDVI, NDWI, SAR VV) to detect land-cover change over time.\n\n"
        "Returns an empty observations list — not an error — when no scenes match.\n"
        "Hard timeout: 300 seconds."
    ),
)
async def search_timeline(request: TimelineSearchRequest) -> TimelineResponse:
    """
    Run generate_timeline() in a thread pool executor so it doesn't
    block the event loop during its many I/O-intensive raster reads.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, generate_timeline, request),
            timeout=TIMELINE_TIMEOUT_S,
        )
        return result
    except asyncio.TimeoutError:
        logger.error(
            "Timeline timed out after %ds for bbox=%s dates=%s/%s",
            TIMELINE_TIMEOUT_S, request.bbox, request.start_date, request.end_date
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"Timeline processing timed out after {TIMELINE_TIMEOUT_S}s. "
                "Try narrowing the date range, raising max cloud cover, or using "
                "a smaller bounding box."
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Timeline failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

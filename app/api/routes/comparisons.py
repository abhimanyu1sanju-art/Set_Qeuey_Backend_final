"""
SatQuery AI — Comparisons Router (Phase 7)

Endpoints:
  POST /api/comparisons          — Run a two-image comparison
  GET  /api/comparisons/{id}     — Retrieve a saved comparison result

Route handlers are intentionally thin — all logic lives in comparison_service.py.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.schemas.comparison import ComparisonRequest, ComparisonResponse
from app.services import comparison_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/comparisons", tags=["Comparisons"])


@router.post(
    "",
    response_model=ComparisonResponse,
    status_code=200,
    summary="Compare two satellite images with AI",
    description=(
        "Submit two previously uploaded images (Image A = before, Image B = after) "
        "for AI-powered change-detection comparison.\n\n"
        "**Required:** `image_id_a`, `image_id_b` (from `POST /api/images/upload`).\n\n"
        "**Optional:** `query` — natural-language question about what to compare.\n"
        "If omitted, a general change-detection analysis is performed.\n\n"
        "**Optional:** `session_id` — associates the result with an existing session. "
        "Both images must already be linked to the session.\n\n"
        "**Workflow:**\n"
        "1. Validates both images exist.\n"
        "2. Checks metadata compatibility (dimensions, format, CRS).\n"
        "3. Sends BOTH images to Gemini in a single request.\n"
        "4. Parses and returns structured change-detection results.\n"
        "5. Saves the result to MongoDB.\n\n"
        "**Returns:** Full comparison result including AI explanation, "
        "detected changes, preprocessing status, and limitations.\n\n"
        "**Errors:**\n"
        "- `400` — images not linked to session\n"
        "- `404` — image or session not found\n"
        "- `422` — images are metadata-incompatible\n"
        "- `503` — AI provider not configured (missing `AI_API_KEY`)\n"
        "- `500` — unexpected internal error\n\n"
        "> **Note:** The AI comparison is qualitative. "
        "Pixel-level spectral analysis requires calibrated multispectral data."
    ),
    response_description="Completed comparison result",
    responses={
        400: {"description": "Images not linked to session"},
        404: {"description": "Image or session not found"},
        422: {"description": "Images are metadata-incompatible"},
        503: {"description": "AI provider not configured — set AI_API_KEY in .env"},
        500: {"description": "Internal server error"},
    },
)
def compare_images(body: ComparisonRequest) -> ComparisonResponse:
    """Run AI-powered two-image change-detection comparison."""
    return comparison_service.run_comparison(
        image_id_a=body.image_id_a,
        image_id_b=body.image_id_b,
        query=body.query or "",
        session_id=body.session_id,
        label_a=body.label_a or "Image A — Before",
        label_b=body.label_b or "Image B — After",
    )


@router.get(
    "/{comparison_id}",
    response_model=ComparisonResponse,
    status_code=200,
    summary="Get a saved comparison result",
    description=(
        "Retrieve a previously saved comparison result by its `comparison_id`.\n\n"
        "Returns HTTP 404 if the comparison does not exist."
    ),
    response_description="Saved comparison result",
    responses={
        404: {"description": "Comparison not found"},
        500: {"description": "Internal server error"},
    },
)
def get_comparison(comparison_id: str) -> ComparisonResponse:
    """Retrieve a saved comparison result by ID."""
    return comparison_service.get_comparison_by_id(comparison_id)

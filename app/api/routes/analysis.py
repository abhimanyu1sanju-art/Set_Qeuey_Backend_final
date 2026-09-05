"""
SatQuery AI — Analysis Router (Phase 4)

Endpoints:
  POST /api/analysis   — Run AI analysis on an uploaded image

Route handler is intentionally thin — all logic lives in analysis_service.py.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.schemas.analysis import AnalysisRequest, AnalysisResponse
from app.services import analysis_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post(
    "",
    response_model=AnalysisResponse,
    status_code=200,
    summary="Analyze an image with AI",
    description=(
        "Submit an uploaded image for AI-powered analysis.\n\n"
        "**Required:** `image_id` (from `POST /api/images/upload`).\n\n"
        "**Optional:** `query` — natural-language question about the image.\n"
        "If omitted, a general scene analysis is performed.\n\n"
        "**Optional:** `session_id` — associates the result with an existing session.\n"
        "When provided the image must already be linked to the session "
        "via `POST /api/sessions/{session_id}/images/{image_id}`.\n\n"
        "**Returns:** Full analysis result including AI answer, provider, model, and status.\n\n"
        "**Errors:**\n"
        "- `400` — invalid request or image not linked to session\n"
        "- `404` — image or session not found\n"
        "- `503` — AI provider not configured (missing `AI_API_KEY`)\n"
        "- `500` — unexpected internal error\n\n"
        "> **Note:** Confidence scores are not available for Gemini. "
        "`confidence` will always be `null` and `confidence_available` will be `false`."
    ),
    response_description="Completed analysis result",
    responses={
        400: {"description": "Invalid request or image not linked to session"},
        404: {"description": "Image or session not found"},
        503: {"description": "AI provider not configured — set AI_API_KEY in .env"},
        500: {"description": "Internal server error"},
    },
)
def analyze_image(body: AnalysisRequest) -> AnalysisResponse:
    """Run AI vision analysis on an uploaded image."""
    return analysis_service.run_analysis(
        image_id=body.image_id,
        query=body.query or "",
        session_id=body.session_id,
        analysis_type=body.analysis_type or "general",
        enable_multi_intent=body.enable_multi_intent,
    )

"""
SatQuery AI — Health Router
GET /api/health
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.db.mongodb import verify_connection

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Health check",
    description=(
        "Returns the operational status of the SatQuery AI backend "
        "and whether MongoDB is reachable. "
        "Always returns HTTP 200 so upstream load-balancers/monitors "
        "can rely on this endpoint even when the database is temporarily unavailable."
    ),
    response_description="Service health status",
)
async def health_check() -> dict[str, Any]:
    """
    Return service health.

    - **status**: always ``"ok"`` — the API itself is running.
    - **database**: ``"connected"`` or ``"unavailable"`` — MongoDB reachability.
    """
    db_ok = verify_connection()

    return {
        "status": "ok",
        "service": "satquery-backend",
        "database": "connected" if db_ok else "unavailable",
    }

"""
SatQuery AI — History Router (Phase 8)

Endpoints:
  GET    /api/history                    — List history (paginated, filterable)
  GET    /api/history/{history_id}       — Get a single history record
  DELETE /api/history/{history_id}       — Delete a single history record
  DELETE /api/history                    — Clear all history for a session

All endpoints are session-scoped.
No API keys, credentials or internal secrets are ever included in responses.
Route handlers are thin — all logic lives in history_service.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query

from app.schemas.history import (
    HistoryClearResponse,
    HistoryDeleteResponse,
    HistoryItem,
    HistoryListResponse,
)
from app.services import history_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["History"])


# ─── GET /api/history ──────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=HistoryListResponse,
    status_code=200,
    summary="List history items",
    description=(
        "Return a paginated list of history records, newest first.\n\n"
        "**Filters:**\n"
        "- `session_id` — scope results to a specific session\n"
        "- `record_type` / `operation_type` — e.g. `'analysis'`, `'comparison'`, `'infrastructure_analysis'`, `'disaster_analysis'`, `'sar_optical_fusion'`, `'timeline_analysis'`, `'spectral_analysis'`, `'change_detection'`\n"
        "- `analysis_type` — e.g. `'general'`, `'vegetation'`, `'change_detection'`\n"
        "- `q` — text search on the query field\n"
        "- `date_from` / `date_to` — ISO 8601 date range filter\n\n"
        "**Pagination:** `page` (1-based) and `limit` (max 100).\n\n"
        "Returns an empty list when no records match — never crashes."
    ),
    response_description="Paginated history list",
    responses={
        400: {"description": "Invalid filter parameters"},
        500: {"description": "Database error"},
    },
)
def list_history(
    session_id: Optional[str] = Query(
        default=None,
        description="Scope to a specific session_id.",
    ),
    record_type: Optional[str] = Query(
        default=None,
        description="Filter by record type, e.g. 'analysis', 'comparison', 'infrastructure_analysis'.",
    ),
    operation_type: Optional[str] = Query(
        default=None,
        description="Alias for record_type. Filter by operation type.",
    ),
    analysis_type: Optional[str] = Query(
        default=None,
        description="Filter by analysis type, e.g. 'general', 'vegetation'.",
    ),
    q: Optional[str] = Query(
        default=None,
        max_length=500,
        description="Text search on the query field.",
    ),
    date_from: Optional[datetime] = Query(
        default=None,
        description="ISO 8601 datetime — include records created at or after this time.",
    ),
    date_to: Optional[datetime] = Query(
        default=None,
        description="ISO 8601 datetime — include records created at or before this time.",
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-based)."),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page."),
) -> HistoryListResponse:
    """Return paginated history, newest first."""
    logger.info(
        "GET /api/history  session_id=%s record_type=%s operation_type=%s q=%r page=%d limit=%d",
        session_id, record_type, operation_type, q, page, limit,
    )
    return history_service.list_history(
        session_id=session_id,
        record_type=record_type,
        operation_type=operation_type,
        analysis_type=analysis_type,
        query_search=q,
        date_from=date_from,
        date_to=date_to,
        page=page,
        limit=limit,
    )


# ─── GET /api/history/{history_id} ────────────────────────────────────────────

@router.get(
    "/{history_id}",
    response_model=HistoryItem,
    status_code=200,
    summary="Get a single history record",
    description=(
        "Return the full detail of a single history record by `history_id`.\n\n"
        "Returns HTTP 404 when the record does not exist.\n\n"
        "Includes the original query, AI answer, image references, timestamps, "
        "detected changes (for comparisons), limitations, and Phase 6 intent results."
    ),
    response_description="Full history record",
    responses={
        400: {"description": "Invalid history_id"},
        404: {"description": "History record not found"},
        500: {"description": "Database error"},
    },
)
def get_history_item(history_id: str) -> HistoryItem:
    """Fetch a single history record by its history_id."""
    logger.info("GET /api/history/%s", history_id)
    return history_service.get_history_item(history_id)


# ─── DELETE /api/history/{history_id} ─────────────────────────────────────────

@router.delete(
    "/{history_id}",
    response_model=HistoryDeleteResponse,
    status_code=200,
    summary="Delete a single history record",
    description=(
        "Permanently delete a single history record by `history_id`.\n\n"
        "Returns HTTP 404 when the record does not exist.\n\n"
        "**Does NOT** delete the underlying session, analysis or comparison documents.\n\n"
        "**Does NOT** affect records belonging to other sessions."
    ),
    response_description="Deletion confirmation",
    responses={
        400: {"description": "Invalid history_id"},
        404: {"description": "History record not found"},
        500: {"description": "Database error"},
    },
)
def delete_history_item(history_id: str) -> HistoryDeleteResponse:
    """Delete a single history record."""
    logger.info("DELETE /api/history/%s", history_id)
    return history_service.delete_history_item(history_id)


# ─── DELETE /api/history ──────────────────────────────────────────────────────

@router.delete(
    "",
    response_model=HistoryClearResponse,
    status_code=200,
    summary="Clear all history for a session",
    description=(
        "Permanently delete **all** history records belonging to `session_id`.\n\n"
        "**`session_id` is required** — clearing history without a session scope is "
        "not allowed to prevent accidental deletion of unrelated data.\n\n"
        "Returns the number of deleted records.\n\n"
        "Returns `deleted_count=0` gracefully when the session has no history.\n\n"
        "**Does NOT** delete the session document, image documents, or other sessions."
    ),
    response_description="Clear confirmation with deleted record count",
    responses={
        400: {"description": "session_id is required"},
        500: {"description": "Database error"},
    },
)
def clear_history(
    session_id: str = Query(
        ...,
        description="Session ID whose history should be cleared. Required.",
    ),
) -> HistoryClearResponse:
    """Clear all history for a given session."""
    logger.info("DELETE /api/history  session_id=%s", session_id)
    return history_service.clear_history(session_id=session_id)

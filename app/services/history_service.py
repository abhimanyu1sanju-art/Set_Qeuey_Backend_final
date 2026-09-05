"""
SatQuery AI — History Service

Provides CRUD operations for the history collection.

History records are written automatically by:
  - analysis_service.run_analysis()           (single-image AI analysis)
  - comparison_service.run_comparison()       (two-image comparison)
  - spectral_service.run_spectral_analysis()  (spectral index analysis)
  - change_service.run_change_detection()     (change detection)
  - infrastructure_service.*                  (infrastructure detection)
  - disaster_service.*                        (disaster detection)
  - fusion_service.*                          (SAR + optical fusion)
  - timeline_service.*                        (historical timeline)

Public API:
  create_history_item(...)      → history_id str
  get_history_item(history_id)  → HistoryItem
  list_history(...)             → HistoryListResponse
  delete_history_item(...)      → HistoryDeleteResponse
  clear_history(session_id)     → HistoryClearResponse

All functions raise HTTPException with clean, human-readable messages.
No secrets (API keys, DB URIs, credentials) are ever exposed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

from app.db.mongodb import get_history_collection
from app.schemas.history import (
    HistoryClearResponse,
    HistoryDeleteResponse,
    HistoryDocument,
    HistoryItem,
    HistoryListItem,
    HistoryListResponse,
)

logger = logging.getLogger(__name__)


# ─── ID Generation ─────────────────────────────────────────────────────────────

def _generate_history_id() -> str:
    """Generate hist_<ULID>."""
    try:
        from ulid import ULID
        return f"hist_{ULID()}"
    except ImportError:
        import uuid
        return f"hist_{uuid.uuid4().hex}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _doc_to_item(doc: dict) -> HistoryItem:
    """Convert a raw MongoDB history document to a HistoryItem response model."""
    return HistoryItem(
        history_id=doc["history_id"],
        session_id=doc.get("session_id"),
        record_type=doc.get("record_type", "analysis"),
        operation_type=doc.get("operation_type"),
        analysis_type=doc.get("analysis_type", "general"),
        analysis_id=doc.get("analysis_id"),
        comparison_id=doc.get("comparison_id"),
        result_id=doc.get("result_id"),
        query=doc.get("query", ""),
        answer=doc.get("answer"),
        image_refs=doc.get("image_refs", []),
        scene_ids=doc.get("scene_ids", []),
        bbox=doc.get("bbox"),
        date_range=doc.get("date_range"),
        satellite=doc.get("satellite"),
        result_summary=doc.get("result_summary"),
        preview_url=doc.get("preview_url"),
        status=doc.get("status", "completed"),
        error=doc.get("error"),
        is_multi_intent=doc.get("is_multi_intent", False),
        detected_intents=doc.get("detected_intents", []),
        intent_results=doc.get("intent_results", []),
        limitations=doc.get("limitations", []),
        detected_changes=doc.get("detected_changes", []),
        label_a=doc.get("label_a"),
        label_b=doc.get("label_b"),
        provider=doc.get("provider"),
        model=doc.get("model"),
        created_at=doc["created_at"],
        updated_at=doc.get("updated_at"),
    )


def _doc_to_list_item(doc: dict) -> HistoryListItem:
    """Convert a raw MongoDB history document to a compact HistoryListItem."""
    raw_answer = doc.get("answer") or ""
    # For satellite ops, build a human-readable preview from result_summary if no text answer
    if not raw_answer and doc.get("result_summary"):
        summary = doc["result_summary"]
        parts = [f"{k}: {v}" for k, v in summary.items() if v is not None][:4]
        raw_answer = ", ".join(parts)
    preview = raw_answer[:200] if raw_answer else None
    return HistoryListItem(
        history_id=doc["history_id"],
        session_id=doc.get("session_id"),
        record_type=doc.get("record_type", "analysis"),
        operation_type=doc.get("operation_type"),
        analysis_type=doc.get("analysis_type", "general"),
        query=doc.get("query", ""),
        answer_preview=preview,
        image_refs=doc.get("image_refs", []),
        scene_ids=doc.get("scene_ids", []),
        result_id=doc.get("result_id"),
        result_summary=doc.get("result_summary"),
        preview_url=doc.get("preview_url"),
        satellite=doc.get("satellite"),
        status=doc.get("status", "completed"),
        is_multi_intent=doc.get("is_multi_intent", False),
        created_at=doc["created_at"],
    )


# ─── Create History Item ───────────────────────────────────────────────────────

def create_history_item(
    *,
    session_id: Optional[str] = None,
    record_type: str,
    analysis_type: str = "general",
    query: str = "",
    answer: Optional[str] = None,
    image_refs: Optional[list[str]] = None,
    status: str,
    # Classic fields
    analysis_id: Optional[str] = None,
    comparison_id: Optional[str] = None,
    error: Optional[str] = None,
    is_multi_intent: bool = False,
    detected_intents: Optional[list[str]] = None,
    intent_results: Optional[list[dict]] = None,
    limitations: Optional[list[str]] = None,
    detected_changes: Optional[list[dict]] = None,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    # Extended fields for satellite operations
    operation_type: Optional[str] = None,
    result_id: Optional[str] = None,
    scene_ids: Optional[list[str]] = None,
    bbox: Optional[list[float]] = None,
    date_range: Optional[dict] = None,
    satellite: Optional[str] = None,
    result_summary: Optional[dict] = None,
    preview_url: Optional[str] = None,
) -> str:
    """
    Insert a new history document into MongoDB.
    Returns the generated history_id.
    Non-fatal: logs a warning on failure but does NOT raise.

    Supports all operation types:
      analysis, comparison, spectral_analysis, change_detection,
      infrastructure_analysis, disaster_analysis, sar_optical_fusion,
      timeline_analysis, satellite_search.
    """
    history_id = _generate_history_id()
    now = _now()

    doc = {
        "history_id": history_id,
        "session_id": session_id,
        "record_type": record_type,
        "operation_type": operation_type or record_type,
        "analysis_type": analysis_type or "general",
        "analysis_id": analysis_id,
        "comparison_id": comparison_id,
        "result_id": result_id,
        "query": (query or "").strip(),
        "answer": answer,
        "image_refs": image_refs or [],
        "scene_ids": scene_ids or [],
        "bbox": bbox,
        "date_range": date_range,
        "satellite": satellite,
        "result_summary": result_summary,
        "preview_url": preview_url,
        "status": status,
        "error": error,
        "is_multi_intent": is_multi_intent,
        "detected_intents": detected_intents or [],
        "intent_results": intent_results or [],
        "limitations": limitations or [],
        "detected_changes": detected_changes or [],
        "label_a": label_a,
        "label_b": label_b,
        "provider": provider,
        "model": model,
        "created_at": now,
        "updated_at": now,
    }

    try:
        col = get_history_collection()
        col.insert_one({**doc})
        logger.info(
            "History item created: history_id=%s record_type=%s session_id=%s status=%s",
            history_id, record_type, session_id, status,
        )
    except Exception as exc:
        logger.warning(
            "Failed to create history item (non-fatal): history_id=%s error=[%s] %s",
            history_id, type(exc).__name__, exc,
        )

    return history_id


# ─── Get Single History Item ───────────────────────────────────────────────────

def get_history_item(history_id: str) -> HistoryItem:
    """
    Fetch a single history item by history_id.
    Raises HTTP 404 if not found, HTTP 500 on DB error.
    """
    if not history_id or not str(history_id).strip():
        raise HTTPException(status_code=400, detail="history_id must not be empty")

    try:
        col = get_history_collection()
        doc = col.find_one({"history_id": history_id}, {"_id": 0})
    except Exception as exc:
        logger.error("DB error fetching history item %s: [%s] %s", history_id, type(exc).__name__, exc)
        raise HTTPException(status_code=500, detail="Database error while fetching history item")

    if not doc:
        raise HTTPException(status_code=404, detail=f"History item '{history_id}' not found")

    return _doc_to_item(doc)


# ─── List History ──────────────────────────────────────────────────────────────

# All known operation/record types
_VALID_RECORD_TYPES = frozenset({
    "analysis",
    "comparison",
    "spectral_analysis",
    "change_detection",
    "infrastructure_analysis",
    "disaster_analysis",
    "sar_optical_fusion",
    "timeline_analysis",
    "satellite_search",
})


def list_history(
    session_id: Optional[str] = None,
    record_type: Optional[str] = None,
    operation_type: Optional[str] = None,
    analysis_type: Optional[str] = None,
    query_search: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    page: int = 1,
    limit: int = 20,
) -> HistoryListResponse:
    """
    Return a paginated list of history items, newest first.

    Filters:
      session_id     — scope to a specific session
      record_type    — any of the known operation types (open list)
      operation_type — alias for record_type filter
      analysis_type  — e.g. 'general', 'vegetation', 'change_detection'
      query_search   — text substring search on query field
      date_from/to   — created_at range filter
    """
    if page < 1:
        page = 1
    if limit < 1:
        limit = 1
    if limit > 100:
        limit = 100

    skip = (page - 1) * limit

    mongo_filter: dict = {}

    if session_id:
        mongo_filter["session_id"] = session_id

    # Accept both record_type and operation_type as filter aliases
    rt_filter = record_type or operation_type
    if rt_filter:
        # No hard whitelist — accept any reasonable value; unknown types simply return empty
        mongo_filter["record_type"] = rt_filter

    if analysis_type:
        mongo_filter["analysis_type"] = analysis_type

    if query_search and query_search.strip():
        # Safe case-insensitive regex search
        import re
        safe_pattern = re.escape(query_search.strip())
        mongo_filter["query"] = {"$regex": safe_pattern, "$options": "i"}

    if date_from or date_to:
        dt_filter: dict = {}
        if date_from:
            dt_filter["$gte"] = date_from
        if date_to:
            dt_filter["$lte"] = date_to
        mongo_filter["created_at"] = dt_filter

    try:
        col = get_history_collection()
        total = col.count_documents(mongo_filter)
        cursor = (
            col.find(mongo_filter, {"_id": 0})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        docs = list(cursor)
    except Exception as exc:
        logger.error("DB error listing history: [%s] %s", type(exc).__name__, exc)
        raise HTTPException(status_code=500, detail="Database error while listing history")

    items = [_doc_to_list_item(d) for d in docs]

    return HistoryListResponse(
        items=items,
        page=page,
        limit=limit,
        total=total,
        session_id=session_id,
    )


# ─── Delete Single History Item ────────────────────────────────────────────────

def delete_history_item(history_id: str) -> HistoryDeleteResponse:
    """
    Delete a single history record by history_id.
    Raises HTTP 404 if not found, HTTP 500 on DB error.
    """
    if not history_id or not str(history_id).strip():
        raise HTTPException(status_code=400, detail="history_id must not be empty")

    # Verify exists first
    try:
        col = get_history_collection()
        doc = col.find_one({"history_id": history_id}, {"_id": 0, "history_id": 1})
    except Exception as exc:
        logger.error("DB error looking up history item %s: %s", history_id, exc)
        raise HTTPException(status_code=500, detail="Database error")

    if not doc:
        raise HTTPException(status_code=404, detail=f"History item '{history_id}' not found")

    try:
        col.delete_one({"history_id": history_id})
        logger.info("History item deleted: history_id=%s", history_id)
    except Exception as exc:
        logger.error("Failed to delete history item %s: %s", history_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete history item")

    return HistoryDeleteResponse(
        history_id=history_id,
        message="History item deleted successfully",
    )


# ─── Clear History (session-scoped) ────────────────────────────────────────────

def clear_history(session_id: Optional[str] = None) -> HistoryClearResponse:
    """
    Delete all history records for a given session.

    Safety:
      - If session_id is provided, only that session's records are deleted.
      - If session_id is None, the operation is rejected to prevent accidental
        global deletion (no authentication layer exists).

    Returns the number of deleted records.
    """
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "session_id is required to clear history. "
                "Clearing history without a session scope is not allowed."
            ),
        )

    try:
        col = get_history_collection()
        result = col.delete_many({"session_id": session_id})
        deleted = result.deleted_count
        logger.info(
            "History cleared: session_id=%s deleted_count=%d",
            session_id, deleted,
        )
    except Exception as exc:
        logger.error("Failed to clear history for session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="Failed to clear history")

    return HistoryClearResponse(
        deleted_count=deleted,
        session_id=session_id,
        message=f"Cleared {deleted} history record(s) for session '{session_id}'",
    )

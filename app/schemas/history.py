"""
SatQuery AI — History Pydantic Schemas

Models for the generic history system.

Supported operation / record types:
  analysis              — single-image AI analysis (Phases 4/5/6)
  comparison            — two-image AI comparison (Phase 7)
  spectral_analysis     — spectral index analysis (NDVI/NDWI/NBR)
  change_detection      — pixel-level change detection
  infrastructure_analysis — building/road/construction detection
  disaster_analysis     — flood/fire/drought detection
  sar_optical_fusion    — SAR + Optical fusion
  timeline_analysis     — historical STAC timeline
  satellite_search      — satellite scene search result

Requests:  (none — history is written automatically by services)
Responses: HistoryItem, HistoryListResponse, HistoryDeleteResponse, HistoryClearResponse
Internal:  HistoryDocument
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── History Item (full detail) ───────────────────────────────────────────────

class HistoryItem(BaseModel):
    """Full history record returned by GET /api/history/{id}."""

    history_id: str = Field(description="Unique history record identifier (hist_<ULID>).")
    session_id: Optional[str] = Field(default=None, description="Parent session ID, if any.")

    # record_type: generic operation classifier
    record_type: str = Field(
        description=(
            "Type of record: 'analysis' | 'comparison' | 'spectral_analysis' | "
            "'change_detection' | 'infrastructure_analysis' | 'disaster_analysis' | "
            "'sar_optical_fusion' | 'timeline_analysis' | 'satellite_search'."
        ),
    )
    # operation_type mirrors record_type for forward compatibility
    operation_type: Optional[str] = Field(
        default=None,
        description="Semantic operation type, same as record_type.",
    )
    analysis_type: str = Field(
        default="general",
        description="Analysis type, e.g. 'general', 'vegetation', 'change_detection', 'multi_intent'.",
    )

    # Source IDs
    analysis_id: Optional[str] = Field(default=None, description="Linked analysis_id (analyses collection).")
    comparison_id: Optional[str] = Field(default=None, description="Linked comparison_id (comparisons collection).")
    result_id: Optional[str] = Field(default=None, description="result_id in the operation's own collection.")

    # Query / result
    query: str = Field(default="", description="User query text or operation description.")
    answer: Optional[str] = Field(default=None, description="AI-generated answer/explanation.")

    # Scene references (for satellite-based operations)
    scene_ids: list[str] = Field(
        default_factory=list,
        description="Sentinel scene IDs involved in this operation.",
    )

    # Image references (for upload-based operations)
    image_refs: list[str] = Field(
        default_factory=list,
        description="List of image_id(s) involved in this record.",
    )

    # Spatial / temporal context
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box [lon_min, lat_min, lon_max, lat_max] if applicable.",
    )
    date_range: Optional[dict] = Field(
        default=None,
        description="Date range dict with 'start' and 'end' keys if applicable.",
    )
    satellite: Optional[str] = Field(
        default=None,
        description="Satellite platform, e.g. 'sentinel-2', 'sentinel-1'.",
    )

    # Result summary (compact — no large arrays)
    result_summary: Optional[dict] = Field(
        default=None,
        description="Compact statistics/metrics summary of the result.",
    )
    preview_url: Optional[str] = Field(
        default=None,
        description="URL to preview PNG, if available.",
    )

    # Status
    status: str = Field(description="'completed' | 'failed' | 'incompatible'.")
    error: Optional[str] = Field(default=None, description="Error message when status='failed'.")

    # Phase 6 multi-intent fields
    is_multi_intent: bool = Field(default=False)
    detected_intents: list[str] = Field(default_factory=list)
    intent_results: list[dict] = Field(default_factory=list)

    # Comparison-specific
    limitations: list[str] = Field(default_factory=list)
    detected_changes: list[dict] = Field(default_factory=list)
    label_a: Optional[str] = None
    label_b: Optional[str] = None

    # Provider info
    provider: Optional[str] = None
    model: Optional[str] = None

    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None


# ─── History List Item (compact) ─────────────────────────────────────────────

class HistoryListItem(BaseModel):
    """Compact history record for list responses."""

    history_id: str
    session_id: Optional[str] = None
    record_type: str
    operation_type: Optional[str] = None
    analysis_type: str = "general"
    query: str = ""
    answer_preview: Optional[str] = Field(
        default=None,
        description="First 200 characters of the answer for list display.",
    )
    image_refs: list[str] = Field(default_factory=list)
    scene_ids: list[str] = Field(default_factory=list)
    result_id: Optional[str] = None
    result_summary: Optional[dict] = None
    preview_url: Optional[str] = None
    satellite: Optional[str] = None
    status: str
    is_multi_intent: bool = False
    created_at: datetime


# ─── List Response ────────────────────────────────────────────────────────────

class HistoryListResponse(BaseModel):
    """Paginated list of history items."""

    items: list[HistoryListItem]
    page: int
    limit: int
    total: int
    session_id: Optional[str] = Field(
        default=None,
        description="If filtered by session, the session_id used.",
    )


# ─── Delete / Clear Responses ────────────────────────────────────────────────

class HistoryDeleteResponse(BaseModel):
    """Response after deleting a single history record."""
    history_id: str
    message: str


class HistoryClearResponse(BaseModel):
    """Response after clearing all history for a session."""
    deleted_count: int
    session_id: Optional[str] = None
    message: str


# ─── Internal MongoDB Document ────────────────────────────────────────────────

class HistoryDocument(BaseModel):
    """
    Represents a history document as stored in MongoDB.
    Used internally — not directly serialised to API responses.
    Never includes API keys, credentials or secrets.
    """

    history_id: str
    session_id: Optional[str] = None
    record_type: str
    operation_type: Optional[str] = None
    analysis_type: str = "general"
    analysis_id: Optional[str] = None
    comparison_id: Optional[str] = None
    result_id: Optional[str] = None
    query: str = ""
    answer: Optional[str] = None
    image_refs: list[str] = Field(default_factory=list)
    scene_ids: list[str] = Field(default_factory=list)
    bbox: Optional[list[float]] = None
    date_range: Optional[dict] = None
    satellite: Optional[str] = None
    result_summary: Optional[dict] = None
    preview_url: Optional[str] = None
    status: str
    error: Optional[str] = None

    # Phase 6 fields
    is_multi_intent: bool = False
    detected_intents: list[str] = Field(default_factory=list)
    intent_results: list[dict] = Field(default_factory=list)

    # Comparison-specific
    limitations: list[str] = Field(default_factory=list)
    detected_changes: list[dict] = Field(default_factory=list)
    label_a: Optional[str] = None
    label_b: Optional[str] = None

    # Provider info (model name only — never the key)
    provider: Optional[str] = None
    model: Optional[str] = None

    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None

"""
SatQuery AI — Analysis Pydantic Schemas (Phase 4 + Phase 6)

Models:
  Requests:  AnalysisRequest
  Responses: AnalysisResponse, AnalysisStatusResponse
             IntentResult (Phase 6), MultiIntentSection (Phase 6)
  Internal:  AnalysisDocument
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Request ──────────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Body for POST /api/analysis."""

    image_id: str = Field(
        ...,
        description="The image_id to analyze (must exist in MongoDB).",
    )
    query: str = Field(
        default="",
        max_length=5000,
        description="Natural-language question about the image. Leave blank for general analysis.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional session_id. When provided the image must be linked to the session "
            "and the result will be stored as a session question/answer."
        ),
    )
    analysis_type: str = Field(
        default="general",
        description="Analysis type hint ('general' for Phase 4). Overridden by multi-intent detection.",
    )
    # Phase 6: opt-out flag — set to False to skip multi-intent detection
    enable_multi_intent: bool = Field(
        default=True,
        description=(
            "Phase 6: When True (default), the query is inspected for multiple intents "
            "and each intent is analyzed separately. Set to False to force single-intent behavior."
        ),
    )


# ─── Finding (structured sub-result) ─────────────────────────────────────────

class AnalysisFinding(BaseModel):
    """
    A structured finding extracted from the AI response.
    Phase 4: findings list is always empty.
    Phase 5+ will populate this from structured AI output.
    """
    label: str
    description: str
    confidence: Optional[float] = None


# ─── Phase 6 — Per-intent result ─────────────────────────────────────────────

class IntentResult(BaseModel):
    """
    Result for a single detected intent within a multi-intent analysis.

    Phase 6: returned in AnalysisResponse.intent_results when the query
    triggered multi-intent analysis.
    """
    intent: str = Field(description="Intent name, e.g. 'vegetation', 'water', 'land_cover'.")
    display_label: str = Field(description="Human-readable intent label.")
    analysis_type: str = Field(description="Analysis type sent to the AI pipeline.")
    status: str = Field(description="'completed' | 'failed'.")
    answer: Optional[str] = Field(default=None, description="AI answer for this intent.")
    error_message: Optional[str] = Field(default=None, description="Error detail if status='failed'.")


# ─── Response ─────────────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    """Full analysis result returned by POST /api/analysis."""

    analysis_id: str
    image_id: str
    session_id: Optional[str]
    query: str
    analysis_type: str
    status: str = Field(
        description="'pending' | 'processing' | 'completed' | 'failed'",
    )
    answer: Optional[str] = Field(
        default=None,
        description="The AI's text response. Null when status is not 'completed'.",
    )
    findings: list[AnalysisFinding] = Field(
        default_factory=list,
        description="Structured findings (Phase 5+). Empty in Phase 4.",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Null unless the provider returns calibrated confidence.",
    )
    confidence_available: bool = Field(
        default=False,
        description="False for Gemini (does not return calibrated confidence).",
    )
    provider: str
    model: str
    created_at: datetime
    completed_at: Optional[datetime] = None

    # ── Phase 6 additions (None/empty for single-intent responses) ──────────
    is_multi_intent: bool = Field(
        default=False,
        description="True when the query triggered multi-intent analysis.",
    )
    detected_intents: list[str] = Field(
        default_factory=list,
        description="Phase 6: List of detected intent names, e.g. ['vegetation', 'water'].",
    )
    intent_results: list[IntentResult] = Field(
        default_factory=list,
        description="Phase 6: Per-intent analysis results. Empty for single-intent responses.",
    )


class AnalysisStatusResponse(BaseModel):
    """Compact status check response (for polling, Phase 5+)."""
    analysis_id: str
    status: str
    answer: Optional[str] = None


# ─── Internal MongoDB Document ────────────────────────────────────────────────

class AnalysisDocument(BaseModel):
    """
    Represents an analysis document stored in MongoDB.
    Used internally — not serialised directly to API responses.
    """
    analysis_id: str
    session_id: Optional[str]
    image_id: str
    query: str
    analysis_type: str
    status: str
    answer: Optional[str] = None
    findings: list[dict] = Field(default_factory=list)
    confidence: Optional[float] = None
    confidence_available: bool = False
    provider: str
    model: str
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    # Phase 6 extra fields stored in MongoDB
    is_multi_intent: bool = False
    detected_intents: list[str] = Field(default_factory=list)
    intent_results: list[dict] = Field(default_factory=list)

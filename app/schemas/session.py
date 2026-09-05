"""
SatQuery AI — Session Pydantic Schemas (Phase 3)

Models:
  Requests:  SessionCreateRequest, SessionQuestionCreateRequest,
             SessionQuestionUpdateRequest
  Responses: SessionResponse, SessionListItem, SessionListResponse,
             SessionImagesResponse, SessionQuestionResponse,
             SessionQuestionsResponse, SessionLinkImageResponse,
             SessionDeleteResponse
  Internal:  QuestionDocument, SessionDocument
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Question status / Session status literals ────────────────────────────────

QUESTION_STATUSES = {"pending", "processing", "completed", "failed"}
SESSION_STATUSES  = {"active", "completed", "deleted"}


# ─── Internal sub-document ────────────────────────────────────────────────────

class QuestionDocument(BaseModel):
    """A question (with optional answer) stored inside a session document."""

    question_id: str
    question: str
    image_id: str
    analysis_type: str = "general"
    answer: Optional[str] = None
    status: str = "pending"
    created_at: datetime
    completed_at: Optional[datetime] = None


# ─── Internal Session Document (MongoDB) ─────────────────────────────────────

class SessionDocument(BaseModel):
    """
    Represents a session document as stored in MongoDB.
    Used internally — not serialised directly to API responses.
    """
    session_id: str
    title: str
    mode: str = "single"
    image_ids: list[str] = Field(default_factory=list)
    questions: list[QuestionDocument] = Field(default_factory=list)
    status: str = "active"
    created_at: datetime
    updated_at: datetime


# ─── Request Models ───────────────────────────────────────────────────────────

class SessionCreateRequest(BaseModel):
    """Body for POST /api/sessions."""

    title: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional session title. Defaults to 'New Analysis'.",
    )
    mode: Optional[str] = Field(
        default="single",
        description="Analysis mode: 'single' or 'comparison'.",
    )


class SessionQuestionCreateRequest(BaseModel):
    """Body for POST /api/sessions/{session_id}/questions."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The question to ask about the image.",
    )
    image_id: str = Field(
        ...,
        description="The image_id this question relates to (must be linked to the session).",
    )
    analysis_type: Optional[str] = Field(
        default="general",
        description="Analysis type hint (e.g. 'general', 'change_detection', 'building_detection').",
    )


class SessionQuestionUpdateRequest(BaseModel):
    """Body for PATCH /api/sessions/{session_id}/questions/{question_id}."""

    answer: Optional[str] = Field(
        default=None,
        max_length=50000,
        description="The answer / analysis result text.",
    )
    status: Optional[str] = Field(
        default=None,
        description="New status: 'pending' | 'processing' | 'completed' | 'failed'.",
    )


# ─── Response Models ──────────────────────────────────────────────────────────

class SessionQuestionResponse(BaseModel):
    """A single question with its answer/status, returned in API responses."""

    question_id: str
    question: str
    image_id: str
    analysis_type: str
    answer: Optional[str]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]


class SessionResponse(BaseModel):
    """Full session response (create, get)."""

    session_id: str
    title: str
    mode: str
    image_ids: list[str]
    questions: list[SessionQuestionResponse]
    status: str
    created_at: datetime
    updated_at: datetime


class SessionListItem(BaseModel):
    """Compact session summary for the history list."""

    session_id: str
    title: str
    mode: str
    image_count: int
    question_count: int
    status: str
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    """Paginated list of sessions."""

    sessions: list[SessionListItem]
    page: int
    limit: int
    total: int


class ImageSummary(BaseModel):
    """Compact image metadata for session image list."""

    image_id: str
    filename: str
    mime_type: str
    width: int
    height: int
    is_geospatial: bool
    created_at: datetime


class SessionImagesResponse(BaseModel):
    """Response for GET /api/sessions/{session_id}/images."""

    session_id: str
    images: list[ImageSummary]


class SessionLinkImageResponse(BaseModel):
    """Response for POST /api/sessions/{session_id}/images/{image_id}."""

    session_id: str
    image_id: str
    message: str


class SessionQuestionsResponse(BaseModel):
    """Response for GET /api/sessions/{session_id}/questions."""

    session_id: str
    questions: list[SessionQuestionResponse]


class SessionDeleteResponse(BaseModel):
    """Response for DELETE /api/sessions/{session_id}."""

    session_id: str
    message: str = "Session deleted successfully"


# ─── Phase 5 — Follow-up / Ask schemas ───────────────────────────────────────

class SessionAskRequest(BaseModel):
    """
    Body for POST /api/sessions/{session_id}/ask

    The backend resolves image_id automatically from the session.
    No need to send image_id from the frontend.
    """
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The follow-up question about the session's satellite image.",
    )
    analysis_type: Optional[str] = Field(
        default="general",
        description="Analysis type hint (e.g. 'general', 'vegetation', 'building_detection').",
    )
    # Phase 6: opt-out flag — matches AnalysisRequest.enable_multi_intent
    enable_multi_intent: bool = Field(
        default=True,
        description=(
            "Phase 6: When True (default), the question is inspected for multiple intents. "
            "Set to False to force single-intent behavior."
        ),
    )


class SessionAskResponse(BaseModel):
    """
    Response for POST /api/sessions/{session_id}/ask

    Returns the full analysis result plus conversation position.
    """
    session_id: str
    question_number: int = Field(
        description="1-based index of this question within the session (Q1, Q2, Q3…).",
    )
    analysis_id: str
    image_id: str
    question: str
    answer: Optional[str]
    analysis_type: str
    status: str
    provider: str
    model: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    # Phase 6 fields — mirror AnalysisResponse
    is_multi_intent: bool = Field(
        default=False,
        description="True when the question triggered multi-intent analysis.",
    )
    detected_intents: list[str] = Field(
        default_factory=list,
        description="Phase 6: detected intent names, e.g. ['vegetation', 'water'].",
    )
    intent_results: list[Any] = Field(
        default_factory=list,
        description="Phase 6: per-intent analysis results (same structure as AnalysisResponse).",
    )


"""
SatQuery AI — Sessions Router (Phase 3)

Endpoints:
  POST   /api/sessions                                        — Create a new session
  GET    /api/sessions                                        — List sessions (paginated)
  GET    /api/sessions/{session_id}                          — Get full session
  DELETE /api/sessions/{session_id}                          — Delete session

  POST   /api/sessions/{session_id}/images/{image_id}        — Link image to session
  GET    /api/sessions/{session_id}/images                   — List session images

  POST   /api/sessions/{session_id}/questions                — Add a question
  GET    /api/sessions/{session_id}/questions                — List questions
  PATCH  /api/sessions/{session_id}/questions/{question_id}  — Update question answer/status

Route handlers are thin — all logic lives in session_service.py.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from app.schemas.session import (
    SessionAskRequest,
    SessionAskResponse,
    SessionCreateRequest,
    SessionDeleteResponse,
    SessionImagesResponse,
    SessionLinkImageResponse,
    SessionListResponse,
    SessionQuestionCreateRequest,
    SessionQuestionResponse,
    SessionQuestionsResponse,
    SessionQuestionUpdateRequest,
    SessionResponse,
)
from app.services import session_service
from app.services import analysis_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


# ─── POST /api/sessions ───────────────────────────────────────────────────────

@router.post(
    "",
    response_model=SessionResponse,
    status_code=201,
    summary="Create a new analysis session",
    description=(
        "Create a new analysis session.\n\n"
        "`title` is optional — defaults to 'New Analysis'.\n\n"
        "`mode` is optional — 'single' (default) or 'comparison'.\n\n"
        "Returns the new session with an empty `image_ids` and `questions` list."
    ),
    response_description="Newly created session",
)
def create_session(body: SessionCreateRequest) -> SessionResponse:
    """Create a new analysis session."""
    return session_service.create_session(
        title=body.title,
        mode=body.mode or "single",
    )


# ─── GET /api/sessions ────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=SessionListResponse,
    status_code=200,
    summary="List all sessions",
    description=(
        "Return a paginated list of sessions sorted by `updated_at` DESC.\n\n"
        "Deleted sessions are excluded from the list."
    ),
    response_description="Paginated list of session summaries",
)
def list_sessions(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> SessionListResponse:
    """Return paginated sessions, newest first."""
    return session_service.list_sessions(page=page, limit=limit)


# ─── GET /api/sessions/{session_id} ──────────────────────────────────────────

@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    status_code=200,
    summary="Get a session by ID",
    description=(
        "Return full session details including all linked `image_ids` and `questions`.\n\n"
        "Returns HTTP 404 if the session does not exist."
    ),
    response_description="Full session document",
)
def get_session(session_id: str) -> SessionResponse:
    """Fetch a session by its session_id."""
    return session_service.get_session(session_id)


# ─── DELETE /api/sessions/{session_id} ───────────────────────────────────────

@router.delete(
    "/{session_id}",
    response_model=SessionDeleteResponse,
    status_code=200,
    summary="Delete a session",
    description=(
        "Delete a session document from MongoDB.\n\n"
        "**Does NOT** delete image files, image documents, or other sessions.\n\n"
        "Returns HTTP 404 if the session does not exist."
    ),
    response_description="Deletion confirmation",
)
def delete_session(session_id: str) -> SessionDeleteResponse:
    """Delete a session by its session_id (images are NOT deleted)."""
    result = session_service.delete_session(session_id)
    return SessionDeleteResponse(**result)


# ─── POST /api/sessions/{session_id}/images/{image_id} ───────────────────────

@router.post(
    "/{session_id}/images/{image_id}",
    response_model=SessionLinkImageResponse,
    status_code=200,
    summary="Link an image to a session",
    description=(
        "Link an existing uploaded image to an existing session.\n\n"
        "**Idempotent**: calling this multiple times with the same image_id is safe.\n\n"
        "Only the `image_id` reference is stored — no file is duplicated.\n\n"
        "Returns HTTP 404 if session or image does not exist."
    ),
    response_description="Link confirmation",
)
def link_image(session_id: str, image_id: str) -> SessionLinkImageResponse:
    """Link an existing image to a session."""
    result = session_service.link_image_to_session(session_id, image_id)
    return SessionLinkImageResponse(**result)


# ─── GET /api/sessions/{session_id}/images ────────────────────────────────────

@router.get(
    "/{session_id}/images",
    response_model=SessionImagesResponse,
    status_code=200,
    summary="Get images linked to a session",
    description=(
        "Return metadata for all images currently linked to the session.\n\n"
        "Broken references (images that have since been deleted) are silently skipped.\n\n"
        "Returns HTTP 404 if the session does not exist."
    ),
    response_description="List of image metadata summaries",
)
def get_session_images(session_id: str) -> SessionImagesResponse:
    """Return metadata for all images linked to this session."""
    return session_service.get_session_images(session_id)


# ─── POST /api/sessions/{session_id}/questions ───────────────────────────────

@router.post(
    "/{session_id}/questions",
    response_model=SessionQuestionResponse,
    status_code=201,
    summary="Add a question to a session",
    description=(
        "Append a new question to the session.\n\n"
        "The `image_id` must be already linked to the session.\n\n"
        "`analysis_type` is optional (defaults to 'general').\n\n"
        "The question is created with `status='pending'` and `answer=null`."
    ),
    response_description="Newly created question",
)
def add_question(
    session_id: str,
    body: SessionQuestionCreateRequest,
) -> SessionQuestionResponse:
    """Add a question to a session."""
    return session_service.add_question(
        session_id=session_id,
        question=body.question,
        image_id=body.image_id,
        analysis_type=body.analysis_type or "general",
    )


# ─── GET /api/sessions/{session_id}/questions ────────────────────────────────

@router.get(
    "/{session_id}/questions",
    response_model=SessionQuestionsResponse,
    status_code=200,
    summary="Get all questions in a session",
    description=(
        "Return all questions for the session in chronological order.\n\n"
        "Returns HTTP 404 if the session does not exist."
    ),
    response_description="Ordered list of questions with answers/statuses",
)
def get_session_questions(session_id: str) -> SessionQuestionsResponse:
    """Return all questions in the session."""
    return session_service.get_session_questions(session_id)


# ─── PATCH /api/sessions/{session_id}/questions/{question_id} ────────────────

@router.patch(
    "/{session_id}/questions/{question_id}",
    response_model=SessionQuestionResponse,
    status_code=200,
    summary="Update a question's answer or status",
    description=(
        "Update the `answer` and/or `status` of an existing question.\n\n"
        "When `status` is set to `'completed'`, `completed_at` is automatically set to now.\n\n"
        "When `status` is set to `'failed'`, `completed_at` is cleared.\n\n"
        "Also updates `session.updated_at`.\n\n"
        "Returns HTTP 404 if session or question does not exist."
    ),
    response_description="Updated question",
)
def update_question(
    session_id: str,
    question_id: str,
    body: SessionQuestionUpdateRequest,
) -> SessionQuestionResponse:
    """Patch a question's answer and/or status."""
    return session_service.update_question(
        session_id=session_id,
        question_id=question_id,
        answer=body.answer,
        status=body.status,
    )


# ─── POST /api/sessions/{session_id}/ask (Phase 5) ─────────────────────────────

@router.post(
    "/{session_id}/ask",
    response_model=SessionAskResponse,
    status_code=200,
    summary="Ask a follow-up question in an existing session",
    description=(
        "**Phase 5 — Conversation Follow-up**\n\n"
        "Send a follow-up question about the satellite image already associated "
        "with this session. The backend will:\n\n"
        "1. Validate the session exists.\n"
        "2. Auto-resolve the image (no need to send `image_id`).\n"
        "3. Retrieve relevant previous Q\u0026A context (last 5 turns).\n"
        "4. Send the image + context + new question to Gemini.\n"
        "5. Persist the answer to MongoDB.\n"
        "6. Return the answer.\n\n"
        "**The image is never uploaded again.** Only the original `session_id` is needed.\n\n"
        "**Returns:** Full analysis result including AI answer, provider, model, and status."
    ),
    response_description="AI answer to the follow-up question",
    responses={
        400: {"description": "Session has no linked image or invalid request"},
        404: {"description": "Session not found"},
        503: {"description": "AI provider not configured"},
        500: {"description": "Internal server error"},
    },
)
def ask_followup(
    session_id: str,
    body: SessionAskRequest,
) -> SessionAskResponse:
    """
    Phase 5: Ask a follow-up question within an existing session.
    The image and previous conversation context are retrieved automatically.
    """
    result, question_number = analysis_service.run_analysis_followup(
        session_id=session_id,
        question=body.question,
        analysis_type=body.analysis_type or "general",
        enable_multi_intent=body.enable_multi_intent,
    )
    return SessionAskResponse(
        session_id=session_id,
        question_number=question_number,
        analysis_id=result.analysis_id,
        image_id=result.image_id,
        question=result.query,
        answer=result.answer,
        analysis_type=result.analysis_type,
        status=result.status,
        provider=result.provider,
        model=result.model,
        created_at=result.created_at,
        completed_at=result.completed_at,
        # Phase 6
        is_multi_intent=result.is_multi_intent,
        detected_intents=result.detected_intents,
        intent_results=[r.model_dump() for r in result.intent_results],
    )

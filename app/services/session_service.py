"""
SatQuery AI — Session Service (Phase 3)

Orchestrates all session operations:

  sessions.py (router)
      ↓
  session_service.py   ← this file
      ↓
  MongoDB  (sessions collection)
      ↓
  MongoDB  (images collection — for cross-reference lookups)

Session IDs:   ses_<ULID>
Question IDs:  q_<ULID>

Session deletion NEVER deletes image files or image documents.
Image deletion removes the image_id reference from all sessions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException

from app.db.mongodb import get_images_collection, get_sessions_collection
from app.schemas.session import (
    ImageSummary,
    QuestionDocument,
    SessionDocument,
    SessionImagesResponse,
    SessionListItem,
    SessionListResponse,
    SessionQuestionResponse,
    SessionQuestionsResponse,
    SessionResponse,
)

logger = logging.getLogger(__name__)


# ─── ID Generation ────────────────────────────────────────────────────────────

def _generate_session_id() -> str:
    """Generate ses_<ULID> — mirrors generate_image_id() in image_utils."""
    try:
        from ulid import ULID
        return f"ses_{ULID()}"
    except ImportError:
        import uuid
        logger.warning("python-ulid not available; falling back to UUID4 for session_id")
        return f"ses_{uuid.uuid4().hex}"


def _generate_question_id() -> str:
    """Generate q_<ULID>."""
    try:
        from ulid import ULID
        return f"q_{ULID()}"
    except ImportError:
        import uuid
        return f"q_{uuid.uuid4().hex}"


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _get_session_doc(session_id: str) -> dict:
    """
    Fetch raw session document from MongoDB.
    Raises HTTP 404 if not found, HTTP 500 on DB error.
    """
    try:
        col = get_sessions_collection()
        doc = col.find_one({"session_id": session_id}, {"_id": 0})
    except Exception as exc:
        logger.error("MongoDB query failed (sessions): %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")
    return doc


def _doc_to_session_response(doc: dict) -> SessionResponse:
    """Parse a raw MongoDB session document into a SessionResponse."""
    questions = [
        SessionQuestionResponse(
            question_id=q["question_id"],
            question=q["question"],
            image_id=q["image_id"],
            analysis_type=q.get("analysis_type", "general"),
            answer=q.get("answer"),
            status=q.get("status", "pending"),
            created_at=_parse_dt(q.get("created_at")),
            completed_at=_parse_dt(q.get("completed_at")),
        )
        for q in doc.get("questions", [])
    ]
    return SessionResponse(
        session_id=doc["session_id"],
        title=doc.get("title", "New Analysis"),
        mode=doc.get("mode", "single"),
        image_ids=doc.get("image_ids", []),
        questions=questions,
        status=doc.get("status", "active"),
        created_at=_parse_dt(doc.get("created_at")),
        updated_at=_parse_dt(doc.get("updated_at")),
    )


def _parse_dt(value: Any) -> datetime:
    """Safely parse a datetime value from MongoDB (may be datetime or ISO string)."""
    if value is None:
        return _now()
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return _now()


# ─── Create Session ───────────────────────────────────────────────────────────

def create_session(title: Optional[str], mode: str = "single") -> SessionResponse:
    """
    Create a new analysis session and insert it into MongoDB.
    Returns a SessionResponse.
    """
    session_id = _generate_session_id()
    now = _now()
    resolved_title = (title or "").strip() or "New Analysis"

    doc = {
        "session_id": session_id,
        "title": resolved_title,
        "mode": mode,
        "image_ids": [],
        "questions": [],
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }

    try:
        col = get_sessions_collection()
        col.insert_one(doc)
        logger.info("Session created: session_id=%s title=%r mode=%s", session_id, resolved_title, mode)
    except Exception as exc:
        logger.error("Failed to insert session: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create session")

    return _doc_to_session_response({**doc})


# ─── Get Session ──────────────────────────────────────────────────────────────

def get_session(session_id: str) -> SessionResponse:
    """Fetch and return a full session by its session_id."""
    doc = _get_session_doc(session_id)
    return _doc_to_session_response(doc)


# ─── List Sessions ────────────────────────────────────────────────────────────

def list_sessions(page: int = 1, limit: int = 20) -> SessionListResponse:
    """
    Return a paginated list of sessions, sorted by updated_at DESC.
    Only non-deleted sessions are included.
    """
    skip = (page - 1) * limit
    try:
        col = get_sessions_collection()
        query = {"status": {"$ne": "deleted"}}
        total = col.count_documents(query)
        cursor = (
            col.find(query, {"_id": 0})
            .sort("updated_at", -1)
            .skip(skip)
            .limit(limit)
        )
        docs = list(cursor)
    except Exception as exc:
        logger.error("Failed to list sessions: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    items = [
        SessionListItem(
            session_id=d["session_id"],
            title=d.get("title", "New Analysis"),
            mode=d.get("mode", "single"),
            image_count=len(d.get("image_ids", [])),
            question_count=len(d.get("questions", [])),
            status=d.get("status", "active"),
            created_at=_parse_dt(d.get("created_at")),
            updated_at=_parse_dt(d.get("updated_at")),
        )
        for d in docs
    ]

    return SessionListResponse(sessions=items, page=page, limit=limit, total=total)


# ─── Link Image to Session ────────────────────────────────────────────────────

def link_image_to_session(session_id: str, image_id: str) -> dict:
    """
    Link an existing image to an existing session (idempotent).
    Validates both session and image exist.
    Returns a dict with session_id, image_id, and message.
    """
    # Validate session
    doc = _get_session_doc(session_id)  # raises 404 if not found

    # Validate image
    try:
        img_col = get_images_collection()
        img_doc = img_col.find_one({"image_id": image_id}, {"_id": 0, "image_id": 1})
    except Exception as exc:
        logger.error("MongoDB query failed (images): %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    if not img_doc:
        raise HTTPException(status_code=404, detail="Image not found")

    # Check for duplicate
    if image_id in doc.get("image_ids", []):
        logger.info("Image already linked: session_id=%s image_id=%s", session_id, image_id)
        return {
            "session_id": session_id,
            "image_id": image_id,
            "message": "Image already linked to session",
        }

    # Append image_id and update updated_at
    try:
        col = get_sessions_collection()
        col.update_one(
            {"session_id": session_id},
            {
                "$push": {"image_ids": image_id},
                "$set": {"updated_at": _now()},
            },
        )
        logger.info("Image linked: session_id=%s image_id=%s", session_id, image_id)
    except Exception as exc:
        logger.error("Failed to link image to session: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to link image")

    return {
        "session_id": session_id,
        "image_id": image_id,
        "message": "Image linked to session successfully",
    }


# ─── Get Session Images ───────────────────────────────────────────────────────

def get_session_images(session_id: str) -> SessionImagesResponse:
    """
    Return metadata for all images linked to a session.
    Handles broken references (deleted images) safely by skipping them.
    """
    doc = _get_session_doc(session_id)
    image_ids: list[str] = doc.get("image_ids", [])

    images: list[ImageSummary] = []
    if image_ids:
        try:
            img_col = get_images_collection()
            cursor = img_col.find({"image_id": {"$in": image_ids}}, {"_id": 0})
            for img in cursor:
                try:
                    images.append(
                        ImageSummary(
                            image_id=img["image_id"],
                            filename=img.get("filename", "unknown"),
                            mime_type=img.get("mime_type", ""),
                            width=img.get("width", 0),
                            height=img.get("height", 0),
                            is_geospatial=img.get("is_geospatial", False),
                            created_at=_parse_dt(img.get("created_at")),
                        )
                    )
                except Exception as exc:
                    logger.warning("Skipping malformed image doc %s: %s", img.get("image_id"), exc)
        except Exception as exc:
            logger.error("Failed to fetch session images: %s", exc)
            raise HTTPException(status_code=500, detail="Database error")

    return SessionImagesResponse(session_id=session_id, images=images)


# ─── Add Question ─────────────────────────────────────────────────────────────

def add_question(
    session_id: str,
    question: str,
    image_id: str,
    analysis_type: str = "general",
) -> SessionQuestionResponse:
    """
    Append a new question to a session's questions array.
    Validates that the image_id is actually linked to the session.
    Returns the created question.
    """
    doc = _get_session_doc(session_id)

    # Validate image belongs to session
    if image_id not in doc.get("image_ids", []):
        raise HTTPException(
            status_code=400,
            detail=f"Image '{image_id}' is not linked to session '{session_id}'",
        )

    question_id = _generate_question_id()
    now = _now()

    q_doc: dict = {
        "question_id": question_id,
        "question": question,
        "image_id": image_id,
        "analysis_type": analysis_type or "general",
        "answer": None,
        "status": "pending",
        "created_at": now,
        "completed_at": None,
    }

    try:
        col = get_sessions_collection()
        col.update_one(
            {"session_id": session_id},
            {
                "$push": {"questions": q_doc},
                "$set": {"updated_at": now},
            },
        )
        logger.info(
            "Question added: session_id=%s question_id=%s image_id=%s",
            session_id, question_id, image_id,
        )
    except Exception as exc:
        logger.error("Failed to add question: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to add question")

    return SessionQuestionResponse(
        question_id=question_id,
        question=question,
        image_id=image_id,
        analysis_type=analysis_type or "general",
        answer=None,
        status="pending",
        created_at=now,
        completed_at=None,
    )


# ─── Update Question Answer ───────────────────────────────────────────────────

def update_question(
    session_id: str,
    question_id: str,
    answer: Optional[str],
    status: Optional[str],
) -> SessionQuestionResponse:
    """
    Update an existing question's answer and/or status.
    Sets completed_at when status becomes 'completed'.
    Always updates session.updated_at.
    """
    doc = _get_session_doc(session_id)

    # Find the question in the embedded array
    questions: list[dict] = doc.get("questions", [])
    q_index = next(
        (i for i, q in enumerate(questions) if q["question_id"] == question_id),
        None,
    )
    if q_index is None:
        raise HTTPException(status_code=404, detail="Question not found")

    q = questions[q_index]
    now = _now()

    # Build update fields for the specific array element
    set_fields: dict = {"updated_at": now}

    if answer is not None:
        set_fields[f"questions.{q_index}.answer"] = answer

    if status is not None:
        if status not in {"pending", "processing", "completed", "failed"}:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Must be: pending, processing, completed, failed",
            )
        set_fields[f"questions.{q_index}.status"] = status
        if status == "completed":
            set_fields[f"questions.{q_index}.completed_at"] = now
        elif status == "failed":
            set_fields[f"questions.{q_index}.completed_at"] = None

    try:
        col = get_sessions_collection()
        col.update_one({"session_id": session_id}, {"$set": set_fields})
        logger.info(
            "Question updated: session_id=%s question_id=%s status=%s",
            session_id, question_id, status,
        )
    except Exception as exc:
        logger.error("Failed to update question: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update question")

    # Reload to return fresh data
    updated_doc = _get_session_doc(session_id)
    updated_questions = updated_doc.get("questions", [])
    uq = next(
        (x for x in updated_questions if x["question_id"] == question_id), q
    )

    return SessionQuestionResponse(
        question_id=uq["question_id"],
        question=uq["question"],
        image_id=uq["image_id"],
        analysis_type=uq.get("analysis_type", "general"),
        answer=uq.get("answer"),
        status=uq.get("status", "pending"),
        created_at=_parse_dt(uq.get("created_at")),
        completed_at=_parse_dt(uq.get("completed_at")) if uq.get("completed_at") else None,
    )


# ─── Get Session Questions ────────────────────────────────────────────────────

def get_session_questions(session_id: str) -> SessionQuestionsResponse:
    """Return all questions in the session in chronological order."""
    doc = _get_session_doc(session_id)

    questions = [
        SessionQuestionResponse(
            question_id=q["question_id"],
            question=q["question"],
            image_id=q["image_id"],
            analysis_type=q.get("analysis_type", "general"),
            answer=q.get("answer"),
            status=q.get("status", "pending"),
            created_at=_parse_dt(q.get("created_at")),
            completed_at=_parse_dt(q.get("completed_at")) if q.get("completed_at") else None,
        )
        for q in doc.get("questions", [])
    ]

    return SessionQuestionsResponse(session_id=session_id, questions=questions)


# ─── Delete Session ───────────────────────────────────────────────────────────

def delete_session(session_id: str) -> dict:
    """
    Delete a session document from MongoDB.
    DOES NOT delete image files or image documents.
    DOES NOT affect other sessions.
    Also removes history records linked to this session (Phase 8).
    """
    _get_session_doc(session_id)  # raises 404 if not found

    try:
        col = get_sessions_collection()
        col.delete_one({"session_id": session_id})
        logger.info("Session deleted: session_id=%s", session_id)
    except Exception as exc:
        logger.error("Failed to delete session: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete session")

    # Phase 8: cascade-delete history records for this session (non-fatal)
    try:
        from app.services import history_service as _hs
        cleared = _hs.clear_history(session_id=session_id)
        logger.info(
            "Session history cleared on session delete: session_id=%s deleted_count=%d",
            session_id, cleared.deleted_count,
        )
    except Exception as exc:
        logger.warning(
            "Could not clear history for deleted session %s: %s", session_id, exc
        )

    return {"session_id": session_id, "message": "Session deleted successfully"}


# ─── Remove image_id from all sessions (called on image deletion) ─────────────

def remove_image_from_all_sessions(image_id: str) -> None:
    """
    Remove an image_id reference from every session that contains it.
    Called by image_service.delete_image_by_id() as a cleanup step.
    Non-fatal: logs a warning if it fails but does not raise.
    """
    try:
        col = get_sessions_collection()
        result = col.update_many(
            {"image_ids": image_id},
            {
                "$pull": {"image_ids": image_id},
                "$set": {"updated_at": _now()},
            },
        )
        if result.modified_count:
            logger.info(
                "Removed image_id=%s from %d session(s)", image_id, result.modified_count
            )
    except Exception as exc:
        logger.warning(
            "Could not remove image_id=%s from sessions: %s", image_id, exc
        )

"""
SatQuery AI — Analysis Service (Phase 4 + Phase 6)

Orchestrates the complete analysis pipeline:

  analysis.py (router)
      ↓
  analysis_service.py   ← this file
      ↓
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Validate image exists                                │
  │ 2. Validate session (if provided)                       │
  │ 3. Validate image linked to session                     │
  │ 4. Create analysis record (pending)                     │
  │ 5a. Single-intent: call AI provider directly (Phase 4)  │
  │ 5b. Multi-intent:  delegate to multi_intent_service     │
  │                    (Phase 6) — one call per intent      │
  │ 6. Parse / combine response                             │
  │ 7. Update analysis record (+ Phase 6 fields)            │
  │ 8. Update session question                              │
  │ 9. Return AnalysisResponse                              │
  └─────────────────────────────────────────────────────────┘
      ↓
  MongoDB (analyses + sessions collections)

This service does NOT:
  - Know about HTTP requests/responses.
  - Touch the AI API key directly (handled by provider.py / gemini_provider.py).
  - Contain prompt text (all in ai/prompts.py).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from app.ai.base import AIRequest
from app.ai.intent_detector import detect_intents, is_multi_intent
from app.ai.prompts import (
    SATELLITE_ANALYSIS_SYSTEM_PROMPT,
    build_final_user_message,
    build_conversation_context_message,
)
from app.ai.provider import get_provider
from app.ai.response_parser import parse_response
from app.core.config import settings
from app.db.mongodb import get_analyses_collection, get_images_collection, get_sessions_collection
from app.schemas.analysis import AnalysisResponse, IntentResult
from app.services import history_service

logger = logging.getLogger(__name__)


# ─── ID Generation ─────────────────────────────────────────────────────────────

def _generate_analysis_id() -> str:
    """Generate analysis_<ULID>."""
    try:
        from ulid import ULID
        return f"analysis_{ULID()}"
    except ImportError:
        import uuid
        return f"analysis_{uuid.uuid4().hex}"


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _get_image_doc(image_id: str) -> dict:
    """Fetch image document; raises HTTP 404 if missing."""
    try:
        col = get_images_collection()
        doc = col.find_one({"image_id": image_id}, {"_id": 0})
    except Exception as exc:
        logger.error("DB error fetching image %s: %s", image_id, exc)
        raise HTTPException(status_code=500, detail="Database error")
    if not doc:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")
    return doc


def _get_session_doc(session_id: str) -> dict:
    """Fetch session document; raises HTTP 404 if missing."""
    try:
        col = get_sessions_collection()
        doc = col.find_one({"session_id": session_id}, {"_id": 0})
    except Exception as exc:
        logger.error("DB error fetching session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail="Database error")
    if not doc:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return doc


def _resolve_image_path(image_doc: dict) -> Path:
    """
    Resolve the filesystem path of the image to send to the AI provider.
    Prefers the 'original' path from storage metadata.
    Falls back to the 'processed' path if original is missing.
    Raises HTTP 500 if neither exists.
    """
    storage = image_doc.get("storage", {}) or {}
    original = storage.get("original", "")
    processed = storage.get("processed", "")

    for path_str in (original, processed):
        if path_str:
            p = Path(path_str)
            if p.exists():
                return p

    # Fallback: reconstruct from image_id + extension
    image_id = image_doc["image_id"]
    ext = image_doc.get("extension", ".jpg")
    original_dir = settings.original_path
    candidate = original_dir / f"{image_id}{ext}"
    if candidate.exists():
        return candidate

    logger.error("Image file not found on disk for image_id=%s", image_id)
    raise HTTPException(
        status_code=500,
        detail="Image file is missing from storage. It may have been deleted.",
    )


# ─── Main Entry Point ──────────────────────────────────────────────────────────

def run_analysis(
    image_id: str,
    query: str,
    session_id: Optional[str],
    analysis_type: str = "general",
    enable_multi_intent: bool = True,
) -> AnalysisResponse:
    """
    Run a complete AI analysis pipeline.

    Steps:
      1.  Check AI is configured (503 if not).
      2.  Validate image exists.
      3.  Validate session exists (if provided).
      4.  Validate image is linked to session (if session provided).
      5.  Create analysis document with status='processing'.
      6.  Resolve image file path.
      7a. [Single-intent] Build prompt → call AI provider → parse.
      7b. [Multi-intent]  Detect intents → run each via multi_intent_service.
      8.  Update analysis document (completed or failed) + Phase 6 fields.
      9.  Update session question/answer (if session provided).
      10. Return AnalysisResponse.

    Phase 6:
      When `enable_multi_intent=True` (default) and the query contains two or
      more detected intents, the multi-intent path is taken automatically.
      The caller can pass `enable_multi_intent=False` to force single-intent
      behavior regardless of query content.
    """

    # 1. Check configuration (raises HTTP 503 if API key is missing)
    provider = get_provider()

    # 2. Validate image
    image_doc = _get_image_doc(image_id)
    mime_type = image_doc.get("mime_type", "image/jpeg")
    is_geospatial = image_doc.get("is_geospatial", False)

    # 3 & 4. Validate session + image linkage
    if session_id:
        session_doc = _get_session_doc(session_id)
        if image_id not in session_doc.get("image_ids", []):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Image '{image_id}' is not linked to session '{session_id}'. "
                    "Link the image first via POST /api/sessions/{session_id}/images/{image_id}."
                ),
            )

    # 5. Create analysis record (status=processing)
    analysis_id = _generate_analysis_id()
    now = _now()
    query_clean = (query or "").strip()

    analysis_doc: dict = {
        "analysis_id": analysis_id,
        "session_id": session_id,
        "image_id": image_id,
        "query": query_clean,
        "analysis_type": analysis_type or "general",
        "status": "processing",
        "answer": None,
        "findings": [],
        "confidence": None,
        "confidence_available": False,
        "provider": provider.provider_name,
        "model": provider.model_name,
        "error_message": None,
        "created_at": now,
        "completed_at": None,
        # Phase 6 fields (populated below when multi-intent)
        "is_multi_intent": False,
        "detected_intents": [],
        "intent_results": [],
    }

    try:
        get_analyses_collection().insert_one({**analysis_doc})
        logger.info(
            "Analysis record created: analysis_id=%s image_id=%s session_id=%s",
            analysis_id, image_id, session_id,
        )
    except Exception as exc:
        logger.error("Failed to insert analysis record: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create analysis record")

    # 6. Resolve image path
    image_path = _resolve_image_path(image_doc)

    # Retrieve conversation history for context (Phase 5)
    conversation_history: list = []
    if session_id:
        conversation_history = _get_conversation_history(session_id, max_turns=5)

    # ── Phase 6: multi-intent detection ──────────────────────────────────────
    use_multi = (
        enable_multi_intent
        and bool(query_clean)  # empty query falls back to general
        and is_multi_intent(query_clean)
    )

    if use_multi:
        return _run_multi_intent_analysis(
            analysis_id=analysis_id,
            image_id=image_id,
            session_id=session_id,
            query_clean=query_clean,
            image_path=image_path,
            mime_type=mime_type,
            is_geospatial=is_geospatial,
            provider=provider,
            conversation_history=conversation_history,
            now=now,
        )

    # ── Single-intent path (Phase 4 / unchanged) ──────────────────────────────
    # 7. Build prompt
    if conversation_history:
        user_message = build_conversation_context_message(
            query_clean, analysis_type, is_geospatial, conversation_history
        )
    else:
        user_message = build_final_user_message(query_clean, analysis_type, is_geospatial)

    ai_request = AIRequest(
        image_path=image_path,
        mime_type=mime_type,
        query=user_message,
        system_prompt=SATELLITE_ANALYSIS_SYSTEM_PROMPT,
        analysis_type=analysis_type or "general",
        conversation_history=conversation_history,
    )

    # 8. Call AI provider
    logger.info("Calling AI provider: %s / %s", provider.provider_name, provider.model_name)
    raw_response = provider.analyze(ai_request)

    # 9. Parse response
    parsed = parse_response(raw_response)

    # 10. Update analysis document
    completed_at = _now() if parsed.status == "completed" else None
    update_fields: dict = {
        "status": parsed.status,
        "answer": parsed.answer,
        "findings": parsed.findings,
        "confidence": parsed.confidence,
        "confidence_available": parsed.confidence_available,
        "error_message": parsed.error_message,
        "completed_at": completed_at,
        "is_multi_intent": False,
        "detected_intents": detect_intents(query_clean) if query_clean else ["general"],
        "intent_results": [],
    }

    try:
        get_analyses_collection().update_one(
            {"analysis_id": analysis_id},
            {"$set": update_fields},
        )
        logger.info(
            "Analysis record updated: analysis_id=%s status=%s",
            analysis_id, parsed.status,
        )
    except Exception as exc:
        logger.error("Failed to update analysis record: %s", exc)
        # Don't raise — we still have the result; log and continue

    # 11. Update session question/answer (if session provided)
    if session_id:
        _update_session_question(
            session_id=session_id,
            image_id=image_id,
            query=query_clean,
            analysis_id=analysis_id,
            answer=parsed.answer,
            status=parsed.status,
            analysis_type=analysis_type,
        )

    # 12. Write history record (Phase 8)
    history_service.create_history_item(
        session_id=session_id,
        record_type="analysis",
        analysis_type=analysis_type or "general",
        query=query_clean,
        answer=parsed.answer,
        image_refs=[image_id],
        status=parsed.status,
        analysis_id=analysis_id,
        error=parsed.error_message,
        is_multi_intent=False,
        detected_intents=detect_intents(query_clean) if query_clean else ["general"],
        provider=provider.provider_name,
        model=provider.model_name,
    )

    # 13. Return response
    return AnalysisResponse(
        analysis_id=analysis_id,
        image_id=image_id,
        session_id=session_id,
        query=query_clean,
        analysis_type=analysis_type or "general",
        status=parsed.status,
        answer=parsed.answer,
        findings=[],
        confidence=parsed.confidence,
        confidence_available=parsed.confidence_available,
        provider=provider.provider_name,
        model=provider.model_name,
        created_at=now,
        completed_at=completed_at,
        is_multi_intent=False,
        detected_intents=detect_intents(query_clean) if query_clean else ["general"],
        intent_results=[],
    )


# ─── Phase 6 — Multi-Intent Analysis ─────────────────────────────────────────

def _run_multi_intent_analysis(
    analysis_id: str,
    image_id: str,
    session_id: Optional[str],
    query_clean: str,
    image_path: Path,
    mime_type: str,
    is_geospatial: bool,
    provider,
    conversation_history: list,
    now: datetime,
) -> AnalysisResponse:
    """
    Phase 6 multi-intent analysis path.

    Delegates to multi_intent_service.run_multi_intent(), then persists the
    combined result to MongoDB and the session in the same way as the
    single-intent path.  Returns a full AnalysisResponse.
    """
    from app.services.multi_intent_service import run_multi_intent

    logger.info(
        "Multi-intent path: analysis_id=%s query=%r",
        analysis_id, query_clean[:80],
    )

    multi_result = run_multi_intent(
        query=query_clean,
        image_path=image_path,
        mime_type=mime_type,
        is_geospatial=is_geospatial,
        provider=provider,
        conversation_history=conversation_history,
    )

    # Serialise per-intent results for MongoDB + response
    intent_results_dicts = [
        {
            "intent":        r.intent,
            "display_label": r.display_label,
            "analysis_type": r.analysis_type,
            "status":        r.status,
            "answer":        r.answer,
            "error_message": r.error_message,
        }
        for r in multi_result.intent_results
    ]

    # Determine overall analysis_type (use first succeeded intent, else 'multi')
    final_analysis_type = "multi_intent"
    for r in multi_result.intent_results:
        if r.status == "completed":
            final_analysis_type = r.analysis_type
            break

    completed_at = _now() if multi_result.overall_status == "completed" else None

    # Update analysis document with Phase 6 fields
    update_fields: dict = {
        "status":             multi_result.overall_status,
        "answer":             multi_result.combined_answer,
        "findings":           [],
        "confidence":         None,
        "confidence_available": False,
        "error_message":      multi_result.error_message,
        "completed_at":       completed_at,
        "analysis_type":      final_analysis_type,
        "is_multi_intent":    True,
        "detected_intents":   multi_result.detected_intents,
        "intent_results":     intent_results_dicts,
    }

    try:
        get_analyses_collection().update_one(
            {"analysis_id": analysis_id},
            {"$set": update_fields},
        )
        logger.info(
            "Multi-intent analysis record updated: analysis_id=%s status=%s intents=%s",
            analysis_id, multi_result.overall_status, multi_result.detected_intents,
        )
    except Exception as exc:
        logger.error("Failed to update multi-intent analysis record: %s", exc)

    # Update session Q&A (single combined record per user query)
    if session_id:
        _update_session_question(
            session_id=session_id,
            image_id=image_id,
            query=query_clean,
            analysis_id=analysis_id,
            answer=multi_result.combined_answer,
            status=multi_result.overall_status,
            analysis_type=final_analysis_type,
        )

    # Write history record (Phase 8)
    history_service.create_history_item(
        session_id=session_id,
        record_type="analysis",
        analysis_type=final_analysis_type,
        query=query_clean,
        answer=multi_result.combined_answer,
        image_refs=[image_id],
        status=multi_result.overall_status,
        analysis_id=analysis_id,
        error=multi_result.error_message,
        is_multi_intent=True,
        detected_intents=multi_result.detected_intents,
        intent_results=intent_results_dicts,
        provider=provider.provider_name,
        model=provider.model_name,
    )

    # Build IntentResult Pydantic models for the response
    intent_result_models = [
        IntentResult(
            intent=r.intent,
            display_label=r.display_label,
            analysis_type=r.analysis_type,
            status=r.status,
            answer=r.answer,
            error_message=r.error_message,
        )
        for r in multi_result.intent_results
    ]

    return AnalysisResponse(
        analysis_id=analysis_id,
        image_id=image_id,
        session_id=session_id,
        query=query_clean,
        analysis_type=final_analysis_type,
        status=multi_result.overall_status,
        answer=multi_result.combined_answer,
        findings=[],
        confidence=None,
        confidence_available=False,
        provider=provider.provider_name,
        model=provider.model_name,
        created_at=now,
        completed_at=completed_at,
        is_multi_intent=True,
        detected_intents=multi_result.detected_intents,
        intent_results=intent_result_models,
    )


# ─── Session Question Update ───────────────────────────────────────────────────

def _update_session_question(
    session_id: str,
    image_id: str,
    query: str,
    analysis_id: str,
    answer: Optional[str],
    status: str,
    analysis_type: str,
) -> None:
    """
    Find or create the matching question in the session and update it with
    the analysis result.

    Strategy:
      1. Look for an existing 'pending' question with matching query + image_id.
      2. If found: update answer, status, analysis_id, completed_at.
      3. If not found: append a new completed question (idempotent upsert).
    """
    now = _now()
    try:
        col = get_sessions_collection()
        session_doc = col.find_one({"session_id": session_id}, {"_id": 0})
        if not session_doc:
            return

        questions: list[dict] = session_doc.get("questions", [])

        # Look for pending question matching this query + image_id
        match_idx = next(
            (
                i for i, q in enumerate(questions)
                if q.get("image_id") == image_id
                and q.get("question", "").strip() == (query or "").strip()
                and q.get("status") in ("pending", "processing")
            ),
            None,
        )

        if match_idx is not None:
            # Update existing question
            set_fields: dict = {
                f"questions.{match_idx}.answer": answer,
                f"questions.{match_idx}.status": status,
                f"questions.{match_idx}.analysis_id": analysis_id,
                "updated_at": now,
            }
            if status in ("completed", "failed"):
                set_fields[f"questions.{match_idx}.completed_at"] = now
            col.update_one({"session_id": session_id}, {"$set": set_fields})
            logger.info(
                "Session question updated: session_id=%s idx=%d status=%s",
                session_id, match_idx, status,
            )
        else:
            # Append new question with result (handles the case where
            # the question wasn't pre-created via the sessions API)
            from app.services.session_service import _generate_question_id
            q_doc = {
                "question_id": _generate_question_id(),
                "question": query or "(auto analysis)",
                "image_id": image_id,
                "analysis_type": analysis_type or "general",
                "answer": answer,
                "status": status,
                "analysis_id": analysis_id,
                "created_at": now,
                "completed_at": now if status in ("completed", "failed") else None,
            }
            col.update_one(
                {"session_id": session_id},
                {
                    "$push": {"questions": q_doc},
                    "$set": {"updated_at": now},
                },
            )
            logger.info(
                "New question appended to session: session_id=%s status=%s",
                session_id, status,
            )

    except Exception as exc:
        # Non-fatal: analysis result is already saved; just log the failure
        logger.warning(
            "Could not update session question for session_id=%s: %s",
            session_id, exc,
        )


# ─── Phase 5 — Conversation History Retrieval ────────────────────────────────

def _get_conversation_history(session_id: str, max_turns: int = 5) -> list:
    """
    Fetch the last `max_turns` completed Q&A pairs from a session.

    Returns a list of dicts: [{"question": str, "answer": str}, ...]
    ordered chronologically (oldest first).

    Only questions with status='completed' and a non-empty answer are included.
    Failed/pending questions are excluded so bad answers don't poison context.
    """
    try:
        col = get_sessions_collection()
        doc = col.find_one({"session_id": session_id}, {"_id": 0, "questions": 1})
        if not doc:
            return []

        questions: list[dict] = doc.get("questions", [])

        # Filter: only completed questions with real answers
        completed = [
            q for q in questions
            if q.get("status") == "completed"
            and q.get("answer")
            and q.get("question")
        ]

        # Take last max_turns (chronological order is preserved by $push in MongoDB)
        recent = completed[-max_turns:] if len(completed) > max_turns else completed

        return [
            {"question": q["question"].strip(), "answer": q["answer"].strip()}
            for q in recent
        ]

    except Exception as exc:
        logger.warning(
            "Could not retrieve conversation history for session_id=%s: %s",
            session_id, exc,
        )
        return []


# ─── Phase 5 — Follow-up Entry Point ─────────────────────────────────────────

def run_analysis_followup(
    session_id: str,
    question: str,
    analysis_type: str = "general",
    enable_multi_intent: bool = True,
) -> tuple:
    """
    Phase 5 follow-up entry point.

    Automatically resolves the image_id from the session so the frontend
    does NOT need to supply it.  Returns (AnalysisResponse, question_number).

    Steps:
      1. Fetch session → validate it exists + has at least one image.
      2. Use the primary (first) image_id.
      3. Count existing questions to compute question_number.
      4. Delegate to run_analysis() which handles context fetching + AI call.
    """
    session_doc = _get_session_doc(session_id)  # raises 404 if not found

    image_ids: list[str] = session_doc.get("image_ids", [])
    if not image_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Session '{session_id}' has no linked images. "
                "Upload and link an image before asking questions."
            ),
        )

    image_id = image_ids[0]  # Primary image

    # Count existing questions (completed or not) to compute position
    all_questions: list = session_doc.get("questions", [])
    question_number = len(all_questions) + 1  # This question will be Q{question_number}

    # Delegate to run_analysis which handles context, AI call, and persistence
    result = run_analysis(
        image_id=image_id,
        query=question,
        session_id=session_id,
        analysis_type=analysis_type or "general",
        enable_multi_intent=enable_multi_intent,
    )

    return result, question_number

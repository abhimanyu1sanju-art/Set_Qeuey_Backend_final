"""
SatQuery AI — AI Response Parser (Phase 4)

Converts provider-specific raw output (AIRawResponse) into the standardized
SatQuery AnalysisResult structure.

Rules:
  - Never let provider formats leak beyond this module.
  - Handle empty, malformed, or error responses gracefully.
  - Never invent or fabricate any field.
  - confidence = None / confidence_available = False unless the provider
    returned a genuine calibrated confidence value (Gemini does not).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.ai.base import AIRawResponse

logger = logging.getLogger(__name__)


# ─── Parsed Result ─────────────────────────────────────────────────────────────

class ParsedAnalysisResult:
    """
    Normalised analysis result ready to be saved to MongoDB and returned to the frontend.

    Fields mirror the MongoDB analyses document structure.
    """

    __slots__ = (
        "answer",
        "findings",
        "confidence",
        "confidence_available",
        "status",
        "error_message",
        "provider",
        "model",
        "raw_text",
    )

    def __init__(
        self,
        answer: Optional[str],
        findings: list,
        confidence: Optional[float],
        confidence_available: bool,
        status: str,
        error_message: Optional[str],
        provider: str,
        model: str,
        raw_text: str = "",
    ) -> None:
        self.answer = answer
        self.findings = findings
        self.confidence = confidence
        self.confidence_available = confidence_available
        self.status = status
        self.error_message = error_message
        self.provider = provider
        self.model = model
        self.raw_text = raw_text

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "findings": self.findings,
            "confidence": self.confidence,
            "confidence_available": self.confidence_available,
            "status": self.status,
            "error_message": self.error_message,
            "provider": self.provider,
            "model": self.model,
        }


# ─── Parser ───────────────────────────────────────────────────────────────────

def parse_response(raw: AIRawResponse) -> ParsedAnalysisResult:
    """
    Convert a raw provider response into a normalized ParsedAnalysisResult.

    Handles:
      - successful text responses
      - empty text (no useful content)
      - provider-flagged errors
      - unexpected/malformed payloads
    """
    provider = raw.provider or "unknown"
    model = raw.model or "unknown"

    # ── Error path ────────────────────────────────────────────────────────────
    if not raw.success:
        error_msg = raw.error_message or "AI provider returned an error."
        logger.warning("AI provider error [%s/%s]: %s", provider, model, error_msg)
        return ParsedAnalysisResult(
            answer=None,
            findings=[],
            confidence=None,
            confidence_available=False,
            status="failed",
            error_message=error_msg,
            provider=provider,
            model=model,
            raw_text=raw.text or "",
        )

    # ── Empty response ────────────────────────────────────────────────────────
    text = (raw.text or "").strip()
    if not text:
        logger.warning("AI provider [%s/%s] returned empty text.", provider, model)
        return ParsedAnalysisResult(
            answer=None,
            findings=[],
            confidence=None,
            confidence_available=False,
            status="failed",
            error_message="AI provider returned an empty response.",
            provider=provider,
            model=model,
            raw_text="",
        )

    # ── Successful response ───────────────────────────────────────────────────
    # Phase 4: findings are not yet extracted structurally (Phase 5 addition).
    # confidence is None — Gemini Flash does not return calibrated confidence.
    logger.info(
        "AI response parsed [%s/%s]: %d chars", provider, model, len(text)
    )
    return ParsedAnalysisResult(
        answer=text,
        findings=[],                 # Phase 5: structured finding extraction
        confidence=None,             # Gemini does not return calibrated confidence
        confidence_available=False,
        status="completed",
        error_message=None,
        provider=provider,
        model=model,
        raw_text=text,
    )

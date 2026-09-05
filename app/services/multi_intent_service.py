"""
SatQuery AI — Multi-Intent Analysis Service (Phase 6)

Orchestrates parallel analysis for queries that contain multiple distinct
analysis intents (e.g. "analyze vegetation, water and land cover").

Design rules:
  - This module is ONLY called by analysis_service.run_analysis() when
    is_multi_intent() returns True and enable_multi_intent is True.
  - Never touches HTTP request/response objects.
  - Never touches the AI API key directly.
  - Never duplicates session/MongoDB logic — the caller handles persistence.
  - Per-intent failures are non-fatal; partial results are always preserved.
  - Returns a MultiIntentResult dataclass that the caller uses to build the
    final AnalysisResponse and persist to MongoDB.

Flow:
  1. detect_intents(query)            → ["vegetation", "water", "land_cover"]
  2. for each intent:
       build_intent_sub_query()       → focused question string
       provider.analyze(AIRequest)    → AIRawResponse
       parse_response()               → ParsedAnalysisResult
  3. combine_results()                → single markdown answer string
  4. return MultiIntentResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.ai.base import AIRequest, BaseAIProvider
from app.ai.intent_detector import (
    build_intent_sub_query,
    detect_intents,
    get_analysis_type,
    get_display_label,
)
from app.ai.prompts import SATELLITE_ANALYSIS_SYSTEM_PROMPT
from app.ai.response_parser import parse_response, ParsedAnalysisResult

logger = logging.getLogger(__name__)


# ─── Per-intent result ─────────────────────────────────────────────────────────

@dataclass
class IntentAnalysisResult:
    """
    Analysis result for one detected intent.

    Mirrors app.schemas.analysis.IntentResult but is a plain dataclass
    so the service layer has no Pydantic dependency.
    """
    intent: str
    display_label: str
    analysis_type: str
    status: str                         # 'completed' | 'failed'
    answer: Optional[str] = None
    error_message: Optional[str] = None


# ─── Combined result ───────────────────────────────────────────────────────────

@dataclass
class MultiIntentResult:
    """
    Complete multi-intent analysis result returned to analysis_service.

    Fields:
        detected_intents   — ordered, deduplicated list of intent names
        intent_results     — one IntentAnalysisResult per intent
        combined_answer    — single markdown string combining all intent answers
        overall_status     — 'completed' if ≥1 succeeded, else 'failed'
        error_message      — set only if ALL intents failed
    """
    detected_intents: list[str] = field(default_factory=list)
    intent_results: list[IntentAnalysisResult] = field(default_factory=list)
    combined_answer: Optional[str] = None
    overall_status: str = "failed"
    error_message: Optional[str] = None


# ─── Result combiner ───────────────────────────────────────────────────────────

def _combine_results(
    original_query: str,
    intent_results: list[IntentAnalysisResult],
) -> str:
    """
    Build a single structured markdown answer from per-intent results.

    Format:
      ## <Display Label>
      <answer or error notice>

      ## Overall Summary
      <summary of what was analyzed>
    """
    sections: list[str] = []

    succeeded = [r for r in intent_results if r.status == "completed" and r.answer]
    failed    = [r for r in intent_results if r.status != "completed" or not r.answer]

    for result in intent_results:
        heading = f"## {result.display_label}"
        if result.status == "completed" and result.answer:
            sections.append(f"{heading}\n\n{result.answer.strip()}")
        else:
            err = result.error_message or "Analysis could not be completed for this intent."
            sections.append(
                f"{heading}\n\n"
                f"> ⚠️ **Analysis unavailable:** {err}"
            )

    # Overall summary
    if succeeded:
        intent_names = ", ".join(r.display_label for r in succeeded)
        summary_parts = [
            f"## Overall Summary\n\n"
            f"This analysis covered **{len(intent_results)}** intent(s) "
            f"for the query: *\"{original_query.strip()}\"*\n\n"
            f"**Successful:** {intent_names}."
        ]
        if failed:
            failed_names = ", ".join(r.display_label for r in failed)
            summary_parts.append(
                f"\n**Could not complete:** {failed_names} "
                f"(see individual section(s) above for details)."
            )
        sections.append("".join(summary_parts))
    else:
        sections.append(
            "## Overall Summary\n\n"
            "None of the requested analyses could be completed. "
            "Please check that the AI provider is configured correctly."
        )

    return "\n\n---\n\n".join(sections)


# ─── Per-intent runner ─────────────────────────────────────────────────────────

def _run_single_intent(
    intent: str,
    original_query: str,
    image_path: Path,
    mime_type: str,
    is_geospatial: bool,
    provider: BaseAIProvider,
    conversation_history: list,
) -> IntentAnalysisResult:
    """
    Execute the AI analysis pipeline for one specific intent.

    Returns IntentAnalysisResult — never raises.
    """
    display_label = get_display_label(intent)
    analysis_type = get_analysis_type(intent)

    try:
        # Build a focused sub-query for this intent
        sub_query = build_intent_sub_query(intent, original_query)

        # Append geospatial limitation note if needed
        if is_geospatial:
            from app.ai.prompts import GEOTIFF_LIMITATION_NOTE
            sub_query += GEOTIFF_LIMITATION_NOTE

        # Prepend conversation context if available
        if conversation_history:
            from app.ai.prompts import build_conversation_context_message
            sub_query = build_conversation_context_message(
                sub_query, analysis_type, is_geospatial, conversation_history
            )

        ai_request = AIRequest(
            image_path=image_path,
            mime_type=mime_type,
            query=sub_query,
            system_prompt=SATELLITE_ANALYSIS_SYSTEM_PROMPT,
            analysis_type=analysis_type,
            conversation_history=conversation_history,
        )

        raw_response = provider.analyze(ai_request)
        parsed: ParsedAnalysisResult = parse_response(raw_response)

        logger.info(
            "Intent '%s' analysis: status=%s chars=%s",
            intent,
            parsed.status,
            len(parsed.answer or ""),
        )

        return IntentAnalysisResult(
            intent=intent,
            display_label=display_label,
            analysis_type=analysis_type,
            status=parsed.status,
            answer=parsed.answer,
            error_message=parsed.error_message,
        )

    except Exception as exc:
        # Catch-all: individual intent failure must not crash the entire pipeline
        logger.error(
            "Unexpected error running intent '%s': %s", intent, exc, exc_info=True
        )
        return IntentAnalysisResult(
            intent=intent,
            display_label=display_label,
            analysis_type=analysis_type,
            status="failed",
            answer=None,
            error_message="An unexpected error occurred during analysis.",
        )


# ─── Public entry point ────────────────────────────────────────────────────────

def run_multi_intent(
    query: str,
    image_path: Path,
    mime_type: str,
    is_geospatial: bool,
    provider: BaseAIProvider,
    conversation_history: Optional[list] = None,
) -> MultiIntentResult:
    """
    Run multi-intent analysis for a query containing ≥2 detected intents.

    Steps:
      1. Detect all intents in the query (deduplicated, ordered).
      2. For each intent, run the appropriate AI analysis pipeline.
      3. Combine per-intent results into a structured markdown answer.
      4. Return a MultiIntentResult — never raises.

    Parameters
    ----------
    query               : Original user query string.
    image_path          : Absolute path to the image file.
    mime_type           : MIME type of the image.
    is_geospatial       : Whether the image is a GeoTIFF.
    provider            : Configured AI provider instance.
    conversation_history: Previous Q&A turns for follow-up context.
    """
    conversation_history = conversation_history or []
    query_clean = (query or "").strip()

    # Step 1: Detect intents
    intents = detect_intents(query_clean)
    logger.info(
        "Multi-intent analysis started: query=%r intents=%s",
        query_clean[:80],
        intents,
    )

    # Step 2: Run each intent
    intent_results: list[IntentAnalysisResult] = []
    for intent in intents:
        result = _run_single_intent(
            intent=intent,
            original_query=query_clean,
            image_path=image_path,
            mime_type=mime_type,
            is_geospatial=is_geospatial,
            provider=provider,
            conversation_history=conversation_history,
        )
        intent_results.append(result)

    # Step 3: Combine results
    combined_answer = _combine_results(query_clean, intent_results)

    # Step 4: Determine overall status
    succeeded = any(r.status == "completed" for r in intent_results)
    overall_status = "completed" if succeeded else "failed"

    error_message: Optional[str] = None
    if not succeeded:
        failed_intents = ", ".join(r.intent for r in intent_results)
        error_message = f"All intent analyses failed: {failed_intents}."

    logger.info(
        "Multi-intent analysis complete: intents=%s overall_status=%s",
        intents,
        overall_status,
    )

    return MultiIntentResult(
        detected_intents=intents,
        intent_results=intent_results,
        combined_answer=combined_answer,
        overall_status=overall_status,
        error_message=error_message,
    )

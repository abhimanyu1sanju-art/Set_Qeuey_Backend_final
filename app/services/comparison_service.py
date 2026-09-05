"""
SatQuery AI — Comparison Service (Phase 7)

Orchestrates the full two-image comparison pipeline:

  comparisons.py (router)
      ↓
  comparison_service.py   ← this file
      ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. Validate both images exist (404 if either missing)           │
  │ 2. Validate session (if provided)                               │
  │ 3. Validate both images linked to session (if session provided) │
  │ 4. Check metadata compatibility                                 │
  │ 5. Create comparison record (status=processing)                 │
  │ 6. Resolve image file paths                                     │
  │ 7. Preprocess / prepare images for AI                           │
  │ 8. Send BOTH images to Gemini in one request                    │
  │ 9. Parse AI response → detected changes + explanation           │
  │ 10. Update comparison record (completed or failed)              │
  │ 11. Return ComparisonResponse                                   │
  └─────────────────────────────────────────────────────────────────┘
      ↓
  MongoDB (comparisons collection)

Design rules:
  - Never exposes AI API key.
  - Never invents comparison results.
  - Per-image path resolution reuses GeminiProvider logic for TIFF handling.
  - Incompatible images return HTTP 422 (Unprocessable Entity), not 500.
  - A failed AI call returns status='failed' with error_message — not 500.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from app.ai.comparison_prompts import (
    COMPARISON_SYSTEM_PROMPT,
    build_comparison_prompt,
    build_meta_summary,
)
from app.core.config import settings
from app.db.mongodb import (
    get_analyses_collection,
    get_comparisons_collection,
    get_images_collection,
    get_sessions_collection,
)
from app.schemas.comparison import (
    CompatibilityResult,
    ComparisonResponse,
    DetectedChange,
    ImageMetaSummary,
    PreprocessingResult,
)
from app.services import history_service

logger = logging.getLogger(__name__)

# Gemini supported MIME types (mirrors gemini_provider.py)
_GEMINI_SUPPORTED_MIME = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/gif", "image/heic", "image/heif",
}


# ─── ID helpers ───────────────────────────────────────────────────────────────

def _generate_comparison_id() -> str:
    try:
        from ulid import ULID
        return f"cmp_{ULID()}"
    except ImportError:
        import uuid
        return f"cmp_{uuid.uuid4().hex}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─── MongoDB helpers ──────────────────────────────────────────────────────────

def _get_image_doc(image_id: str) -> dict:
    """Fetch an image document by image_id. Raises HTTP 404 if not found."""
    try:
        col = get_images_collection()
        doc = col.find_one({"image_id": image_id}, {"_id": 0})
    except Exception as exc:
        logger.error("MongoDB query failed for image %s: %s", image_id, exc)
        raise HTTPException(status_code=500, detail="Database error")
    if not doc:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")
    return doc


def _get_session_doc(session_id: str) -> dict:
    """Fetch session doc. Raises 404 if not found."""
    try:
        col = get_sessions_collection()
        doc = col.find_one({"session_id": session_id}, {"_id": 0})
    except Exception as exc:
        logger.error("MongoDB session query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")
    if not doc:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return doc


# ─── Image file path resolver ─────────────────────────────────────────────────

def _resolve_image_file(image_doc: dict) -> tuple[Path, str]:
    """
    Return (path, effective_mime) for the image file to send to AI.

    For TIFF/GeoTIFF: use the processed JPEG derivative (same logic as GeminiProvider).
    For standard formats: use the stored processed path as-is.
    """
    storage = image_doc.get("storage", {})
    processed_str = storage.get("processed", "")
    original_str = storage.get("original", "")
    mime = image_doc.get("mime_type", "image/jpeg").lower()

    is_tiff = (
        mime in ("image/tiff", "image/geotiff")
        or (processed_str and Path(processed_str).suffix.lower() in (".tif", ".tiff"))
    )

    if is_tiff:
        # Look for the JPEG derivative in processed directory
        stem = Path(processed_str or original_str).stem
        processed_dir = settings.processed_path
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = processed_dir / f"{stem}{ext}"
            if candidate.exists():
                logger.info("TIFF: using processed derivative %s", candidate)
                return candidate, "image/jpeg"
        # Fallback: try processed path directly
        if processed_str and Path(processed_str).exists():
            return Path(processed_str), "image/jpeg"
        raise HTTPException(
            status_code=500,
            detail=f"TIFF processed derivative not found for image {image_doc['image_id']}",
        )

    # Standard image: use processed copy
    if processed_str and Path(processed_str).exists():
        effective_mime = mime if mime in _GEMINI_SUPPORTED_MIME else "image/jpeg"
        return Path(processed_str), effective_mime

    # Fallback to original
    if original_str and Path(original_str).exists():
        effective_mime = mime if mime in _GEMINI_SUPPORTED_MIME else "image/jpeg"
        return Path(original_str), effective_mime

    raise HTTPException(
        status_code=500,
        detail=f"Image file not found on disk for {image_doc['image_id']}",
    )


# ─── Metadata compatibility check ─────────────────────────────────────────────

def _check_compatibility(doc_a: dict, doc_b: dict) -> CompatibilityResult:
    """
    Compare metadata of Image A and Image B.
    Returns CompatibilityResult — never raises.

    Only blocks on hard incompatibilities (e.g. one is not an image).
    Records warnings for soft mismatches (dimension difference, different CRS).
    """
    reasons: list[str] = []
    warnings: list[str] = []

    mime_a = doc_a.get("mime_type", "").lower()
    mime_b = doc_b.get("mime_type", "").lower()

    # Hard block: if either is not a recognised image type
    image_mimes = {
        "image/jpeg", "image/jpg", "image/png",
        "image/tiff", "image/geotiff", "image/webp",
    }
    if mime_a not in image_mimes:
        reasons.append(f"Image A has unsupported MIME type: {mime_a}")
    if mime_b not in image_mimes:
        reasons.append(f"Image B has unsupported MIME type: {mime_b}")

    if reasons:
        return CompatibilityResult(compatible=False, reasons=reasons, warnings=warnings)

    # Dimension mismatch (warning, not block)
    w_a, h_a = doc_a.get("width", 0), doc_a.get("height", 0)
    w_b, h_b = doc_b.get("width", 0), doc_b.get("height", 0)
    if w_a > 0 and w_b > 0 and (abs(w_a - w_b) > 50 or abs(h_a - h_b) > 50):
        warnings.append(
            f"Images have different dimensions: "
            f"A={w_a}×{h_a}px, B={w_b}×{h_b}px. "
            "Comparison will be qualitative only."
        )

    # Format mismatch (warning)
    fmt_a = doc_a.get("format", "").upper()
    fmt_b = doc_b.get("format", "").upper()
    if fmt_a and fmt_b and fmt_a != fmt_b:
        warnings.append(
            f"Images have different formats: A={fmt_a}, B={fmt_b}. "
            "Visual comparison is still possible."
        )

    # Geospatial mismatch
    geo_a = doc_a.get("geospatial") or {}
    geo_b = doc_b.get("geospatial") or {}
    is_geo_a = doc_a.get("is_geospatial", False)
    is_geo_b = doc_b.get("is_geospatial", False)

    if is_geo_a and is_geo_b:
        crs_a = (geo_a.get("crs") or "").upper()
        crs_b = (geo_b.get("crs") or "").upper()
        if crs_a and crs_b and crs_a != crs_b:
            warnings.append(
                f"Images have different CRS: A={crs_a}, B={crs_b}. "
                "Precise geospatial alignment is not possible. "
                "Comparison will be qualitative."
            )
        res_a = geo_a.get("resolution")
        res_b = geo_b.get("resolution")
        if res_a and res_b and res_a != res_b:
            warnings.append(
                f"Images have different resolutions: A={res_a}, B={res_b}. "
                "Comparison is qualitative only."
            )
    elif is_geo_a != is_geo_b:
        warnings.append(
            "One image is a GeoTIFF and the other is not. "
            "Geospatial alignment cannot be performed."
        )

    return CompatibilityResult(compatible=True, reasons=[], warnings=warnings)


# ─── Preprocessing record ──────────────────────────────────────────────────────

def _build_preprocessing(doc_a: dict, doc_b: dict, warnings: list[str]) -> PreprocessingResult:
    """
    Record what preprocessing/alignment was performed.
    This is honest: we only record what actually happened.
    """
    is_geo_a = doc_a.get("is_geospatial", False)
    is_geo_b = doc_b.get("is_geospatial", False)

    geo_a = doc_a.get("geospatial") or {}
    geo_b = doc_b.get("geospatial") or {}
    crs_a = (geo_a.get("crs") or "").upper()
    crs_b = (geo_b.get("crs") or "").upper()

    alignment_attempted = False
    alignment_succeeded = False
    alignment_method = "none"
    alignment_note = None
    limitations: list[str] = list(warnings)

    if is_geo_a and is_geo_b and crs_a and crs_b:
        alignment_attempted = True
        if crs_a == crs_b:
            alignment_succeeded = True
            alignment_method = "geospatial_crs_match"
            alignment_note = (
                "Both images share the same CRS. The AI model receives both "
                "images and is aware they share the same projection."
            )
        else:
            alignment_succeeded = False
            alignment_method = "none"
            alignment_note = (
                "CRS mismatch — pixel-level alignment was not performed. "
                "Comparison is based on visual AI interpretation only."
            )
            limitations.append(
                "Alignment was not performed due to CRS mismatch. "
                "Pixel-level change statistics are not available."
            )
    else:
        alignment_note = (
            "No geospatial alignment was performed. "
            "Comparison is based on visual AI interpretation of both images."
        )
        if not (is_geo_a and is_geo_b):
            limitations.append(
                "One or both images lack geospatial metadata. "
                "Precise alignment and pixel-level statistics are not available."
            )

    return PreprocessingResult(
        image_a_prepared=True,
        image_b_prepared=True,
        alignment_attempted=alignment_attempted,
        alignment_succeeded=alignment_succeeded,
        alignment_method=alignment_method,
        alignment_note=alignment_note,
        limitations=limitations,
    )


# ─── AI call: two images in one request ───────────────────────────────────────

def _call_ai_comparison(
    image_path_a: Path,
    mime_a: str,
    image_path_b: Path,
    mime_b: str,
    prompt: str,
    provider,
) -> str | None:
    """
    Send BOTH images to Gemini in a single request.

    Gemini accepts multiple Part objects in the contents list.
    Returns the AI text response, or None on failure.
    """
    from google import genai
    from google.genai import types as genai_types

    try:
        bytes_a = image_path_a.read_bytes()
        bytes_b = image_path_b.read_bytes()
    except Exception as exc:
        logger.error("Failed to read image files: %s", exc)
        return None

    try:
        client = provider._get_client()
        start = time.monotonic()

        response = client.models.generate_content(
            model=provider.model_name,
            contents=[
                # Image A first, then Image B, then the text prompt
                genai_types.Part.from_bytes(data=bytes_a, mime_type=mime_a),
                "Image A (before/first image):",
                genai_types.Part.from_bytes(data=bytes_b, mime_type=mime_b),
                "Image B (after/second image):",
                prompt,
            ],
            config=genai_types.GenerateContentConfig(
                system_instruction=COMPARISON_SYSTEM_PROMPT,
                max_output_tokens=4096,
            ),
        )

        elapsed = time.monotonic() - start
        logger.info("Comparison AI response in %.2fs", elapsed)

        text = ""
        try:
            text = response.text or ""
        except Exception:
            try:
                text = response.candidates[0].content.parts[0].text or ""
            except Exception:
                text = ""

        return text or None

    except Exception as exc:
        logger.error("Comparison AI call failed: %s", exc, exc_info=True)
        return None


# ─── Change parser ─────────────────────────────────────────────────────────────

_CHANGE_KEYWORDS = {
    "increase": "up", "expand": "up", "grow": "up", "more": "up",
    "additional": "up", "greater": "up", "higher": "up", "gain": "up",
    "appear": "up", "built": "up", "develop": "up", "new": "up",
    "decrease": "down", "reduc": "down", "less": "down", "loss": "down",
    "fewer": "down", "lower": "down", "shrink": "down", "disappear": "down",
    "deforest": "down", "cleared": "down", "lost": "down",
}

_FEATURE_NAMES = [
    ("vegetation", "Vegetation Cover"),
    ("forest", "Vegetation Cover"),
    ("tree", "Vegetation Cover"),
    ("crop", "Agricultural Land"),
    ("agricultural", "Agricultural Land"),
    ("water", "Water Bodies"),
    ("lake", "Water Bodies"),
    ("river", "Water Bodies"),
    ("flood", "Water Bodies"),
    ("built", "Built-up Area"),
    ("urban", "Built-up Area"),
    ("building", "Built-up Area"),
    ("structure", "Built-up Area"),
    ("road", "Road Network"),
    ("highway", "Road Network"),
    ("infrastructure", "Infrastructure"),
    ("land cover", "Land Cover"),
    ("open ground", "Open Ground"),
    ("bare", "Open Ground"),
    ("soil", "Open Ground"),
]


def _parse_detected_changes(ai_text: str) -> list[DetectedChange]:
    """
    Parse the AI response text to extract DetectedChange items.
    Uses keyword matching — never invents changes not mentioned in AI text.
    Returns an empty list if no changes can be reliably extracted.
    """
    if not ai_text:
        return []

    text_lower = ai_text.lower()
    seen_features: set[str] = set()
    changes: list[DetectedChange] = []

    for keyword, display_name in _FEATURE_NAMES:
        if keyword not in text_lower:
            continue
        if display_name in seen_features:
            continue

        # Determine direction from surrounding context
        # Find the sentence(s) containing this keyword
        direction = "unknown"
        for line in ai_text.split("\n"):
            if keyword in line.lower():
                line_lower = line.lower()
                for kw, dir_ in _CHANGE_KEYWORDS.items():
                    if kw in line_lower:
                        direction = dir_
                        break
                if direction != "unknown":
                    break

        # Check for "stable" / "no change" / "unchanged"
        for line in ai_text.split("\n"):
            if keyword in line.lower():
                ll = line.lower()
                if any(w in ll for w in ("stable", "unchanged", "no change", "similar", "consistent")):
                    direction = "stable"
                    break

        label_map = {
            "up": "Appears Increased",
            "down": "Appears Decreased",
            "stable": "Largely Unchanged",
            "unknown": "Change Detected",
        }

        seen_features.add(display_name)
        changes.append(DetectedChange(
            feature=display_name,
            direction=direction,
            label=label_map.get(direction, "Change Detected"),
        ))

    return changes


# ─── Main Entry Point ──────────────────────────────────────────────────────────

def run_comparison(
    image_id_a: str,
    image_id_b: str,
    query: str,
    session_id: Optional[str],
    label_a: str = "Image A — Before",
    label_b: str = "Image B — After",
) -> ComparisonResponse:
    """
    Run the full two-image comparison pipeline.

    Steps:
      1.  Check AI configuration (503 if missing).
      2.  Validate Image A exists.
      3.  Validate Image B exists.
      4.  Validate session + image linkage (if session_id provided).
      5.  Check metadata compatibility.
      6.  If incompatible, return 422 + ComparisonResponse with status='incompatible'.
      7.  Create comparison MongoDB record (status=processing).
      8.  Resolve image file paths for AI.
      9.  Preprocess metadata record.
      10. Build comparison prompt.
      11. Call AI with BOTH images in one request.
      12. Parse AI response → detected changes.
      13. Update MongoDB record (completed or failed).
      14. Return ComparisonResponse.
    """
    from app.ai.provider import get_provider

    # 1. Check AI is configured
    provider = get_provider()

    # 2 & 3. Validate both images
    doc_a = _get_image_doc(image_id_a)
    doc_b = _get_image_doc(image_id_b)

    # 4. Validate session
    if session_id:
        session_doc = _get_session_doc(session_id)
        linked = session_doc.get("image_ids", [])
        missing = []
        if image_id_a not in linked:
            missing.append(f"Image A ('{image_id_a}')")
        if image_id_b not in linked:
            missing.append(f"Image B ('{image_id_b}')")
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{' and '.join(missing)} not linked to session '{session_id}'. "
                    "Link images first via POST /api/sessions/{session_id}/images/{image_id}."
                ),
            )

    # 5. Metadata compatibility check
    compat = _check_compatibility(doc_a, doc_b)

    # Build meta summaries
    geo_a = doc_a.get("geospatial") or {}
    geo_b = doc_b.get("geospatial") or {}

    meta_a = ImageMetaSummary(
        image_id=image_id_a,
        filename=doc_a.get("filename", ""),
        width=doc_a.get("width", 0),
        height=doc_a.get("height", 0),
        format=doc_a.get("format", ""),
        mime_type=doc_a.get("mime_type", ""),
        is_geospatial=doc_a.get("is_geospatial", False),
        crs=geo_a.get("crs"),
        bands=geo_a.get("bands"),
        resolution=geo_a.get("resolution"),
    )
    meta_b = ImageMetaSummary(
        image_id=image_id_b,
        filename=doc_b.get("filename", ""),
        width=doc_b.get("width", 0),
        height=doc_b.get("height", 0),
        format=doc_b.get("format", ""),
        mime_type=doc_b.get("mime_type", ""),
        is_geospatial=doc_b.get("is_geospatial", False),
        crs=geo_b.get("crs"),
        bands=geo_b.get("bands"),
        resolution=geo_b.get("resolution"),
    )

    now = _now()
    comparison_id = _generate_comparison_id()
    query_clean = (query or "").strip()

    # 6. Incompatible → return early with structured error (not 500)
    if not compat.compatible:
        logger.warning(
            "Comparison %s blocked: incompatible images: %s",
            comparison_id, compat.reasons,
        )
        _insert_comparison_doc({
            "comparison_id": comparison_id,
            "session_id": session_id,
            "image_id_a": image_id_a,
            "image_id_b": image_id_b,
            "label_a": label_a,
            "label_b": label_b,
            "query": query_clean,
            "status": "incompatible",
            "meta_a": meta_a.model_dump(),
            "meta_b": meta_b.model_dump(),
            "compatibility": compat.model_dump(),
            "preprocessing": PreprocessingResult().model_dump(),
            "ai_explanation": None,
            "detected_changes": [],
            "limitations": compat.reasons,
            "error_message": "; ".join(compat.reasons),
            "provider": provider.provider_name,
            "model": provider.model_name,
            "created_at": now,
            "completed_at": now,
        })
        return ComparisonResponse(
            comparison_id=comparison_id,
            session_id=session_id,
            image_id_a=image_id_a,
            image_id_b=image_id_b,
            label_a=label_a,
            label_b=label_b,
            meta_a=meta_a,
            meta_b=meta_b,
            compatibility=compat,
            preprocessing=PreprocessingResult(),
            status="incompatible",
            query=query_clean,
            ai_explanation=None,
            detected_changes=[],
            limitations=compat.reasons,
            error_message="; ".join(compat.reasons),
            provider=provider.provider_name,
            model=provider.model_name,
            created_at=now,
            completed_at=now,
        )

    # 7. Create MongoDB record (status=processing)
    processing_doc = {
        "comparison_id": comparison_id,
        "session_id": session_id,
        "image_id_a": image_id_a,
        "image_id_b": image_id_b,
        "label_a": label_a,
        "label_b": label_b,
        "query": query_clean,
        "status": "processing",
        "meta_a": meta_a.model_dump(),
        "meta_b": meta_b.model_dump(),
        "compatibility": compat.model_dump(),
        "preprocessing": {},
        "ai_explanation": None,
        "detected_changes": [],
        "limitations": [],
        "error_message": None,
        "provider": provider.provider_name,
        "model": provider.model_name,
        "created_at": now,
        "completed_at": None,
    }
    _insert_comparison_doc(processing_doc)

    # 8. Resolve image file paths
    try:
        path_a, mime_eff_a = _resolve_image_file(doc_a)
        path_b, mime_eff_b = _resolve_image_file(doc_b)
    except HTTPException as exc:
        _update_comparison_failed(
            comparison_id, f"Could not load image files: {exc.detail}"
        )
        raise

    # 9. Preprocessing record
    preprocessing = _build_preprocessing(doc_a, doc_b, compat.warnings)

    # 10. Build comparison prompt
    meta_a_str = build_meta_summary(
        meta_a.width, meta_a.height, meta_a.format,
        meta_a.is_geospatial, meta_a.crs, meta_a.bands, meta_a.resolution,
    )
    meta_b_str = build_meta_summary(
        meta_b.width, meta_b.height, meta_b.format,
        meta_b.is_geospatial, meta_b.crs, meta_b.bands, meta_b.resolution,
    )
    comparison_prompt = build_comparison_prompt(
        query=query_clean,
        label_a=label_a,
        label_b=label_b,
        meta_a_summary=meta_a_str,
        meta_b_summary=meta_b_str,
        warnings=compat.warnings,
        is_geospatial_a=meta_a.is_geospatial,
        is_geospatial_b=meta_b.is_geospatial,
    )

    # 11. Call AI with BOTH images
    logger.info(
        "Running comparison %s: A=%s B=%s",
        comparison_id, image_id_a, image_id_b,
    )
    ai_text = _call_ai_comparison(
        image_path_a=path_a,
        mime_a=mime_eff_a,
        image_path_b=path_b,
        mime_b=mime_eff_b,
        prompt=comparison_prompt,
        provider=provider,
    )

    # 12. Parse AI response
    detected_changes: list[DetectedChange] = []
    if ai_text:
        detected_changes = _parse_detected_changes(ai_text)

    # Determine overall status
    if ai_text:
        final_status = "completed"
        error_message = None
    else:
        final_status = "failed"
        error_message = "AI provider returned no response. Check API key and model availability."

    completed_at = _now() if final_status == "completed" else None

    # Combine limitations
    all_limitations = list(preprocessing.limitations)
    if not meta_a.is_geospatial or not meta_b.is_geospatial:
        all_limitations.append(
            "Without geospatial metadata, comparison is based solely on visual AI interpretation."
        )
    all_limitations.append(
        "This result is a qualitative AI visual comparison — "
        "not a calibrated remote-sensing spectral analysis."
    )

    # 13. Update MongoDB record
    update_fields = {
        "status": final_status,
        "preprocessing": preprocessing.model_dump(),
        "ai_explanation": ai_text,
        "detected_changes": [c.model_dump() for c in detected_changes],
        "limitations": all_limitations,
        "error_message": error_message,
        "completed_at": completed_at,
    }
    try:
        get_comparisons_collection().update_one(
            {"comparison_id": comparison_id},
            {"$set": update_fields},
        )
        logger.info(
            "Comparison %s updated: status=%s changes=%d",
            comparison_id, final_status, len(detected_changes),
        )
    except Exception as exc:
        logger.error("Failed to update comparison record %s: %s", comparison_id, exc)

    # 14. Write history record (Phase 8)
    history_service.create_history_item(
        session_id=session_id,
        record_type="comparison",
        analysis_type="change_detection",
        query=query_clean,
        answer=ai_text,
        image_refs=[image_id_a, image_id_b],
        status=final_status,
        comparison_id=comparison_id,
        error=error_message,
        limitations=all_limitations,
        detected_changes=[c.model_dump() for c in detected_changes],
        label_a=label_a,
        label_b=label_b,
        provider=provider.provider_name,
        model=provider.model_name,
    )

    # 15. Return structured response
    return ComparisonResponse(
        comparison_id=comparison_id,
        session_id=session_id,
        image_id_a=image_id_a,
        image_id_b=image_id_b,
        label_a=label_a,
        label_b=label_b,
        meta_a=meta_a,
        meta_b=meta_b,
        compatibility=compat,
        preprocessing=preprocessing,
        status=final_status,
        query=query_clean,
        ai_explanation=ai_text,
        detected_changes=detected_changes,
        limitations=all_limitations,
        error_message=error_message,
        provider=provider.provider_name,
        model=provider.model_name,
        created_at=now,
        completed_at=completed_at,
    )


# ─── Get comparison by ID ──────────────────────────────────────────────────────

def get_comparison_by_id(comparison_id: str) -> ComparisonResponse:
    """Fetch a saved comparison from MongoDB. Raises 404 if not found."""
    try:
        col = get_comparisons_collection()
        doc = col.find_one({"comparison_id": comparison_id}, {"_id": 0})
    except Exception as exc:
        logger.error("MongoDB comparison query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Comparison '{comparison_id}' not found",
        )

    try:
        return _doc_to_response(doc)
    except Exception as exc:
        logger.error("Failed to parse comparison document %s: %s", comparison_id, exc)
        raise HTTPException(status_code=500, detail="Comparison data corrupted")


# ─── MongoDB helpers ──────────────────────────────────────────────────────────

def _insert_comparison_doc(doc: dict) -> None:
    """Insert a comparison document. Non-fatal on MongoDB error (log + continue)."""
    try:
        get_comparisons_collection().insert_one({**doc})
        logger.debug("Comparison record inserted: %s", doc.get("comparison_id"))
    except Exception as exc:
        logger.error("Failed to insert comparison record: %s", exc)


def _update_comparison_failed(comparison_id: str, error: str) -> None:
    """Mark a comparison as failed in MongoDB."""
    try:
        get_comparisons_collection().update_one(
            {"comparison_id": comparison_id},
            {"$set": {
                "status": "failed",
                "error_message": error,
                "completed_at": _now(),
            }},
        )
    except Exception as exc:
        logger.error("Failed to mark comparison %s as failed: %s", comparison_id, exc)


def _doc_to_response(doc: dict) -> ComparisonResponse:
    """Convert a raw MongoDB document to a ComparisonResponse."""
    meta_a_raw = doc.get("meta_a", {})
    meta_b_raw = doc.get("meta_b", {})
    compat_raw = doc.get("compatibility", {})
    prep_raw = doc.get("preprocessing", {})

    meta_a = ImageMetaSummary(**meta_a_raw) if meta_a_raw else ImageMetaSummary(
        image_id=doc.get("image_id_a", ""),
        filename="", width=0, height=0, format="",
        mime_type="", is_geospatial=False,
    )
    meta_b = ImageMetaSummary(**meta_b_raw) if meta_b_raw else ImageMetaSummary(
        image_id=doc.get("image_id_b", ""),
        filename="", width=0, height=0, format="",
        mime_type="", is_geospatial=False,
    )
    compat = CompatibilityResult(**compat_raw) if compat_raw else CompatibilityResult(compatible=True)
    prep = PreprocessingResult(**prep_raw) if prep_raw else PreprocessingResult()

    changes = [
        DetectedChange(**c) for c in (doc.get("detected_changes") or [])
        if isinstance(c, dict)
    ]

    created_at = doc.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    if created_at is None:
        created_at = _now()

    completed_at = doc.get("completed_at")
    if isinstance(completed_at, str):
        completed_at = datetime.fromisoformat(completed_at)

    return ComparisonResponse(
        comparison_id=doc["comparison_id"],
        session_id=doc.get("session_id"),
        image_id_a=doc["image_id_a"],
        image_id_b=doc["image_id_b"],
        label_a=doc.get("label_a", "Image A"),
        label_b=doc.get("label_b", "Image B"),
        meta_a=meta_a,
        meta_b=meta_b,
        compatibility=compat,
        preprocessing=prep,
        status=doc.get("status", "unknown"),
        query=doc.get("query", ""),
        ai_explanation=doc.get("ai_explanation"),
        detected_changes=changes,
        limitations=doc.get("limitations", []),
        error_message=doc.get("error_message"),
        provider=doc.get("provider", "unknown"),
        model=doc.get("model", "unknown"),
        created_at=created_at,
        completed_at=completed_at,
    )

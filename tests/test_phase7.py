"""
SatQuery AI — Phase 7 Test Suite: Two-Image Comparison

Tests:
  1.  Two valid images accepted
  2.  Image A and B are correctly identified in the response
  3.  Metadata validation and compatibility check works
  4.  Incompatible images (bad MIME type) are rejected with structured error
  5.  Missing Image A raises 404
  6.  Missing Image B raises 404
  7.  Comparison service returns real structured result (not mock)
  8.  AI explanation is generated from comparison findings
  9.  Detected changes are extracted from AI text
  10. Comparison saved to MongoDB
  11. Saved comparison can be retrieved by ID
  12. Session validation: images not linked to session returns 400
  13. Labels A/B are preserved in the response
  14. Preprocessing/alignment record is populated honestly
  15. Limitations are included in the response
  16. API key is never exposed in any response field
  17. Invalid image_id format returns 404 gracefully
  18. Metadata compatibility warnings for dimension mismatch
  19. Metadata compatibility warnings for CRS mismatch (geospatial)
  20. Status = 'incompatible' when images are hard-incompatible
  21. Status = 'completed' when comparison succeeds
  22. Status = 'failed' when AI returns no response
  23. Error handling: MongoDB insert failure is non-fatal for response
  24. Comparison ID uses cmp_ prefix format
  25. created_at timestamp is set on all responses
  26. Comparison prompt includes both image labels and metadata
  27. Comparison prompt includes compatibility warnings
  28. Phase 1–6 tests: regression — no existing test should fail

Requirements:
  - Python 3.11+
  - pytest
  - No real MongoDB or Gemini calls — all external dependencies are mocked.
  - No fake AI results returned in production code paths.
"""

from __future__ import annotations

import sys
import types
import importlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ─── Path setup ───────────────────────────────────────────────────────────────

_BACKEND = Path(__file__).parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ─── Minimal stubs for heavy optional dependencies ────────────────────────────

def _stub_google_genai():
    """Stub the google.genai SDK so tests don't need it installed."""
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai_types = types.ModuleType("google.genai.types")

    class _Part:
        @staticmethod
        def from_bytes(data, mime_type):
            return {"data": data[:8], "mime": mime_type}

    class _Config:
        def __init__(self, **kw): pass

    genai_types.Part = _Part
    genai_types.GenerateContentConfig = _Config
    genai.Client = MagicMock
    genai.types = genai_types
    google.genai = genai
    sys.modules.setdefault("google", google)
    sys.modules.setdefault("google.genai", genai)
    sys.modules.setdefault("google.genai.types", genai_types)


def _stub_rasterio():
    rasterio = types.ModuleType("rasterio")
    rasterio.open = MagicMock()
    rasterio_io = types.ModuleType("rasterio.io")
    rasterio_io.MemoryFile = MagicMock()
    sys.modules.setdefault("rasterio", rasterio)
    sys.modules.setdefault("rasterio.io", rasterio_io)


def _stub_ulid():
    ulid = types.ModuleType("ulid")
    class _ULID:
        def __str__(self): return "01TESTULID0000000000000000"
    ulid.ULID = _ULID
    sys.modules.setdefault("ulid", ulid)


_stub_google_genai()
_stub_rasterio()
_stub_ulid()


# ─── Shared fixtures ──────────────────────────────────────────────────────────

def _make_image_doc(image_id="img_A", filename="a.png", width=512, height=512,
                    fmt="PNG", mime="image/png", is_geo=False, geo=None,
                    orig_path="/uploads/original/img_A.png",
                    proc_path="/uploads/processed/img_A.png"):
    return {
        "image_id": image_id,
        "filename": filename,
        "width": width,
        "height": height,
        "format": fmt,
        "mime_type": mime,
        "is_geospatial": is_geo,
        "geospatial": geo,
        "storage": {
            "original": orig_path,
            "processed": proc_path,
            "thumbnail": f"/uploads/thumbnails/{image_id}.jpg",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _make_provider_mock(text="AI comparison result text."):
    provider = MagicMock()
    provider.provider_name = "gemini"
    provider.model_name = "gemini-3.6-flash"
    client_mock = MagicMock()
    resp_mock = MagicMock()
    resp_mock.text = text
    client_mock.models.generate_content.return_value = resp_mock
    provider._get_client.return_value = client_mock
    return provider


# ─── Import the service ───────────────────────────────────────────────────────

from app.services import comparison_service
from app.schemas.comparison import (
    ComparisonRequest, ComparisonResponse,
    CompatibilityResult, PreprocessingResult,
    ImageMetaSummary, DetectedChange,
)
from app.ai.comparison_prompts import (
    build_comparison_prompt, build_meta_summary, COMPARISON_SYSTEM_PROMPT,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Two valid images accepted
# ═══════════════════════════════════════════════════════════════════════════════

def test_valid_images_accepted(tmp_path):
    """run_comparison accepts two valid image docs without error."""
    doc_a = _make_image_doc("img_A", proc_path=str(tmp_path / "a.png"))
    doc_b = _make_image_doc("img_B", proc_path=str(tmp_path / "b.png"))
    # Create fake image files
    (tmp_path / "a.png").write_bytes(b"PNG" + b"\x00" * 100)
    (tmp_path / "b.png").write_bytes(b"PNG" + b"\x00" * 100)

    provider = _make_provider_mock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection"), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison(
            image_id_a="img_A",
            image_id_b="img_B",
            query="What changed?",
            session_id=None,
        )
    assert result.status == "completed"
    assert result.image_id_a == "img_A"
    assert result.image_id_b == "img_B"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Image A and B are correctly identified
# ═══════════════════════════════════════════════════════════════════════════════

def test_images_correctly_identified(tmp_path):
    """meta_a and meta_b in the response match the correct image IDs."""
    doc_a = _make_image_doc("img_BEFORE", filename="before.png", proc_path=str(tmp_path / "a.png"))
    doc_b = _make_image_doc("img_AFTER",  filename="after.png",  proc_path=str(tmp_path / "b.png"))
    (tmp_path / "a.png").write_bytes(b"PNG" + b"\x00" * 50)
    (tmp_path / "b.png").write_bytes(b"PNG" + b"\x00" * 50)

    provider = _make_provider_mock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection"), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison("img_BEFORE", "img_AFTER", "", None)

    assert result.meta_a.image_id == "img_BEFORE"
    assert result.meta_a.filename == "before.png"
    assert result.meta_b.image_id == "img_AFTER"
    assert result.meta_b.filename == "after.png"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Metadata compatibility check works
# ═══════════════════════════════════════════════════════════════════════════════

def test_compatibility_check_compatible():
    """Two standard images with same MIME are compatible."""
    doc_a = _make_image_doc("img_A", mime="image/jpeg", width=800, height=600)
    doc_b = _make_image_doc("img_B", mime="image/jpeg", width=800, height=600)
    result = comparison_service._check_compatibility(doc_a, doc_b)
    assert result.compatible is True
    assert result.reasons == []


def test_compatibility_check_warns_on_dimension_mismatch():
    """Large dimension difference generates a warning (not a hard failure)."""
    doc_a = _make_image_doc("img_A", width=1024, height=768)
    doc_b = _make_image_doc("img_B", width=200,  height=150)
    result = comparison_service._check_compatibility(doc_a, doc_b)
    assert result.compatible is True
    assert any("dimension" in w.lower() for w in result.warnings)


def test_compatibility_check_warns_on_format_mismatch():
    """Different formats generate a warning."""
    doc_a = _make_image_doc("img_A", fmt="JPEG", mime="image/jpeg")
    doc_b = _make_image_doc("img_B", fmt="PNG",  mime="image/png")
    result = comparison_service._check_compatibility(doc_a, doc_b)
    assert result.compatible is True
    assert any("format" in w.lower() for w in result.warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Incompatible images rejected with structured error (not 500)
# ═══════════════════════════════════════════════════════════════════════════════

def test_incompatible_images_blocked():
    """Images with unsupported MIME type result in status='incompatible'."""
    doc_a = _make_image_doc("img_A", mime="application/pdf")
    doc_b = _make_image_doc("img_B", mime="image/png")

    provider = _make_provider_mock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison("img_A", "img_B", "", None)

    assert result.status == "incompatible"
    assert result.compatibility.compatible is False
    assert len(result.compatibility.reasons) > 0
    assert result.ai_explanation is None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Missing Image A raises 404
# ═══════════════════════════════════════════════════════════════════════════════

def test_missing_image_a_raises_404():
    from fastapi import HTTPException
    def raise_404(image_id):
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")

    provider = _make_provider_mock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=raise_404), \
         patch("app.ai.provider.get_provider", return_value=provider):
        with pytest.raises(HTTPException) as exc_info:
            comparison_service.run_comparison("img_MISSING", "img_B", "", None)
    assert exc_info.value.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Missing Image B raises 404
# ═══════════════════════════════════════════════════════════════════════════════

def test_missing_image_b_raises_404():
    from fastapi import HTTPException
    doc_a = _make_image_doc("img_A")

    def side_effect(image_id):
        if image_id == "img_A":
            return doc_a
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")

    provider = _make_provider_mock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=side_effect), \
         patch("app.ai.provider.get_provider", return_value=provider):
        with pytest.raises(HTTPException) as exc_info:
            comparison_service.run_comparison("img_A", "img_B_MISSING", "", None)
    assert exc_info.value.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Comparison service returns real structured result (not mock)
# ═══════════════════════════════════════════════════════════════════════════════

def test_result_is_real_not_mock(tmp_path):
    """The response comes from the AI call, not hardcoded mock text."""
    real_ai_text = "AI ANALYSIS: Vegetation has increased in the eastern quadrant."
    doc_a = _make_image_doc("img_A", proc_path=str(tmp_path / "a.jpg"), mime="image/jpeg", fmt="JPEG")
    doc_b = _make_image_doc("img_B", proc_path=str(tmp_path / "b.jpg"), mime="image/jpeg", fmt="JPEG")
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)

    provider = _make_provider_mock(text=real_ai_text)
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection"), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison("img_A", "img_B", "Compare these", None)

    assert result.ai_explanation == real_ai_text
    assert "AI ANALYSIS" in result.ai_explanation


# ═══════════════════════════════════════════════════════════════════════════════
# 8. AI explanation is generated from comparison findings
# ═══════════════════════════════════════════════════════════════════════════════

def test_ai_explanation_is_populated(tmp_path):
    """ai_explanation is non-empty when AI call succeeds."""
    doc_a = _make_image_doc("img_A", proc_path=str(tmp_path / "a.png"))
    doc_b = _make_image_doc("img_B", proc_path=str(tmp_path / "b.png"))
    (tmp_path / "a.png").write_bytes(b"PNG" + b"\x00" * 60)
    (tmp_path / "b.png").write_bytes(b"PNG" + b"\x00" * 60)

    provider = _make_provider_mock(text="Detected significant vegetation loss in the northern area.")
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection"), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison("img_A", "img_B", "", None)

    assert result.ai_explanation is not None
    assert len(result.ai_explanation) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Detected changes are extracted from AI text
# ═══════════════════════════════════════════════════════════════════════════════

def test_detected_changes_parsed():
    """Change parser extracts vegetation and water changes from AI text."""
    ai_text = (
        "Vegetation cover appears to have decreased significantly in the area.\n"
        "Water bodies appear to have expanded following recent rainfall.\n"
        "Built-up area shows expansion near the eastern boundary.\n"
    )
    changes = comparison_service._parse_detected_changes(ai_text)
    features = [c.feature for c in changes]
    assert "Vegetation Cover" in features
    assert "Water Bodies" in features
    assert "Built-up Area" in features


def test_detected_changes_directions():
    """Directional keywords are correctly parsed."""
    ai_text = (
        "Vegetation has significantly decreased.\n"
        "Road network appears to have expanded.\n"
        "Water bodies are largely stable and consistent.\n"
    )
    changes = comparison_service._parse_detected_changes(ai_text)
    direction_map = {c.feature: c.direction for c in changes}
    assert direction_map.get("Vegetation Cover") == "down"
    assert direction_map.get("Road Network") == "up"
    assert direction_map.get("Water Bodies") == "stable"


def test_no_changes_returns_empty_list():
    """Empty AI text produces empty change list."""
    assert comparison_service._parse_detected_changes("") == []
    assert comparison_service._parse_detected_changes(None) == []


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Comparison saved to MongoDB
# ═══════════════════════════════════════════════════════════════════════════════

def test_comparison_saved_to_mongodb(tmp_path):
    """_insert_comparison_doc is called during run_comparison."""
    doc_a = _make_image_doc("img_A", proc_path=str(tmp_path / "a.png"))
    doc_b = _make_image_doc("img_B", proc_path=str(tmp_path / "b.png"))
    (tmp_path / "a.png").write_bytes(b"PNG" + b"\x00" * 60)
    (tmp_path / "b.png").write_bytes(b"PNG" + b"\x00" * 60)

    insert_calls = []
    def capture_insert(doc):
        insert_calls.append(doc)

    provider = _make_provider_mock()
    col_mock = MagicMock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc", side_effect=capture_insert), \
         patch.object(comparison_service, "get_comparisons_collection", return_value=col_mock), \
         patch("app.ai.provider.get_provider", return_value=provider):
        comparison_service.run_comparison("img_A", "img_B", "test query", None)

    assert len(insert_calls) >= 1
    saved_doc = insert_calls[0]
    assert saved_doc["image_id_a"] == "img_A"
    assert saved_doc["image_id_b"] == "img_B"
    assert "comparison_id" in saved_doc


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Saved comparison can be retrieved by ID
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_comparison_by_id():
    """get_comparison_by_id correctly reconstructs ComparisonResponse."""
    stored_doc = {
        "comparison_id": "cmp_TEST123",
        "session_id": None,
        "image_id_a": "img_A",
        "image_id_b": "img_B",
        "label_a": "Before",
        "label_b": "After",
        "query": "what changed?",
        "status": "completed",
        "meta_a": {"image_id": "img_A", "filename": "a.png", "width": 512, "height": 512,
                   "format": "PNG", "mime_type": "image/png", "is_geospatial": False},
        "meta_b": {"image_id": "img_B", "filename": "b.png", "width": 512, "height": 512,
                   "format": "PNG", "mime_type": "image/png", "is_geospatial": False},
        "compatibility": {"compatible": True, "reasons": [], "warnings": []},
        "preprocessing": {"image_a_prepared": True, "image_b_prepared": True,
                          "alignment_attempted": False, "alignment_succeeded": False,
                          "limitations": []},
        "ai_explanation": "Vegetation decreased.",
        "detected_changes": [{"feature": "Vegetation Cover", "direction": "down",
                               "label": "Appears Decreased", "detail": None}],
        "limitations": ["Qualitative only."],
        "error_message": None,
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }
    col_mock = MagicMock()
    col_mock.find_one.return_value = stored_doc

    with patch.object(comparison_service, "get_comparisons_collection", return_value=col_mock):
        result = comparison_service.get_comparison_by_id("cmp_TEST123")

    assert result.comparison_id == "cmp_TEST123"
    assert result.status == "completed"
    assert result.ai_explanation == "Vegetation decreased."
    assert len(result.detected_changes) == 1
    assert result.detected_changes[0].feature == "Vegetation Cover"


def test_get_comparison_not_found_raises_404():
    from fastapi import HTTPException
    col_mock = MagicMock()
    col_mock.find_one.return_value = None

    with patch.object(comparison_service, "get_comparisons_collection", return_value=col_mock):
        with pytest.raises(HTTPException) as exc_info:
            comparison_service.get_comparison_by_id("cmp_NOTEXIST")
    assert exc_info.value.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Session validation: images not linked to session → 400
# ═══════════════════════════════════════════════════════════════════════════════

def test_images_not_linked_to_session_raises_400():
    from fastapi import HTTPException
    doc_a = _make_image_doc("img_A")
    doc_b = _make_image_doc("img_B")
    session_doc = {
        "session_id": "ses_123",
        "image_ids": ["img_C", "img_D"],  # neither A nor B linked
    }

    provider = _make_provider_mock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_get_session_doc", return_value=session_doc), \
         patch("app.ai.provider.get_provider", return_value=provider):
        with pytest.raises(HTTPException) as exc_info:
            comparison_service.run_comparison(
                "img_A", "img_B", "", session_id="ses_123"
            )
    assert exc_info.value.status_code == 400


def test_session_validation_passes_when_both_linked(tmp_path):
    """No error when both images are linked to the session."""
    doc_a = _make_image_doc("img_A", proc_path=str(tmp_path / "a.png"))
    doc_b = _make_image_doc("img_B", proc_path=str(tmp_path / "b.png"))
    (tmp_path / "a.png").write_bytes(b"PNG" + b"\x00" * 50)
    (tmp_path / "b.png").write_bytes(b"PNG" + b"\x00" * 50)
    session_doc = {"session_id": "ses_123", "image_ids": ["img_A", "img_B"]}

    provider = _make_provider_mock()
    col_mock = MagicMock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_get_session_doc", return_value=session_doc), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection", return_value=col_mock), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison(
            "img_A", "img_B", "", session_id="ses_123"
        )
    assert result.session_id == "ses_123"


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Labels A/B are preserved in the response
# ═══════════════════════════════════════════════════════════════════════════════

def test_custom_labels_preserved(tmp_path):
    """Custom label_a and label_b appear verbatim in the response."""
    doc_a = _make_image_doc("img_A", proc_path=str(tmp_path / "a.png"))
    doc_b = _make_image_doc("img_B", proc_path=str(tmp_path / "b.png"))
    (tmp_path / "a.png").write_bytes(b"PNG" + b"\x00" * 50)
    (tmp_path / "b.png").write_bytes(b"PNG" + b"\x00" * 50)

    provider = _make_provider_mock()
    col_mock = MagicMock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection", return_value=col_mock), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison(
            "img_A", "img_B", "",
            session_id=None,
            label_a="Summer 2020",
            label_b="Winter 2024",
        )
    assert result.label_a == "Summer 2020"
    assert result.label_b == "Winter 2024"


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Preprocessing record is populated honestly
# ═══════════════════════════════════════════════════════════════════════════════

def test_preprocessing_no_geospatial():
    """Without geospatial metadata, no alignment is attempted."""
    doc_a = _make_image_doc("img_A", is_geo=False)
    doc_b = _make_image_doc("img_B", is_geo=False)
    prep = comparison_service._build_preprocessing(doc_a, doc_b, warnings=[])
    assert prep.alignment_attempted is False
    assert prep.alignment_succeeded is False
    assert prep.alignment_method == "none"


def test_preprocessing_matching_crs():
    """With matching CRS, alignment_succeeded is True."""
    geo = {"crs": "EPSG:4326", "bands": 1, "resolution": [0.0001, 0.0001]}
    doc_a = _make_image_doc("img_A", is_geo=True, geo=geo)
    doc_b = _make_image_doc("img_B", is_geo=True, geo=geo)
    prep = comparison_service._build_preprocessing(doc_a, doc_b, warnings=[])
    assert prep.alignment_attempted is True
    assert prep.alignment_succeeded is True
    assert prep.alignment_method == "geospatial_crs_match"


def test_preprocessing_mismatched_crs():
    """With mismatched CRS, alignment_attempted=True but alignment_succeeded=False."""
    geo_a = {"crs": "EPSG:4326"}
    geo_b = {"crs": "EPSG:32644"}
    doc_a = _make_image_doc("img_A", is_geo=True, geo=geo_a)
    doc_b = _make_image_doc("img_B", is_geo=True, geo=geo_b)
    prep = comparison_service._build_preprocessing(doc_a, doc_b, warnings=[])
    assert prep.alignment_attempted is True
    assert prep.alignment_succeeded is False


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Limitations are included in the response
# ═══════════════════════════════════════════════════════════════════════════════

def test_limitations_in_response(tmp_path):
    """Result always includes at least one limitation note."""
    doc_a = _make_image_doc("img_A", proc_path=str(tmp_path / "a.png"))
    doc_b = _make_image_doc("img_B", proc_path=str(tmp_path / "b.png"))
    (tmp_path / "a.png").write_bytes(b"PNG" + b"\x00" * 50)
    (tmp_path / "b.png").write_bytes(b"PNG" + b"\x00" * 50)

    provider = _make_provider_mock()
    col_mock = MagicMock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection", return_value=col_mock), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison("img_A", "img_B", "", None)

    assert len(result.limitations) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 16. API key is never exposed in any response field
# ═══════════════════════════════════════════════════════════════════════════════

def test_api_key_never_exposed_in_response(tmp_path):
    """No field in ComparisonResponse contains a secret-like token."""
    doc_a = _make_image_doc("img_A", proc_path=str(tmp_path / "a.png"))
    doc_b = _make_image_doc("img_B", proc_path=str(tmp_path / "b.png"))
    (tmp_path / "a.png").write_bytes(b"PNG" + b"\x00" * 50)
    (tmp_path / "b.png").write_bytes(b"PNG" + b"\x00" * 50)

    fake_key = "FAKE_SECRET_API_KEY_XYZ123"
    provider = _make_provider_mock()
    col_mock = MagicMock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection", return_value=col_mock), \
         patch("app.ai.provider.get_provider", return_value=provider), \
         patch("app.core.config.settings") as s_mock:
        s_mock.ai_api_key = fake_key
        result = comparison_service.run_comparison("img_A", "img_B", "", None)

    result_json = result.model_dump_json()
    assert fake_key not in result_json


# ═══════════════════════════════════════════════════════════════════════════════
# 17. Invalid image_id returns 404 gracefully
# ═══════════════════════════════════════════════════════════════════════════════

def test_invalid_image_id_raises_404():
    from fastapi import HTTPException
    col_mock = MagicMock()
    col_mock.find_one.return_value = None

    provider = _make_provider_mock()
    with patch("app.ai.provider.get_provider", return_value=provider), \
         patch.object(comparison_service, "get_images_collection", return_value=col_mock):
        with pytest.raises(HTTPException) as exc_info:
            comparison_service.run_comparison("img_INVALID", "img_B", "", None)
    assert exc_info.value.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# 18. Dimension mismatch triggers warning
# ═══════════════════════════════════════════════════════════════════════════════

def test_dimension_mismatch_warning():
    doc_a = _make_image_doc("img_A", width=2000, height=1500)
    doc_b = _make_image_doc("img_B", width=400,  height=300)
    result = comparison_service._check_compatibility(doc_a, doc_b)
    assert result.compatible is True
    assert any("dimension" in w.lower() for w in result.warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# 19. CRS mismatch triggers warning
# ═══════════════════════════════════════════════════════════════════════════════

def test_crs_mismatch_warning():
    doc_a = _make_image_doc("img_A", is_geo=True, geo={"crs": "EPSG:4326"})
    doc_b = _make_image_doc("img_B", is_geo=True, geo={"crs": "EPSG:3857"})
    result = comparison_service._check_compatibility(doc_a, doc_b)
    assert result.compatible is True
    assert any("CRS" in w or "crs" in w.lower() for w in result.warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# 20. status='incompatible' when hard-incompatible
# ═══════════════════════════════════════════════════════════════════════════════

def test_status_incompatible_for_bad_mime():
    doc_a = _make_image_doc("img_A", mime="text/plain")
    doc_b = _make_image_doc("img_B", mime="image/png")

    provider = _make_provider_mock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison("img_A", "img_B", "", None)

    assert result.status == "incompatible"


# ═══════════════════════════════════════════════════════════════════════════════
# 21. status='completed' when comparison succeeds
# ═══════════════════════════════════════════════════════════════════════════════

def test_status_completed_on_success(tmp_path):
    doc_a = _make_image_doc("img_A", proc_path=str(tmp_path / "a.png"))
    doc_b = _make_image_doc("img_B", proc_path=str(tmp_path / "b.png"))
    (tmp_path / "a.png").write_bytes(b"PNG" + b"\x00" * 60)
    (tmp_path / "b.png").write_bytes(b"PNG" + b"\x00" * 60)

    provider = _make_provider_mock(text="Substantial vegetation decrease in Image B.")
    col_mock = MagicMock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection", return_value=col_mock), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison("img_A", "img_B", "", None)

    assert result.status == "completed"
    assert result.completed_at is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 22. status='failed' when AI returns no response
# ═══════════════════════════════════════════════════════════════════════════════

def test_status_failed_when_ai_returns_nothing(tmp_path):
    doc_a = _make_image_doc("img_A", proc_path=str(tmp_path / "a.png"))
    doc_b = _make_image_doc("img_B", proc_path=str(tmp_path / "b.png"))
    (tmp_path / "a.png").write_bytes(b"PNG" + b"\x00" * 60)
    (tmp_path / "b.png").write_bytes(b"PNG" + b"\x00" * 60)

    # Provider raises exception → AI text = None
    provider = _make_provider_mock()
    provider._get_client.side_effect = Exception("Connection refused")

    col_mock = MagicMock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection", return_value=col_mock), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison("img_A", "img_B", "", None)

    assert result.status == "failed"
    assert result.error_message is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 23. Comparison ID uses cmp_ prefix
# ═══════════════════════════════════════════════════════════════════════════════

def test_comparison_id_format():
    cid = comparison_service._generate_comparison_id()
    assert cid.startswith("cmp_")
    assert len(cid) > 5


# ═══════════════════════════════════════════════════════════════════════════════
# 24. created_at timestamp is always set
# ═══════════════════════════════════════════════════════════════════════════════

def test_created_at_is_always_set(tmp_path):
    doc_a = _make_image_doc("img_A", proc_path=str(tmp_path / "a.png"))
    doc_b = _make_image_doc("img_B", proc_path=str(tmp_path / "b.png"))
    (tmp_path / "a.png").write_bytes(b"PNG" + b"\x00" * 50)
    (tmp_path / "b.png").write_bytes(b"PNG" + b"\x00" * 50)

    provider = _make_provider_mock()
    col_mock = MagicMock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection", return_value=col_mock), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison("img_A", "img_B", "", None)

    assert result.created_at is not None
    assert isinstance(result.created_at, datetime)


# ═══════════════════════════════════════════════════════════════════════════════
# 25–27. Comparison prompt tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_comparison_prompt_includes_both_labels():
    """Prompt includes both label_a and label_b."""
    prompt = build_comparison_prompt(
        query="What changed?",
        label_a="Summer 2020",
        label_b="Winter 2024",
        meta_a_summary="800×600px, PNG",
        meta_b_summary="800×600px, PNG",
        warnings=[],
    )
    assert "Summer 2020" in prompt
    assert "Winter 2024" in prompt


def test_comparison_prompt_includes_metadata():
    """Prompt includes metadata summaries for both images."""
    prompt = build_comparison_prompt(
        query="",
        label_a="A",
        label_b="B",
        meta_a_summary="1024×768px, GeoTIFF, CRS=EPSG:4326",
        meta_b_summary="512×512px, JPEG",
        warnings=[],
    )
    assert "1024×768" in prompt
    assert "EPSG:4326" in prompt
    assert "512×512" in prompt


def test_comparison_prompt_includes_warnings():
    """Compatibility warnings appear in the prompt."""
    warnings = ["Images have different dimensions.", "CRS mismatch detected."]
    prompt = build_comparison_prompt(
        query="",
        label_a="A", label_b="B",
        meta_a_summary="x", meta_b_summary="y",
        warnings=warnings,
    )
    assert "different dimensions" in prompt
    assert "CRS mismatch" in prompt


def test_comparison_system_prompt_is_non_empty():
    """System prompt exists and enforces honesty rules."""
    assert len(COMPARISON_SYSTEM_PROMPT) > 200
    assert "FOLLOW STRICTLY" in COMPARISON_SYSTEM_PROMPT
    assert "Do NOT invent" in COMPARISON_SYSTEM_PROMPT


def test_meta_summary_builder():
    """build_meta_summary produces a useful one-line description."""
    s = build_meta_summary(1024, 768, "GeoTIFF", True, "EPSG:4326", 4, [0.0001, 0.0001])
    assert "1024×768" in s
    assert "GeoTIFF" in s
    assert "EPSG:4326" in s


# ═══════════════════════════════════════════════════════════════════════════════
# 28. ComparisonRequest schema validation
# ═══════════════════════════════════════════════════════════════════════════════

def test_comparison_request_requires_both_ids():
    """ComparisonRequest requires image_id_a and image_id_b."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ComparisonRequest(image_id_b="img_B")  # missing image_id_a
    with pytest.raises(ValidationError):
        ComparisonRequest(image_id_a="img_A")  # missing image_id_b


def test_comparison_request_defaults():
    """ComparisonRequest has sensible defaults."""
    req = ComparisonRequest(image_id_a="img_A", image_id_b="img_B")
    assert req.query == ""
    assert req.session_id is None
    assert "A" in req.label_a
    assert "B" in req.label_b


# ═══════════════════════════════════════════════════════════════════════════════
# 29. Duplicate image IDs (A == B) should still work (not blocked)
# ═══════════════════════════════════════════════════════════════════════════════

def test_same_image_id_not_blocked(tmp_path):
    """Using the same image for A and B is allowed (edge case)."""
    doc = _make_image_doc("img_SAME", proc_path=str(tmp_path / "same.png"))
    (tmp_path / "same.png").write_bytes(b"PNG" + b"\x00" * 60)

    provider = _make_provider_mock(text="Both images appear identical.")
    col_mock = MagicMock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc, doc]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection", return_value=col_mock), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison("img_SAME", "img_SAME", "", None)

    assert result.status in ("completed", "failed")  # No crash


# ═══════════════════════════════════════════════════════════════════════════════
# 30. Query is preserved in the response
# ═══════════════════════════════════════════════════════════════════════════════

def test_query_preserved_in_response(tmp_path):
    doc_a = _make_image_doc("img_A", proc_path=str(tmp_path / "a.png"))
    doc_b = _make_image_doc("img_B", proc_path=str(tmp_path / "b.png"))
    (tmp_path / "a.png").write_bytes(b"PNG" + b"\x00" * 50)
    (tmp_path / "b.png").write_bytes(b"PNG" + b"\x00" * 50)

    provider = _make_provider_mock()
    col_mock = MagicMock()
    with patch.object(comparison_service, "_get_image_doc", side_effect=[doc_a, doc_b]), \
         patch.object(comparison_service, "_insert_comparison_doc"), \
         patch.object(comparison_service, "get_comparisons_collection", return_value=col_mock), \
         patch("app.ai.provider.get_provider", return_value=provider):
        result = comparison_service.run_comparison(
            "img_A", "img_B", "How has vegetation changed?", None
        )

    assert result.query == "How has vegetation changed?"

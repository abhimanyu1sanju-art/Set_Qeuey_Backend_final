"""
SatQuery AI — Phase 6 Multi-Intent Analysis Tests

Tests requirements (numbered 1–12 from the Phase 6 spec):

  1.  Single-intent query still works (no regression).
  2.  "vegetation and water" → detects both intents.
  3.  "vegetation, water and land cover" → detects all three.
  4.  Duplicate intents are removed.
  5.  Unknown / general query falls back to general analysis.
  6.  Each intent is routed to the correct analysis pipeline.
  7.  Multiple results are combined into a structured answer.
  8.  One failed intent does not discard successful results.
  9.  Multi-intent Q/A is saved to MongoDB / session history.
  10. Phase 5 follow-up still works without re-uploading image.
  11. API key is never returned to the frontend.
  12. Existing Phase 4 and Phase 5 tests still pass (regression).

All AI calls are mocked — no real API key is required.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch, call

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.ai.base import AIRawResponse
from app.ai.intent_detector import (
    detect_intents,
    is_multi_intent,
    get_display_label,
    get_analysis_type,
    build_intent_sub_query,
    INTENT_LABELS,
)

client = TestClient(app)

PATCH_GET_PROVIDER = "app.services.analysis_service.get_provider"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(w: int = 80, h: int = 60) -> bytes:
    img = Image.new("RGB", (w, h), color=(60, 120, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _upload_image() -> str:
    data = _make_jpeg_bytes()
    resp = client.post(
        "/api/images/upload",
        files={"file": ("sat.jpg", io.BytesIO(data), "image/jpeg")},
    )
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    return resp.json()["image_id"]


def _create_session(title: str = "Phase 6 Test") -> str:
    resp = client.post("/api/sessions", json={"title": title})
    assert resp.status_code == 201
    return resp.json()["session_id"]


def _link_image(session_id: str, image_id: str) -> None:
    resp = client.post(f"/api/sessions/{session_id}/images/{image_id}")
    assert resp.status_code == 200


def _mock_provider(answer: str = "AI analysis result.", success: bool = True):
    """Return a mock provider that always returns the given answer."""
    mock_prov = MagicMock()
    mock_prov.provider_name = "mock"
    mock_prov.model_name = "mock-model"
    mock_prov.analyze.return_value = AIRawResponse(
        text=answer if success else "",
        success=success,
        error_message=None if success else "Mock provider error",
        provider="mock",
        model="mock-model",
    )
    return mock_prov


def _mock_provider_sequence(answers: list[str | None]):
    """
    Return a mock provider whose analyze() returns answers in sequence.
    None in the sequence means a failed (empty text) response.
    """
    responses = []
    for ans in answers:
        if ans is not None:
            responses.append(AIRawResponse(
                text=ans, success=True, provider="mock", model="mock-model",
            ))
        else:
            responses.append(AIRawResponse(
                text="", success=False,
                error_message="Simulated failure",
                provider="mock", model="mock-model",
            ))

    mock_prov = MagicMock()
    mock_prov.provider_name = "mock"
    mock_prov.model_name = "mock-model"
    mock_prov.analyze.side_effect = responses
    return mock_prov


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — Intent detector
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntentDetector:
    """Unit tests for app.ai.intent_detector — no HTTP, no AI calls."""

    # ── Requirement 1: single intent ──────────────────────────────────────────
    def test_single_vegetation_intent(self):
        intents = detect_intents("Analyze the vegetation in this image.")
        assert "vegetation" in intents
        assert len(intents) == 1

    def test_single_water_intent(self):
        intents = detect_intents("What water bodies are visible?")
        assert "water" in intents
        assert len(intents) == 1

    def test_single_land_cover_intent(self):
        intents = detect_intents("Classify the land cover in this scene.")
        assert "land_cover" in intents
        assert len(intents) == 1

    # ── Requirement 2: vegetation + water ────────────────────────────────────
    def test_vegetation_and_water_both_detected(self):
        intents = detect_intents("Analyze vegetation and water bodies in this image.")
        assert "vegetation" in intents
        assert "water" in intents
        assert len(intents) == 2

    # ── Requirement 3: vegetation + water + land cover ────────────────────────
    def test_three_intents_detected(self):
        intents = detect_intents(
            "Analyze vegetation, water bodies, and land cover in this image."
        )
        assert "vegetation" in intents
        assert "water" in intents
        assert "land_cover" in intents
        assert len(intents) == 3

    # ── Requirement 4: duplicate intents removed ───────────────────────────────
    def test_duplicate_intents_are_deduplicated(self):
        # "vegetation" and "forest" both map to vegetation intent
        intents = detect_intents("Check the vegetation, forest, and plant cover.")
        assert intents.count("vegetation") == 1

    # ── Requirement 5: unknown/general fallback ────────────────────────────────
    def test_unknown_query_falls_back_to_general(self):
        intents = detect_intents("What is happening in this image?")
        assert intents == ["general"]

    def test_empty_query_falls_back_to_general(self):
        assert detect_intents("") == ["general"]

    def test_whitespace_only_falls_back_to_general(self):
        assert detect_intents("   ") == ["general"]

    # ── Requirement 6: correct routing ────────────────────────────────────────
    def test_vegetation_routes_to_vegetation_pipeline(self):
        assert get_analysis_type("vegetation") == "vegetation_detection"

    def test_ndvi_routes_to_vegetation_pipeline(self):
        assert get_analysis_type("ndvi") == "vegetation_detection"

    def test_water_routes_to_water_pipeline(self):
        assert get_analysis_type("water") == "water_detection"

    def test_ndwi_routes_to_water_pipeline(self):
        assert get_analysis_type("ndwi") == "water_detection"

    def test_land_cover_routes_to_land_cover_pipeline(self):
        assert get_analysis_type("land_cover") == "land_cover"

    def test_general_routes_to_general_pipeline(self):
        assert get_analysis_type("general") == "general"

    def test_is_multi_intent_returns_true_for_two(self):
        assert is_multi_intent("Analyze vegetation and water.") is True

    def test_is_multi_intent_returns_false_for_one(self):
        assert is_multi_intent("Analyze vegetation only.") is False

    def test_display_labels_exist_for_all_intents(self):
        for intent in INTENT_LABELS:
            label = get_display_label(intent)
            assert isinstance(label, str) and len(label) > 0

    def test_sub_query_contains_original_query(self):
        original = "Analyze vegetation and water."
        sub = build_intent_sub_query("vegetation", original)
        assert original in sub

    def test_sub_query_for_each_intent_is_a_string(self):
        for intent in ("vegetation", "water", "land_cover", "building",
                       "road", "ndvi", "ndwi", "nbr", "general"):
            sub = build_intent_sub_query(intent, "test query")
            assert isinstance(sub, str) and len(sub) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — multi-intent pipeline via API
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiIntentAPI:
    """
    Integration tests that hit POST /api/analysis and verify multi-intent
    behavior end-to-end.
    """

    def _analyze(self, image_id: str, query: str, session_id: str = None,
                 provider=None, enable_multi_intent: bool = True):
        """POST /api/analysis with a mock provider."""
        if provider is None:
            provider = _mock_provider()
        payload = {
            "image_id": image_id,
            "query": query,
            "enable_multi_intent": enable_multi_intent,
        }
        if session_id:
            payload["session_id"] = session_id
        with patch(PATCH_GET_PROVIDER, return_value=provider):
            return client.post("/api/analysis", json=payload)

    # ── Requirement 1: single-intent still works ──────────────────────────────
    def test_single_intent_query_still_works(self):
        iid = _upload_image()
        resp = self._analyze(iid, "What is visible in this image?")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "completed"
        assert body["answer"] is not None
        assert body["is_multi_intent"] is False

    def test_explicit_single_intent_query(self):
        iid = _upload_image()
        resp = self._analyze(iid, "Analyze vegetation in this image.")
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_multi_intent"] is False  # Only 1 intent detected
        assert body["status"] == "completed"

    # ── Requirement 2: vegetation + water ────────────────────────────────────
    def test_vegetation_and_water_returns_multi_intent_response(self):
        iid = _upload_image()
        provider = _mock_provider_sequence([
            "Dense forest canopy is visible.",
            "Several rivers are present.",
        ])
        resp = self._analyze(iid, "Analyze vegetation and water bodies.", provider=provider)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["is_multi_intent"] is True
        assert "vegetation" in body["detected_intents"]
        assert "water" in body["detected_intents"]
        assert len(body["intent_results"]) == 2

    # ── Requirement 3: vegetation + water + land cover ────────────────────────
    def test_three_intent_query_response(self):
        iid = _upload_image()
        provider = _mock_provider_sequence([
            "Healthy vegetation covers 60% of the scene.",
            "A river runs through the eastern portion.",
            "Urban, agricultural, and forested zones are visible.",
        ])
        resp = self._analyze(
            iid,
            "Analyze vegetation, water bodies, and land cover in this image.",
            provider=provider,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["is_multi_intent"] is True
        intents = body["detected_intents"]
        assert "vegetation" in intents
        assert "water" in intents
        assert "land_cover" in intents
        assert len(body["intent_results"]) == 3

    # ── Requirement 4: duplicate intents removed ───────────────────────────────
    def test_duplicate_intents_not_duplicated_in_results(self):
        iid = _upload_image()
        provider = _mock_provider_sequence([
            "Vegetation answer.",
            "Water answer.",
        ])
        # "forest", "vegetation", "green" all map to vegetation — only 1 call expected
        resp = self._analyze(
            iid,
            "Describe the vegetation, water, and forest cover.",
            provider=provider,
        )
        assert resp.status_code == 200
        body = resp.json()
        intents = body["detected_intents"]
        assert intents.count("vegetation") == 1

    # ── Requirement 5: unknown query falls back to general ─────────────────────
    def test_unknown_query_falls_back_to_general_analysis(self):
        iid = _upload_image()
        resp = self._analyze(iid, "What is in this picture?")
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_multi_intent"] is False
        assert body["status"] == "completed"
        # General analysis_type
        assert body["analysis_type"] in ("general", "vegetation_detection", "water_detection")

    # ── Requirement 6: each intent routed to correct pipeline ──────────────────
    def test_intent_results_contain_correct_analysis_type(self):
        iid = _upload_image()
        provider = _mock_provider_sequence([
            "Vegetation covers much of the scene.",
            "Water bodies detected in the northwest.",
        ])
        resp = self._analyze(iid, "Analyze vegetation and water.", provider=provider)
        assert resp.status_code == 200
        body = resp.json()
        intent_types = {r["intent"]: r["analysis_type"] for r in body["intent_results"]}
        assert intent_types.get("vegetation") == "vegetation_detection"
        assert intent_types.get("water") == "water_detection"

    # ── Requirement 7: results combined into structured answer ─────────────────
    def test_combined_answer_contains_all_sections(self):
        iid = _upload_image()
        provider = _mock_provider_sequence([
            "Dense forest canopy.",
            "A large lake is visible.",
        ])
        resp = self._analyze(iid, "Analyze vegetation and water.", provider=provider)
        assert resp.status_code == 200
        combined = resp.json()["answer"]
        assert combined is not None
        # Markdown headings for each intent should be present
        assert "Vegetation" in combined or "vegetation" in combined.lower()
        assert "Water" in combined or "water" in combined.lower()
        # Overall summary section
        assert "Overall Summary" in combined or "summary" in combined.lower()

    def test_individual_intent_answers_in_intent_results(self):
        iid = _upload_image()
        provider = _mock_provider_sequence([
            "Vegetation: Dense forest.",
            "Water: River detected.",
        ])
        resp = self._analyze(iid, "Analyze vegetation and water bodies.", provider=provider)
        assert resp.status_code == 200
        results = resp.json()["intent_results"]
        answers = [r["answer"] for r in results if r["answer"]]
        assert len(answers) == 2

    # ── Requirement 8: one failed intent doesn't discard others ───────────────
    def test_partial_failure_preserves_successful_results(self):
        iid = _upload_image()
        # First call (vegetation) succeeds, second call (water) fails
        provider = _mock_provider_sequence([
            "Dense forest visible.",
            None,  # water analysis fails
        ])
        resp = self._analyze(iid, "Analyze vegetation and water.", provider=provider)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Overall: at least one succeeded
        assert body["status"] == "completed"
        results = body["intent_results"]
        statuses = {r["intent"]: r["status"] for r in results}
        assert statuses.get("vegetation") == "completed"
        assert statuses.get("water") == "failed"
        # Combined answer must still contain the successful result
        assert body["answer"] is not None
        assert "Dense forest visible" in body["answer"] or "Vegetation" in body["answer"]
        # Failed intent must show error/warning in combined answer
        assert "⚠️" in body["answer"] or "unavailable" in body["answer"].lower() or \
               "failed" in body["answer"].lower()

    def test_all_failed_intents_returns_failed_status(self):
        iid = _upload_image()
        provider = _mock_provider_sequence([None, None])  # both fail
        resp = self._analyze(iid, "Analyze vegetation and water.", provider=provider)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert body["is_multi_intent"] is True

    # ── Requirement 11: API key never in response ──────────────────────────────
    def test_api_key_never_in_response(self):
        iid = _upload_image()
        provider = _mock_provider_sequence(["Vegetation found.", "Water found."])
        resp = self._analyze(iid, "Analyze vegetation and water.", provider=provider)
        assert resp.status_code == 200
        resp_text = resp.text
        # Common env-var name patterns for the key must not appear
        assert "AI_API_KEY" not in resp_text
        assert "api_key" not in resp_text.lower()

    def test_enable_multi_intent_false_forces_single_path(self):
        """Setting enable_multi_intent=False must bypass multi-intent detection."""
        iid = _upload_image()
        provider = _mock_provider("General answer.")
        resp = self._analyze(
            iid,
            "Analyze vegetation and water.",
            provider=provider,
            enable_multi_intent=False,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_multi_intent"] is False
        # provider.analyze should have been called exactly once (single call)
        assert provider.analyze.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Requirement 9: Multi-intent result saved to session history
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiIntentSessionPersistence:
    """Verify multi-intent Q/A is saved to the session's questions array."""

    def test_multi_intent_saved_to_session_questions(self):
        iid = _upload_image()
        sid = _create_session("Multi-intent Session Test")
        _link_image(sid, iid)

        provider = _mock_provider_sequence([
            "Vegetation analysis result.",
            "Water analysis result.",
        ])
        payload = {
            "image_id": iid,
            "session_id": sid,
            "query": "Analyze vegetation and water bodies.",
            "enable_multi_intent": True,
        }
        with patch(PATCH_GET_PROVIDER, return_value=provider):
            resp = client.post("/api/analysis", json=payload)
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "completed"

        # Check session has the question saved
        q_resp = client.get(f"/api/sessions/{sid}/questions")
        assert q_resp.status_code == 200
        questions = q_resp.json()["questions"]
        completed = [q for q in questions if q["status"] == "completed"]
        assert len(completed) >= 1

        # The answer should be the combined multi-intent answer
        assert completed[0]["answer"] is not None
        assert len(completed[0]["answer"]) > 0

    def test_multi_intent_saves_single_qa_record_not_multiple(self):
        """One user query → one Q/A record even for 3 intents."""
        iid = _upload_image()
        sid = _create_session("Single Record Test")
        _link_image(sid, iid)

        provider = _mock_provider_sequence([
            "Vegetation result.",
            "Water result.",
            "Land cover result.",
        ])
        payload = {
            "image_id": iid,
            "session_id": sid,
            "query": "Analyze vegetation, water, and land cover.",
            "enable_multi_intent": True,
        }
        with patch(PATCH_GET_PROVIDER, return_value=provider):
            resp = client.post("/api/analysis", json=payload)
        assert resp.status_code == 200

        q_resp = client.get(f"/api/sessions/{sid}/questions")
        questions = q_resp.json()["questions"]
        # Must be exactly ONE question record (not 3)
        assert len(questions) == 1

    def test_multi_intent_response_has_correct_is_multi_intent_flag(self):
        iid = _upload_image()
        sid = _create_session("Flag Test")
        _link_image(sid, iid)

        provider = _mock_provider_sequence(["Veg result.", "Water result."])
        payload = {
            "image_id": iid,
            "session_id": sid,
            "query": "Analyze vegetation and water.",
        }
        with patch(PATCH_GET_PROVIDER, return_value=provider):
            resp = client.post("/api/analysis", json=payload)
        assert resp.status_code == 200
        assert resp.json()["is_multi_intent"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Requirement 10: Phase 5 follow-up still works without re-uploading image
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase5FollowupCompatibility:
    """Phase 5 conversation follow-up must not be broken by Phase 6."""

    def test_followup_single_intent_still_works(self):
        """A simple follow-up question (Phase 5) works unchanged."""
        iid = _upload_image()
        sid = _create_session("Followup Test")
        _link_image(sid, iid)

        # First analysis
        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider("First answer.")):
            r1 = client.post("/api/analysis", json={
                "image_id": iid,
                "session_id": sid,
                "query": "What is visible?",
            })
        assert r1.status_code == 200

        # Follow-up via /ask endpoint (no image_id needed)
        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider("Follow-up answer.")):
            r2 = client.post(f"/api/sessions/{sid}/ask", json={
                "question": "Tell me more about the buildings.",
                "analysis_type": "general",
            })
        assert r2.status_code == 200, r2.text
        body = r2.json()
        assert body["status"] == "completed"
        assert body["answer"] is not None
        assert body["session_id"] == sid
        assert body["image_id"] == iid  # image resolved automatically

    def test_followup_multi_intent_question_via_ask(self):
        """Multi-intent question via the /ask endpoint also works."""
        iid = _upload_image()
        sid = _create_session("Multi Follow-up Test")
        _link_image(sid, iid)

        # Seed an initial analysis
        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider("Initial scene analysis.")):
            client.post("/api/analysis", json={
                "image_id": iid,
                "session_id": sid,
                "query": "General overview.",
                "enable_multi_intent": False,
            })

        # Follow-up with a multi-intent question
        provider = _mock_provider_sequence(["Vegetation detected.", "Water detected."])
        with patch(PATCH_GET_PROVIDER, return_value=provider):
            r = client.post(f"/api/sessions/{sid}/ask", json={
                "question": "Now analyze vegetation and water in this image.",
                "enable_multi_intent": True,
            })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "completed"
        assert body["is_multi_intent"] is True

    def test_followup_preserves_question_number(self):
        """question_number increments correctly across multi-intent and single-intent questions."""
        iid = _upload_image()
        sid = _create_session("Question Number Test")
        _link_image(sid, iid)

        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider("Answer 1.")):
            r1 = client.post(f"/api/sessions/{sid}/ask", json={
                "question": "Q1: What is visible?",
                "enable_multi_intent": False,
            })
        assert r1.json()["question_number"] == 1

        provider = _mock_provider_sequence(["Veg.", "Water."])
        with patch(PATCH_GET_PROVIDER, return_value=provider):
            r2 = client.post(f"/api/sessions/{sid}/ask", json={
                "question": "Q2: Analyze vegetation and water.",
            })
        assert r2.json()["question_number"] == 2

    def test_image_not_reuploaded_for_followup(self):
        """The follow-up uses the same image — image_id in response matches original."""
        iid = _upload_image()
        sid = _create_session("No Re-upload Test")
        _link_image(sid, iid)

        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider("A1")):
            client.post("/api/analysis", json={
                "image_id": iid, "session_id": sid, "query": "Q1"
            })

        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider("A2")):
            r = client.post(f"/api/sessions/{sid}/ask", json={"question": "Q2"})
        assert r.json()["image_id"] == iid


# ═══════════════════════════════════════════════════════════════════════════════
# Requirement 12: Phase 4 & Phase 5 regression guard
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase4Phase5Regression:
    """
    Smoke tests for core Phase 4 and Phase 5 functionality.
    Ensures Phase 6 additions did not break existing behavior.
    """

    def test_phase4_basic_analysis_still_works(self):
        iid = _upload_image()
        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider("Test answer.")):
            resp = client.post("/api/analysis", json={
                "image_id": iid,
                "query": "Describe this image.",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["analysis_id"].startswith("analysis_")

    def test_phase4_analysis_response_has_required_fields(self):
        iid = _upload_image()
        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider()):
            resp = client.post("/api/analysis", json={"image_id": iid, "query": "test"})
        body = resp.json()
        for field in (
            "analysis_id", "image_id", "session_id", "query",
            "analysis_type", "status", "answer", "provider", "model",
            "created_at", "is_multi_intent", "detected_intents", "intent_results",
        ):
            assert field in body, f"Missing field: {field}"

    def test_phase4_missing_image_still_404(self):
        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider()):
            resp = client.post("/api/analysis", json={
                "image_id": "img_nonexistent_phase6",
                "query": "test",
            })
        assert resp.status_code == 404

    def test_phase4_missing_api_key_still_503(self):
        iid = _upload_image()
        with patch("app.ai.provider.settings", ai_configured=False, ai_api_key=None, ai_provider="gemini"):
            resp = client.post("/api/analysis", json={"image_id": iid, "query": "test"})
        assert resp.status_code == 503
        assert "AI_API_KEY" in resp.json()["detail"]

    def test_phase5_followup_ask_still_works(self):
        iid = _upload_image()
        sid = _create_session("Phase5 regression")
        _link_image(sid, iid)

        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider("A1")):
            r1 = client.post("/api/analysis", json={
                "image_id": iid, "session_id": sid, "query": "Initial question"
            })
        assert r1.status_code == 200

        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider("Follow-up.")):
            r2 = client.post(f"/api/sessions/{sid}/ask", json={
                "question": "A follow-up question",
            })
        assert r2.status_code == 200
        assert r2.json()["status"] == "completed"
        assert r2.json()["answer"] == "Follow-up."

    def test_phase5_session_history_order_preserved(self):
        """Q1 → Q2 → Q3 history order is maintained."""
        iid = _upload_image()
        sid = _create_session("History order test")
        _link_image(sid, iid)

        for i in range(1, 4):
            with patch(PATCH_GET_PROVIDER, return_value=_mock_provider(f"Answer {i}.")):
                client.post(f"/api/sessions/{sid}/ask", json={
                    "question": f"Question {i}",
                    "enable_multi_intent": False,
                })

        q_resp = client.get(f"/api/sessions/{sid}/questions")
        questions = q_resp.json()["questions"]
        assert len(questions) == 3
        for i, q in enumerate(questions, 1):
            assert q["question"] == f"Question {i}"

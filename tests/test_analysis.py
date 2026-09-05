"""
SatQuery AI — Analysis System Tests (Phase 4)

Tests run WITHOUT a real AI API key.
A mock provider is injected via monkeypatching for all AI-dependent tests.

Key fix: patch target is app.services.analysis_service.get_provider
         (where it is actually called, not where it is defined).
"""

from __future__ import annotations

import io
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.ai.base import AIRawResponse

client = TestClient(app)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(w: int = 80, h: int = 60) -> bytes:
    img = Image.new("RGB", (w, h), color=(50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _upload_image() -> str:
    data = _make_jpeg_bytes()
    resp = client.post(
        "/api/images/upload",
        files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
    )
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    return resp.json()["image_id"]


def _create_session(mode: str = "single") -> str:
    resp = client.post("/api/sessions", json={"title": "Analysis Test", "mode": mode})
    assert resp.status_code == 201
    return resp.json()["session_id"]


def _link_image(session_id: str, image_id: str) -> None:
    resp = client.post(f"/api/sessions/{session_id}/images/{image_id}")
    assert resp.status_code == 200


# ─── Mock Provider Factory ────────────────────────────────────────────────────

def _mock_provider(answer: str = "This is a test analysis answer.", success: bool = True):
    """Return a mock provider object."""
    mock_prov = MagicMock()
    mock_prov.provider_name = "mock"
    mock_prov.model_name = "mock-model-1"
    mock_prov.analyze.return_value = AIRawResponse(
        text=answer if success else "",
        success=success,
        error_message=None if success else "Mock provider error",
        provider="mock",
        model="mock-model-1",
    )
    return mock_prov


# Correct patch target: the get_provider imported into analysis_service
PATCH_GET_PROVIDER = "app.services.analysis_service.get_provider"


# ─── Test: Missing API Key ────────────────────────────────────────────────────

class TestMissingApiKey:

    def test_no_api_key_returns_503(self):
        """POST /api/analysis without API key → HTTP 503."""
        iid = _upload_image()
        # Patch the settings object's ai_configured property via the provider module
        with patch(
            "app.ai.provider.settings",
            ai_configured=False,
            ai_api_key=None,
            ai_provider="gemini",
        ):
            resp = client.post("/api/analysis", json={
                "image_id": iid,
                "query": "What is visible?",
            })
        assert resp.status_code == 503, resp.text
        assert "AI provider is not configured" in resp.json()["detail"]
        assert "AI_API_KEY" in resp.json()["detail"]


# ─── Test: Input Validation ───────────────────────────────────────────────────

class TestInputValidation:

    def test_missing_image_returns_404(self):
        """POST /api/analysis with non-existent image_id → 404."""
        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider()):
            resp = client.post("/api/analysis", json={
                "image_id": "img_does_not_exist_xyz",
                "query": "Describe this image.",
            })
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_missing_session_returns_404(self):
        """POST /api/analysis with non-existent session_id → 404."""
        iid = _upload_image()
        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider()):
            resp = client.post("/api/analysis", json={
                "image_id": iid,
                "query": "What is visible?",
                "session_id": "ses_does_not_exist_xyz",
            })
        assert resp.status_code == 404

    def test_image_not_linked_to_session_returns_400(self):
        """Image exists and session exists but image not linked → 400."""
        iid = _upload_image()
        sid = _create_session()
        # Do NOT link image to session

        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider()):
            resp = client.post("/api/analysis", json={
                "image_id": iid,
                "query": "Analyze.",
                "session_id": sid,
            })
        assert resp.status_code == 400
        body = resp.json()["detail"]
        assert "not linked" in body.lower() or "image" in body.lower()


# ─── Test: Successful Analysis ────────────────────────────────────────────────

class TestSuccessfulAnalysis:

    def _run_mocked(self, image_id: str, query: str = "", session_id: str = None):
        payload = {"image_id": image_id, "query": query}
        if session_id:
            payload["session_id"] = session_id
        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider()):
            return client.post("/api/analysis", json=payload)

    def test_basic_analysis(self):
        """POST /api/analysis with valid image → 200, answer populated."""
        iid = _upload_image()
        resp = self._run_mocked(iid, "What is visible in this image?")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "completed"
        assert body["answer"] is not None
        assert len(body["answer"]) > 0
        assert body["image_id"] == iid

    def test_analysis_id_format(self):
        """analysis_id must start with 'analysis_'."""
        iid = _upload_image()
        resp = self._run_mocked(iid)
        assert resp.status_code == 200
        assert resp.json()["analysis_id"].startswith("analysis_")

    def test_confidence_is_null(self):
        """Confidence must be null (Gemini does not provide calibrated confidence)."""
        iid = _upload_image()
        resp = self._run_mocked(iid)
        assert resp.status_code == 200
        body = resp.json()
        assert body["confidence"] is None
        assert body["confidence_available"] is False

    def test_provider_model_fields(self):
        """Provider and model fields are returned in the response."""
        iid = _upload_image()
        resp = self._run_mocked(iid)
        assert resp.status_code == 200
        body = resp.json()
        assert "provider" in body
        assert "model" in body
        assert body["provider"] == "mock"
        assert body["model"] == "mock-model-1"

    def test_explicit_query_preserved(self):
        """The user's query is preserved in the response."""
        iid = _upload_image()
        q = "Identify all visible roads."
        resp = self._run_mocked(iid, q)
        assert resp.status_code == 200
        assert resp.json()["query"] == q

    def test_empty_query_allowed(self):
        """Empty query is allowed and triggers general analysis."""
        iid = _upload_image()
        resp = self._run_mocked(iid, "")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_session_id_none_when_no_session(self):
        """session_id in response is null when no session was passed."""
        iid = _upload_image()
        resp = self._run_mocked(iid, "Test.")
        assert resp.status_code == 200
        assert resp.json()["session_id"] is None


# ─── Test: Session Integration ────────────────────────────────────────────────

class TestSessionIntegration:

    def test_analysis_with_session(self):
        """Analysis with valid session_id returns session_id in response."""
        iid = _upload_image()
        sid = _create_session()
        _link_image(sid, iid)

        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider()):
            resp = client.post("/api/analysis", json={
                "image_id": iid,
                "query": "Describe the scene.",
                "session_id": sid,
            })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"] == sid
        assert body["status"] == "completed"

    def test_analysis_stores_question_in_session(self):
        """After analysis with session_id, session should contain the question."""
        iid = _upload_image()
        sid = _create_session()
        _link_image(sid, iid)

        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider("Buildings detected.")):
            resp = client.post("/api/analysis", json={
                "image_id": iid,
                "query": "Identify buildings.",
                "session_id": sid,
            })
        assert resp.status_code == 200

        # Check session questions updated
        q_resp = client.get(f"/api/sessions/{sid}/questions")
        assert q_resp.status_code == 200
        questions = q_resp.json()["questions"]
        assert len(questions) >= 1
        completed = [q for q in questions if q["status"] == "completed"]
        assert len(completed) >= 1
        assert completed[0]["answer"] is not None


# ─── Test: Provider Errors ────────────────────────────────────────────────────

class TestProviderErrors:

    def test_provider_error_returns_failed_status(self):
        """When provider returns success=False → status=failed, answer=null."""
        iid = _upload_image()
        with patch(PATCH_GET_PROVIDER, return_value=_mock_provider(success=False)):
            resp = client.post("/api/analysis", json={
                "image_id": iid,
                "query": "What is visible?",
            })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "failed"
        assert body["answer"] is None

    def test_provider_empty_response_returns_failed(self):
        """Provider returning empty text → status=failed."""
        iid = _upload_image()
        empty_prov = MagicMock()
        empty_prov.provider_name = "mock"
        empty_prov.model_name = "mock-model"
        empty_prov.analyze.return_value = AIRawResponse(
            text="",
            success=True,
            provider="mock",
            model="mock-model",
        )
        with patch(PATCH_GET_PROVIDER, return_value=empty_prov):
            resp = client.post("/api/analysis", json={
                "image_id": iid,
                "query": "Anything visible?",
            })
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"
        assert resp.json()["answer"] is None

    def test_provider_exception_handled(self):
        """If provider.analyze() returns failed → 200 with failed status."""
        iid = _upload_image()
        exc_prov = MagicMock()
        exc_prov.provider_name = "mock"
        exc_prov.model_name = "mock-model"
        exc_prov.analyze.return_value = AIRawResponse(
            text="",
            success=False,
            error_message="Simulated timeout",
            provider="mock",
            model="mock-model",
        )
        with patch(PATCH_GET_PROVIDER, return_value=exc_prov):
            resp = client.post("/api/analysis", json={
                "image_id": iid,
                "query": "Test.",
            })
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"


# ─── Test: Response Parser ────────────────────────────────────────────────────

class TestResponseParser:
    """Unit tests for the response parser — no HTTP involved."""

    def test_parse_successful_response(self):
        from app.ai.response_parser import parse_response
        raw = AIRawResponse(
            text="This satellite image shows urban development.",
            success=True,
            provider="mock",
            model="mock-1",
        )
        result = parse_response(raw)
        assert result.status == "completed"
        assert result.answer == "This satellite image shows urban development."
        assert result.confidence is None
        assert result.confidence_available is False

    def test_parse_error_response(self):
        from app.ai.response_parser import parse_response
        raw = AIRawResponse(
            text="",
            success=False,
            error_message="API quota exceeded",
            provider="mock",
            model="mock-1",
        )
        result = parse_response(raw)
        assert result.status == "failed"
        assert result.answer is None

    def test_parse_empty_text(self):
        from app.ai.response_parser import parse_response
        raw = AIRawResponse(text="", success=True, provider="mock", model="mock-1")
        result = parse_response(raw)
        assert result.status == "failed"


# ─── Regression: Existing APIs ────────────────────────────────────────────────

class TestRegressionPhase1to3:
    """Verify Phases 1–3 still work after Phase 4 additions."""

    def test_health_endpoint(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        # Accept either 'ok', 'healthy', or 'degraded'
        assert resp.json()["status"] in ("ok", "healthy", "degraded")

    def test_image_upload_regression(self):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/images/upload",
            files={"file": ("reg.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["image_id"].startswith("img_")

    def test_session_create_regression(self):
        resp = client.post("/api/sessions", json={"title": "Regression"})
        assert resp.status_code == 201
        assert resp.json()["session_id"].startswith("ses_")

    def test_analysis_route_exists(self):
        """POST /api/analysis endpoint exists and rejects unconfigured key."""
        # Patch the whole settings used inside provider.py
        mock_settings = MagicMock()
        mock_settings.ai_configured = False
        mock_settings.ai_api_key = None
        mock_settings.ai_provider = "gemini"

        with patch("app.ai.provider.settings", mock_settings):
            resp = client.post("/api/analysis", json={"image_id": "img_x", "query": "test"})
        # 503 (key missing) or 404 (image not found if key check fails first)
        assert resp.status_code in (503, 404)

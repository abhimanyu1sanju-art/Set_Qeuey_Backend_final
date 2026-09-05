"""
SatQuery AI — Session System Tests (Phase 3)

Tests:
  1.  Create session with title
  2.  Create session with default title (no title provided)
  3.  Get session (valid)
  4.  Get nonexistent session → 404
  5.  List sessions (paginated)
  6.  Link valid image to session
  7.  Link nonexistent image → 404
  8.  Link image to nonexistent session → 404
  9.  Prevent duplicate image link (idempotent)
  10. Get session images
  11. Add question to session
  12. Reject question whose image_id is not in session
  13. Get session questions
  14. Update question answer + status
  15. Delete session
  16. Verify deleting session does NOT delete image
  17. Verify existing image APIs still work (regression)
"""

from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(width: int = 80, height: int = 60) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _upload_image() -> str:
    """Upload a JPEG and return the image_id."""
    data = _make_jpeg_bytes()
    resp = client.post(
        "/api/images/upload",
        files={"file": ("test.jpg", io.BytesIO(data), "image/jpeg")},
    )
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    return resp.json()["image_id"]


def _create_session(title: str | None = "Test Session", mode: str = "single") -> str:
    """Create a session and return the session_id."""
    body = {}
    if title is not None:
        body["title"] = title
    if mode != "single":
        body["mode"] = mode
    resp = client.post("/api/sessions", json=body)
    assert resp.status_code == 201, f"Create session failed: {resp.text}"
    return resp.json()["session_id"]


# ─── Test: Create Session ─────────────────────────────────────────────────────

class TestCreateSession:

    def test_create_session_with_title(self):
        """POST /api/sessions with title → 201, correct title returned."""
        resp = client.post("/api/sessions", json={"title": "My Analysis"})
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["title"] == "My Analysis"
        assert body["session_id"].startswith("ses_")
        assert body["image_ids"] == []
        assert body["questions"] == []
        assert body["status"] == "active"
        assert "created_at" in body
        assert "updated_at" in body

    def test_create_session_default_title(self):
        """POST /api/sessions with no title → 201, title = 'New Analysis'."""
        resp = client.post("/api/sessions", json={})
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["title"] == "New Analysis"
        assert body["session_id"].startswith("ses_")

    def test_create_session_empty_title_gets_default(self):
        """POST /api/sessions with empty title → defaults to 'New Analysis'."""
        resp = client.post("/api/sessions", json={"title": ""})
        assert resp.status_code == 201, resp.text
        assert resp.json()["title"] == "New Analysis"

    def test_create_session_comparison_mode(self):
        """POST /api/sessions with mode='comparison' → stored correctly."""
        resp = client.post("/api/sessions", json={"title": "Compare", "mode": "comparison"})
        assert resp.status_code == 201, resp.text
        assert resp.json()["mode"] == "comparison"


# ─── Test: Get Session ────────────────────────────────────────────────────────

class TestGetSession:

    def test_get_existing_session(self):
        """GET /api/sessions/{session_id} → 200 with full session data."""
        sid = _create_session("Fetch Test")
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"] == sid
        assert body["title"] == "Fetch Test"

    def test_get_nonexistent_session(self):
        """GET /api/sessions/nonexistent → 404."""
        resp = client.get("/api/sessions/ses_does_not_exist_xyz_999")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Session not found"


# ─── Test: List Sessions ──────────────────────────────────────────────────────

class TestListSessions:

    def test_list_sessions(self):
        """GET /api/sessions → 200 with paginated structure."""
        # Create at least one session to ensure the list is not empty
        _create_session("List Test Session")
        resp = client.get("/api/sessions?page=1&limit=10")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "sessions" in body
        assert "page" in body
        assert "limit" in body
        assert "total" in body
        assert body["page"] == 1
        assert body["limit"] == 10
        assert isinstance(body["sessions"], list)

    def test_list_sessions_pagination(self):
        """Pagination parameters are respected."""
        resp = client.get("/api/sessions?page=1&limit=2")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["sessions"]) <= 2

    def test_list_sessions_items_have_expected_fields(self):
        """Each session list item has required summary fields."""
        _create_session("Field Check")
        resp = client.get("/api/sessions?page=1&limit=5")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        if sessions:
            s = sessions[0]
            for field in ("session_id", "title", "mode", "image_count", "question_count", "status", "created_at", "updated_at"):
                assert field in s, f"Missing field: {field}"


# ─── Test: Link Image ─────────────────────────────────────────────────────────

class TestLinkImage:

    def test_link_valid_image(self):
        """POST /api/sessions/{sid}/images/{iid} with valid IDs → 200."""
        sid = _create_session("Link Test")
        iid = _upload_image()
        resp = client.post(f"/api/sessions/{sid}/images/{iid}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"] == sid
        assert body["image_id"] == iid
        assert "linked" in body["message"].lower() or "successfully" in body["message"].lower()

        # Confirm it is now in the session
        get_resp = client.get(f"/api/sessions/{sid}")
        assert iid in get_resp.json()["image_ids"]

    def test_link_nonexistent_image(self):
        """POST /api/sessions/{sid}/images/fake_id → 404 (image not found)."""
        sid = _create_session("Link Fail Test")
        resp = client.post(f"/api/sessions/{sid}/images/img_does_not_exist_abc")
        assert resp.status_code == 404
        assert "image" in resp.json()["detail"].lower()

    def test_link_nonexistent_session(self):
        """POST /api/sessions/fake_sid/images/{iid} → 404 (session not found)."""
        iid = _upload_image()
        resp = client.post(f"/api/sessions/ses_fake_xyz_999/images/{iid}")
        assert resp.status_code == 404
        assert "session" in resp.json()["detail"].lower()

    def test_prevent_duplicate_image_link(self):
        """Linking the same image twice is idempotent — no duplicate in image_ids."""
        sid = _create_session("Duplicate Link Test")
        iid = _upload_image()

        resp1 = client.post(f"/api/sessions/{sid}/images/{iid}")
        assert resp1.status_code == 200

        resp2 = client.post(f"/api/sessions/{sid}/images/{iid}")
        assert resp2.status_code == 200
        assert "already linked" in resp2.json()["message"].lower()

        # Confirm image_ids has only one copy
        get_resp = client.get(f"/api/sessions/{sid}")
        assert get_resp.json()["image_ids"].count(iid) == 1


# ─── Test: Get Session Images ─────────────────────────────────────────────────

class TestGetSessionImages:

    def test_get_session_images(self):
        """GET /api/sessions/{sid}/images returns metadata for linked images."""
        sid = _create_session("Images Test")
        iid = _upload_image()
        client.post(f"/api/sessions/{sid}/images/{iid}")

        resp = client.get(f"/api/sessions/{sid}/images")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"] == sid
        assert isinstance(body["images"], list)
        assert len(body["images"]) == 1
        assert body["images"][0]["image_id"] == iid

    def test_get_session_images_empty(self):
        """GET /api/sessions/{sid}/images with no images linked → empty list."""
        sid = _create_session("No Images Test")
        resp = client.get(f"/api/sessions/{sid}/images")
        assert resp.status_code == 200
        assert resp.json()["images"] == []


# ─── Test: Questions ──────────────────────────────────────────────────────────

class TestSessionQuestions:

    def _setup(self):
        """Create session + upload + link image. Returns (sid, iid)."""
        sid = _create_session("Question Test")
        iid = _upload_image()
        client.post(f"/api/sessions/{sid}/images/{iid}")
        return sid, iid

    def test_add_question(self):
        """POST /api/sessions/{sid}/questions → 201, question created."""
        sid, iid = self._setup()
        resp = client.post(
            f"/api/sessions/{sid}/questions",
            json={"question": "What is visible?", "image_id": iid, "analysis_type": "general"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["question_id"].startswith("q_")
        assert body["question"] == "What is visible?"
        assert body["image_id"] == iid
        assert body["status"] == "pending"
        assert body["answer"] is None

    def test_reject_question_image_not_in_session(self):
        """Adding question with image not linked to session → 400."""
        sid = _create_session("Reject Question Test")
        # Don't link any image
        iid = _upload_image()
        resp = client.post(
            f"/api/sessions/{sid}/questions",
            json={"question": "What is this?", "image_id": iid},
        )
        assert resp.status_code == 400

    def test_get_session_questions(self):
        """GET /api/sessions/{sid}/questions → returns all questions."""
        sid, iid = self._setup()
        client.post(
            f"/api/sessions/{sid}/questions",
            json={"question": "Q1", "image_id": iid},
        )
        client.post(
            f"/api/sessions/{sid}/questions",
            json={"question": "Q2", "image_id": iid},
        )

        resp = client.get(f"/api/sessions/{sid}/questions")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"] == sid
        questions = body["questions"]
        assert len(questions) == 2
        assert questions[0]["question"] == "Q1"
        assert questions[1]["question"] == "Q2"

    def test_update_question_answer(self):
        """PATCH /api/sessions/{sid}/questions/{qid} → answer stored, status updated."""
        sid, iid = self._setup()
        q_resp = client.post(
            f"/api/sessions/{sid}/questions",
            json={"question": "Describe the scene.", "image_id": iid},
        )
        qid = q_resp.json()["question_id"]

        patch_resp = client.patch(
            f"/api/sessions/{sid}/questions/{qid}",
            json={"answer": "Urban landscape with vegetation.", "status": "completed"},
        )
        assert patch_resp.status_code == 200, patch_resp.text
        body = patch_resp.json()
        assert body["answer"] == "Urban landscape with vegetation."
        assert body["status"] == "completed"
        assert body["completed_at"] is not None

    def test_update_question_failed_status(self):
        """PATCH with status='failed' → answer cleared, completed_at null."""
        sid, iid = self._setup()
        q_resp = client.post(
            f"/api/sessions/{sid}/questions",
            json={"question": "Analyze terrain.", "image_id": iid},
        )
        qid = q_resp.json()["question_id"]

        patch_resp = client.patch(
            f"/api/sessions/{sid}/questions/{qid}",
            json={"status": "failed"},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["status"] == "failed"


# ─── Test: Delete Session ─────────────────────────────────────────────────────

class TestDeleteSession:

    def test_delete_session(self):
        """DELETE /api/sessions/{sid} → 200, session gone."""
        sid = _create_session("Delete Me")
        resp = client.delete(f"/api/sessions/{sid}")
        assert resp.status_code == 200, resp.text
        assert resp.json()["session_id"] == sid

        # Confirm gone
        get_resp = client.get(f"/api/sessions/{sid}")
        assert get_resp.status_code == 404

    def test_delete_nonexistent_session(self):
        """DELETE /api/sessions/fake → 404."""
        resp = client.delete("/api/sessions/ses_fake_does_not_exist_abc")
        assert resp.status_code == 404

    def test_delete_session_does_not_delete_image(self):
        """Deleting a session must NOT delete the linked image."""
        sid = _create_session("Safe Delete")
        iid = _upload_image()
        client.post(f"/api/sessions/{sid}/images/{iid}")

        # Delete the session
        del_resp = client.delete(f"/api/sessions/{sid}")
        assert del_resp.status_code == 200

        # Image must still exist
        img_resp = client.get(f"/api/images/{iid}")
        assert img_resp.status_code == 200, "Image should still exist after session deletion"
        assert img_resp.json()["image_id"] == iid


# ─── Regression: Existing Image APIs ─────────────────────────────────────────

class TestImageApiRegression:
    """Verify Phase 2 image APIs still work correctly after Phase 3 additions."""

    def test_image_upload_still_works(self):
        data = _make_jpeg_bytes()
        resp = client.post(
            "/api/images/upload",
            files={"file": ("reg_test.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["image_id"].startswith("img_")
        assert resp.json()["status"] == "ready"

    def test_image_get_still_works(self):
        iid = _upload_image()
        resp = client.get(f"/api/images/{iid}")
        assert resp.status_code == 200
        assert resp.json()["image_id"] == iid

    def test_image_delete_still_works(self):
        iid = _upload_image()
        resp = client.delete(f"/api/images/{iid}")
        assert resp.status_code == 200
        assert resp.json()["image_id"] == iid

        # Confirm gone
        assert client.get(f"/api/images/{iid}").status_code == 404

    def test_image_delete_removes_from_session(self):
        """Deleting an image removes its reference from linked sessions."""
        sid = _create_session("Image Delete Cleanup")
        iid = _upload_image()
        client.post(f"/api/sessions/{sid}/images/{iid}")

        # Confirm it's linked
        assert iid in client.get(f"/api/sessions/{sid}").json()["image_ids"]

        # Delete image
        client.delete(f"/api/images/{iid}")

        # image_id should be removed from session
        session_data = client.get(f"/api/sessions/{sid}").json()
        assert iid not in session_data["image_ids"]

    def test_image_get_nonexistent_still_404(self):
        resp = client.get("/api/images/img_totally_fake_xyz_000")
        assert resp.status_code == 404

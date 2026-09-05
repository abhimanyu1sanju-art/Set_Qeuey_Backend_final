"""
SatQuery AI — Image Upload Tests (Phase 2)

Tests:
  - Valid JPEG upload → HTTP 200, status=ready
  - Valid PNG upload → HTTP 200
  - Valid TIFF upload → HTTP 200
  - Invalid TXT file → HTTP 400
  - Fake JPG (non-image bytes with .jpg extension) → HTTP 400
  - Oversized file → HTTP 413
  - Corrupt image → HTTP 400
  - GET existing image → HTTP 200
  - GET non-existing image → HTTP 404
  - DELETE image → HTTP 200, files removed
"""

from __future__ import annotations

import io
import os
import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ─── Image Factories ──────────────────────────────────────────────────────────

def make_jpeg_bytes(width: int = 100, height: int = 80) -> bytes:
    """Create a minimal valid JPEG in memory using Pillow."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_png_bytes(width: int = 80, height: int = 60) -> bytes:
    """Create a minimal valid PNG in memory using Pillow."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_tiff_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create a minimal valid TIFF in memory using Pillow."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(50, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    return buf.getvalue()


def make_fake_jpeg_bytes() -> bytes:
    """Return non-image bytes with no valid image signature."""
    return b"This is not an image. It is plain text masquerading as JPEG."


def make_corrupt_jpeg_bytes() -> bytes:
    """Return truncated JPEG header bytes (corrupt)."""
    # Valid JPEG starts with FF D8, but we cut it short
    return bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10])  # truncated


# ─── Upload Helper ────────────────────────────────────────────────────────────

def upload_file(
    data: bytes,
    filename: str,
    content_type: str = "image/jpeg",
) -> "requests.Response":
    return client.post(
        "/api/images/upload",
        files={"file": (filename, io.BytesIO(data), content_type)},
    )


# ─── Tests: Upload ────────────────────────────────────────────────────────────

class TestImageUpload:

    def test_valid_jpeg_upload(self):
        """Valid JPEG file should return HTTP 200 with status=ready."""
        data = make_jpeg_bytes()
        resp = upload_file(data, "test.jpg", "image/jpeg")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ready"
        assert body["format"] == "JPEG"
        assert body["is_geospatial"] is False
        assert body["image_id"].startswith("img_")
        assert body["width"] == 100
        assert body["height"] == 80
        assert body["size"] == len(data)

    def test_valid_png_upload(self):
        """Valid PNG file should return HTTP 200."""
        data = make_png_bytes()
        resp = upload_file(data, "test.png", "image/png")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ready"
        assert body["format"] == "PNG"
        assert body["image_id"].startswith("img_")

    def test_valid_tiff_upload(self):
        """Valid TIFF file should return HTTP 200."""
        data = make_tiff_bytes()
        resp = upload_file(data, "test.tif", "image/tiff")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ready"
        assert body["image_id"].startswith("img_")

    def test_invalid_txt_file(self):
        """TXT file should be rejected with HTTP 400."""
        resp = upload_file(b"hello world", "document.txt", "text/plain")
        assert resp.status_code == 400, resp.text
        assert "Unsupported" in resp.json()["detail"] or "extension" in resp.json()["detail"].lower()

    def test_fake_jpg_file(self):
        """Non-image bytes with .jpg extension should be rejected with HTTP 400."""
        resp = upload_file(make_fake_jpeg_bytes(), "fake.jpg", "image/jpeg")
        assert resp.status_code == 400, resp.text

    def test_oversized_file(self, monkeypatch):
        """File exceeding MAX_FILE_SIZE_MB should return HTTP 413."""
        from app.core.config import settings
        # Temporarily set max to 1 byte to force the limit
        monkeypatch.setattr(settings, "max_file_size_mb", 0)

        resp = upload_file(make_jpeg_bytes(), "big.jpg", "image/jpeg")
        assert resp.status_code == 413, resp.text
        assert "exceeds" in resp.json()["detail"].lower()

    def test_corrupt_image(self):
        """Corrupt image bytes should return HTTP 400."""
        resp = upload_file(make_corrupt_jpeg_bytes(), "corrupt.jpg", "image/jpeg")
        assert resp.status_code == 400, resp.text


# ─── Tests: Get & Delete ──────────────────────────────────────────────────────

class TestImageGetDelete:

    def _upload_jpeg(self) -> str:
        """Upload a JPEG and return the image_id."""
        data = make_jpeg_bytes()
        resp = upload_file(data, "gettest.jpg", "image/jpeg")
        assert resp.status_code == 200
        return resp.json()["image_id"]

    def test_get_existing_image(self):
        """GET /api/images/{image_id} for an uploaded image returns HTTP 200."""
        image_id = self._upload_jpeg()
        resp = client.get(f"/api/images/{image_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["image_id"] == image_id
        assert body["status"] == "ready"

    def test_get_nonexistent_image(self):
        """GET /api/images/nonexistent_id should return HTTP 404."""
        resp = client.get("/api/images/img_does_not_exist_xyz_abc_123")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Image not found"

    def test_delete_image(self):
        """DELETE /api/images/{image_id} should return HTTP 200 and remove files."""
        from app.services.image_service import get_image_by_id

        image_id = self._upload_jpeg()

        # Grab storage paths before deletion
        doc = get_image_by_id(image_id)
        original_path = doc.storage.original
        processed_path = doc.storage.processed
        thumbnail_path = doc.storage.thumbnail

        resp = client.delete(f"/api/images/{image_id}")
        assert resp.status_code == 200, resp.text
        assert resp.json()["image_id"] == image_id

        # Verify files are gone
        if original_path:
            assert not Path(original_path).exists(), "Original file should be deleted"
        if processed_path:
            assert not Path(processed_path).exists(), "Processed file should be deleted"
        if thumbnail_path:
            assert not Path(thumbnail_path).exists(), "Thumbnail file should be deleted"

        # Verify MongoDB record is gone
        get_resp = client.get(f"/api/images/{image_id}")
        assert get_resp.status_code == 404

    def test_delete_nonexistent_image(self):
        """DELETE on a non-existent image_id should return HTTP 404."""
        resp = client.delete("/api/images/img_totally_fake_xyz_999")
        assert resp.status_code == 404

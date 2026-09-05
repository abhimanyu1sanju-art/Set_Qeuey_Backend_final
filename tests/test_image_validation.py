"""
SatQuery AI — Image Validation Unit Tests (Phase 2)

Tests for standalone utility functions in:
  - app/utils/file_validation.py
  - app/utils/image_utils.py

These are unit tests that do not require HTTP or MongoDB.
"""

from __future__ import annotations

import io
import pytest


# ─── Extension Validation ─────────────────────────────────────────────────────

class TestExtensionValidation:

    def test_jpg_allowed(self):
        from app.utils.file_validation import validate_extension
        assert validate_extension("photo.jpg") == ".jpg"

    def test_jpeg_allowed(self):
        from app.utils.file_validation import validate_extension
        assert validate_extension("PHOTO.JPEG") == ".jpeg"

    def test_png_allowed(self):
        from app.utils.file_validation import validate_extension
        assert validate_extension("image.png") == ".png"

    def test_tif_allowed(self):
        from app.utils.file_validation import validate_extension
        assert validate_extension("geo.tif") == ".tif"

    def test_tiff_allowed(self):
        from app.utils.file_validation import validate_extension
        assert validate_extension("geo.tiff") == ".tiff"

    def test_txt_rejected(self):
        from fastapi import HTTPException
        from app.utils.file_validation import validate_extension
        with pytest.raises(HTTPException) as exc_info:
            validate_extension("document.txt")
        assert exc_info.value.status_code == 400

    def test_pdf_rejected(self):
        from fastapi import HTTPException
        from app.utils.file_validation import validate_extension
        with pytest.raises(HTTPException) as exc_info:
            validate_extension("report.pdf")
        assert exc_info.value.status_code == 400

    def test_exe_rejected(self):
        from fastapi import HTTPException
        from app.utils.file_validation import validate_extension
        with pytest.raises(HTTPException) as exc_info:
            validate_extension("malware.exe")
        assert exc_info.value.status_code == 400

    def test_svg_rejected(self):
        from fastapi import HTTPException
        from app.utils.file_validation import validate_extension
        with pytest.raises(HTTPException) as exc_info:
            validate_extension("icon.svg")
        assert exc_info.value.status_code == 400

    def test_mp4_rejected(self):
        from fastapi import HTTPException
        from app.utils.file_validation import validate_extension
        with pytest.raises(HTTPException) as exc_info:
            validate_extension("video.mp4")
        assert exc_info.value.status_code == 400

    def test_zip_rejected(self):
        from fastapi import HTTPException
        from app.utils.file_validation import validate_extension
        with pytest.raises(HTTPException) as exc_info:
            validate_extension("archive.zip")
        assert exc_info.value.status_code == 400


# ─── Safe Filename ────────────────────────────────────────────────────────────

class TestSafeFilename:

    def test_path_traversal_stripped(self):
        from app.utils.file_validation import sanitise_filename
        result = sanitise_filename("../../malicious.exe")
        # Should only return the basename
        assert ".." not in result
        assert "/" not in result
        assert "\\" not in result

    def test_normal_filename_preserved(self):
        from app.utils.file_validation import sanitise_filename
        result = sanitise_filename("satellite_image.jpg")
        assert "satellite_image" in result

    def test_unsafe_chars_replaced(self):
        from app.utils.file_validation import sanitise_filename
        result = sanitise_filename("image (1).jpg")
        assert "(" not in result
        assert ")" not in result


# ─── Content Validation ───────────────────────────────────────────────────────

class TestContentValidation:

    def test_valid_jpeg_passes(self):
        from PIL import Image
        from app.utils.file_validation import validate_image_content
        img = Image.new("RGB", (50, 50), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        # Should not raise
        validate_image_content(buf.getvalue(), "test.jpg")

    def test_valid_png_passes(self):
        from PIL import Image
        from app.utils.file_validation import validate_image_content
        img = Image.new("RGB", (50, 50), color=(0, 255, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        validate_image_content(buf.getvalue(), "test.png")

    def test_text_file_rejected(self):
        from fastapi import HTTPException
        from app.utils.file_validation import validate_image_content
        with pytest.raises(HTTPException) as exc_info:
            validate_image_content(b"This is not an image", "test.jpg")
        assert exc_info.value.status_code == 400

    def test_corrupt_bytes_rejected(self):
        from fastapi import HTTPException
        from app.utils.file_validation import validate_image_content
        with pytest.raises(HTTPException) as exc_info:
            validate_image_content(bytes([0xFF, 0xD8, 0x00, 0x01]), "corrupt.jpg")
        assert exc_info.value.status_code == 400


# ─── Image ID Generation ──────────────────────────────────────────────────────

class TestImageIdGeneration:

    def test_image_id_format(self):
        from app.utils.image_utils import generate_image_id
        image_id = generate_image_id()
        assert image_id.startswith("img_")
        assert len(image_id) > 10

    def test_image_ids_are_unique(self):
        from app.utils.image_utils import generate_image_id
        ids = {generate_image_id() for _ in range(100)}
        assert len(ids) == 100, "Generated IDs must be unique"


# ─── Metadata Extraction ──────────────────────────────────────────────────────

class TestMetadataExtraction:

    def _make_jpeg(self, w: int = 120, h: int = 80) -> bytes:
        from PIL import Image
        img = Image.new("RGB", (w, h))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_jpeg_dimensions(self):
        from app.utils.image_utils import extract_image_metadata
        meta = extract_image_metadata(self._make_jpeg(120, 80), "test.jpg")
        assert meta["width"] == 120
        assert meta["height"] == 80
        assert meta["format"] == "JPEG"
        assert meta["mime_type"] == "image/jpeg"
        assert meta["is_geospatial"] is False

    def test_png_format(self):
        from PIL import Image
        from app.utils.image_utils import extract_image_metadata
        img = Image.new("RGB", (64, 64))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        meta = extract_image_metadata(buf.getvalue(), "test.png")
        assert meta["format"] == "PNG"
        assert meta["mime_type"] == "image/png"

    def test_tiff_not_geospatial(self):
        from PIL import Image
        from app.utils.image_utils import extract_image_metadata
        img = Image.new("RGB", (64, 64))
        buf = io.BytesIO()
        img.save(buf, format="TIFF")
        meta = extract_image_metadata(buf.getvalue(), "test.tiff")
        # A plain TIFF without geospatial tags should not be marked as geospatial
        assert meta["is_geospatial"] is False

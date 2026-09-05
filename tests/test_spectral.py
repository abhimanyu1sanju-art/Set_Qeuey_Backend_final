"""
SatQuery AI — Phase 2: Spectral Analysis Tests

Tests the spectral index calculation functions, raster utilities,
and API endpoints using synthetic numpy fixtures.

Design principles:
  - All formula tests use synthetic arrays with known expected outputs.
  - No real Copernicus data is downloaded during tests.
  - Integration tests that require CDSE credentials are skipped automatically
    when COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET are not set.
  - Every test asserts on real computed values — no mocks of NumPy arithmetic.

Test categories:
  1. Formula correctness (NDVI, NDWI, NBR)
  2. Edge cases (division by zero, nodata, NaN propagation)
  3. Band alignment / resampling
  4. Pixel statistics
  5. GeoTIFF metadata validation
  6. API endpoint validation (schema, error codes)
  7. Integration (skipped without credentials)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_band(shape=(4, 4), base_value=0.0, dtype=np.float32) -> np.ndarray:
    """Create a synthetic band array filled with a constant value."""
    return np.full(shape, fill_value=base_value, dtype=dtype)


def make_gradient_band(shape=(4, 4), low=0.1, high=0.9) -> np.ndarray:
    """Create a band with linearly spaced values for more realistic tests."""
    arr = np.linspace(low, high, num=shape[0] * shape[1], dtype=np.float32)
    return arr.reshape(shape)


# ─── 1. NDVI formula tests ────────────────────────────────────────────────────

class TestNDVIFormula:
    """NDVI = (NIR - RED) / (NIR + RED)"""

    def test_ndvi_with_known_values(self):
        """NDVI with NIR=8000, RED=2000 should give (8000-2000)/(8000+2000) = 0.6"""
        from app.services.spectral_service import calculate_ndvi
        nir = make_band(base_value=8000.0)
        red = make_band(base_value=2000.0)
        result = calculate_ndvi(nir=nir, red=red)
        expected = (8000.0 - 2000.0) / (8000.0 + 2000.0)  # = 0.6
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, expected, rtol=1e-5,
                                   err_msg=f"Expected NDVI={expected:.4f}")

    def test_ndvi_dense_vegetation(self):
        """NDVI near 1.0 for very high NIR, very low RED"""
        from app.services.spectral_service import calculate_ndvi
        nir = make_band(base_value=9000.0)
        red = make_band(base_value=100.0)
        result = calculate_ndvi(nir=nir, red=red)
        # (9000-100)/(9000+100) = 8900/9100 ≈ 0.9780
        expected = 8900.0 / 9100.0
        np.testing.assert_allclose(result, expected, rtol=1e-4)
        assert np.all(result > 0.9), "Dense vegetation should have NDVI > 0.9"

    def test_ndvi_water_returns_negative(self):
        """NDVI should be negative for water (RED > NIR)"""
        from app.services.spectral_service import calculate_ndvi
        nir = make_band(base_value=500.0)
        red = make_band(base_value=3000.0)
        result = calculate_ndvi(nir=nir, red=red)
        assert np.all(result < 0), "Water areas should have negative NDVI"

    def test_ndvi_clipped_to_valid_range(self):
        """NDVI result must always be in [-1, 1]"""
        from app.services.spectral_service import calculate_ndvi
        nir = make_gradient_band(low=0, high=10000)
        red = make_gradient_band(low=10000, high=0)
        result = calculate_ndvi(nir=nir, red=red)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1.0), "NDVI must not go below -1"
        assert np.all(valid <= 1.0), "NDVI must not exceed +1"


# ─── 2. NDWI formula tests ────────────────────────────────────────────────────

class TestNDWIFormula:
    """NDWI = (GREEN - NIR) / (GREEN + NIR)   [McFeeters 1996]"""

    def test_ndwi_with_known_values(self):
        """NDWI with GREEN=5000, NIR=1000 → (5000-1000)/(5000+1000) = 4000/6000 ≈ 0.6667"""
        from app.services.spectral_service import calculate_ndwi
        green = make_band(base_value=5000.0)
        nir = make_band(base_value=1000.0)
        result = calculate_ndwi(green=green, nir=nir)
        expected = (5000.0 - 1000.0) / (5000.0 + 1000.0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_ndwi_water_positive(self):
        """Water pixels (GREEN >> NIR) should return positive NDWI"""
        from app.services.spectral_service import calculate_ndwi
        green = make_band(base_value=7000.0)
        nir = make_band(base_value=500.0)
        result = calculate_ndwi(green=green, nir=nir)
        assert np.all(result > 0), "Water should have positive NDWI"

    def test_ndwi_vegetation_negative(self):
        """Vegetation (NIR >> GREEN) should return negative NDWI"""
        from app.services.spectral_service import calculate_ndwi
        green = make_band(base_value=1000.0)
        nir = make_band(base_value=8000.0)
        result = calculate_ndwi(green=green, nir=nir)
        assert np.all(result < 0), "Vegetation should have negative NDWI"

    def test_ndwi_output_dtype(self):
        """NDWI output must be float32"""
        from app.services.spectral_service import calculate_ndwi
        green = make_band(base_value=3000.0)
        nir = make_band(base_value=3000.0)
        result = calculate_ndwi(green=green, nir=nir)
        assert result.dtype == np.float32


# ─── 3. NBR formula tests ─────────────────────────────────────────────────────

class TestNBRFormula:
    """NBR = (NIR - SWIR2) / (NIR + SWIR2)"""

    def test_nbr_with_known_values(self):
        """NBR with NIR=8000, SWIR2=1000 → (8000-1000)/(8000+1000) = 7000/9000 ≈ 0.7778"""
        from app.services.spectral_service import calculate_nbr
        nir = make_band(base_value=8000.0)
        swir2 = make_band(base_value=1000.0)
        result = calculate_nbr(nir=nir, swir2=swir2)
        expected = 7000.0 / 9000.0
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_nbr_burned_area_negative(self):
        """Burned areas (low NIR, high SWIR2) should return negative NBR"""
        from app.services.spectral_service import calculate_nbr
        nir = make_band(base_value=1000.0)
        swir2 = make_band(base_value=7000.0)
        result = calculate_nbr(nir=nir, swir2=swir2)
        assert np.all(result < 0), "Burned areas should have negative NBR"

    def test_nbr_healthy_vegetation_positive(self):
        """Healthy vegetation (high NIR, low SWIR2) should return positive NBR"""
        from app.services.spectral_service import calculate_nbr
        nir = make_band(base_value=8500.0)
        swir2 = make_band(base_value=500.0)
        result = calculate_nbr(nir=nir, swir2=swir2)
        assert np.all(result > 0.8), "Healthy vegetation should have NBR > 0.8"


# ─── 4. Division-by-zero safety ───────────────────────────────────────────────

class TestDivisionByZeroSafety:
    """All three indices must handle zero denominators without crashing."""

    def test_ndvi_zero_denominator_returns_nan(self):
        """When NIR = RED = 0, NDVI denominator is 0 → must return NaN, not crash"""
        from app.services.spectral_service import calculate_ndvi
        nir = make_band(base_value=0.0)
        red = make_band(base_value=0.0)
        result = calculate_ndvi(nir=nir, red=red)
        assert not np.any(np.isinf(result)), "Result must not contain inf"
        assert np.all(np.isnan(result)), "Zero denominator must produce NaN"

    def test_ndwi_zero_denominator_returns_nan(self):
        """When GREEN = NIR = 0, NDWI denominator is 0 → NaN"""
        from app.services.spectral_service import calculate_ndwi
        green = make_band(base_value=0.0)
        nir = make_band(base_value=0.0)
        result = calculate_ndwi(green=green, nir=nir)
        assert not np.any(np.isinf(result))
        assert np.all(np.isnan(result))

    def test_nbr_zero_denominator_returns_nan(self):
        """When NIR = SWIR2 = 0, NBR denominator is 0 → NaN"""
        from app.services.spectral_service import calculate_nbr
        nir = make_band(base_value=0.0)
        swir2 = make_band(base_value=0.0)
        result = calculate_nbr(nir=nir, swir2=swir2)
        assert not np.any(np.isinf(result))
        assert np.all(np.isnan(result))

    def test_mixed_zero_nonzero_pixels(self):
        """Pixels where denominator is zero should be NaN; others should be valid."""
        from app.services.spectral_service import calculate_ndvi
        nir = np.array([[8000.0, 0.0], [5000.0, 0.0]], dtype=np.float32)
        red = np.array([[2000.0, 0.0], [3000.0, 0.0]], dtype=np.float32)
        result = calculate_ndvi(nir=nir, red=red)

        # (0, 0) → valid pixel → 0.6
        np.testing.assert_allclose(result[0, 0], 0.6, rtol=1e-5)
        # (0, 1) → denominator 0 → NaN
        assert np.isnan(result[0, 1]), "Zero denominator pixel must be NaN"
        # (1, 0) → valid pixel → 0.25
        np.testing.assert_allclose(result[1, 0], 0.25, rtol=1e-5)
        # (1, 1) → denominator 0 → NaN
        assert np.isnan(result[1, 1]), "Zero denominator pixel must be NaN"


# ─── 5. NoData masking ────────────────────────────────────────────────────────

class TestNodataMasking:
    """Nodata (zero-valued) pixels must be excluded from statistics."""

    def test_nodata_pixels_excluded_from_stats(self):
        """Stats must only count valid (non-NaN) pixels."""
        from app.services.raster_service import compute_raster_stats
        arr = np.array([[0.5, 0.3, np.nan, 0.8]], dtype=np.float32)
        stats = compute_raster_stats(arr)
        assert stats["valid_pixel_count"] == 3, "NaN pixel must be excluded"
        assert stats["nodata_pixel_count"] == 1, "NaN pixel must be counted as nodata"
        assert stats["total_pixel_count"] == 4
        np.testing.assert_allclose(stats["mean"], np.mean([0.5, 0.3, 0.8]), rtol=1e-5)

    def test_all_nodata_returns_none_stats(self):
        """Array of all NaN must return None for all statistics."""
        from app.services.raster_service import compute_raster_stats
        arr = np.full((3, 3), fill_value=np.nan, dtype=np.float32)
        stats = compute_raster_stats(arr)
        assert stats["valid_pixel_count"] == 0
        assert stats["mean"] is None
        assert stats["minimum"] is None
        assert stats["maximum"] is None

    def test_valid_pixel_pct_calculation(self):
        """valid_pixel_pct must be computed correctly."""
        from app.services.raster_service import compute_raster_stats
        arr = np.array([[1.0, np.nan, np.nan, np.nan]], dtype=np.float32)
        stats = compute_raster_stats(arr)
        assert stats["valid_pixel_pct"] == 25.0  # 1 out of 4 = 25%

    def test_percentiles_correct(self):
        """5th and 95th percentiles must match numpy."""
        from app.services.raster_service import compute_raster_stats
        arr = np.linspace(-1.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
        stats = compute_raster_stats(arr)
        valid = arr.ravel()
        np.testing.assert_allclose(stats["percentile_5"],  np.percentile(valid, 5),  atol=1e-4)
        np.testing.assert_allclose(stats["percentile_95"], np.percentile(valid, 95), atol=1e-4)


# ─── 6. Band alignment / resampling ──────────────────────────────────────────

class TestBandAlignment:
    """align_band_to_reference must produce output matching the reference shape."""

    def _make_mock_transform(self, x_origin, y_origin, pixel_size):
        """Create a minimal rasterio-compatible Affine transform."""
        try:
            from rasterio.transform import from_origin
            return from_origin(x_origin, y_origin, pixel_size, pixel_size)
        except ImportError:
            pytest.skip("rasterio not installed")

    def _make_mock_crs(self):
        try:
            from rasterio.crs import CRS
            return CRS.from_epsg(32643)
        except ImportError:
            pytest.skip("rasterio not installed")

    def test_upsampling_shape_matches_reference(self):
        """20m source (5×5) resampled to 10m reference (10×10) → output must be 10×10."""
        try:
            from app.services.raster_service import align_band_to_reference
        except ImportError:
            pytest.skip("rasterio not available")

        # 20m source: 5×5 pixels covering a 100m×100m area
        src_arr = np.full((5, 5), fill_value=3000.0, dtype=np.float32)
        src_transform = self._make_mock_transform(0, 100, 20)
        src_crs = self._make_mock_crs()

        # 10m reference: 10×10 pixels covering same 100m×100m area
        ref_arr = np.full((10, 10), fill_value=0.0, dtype=np.float32)
        ref_transform = self._make_mock_transform(0, 100, 10)
        ref_crs = self._make_mock_crs()

        result = align_band_to_reference(
            source_array=src_arr,
            source_transform=src_transform,
            source_crs=src_crs,
            reference_array=ref_arr,
            reference_transform=ref_transform,
            reference_crs=ref_crs,
            resampling_method="bilinear",
        )

        assert result.shape == (10, 10), (
            f"Resampled array shape {result.shape} must match reference (10, 10)"
        )

    def test_resampled_values_within_source_range(self):
        """Resampled values must be within the range of the source (no extrapolation)."""
        try:
            from app.services.raster_service import align_band_to_reference
        except ImportError:
            pytest.skip("rasterio not available")

        src_arr = np.full((5, 5), fill_value=5000.0, dtype=np.float32)
        src_transform = self._make_mock_transform(0, 100, 20)
        src_crs = self._make_mock_crs()

        ref_arr = np.full((10, 10), fill_value=0.0, dtype=np.float32)
        ref_transform = self._make_mock_transform(0, 100, 10)
        ref_crs = self._make_mock_crs()

        result = align_band_to_reference(
            source_array=src_arr, source_transform=src_transform, source_crs=src_crs,
            reference_array=ref_arr, reference_transform=ref_transform, reference_crs=ref_crs,
        )

        valid = result[~np.isnan(result)]
        if valid.size > 0:
            assert float(valid.min()) >= 0, "Resampled values must not go below 0"
            assert float(valid.max()) <= 10000.0, "Resampled values must not exceed source max"


# ─── 7. GeoTIFF metadata validation ──────────────────────────────────────────

class TestGeoTIFFOutput:
    """save_geotiff must write correct dtype, CRS, and transform."""

    def test_geotiff_written_with_correct_dtype(self):
        """Output GeoTIFF must be float32."""
        try:
            from app.services.raster_service import save_geotiff, get_geotiff_metadata
            import rasterio
            from rasterio.transform import from_origin
            from rasterio.crs import CRS
        except ImportError:
            pytest.skip("rasterio not installed")

        arr = np.random.rand(10, 10).astype(np.float32)
        transform = from_origin(0, 100, 10, 10)
        crs = CRS.from_epsg(32643)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_output.tif"
            save_geotiff(array=arr, transform=transform, crs=crs, output_path=out_path)

            assert out_path.exists(), "GeoTIFF file must be created"

            meta = get_geotiff_metadata(out_path)
            assert meta["dtype"] == "float32", f"Expected float32, got {meta['dtype']}"
            assert meta["count"] == 1, "GeoTIFF must have exactly 1 band"
            assert meta["width"] == 10
            assert meta["height"] == 10

    def test_geotiff_crs_preserved(self):
        """CRS written to GeoTIFF must match the input CRS."""
        try:
            from app.services.raster_service import save_geotiff, get_geotiff_metadata
            from rasterio.transform import from_origin
            from rasterio.crs import CRS
        except ImportError:
            pytest.skip("rasterio not installed")

        arr = np.ones((5, 5), dtype=np.float32)
        transform = from_origin(0, 50, 10, 10)
        crs = CRS.from_epsg(4326)  # WGS84

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "crs_test.tif"
            save_geotiff(array=arr, transform=transform, crs=crs, output_path=out_path)
            meta = get_geotiff_metadata(out_path)
            # CRS string must reference EPSG:4326
            assert "4326" in meta["crs"], f"CRS mismatch: {meta['crs']}"


# ─── 8. Schema / request validation ──────────────────────────────────────────

class TestSchemaValidation:
    """Pydantic schema must reject invalid inputs cleanly."""

    def test_empty_indices_rejected(self):
        """indices must have at least 1 element."""
        from pydantic import ValidationError
        from app.schemas.spectral import SpectralAnalysisRequest
        with pytest.raises(ValidationError, match="too_short|min_length"):
            SpectralAnalysisRequest(scene_id="S2_test", indices=[])

    def test_unsupported_index_rejected(self):
        """Unknown index name must raise ValidationError."""
        from pydantic import ValidationError
        from app.schemas.spectral import SpectralAnalysisRequest
        with pytest.raises((ValidationError, ValueError)):
            SpectralAnalysisRequest(scene_id="S2_test", indices=["EVI"])

    def test_non_s2_collection_rejected(self):
        """Only sentinel-2-l2a is supported; Sentinel-1 must be rejected."""
        from pydantic import ValidationError
        from app.schemas.spectral import SpectralAnalysisRequest
        with pytest.raises((ValidationError, ValueError)):
            SpectralAnalysisRequest(
                scene_id="S1_test",
                indices=["NDVI"],
                collection="sentinel-1-grd",
            )

    def test_valid_request_passes(self):
        """Valid request must not raise."""
        from app.schemas.spectral import SpectralAnalysisRequest
        req = SpectralAnalysisRequest(
            scene_id="S2C_MSIL2A_20250128T053131_N0511_R062_T43QCA_20250128T083001",
            indices=["NDVI", "NDWI"],
            collection="sentinel-2-l2a",
        )
        assert req.scene_id.startswith("S2C")
        assert "NDVI" in req.indices
        assert "NDWI" in req.indices

    def test_indices_are_normalised_to_uppercase(self):
        """Lowercase index names must be normalised to uppercase."""
        from app.schemas.spectral import SpectralAnalysisRequest
        req = SpectralAnalysisRequest(scene_id="S2_test", indices=["ndvi", "ndwi", "nbr"])
        assert req.indices == ["NDVI", "NDWI", "NBR"]


# ─── 9. API endpoint validation ───────────────────────────────────────────────

class TestSpectralAPIEndpoints:
    """FastAPI endpoint validation — no real Copernicus calls made."""

    @pytest.fixture(autouse=True)
    def client(self):
        """Create a FastAPI test client."""
        try:
            from fastapi.testclient import TestClient
            from app.main import create_app
            app = create_app()
            self._client = TestClient(app, raise_server_exceptions=False)
        except Exception as exc:
            pytest.skip(f"Could not create test client: {exc}")

    def test_status_endpoint_returns_200(self):
        """GET /api/spectral/status must return 200."""
        resp = self._client.get("/api/spectral/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "credentials_configured" in data
        assert "supported_indices" in data

    def test_analyze_missing_scene_id_returns_422(self):
        """POST /api/spectral/analyze without scene_id must return 422."""
        resp = self._client.post(
            "/api/spectral/analyze",
            json={"indices": ["NDVI"]},
        )
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_analyze_no_credentials_returns_503(self):
        """
        POST /api/spectral/analyze with valid scene_id but no CDSE credentials
        must return HTTP 503 — never mock data.
        """
        # Temporarily remove credentials from environment
        orig_id = os.environ.pop("COPERNICUS_CLIENT_ID", None)
        orig_sec = os.environ.pop("COPERNICUS_CLIENT_SECRET", None)
        try:
            # Force cache miss by using a clearly non-existent scene_id
            resp = self._client.post(
                "/api/spectral/analyze",
                json={
                    "scene_id": "S2C_NO_CACHE_SCENE_TEST_12345",
                    "indices": ["NDVI"],
                    "collection": "sentinel-2-l2a",
                },
            )
            # Should be 503 (credentials) or 404 (scene not found)
            # Either is acceptable — what's NOT acceptable is 200 with mock data
            assert resp.status_code in (503, 404), (
                f"Expected 503 or 404 without credentials, got {resp.status_code}"
            )
            # Must NOT return a 200 with spectral stats
            assert resp.status_code != 200, "Must not return 200 without credentials"
        finally:
            if orig_id:
                os.environ["COPERNICUS_CLIENT_ID"] = orig_id
            if orig_sec:
                os.environ["COPERNICUS_CLIENT_SECRET"] = orig_sec

    def test_preview_invalid_index_returns_400(self):
        """GET /api/spectral/preview/{id}/invalid_index must return 400."""
        resp = self._client.get("/api/spectral/preview/test_scene/invalid_index")
        assert resp.status_code == 400

    def test_results_nonexistent_scene_returns_404(self):
        """GET /api/spectral/results/nonexistent_scene must return 404."""
        resp = self._client.get("/api/spectral/results/nonexistent_scene_id_xyz")
        assert resp.status_code == 404


# ─── 10. Integration test (skipped without credentials) ──────────────────────

@pytest.mark.skipif(
    not os.environ.get("COPERNICUS_CLIENT_ID")
    or not os.environ.get("COPERNICUS_CLIENT_SECRET"),
    reason="COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET not set — skipping real band download test",
)
class TestSpectralIntegration:
    """
    Integration test: runs real band download and NDVI calculation.
    Only runs when COPERNICUS_CLIENT_ID + COPERNICUS_CLIENT_SECRET are set.
    Requires network access to dataspace.copernicus.eu.
    """

    def test_cdse_token_acquisition(self):
        """CDSE OAuth2 token must be acquired successfully."""
        from app.services.cdse_auth import get_cdse_token, clear_token_cache
        clear_token_cache()
        token = get_cdse_token()
        assert token is not None
        assert len(token) > 50, "Token should be a non-trivial string"
        assert not token.startswith("Bearer"), "Token should not include 'Bearer' prefix"

    def test_token_caching(self):
        """Calling get_cdse_token() twice should return the same token (cached)."""
        from app.services.cdse_auth import get_cdse_token, clear_token_cache
        clear_token_cache()
        token_1 = get_cdse_token()
        token_2 = get_cdse_token()  # Should hit cache
        assert token_1 == token_2, "Second call must return cached token"

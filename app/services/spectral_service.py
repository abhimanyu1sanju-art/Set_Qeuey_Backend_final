"""
SatQuery AI — Spectral Analysis Service (Phase 2)

Orchestrates real Sentinel-2 L2A spectral index calculations.

Architecture:
  spectral.py (router)
      ↓
  spectral_service.py   ← this file
      ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │ 1. Validate scene_id — fetch from MongoDB / STAC API            │
  │ 2. Check MongoDB cache — return result if already processed     │
  │ 3. Acquire CDSE Bearer token (cdse_auth.py)                     │
  │ 4. Resolve band asset HTTPS URLs from scene assets              │
  │ 5. Configure GDAL auth                                           │
  │ 6. Read required raster bands via /vsicurl/ (raster_service.py) │
  │ 7. Align band resolutions (NBR: B12 20m → B08 10m grid)         │
  │ 8. Calculate spectral indices (pure NumPy)                       │
  │ 9. Compute pixel statistics                                       │
  │ 10. Write GeoTIFF output files                                   │
  │ 11. Generate PNG preview for frontend                             │
  │ 12. Save analysis record to MongoDB                               │
  │ 13. Return SpectralAnalysisResponse                              │
  └──────────────────────────────────────────────────────────────────┘

Index formulas (all deterministic NumPy — Gemini is NOT involved):
  NDVI = (B08 - B04) / (B08 + B04)   NIR - RED / NIR + RED
  NDWI = (B03 - B08) / (B03 + B08)   GREEN - NIR / GREEN + NIR
  NBR  = (B08 - B12) / (B08 + B12)   NIR - SWIR2 / NIR + SWIR2

Division-by-zero protection: np.where(denom != 0, numer/denom, np.nan)

Resolution handling:
  B02/B03/B04/B08 are 10m native.
  B11/B12 are 20m native.
  NBR resamples B12 (20m) → 10m using bilinear interpolation.
  All outputs are at the working resolution of the primary band pair.

Caching strategy:
  Cache key: (scene_id, index_type)
  Cache store: MongoDB spectral_analyses + on-disk GeoTIFF presence check.
  If GeoTIFF exists on disk AND MongoDB record exists → return cached.
  force_reprocess=True bypasses cache.

Never:
  - Returns mock or estimated spectral values.
  - Invokes Gemini for numerical calculations.
  - Exposes Bearer tokens to the frontend or in logs.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import HTTPException

from app.core.config import settings
from app.db.mongodb import get_spectral_analyses_collection, get_satellite_scenes_collection
from app.schemas.spectral import (
    INDEX_BANDS,
    IndexResult,
    IndexStats,
    SpectralAnalysisRecord,
    SpectralAnalysisResponse,
)
from app.services import cdse_auth, raster_service
from app.services.satellite_service import get_scene_metadata

logger = logging.getLogger(__name__)

# ─── ID generation ────────────────────────────────────────────────────────────

def _generate_record_id() -> str:
    try:
        from ulid import ULID
        return f"spectral_{ULID()}"
    except ImportError:
        import uuid
        return f"spectral_{uuid.uuid4().hex}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Output directory ─────────────────────────────────────────────────────────

def get_output_dir(scene_id: str) -> Path:
    """Return the output directory for a scene's spectral products."""
    base = Path(settings.spectral_output_dir).resolve()
    return base / _safe_scene_dirname(scene_id)


def _safe_scene_dirname(scene_id: str) -> str:
    """Make scene_id safe for use as a directory name."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in scene_id)


def ensure_output_dirs() -> None:
    """Create the spectral output directory tree on startup."""
    base = Path(settings.spectral_output_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    logger.info("Spectral output directory ready: %s", base)


# ─── MongoDB cache helpers ────────────────────────────────────────────────────

def _load_cached_analysis(scene_id: str, index_type: str) -> Optional[dict]:
    """
    Load a cached spectral analysis record from MongoDB.
    Returns the record dict if found and the output GeoTIFF exists on disk.
    Returns None if not cached or output file is missing.
    """
    try:
        col = get_spectral_analyses_collection()
        doc = col.find_one(
            {"scene_id": scene_id, "index_type": index_type, "status": "completed"},
            {"_id": 0},
        )
        if not doc:
            return None

        # Verify the GeoTIFF still exists on disk
        output_path = doc.get("output_path")
        if output_path and not Path(output_path).exists():
            logger.info(
                "Cached record for %s/%s found but GeoTIFF missing — will reprocess",
                scene_id, index_type,
            )
            return None

        logger.info("Cache hit: %s/%s", scene_id, index_type)
        return doc

    except Exception as exc:
        logger.warning("Cache lookup failed for %s/%s: %s", scene_id, index_type, exc)
        return None


def _save_analysis_record(record: SpectralAnalysisRecord) -> None:
    """Upsert a spectral analysis record to MongoDB. Non-fatal on failure."""
    try:
        col = get_spectral_analyses_collection()
        doc = record.model_dump()
        col.update_one(
            {"scene_id": record.scene_id, "index_type": record.index_type},
            {"$set": doc},
            upsert=True,
        )
    except Exception as exc:
        logger.warning(
            "Failed to save spectral record %s/%s (non-fatal): %s",
            record.scene_id, record.index_type, exc,
        )


def _save_failed_record(
    scene_id: str, index_type: str, error_msg: str
) -> None:
    """Record a failed analysis attempt in MongoDB."""
    try:
        col = get_spectral_analyses_collection()
        col.update_one(
            {"scene_id": scene_id, "index_type": index_type},
            {
                "$set": {
                    "record_id": _generate_record_id(),
                    "scene_id": scene_id,
                    "collection": "sentinel-2-l2a",
                    "index_type": index_type,
                    "status": "failed",
                    "source": "Copernicus Data Space",
                    "created_at": _now(),
                    "error_message": error_msg,
                    "output_path": None,
                }
            },
            upsert=True,
        )
    except Exception:
        pass  # Non-fatal


# ─── Index calculation functions (pure NumPy) ─────────────────────────────────

def _safe_index(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Compute (numerator / denominator) safely.

    - Division by zero → NaN (not 0, not inf, not crash)
    - NaN propagation preserved
    - Result clipped to [-1, 1] (valid spectral index range)

    This function is deterministic and contains no AI/heuristic logic.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(
            denominator != 0,
            numerator / denominator,
            np.nan,
        )
    # Clip to valid spectral range (values outside are sensor artefacts)
    result = np.where(
        np.isfinite(result),
        np.clip(result, -1.0, 1.0),
        np.nan,
    )
    return result.astype(np.float32)


def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    NDVI = (NIR - RED) / (NIR + RED)

    Bands: B08 (NIR, 10m) and B04 (RED, 10m)

    Valid range: approximately -1 to +1
      < 0     : water, clouds, bare rock
      0 – 0.2 : sparse vegetation, bare soil
      0.2–0.4 : light vegetation
      > 0.4   : dense/healthy vegetation
    """
    numerator = nir.astype(np.float32) - red.astype(np.float32)
    denominator = nir.astype(np.float32) + red.astype(np.float32)
    return _safe_index(numerator, denominator)


def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    NDWI = (GREEN - NIR) / (GREEN + NIR)

    Formulation: McFeeters (1996)
    Bands: B03 (GREEN, 10m) and B08 (NIR, 10m)

    This is the McFeeters NDWI, not the Gao (1996) moisture index.
    Positive values indicate water presence; negative values indicate vegetation/land.
    """
    numerator = green.astype(np.float32) - nir.astype(np.float32)
    denominator = green.astype(np.float32) + nir.astype(np.float32)
    return _safe_index(numerator, denominator)


def calculate_nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    NBR = (NIR - SWIR2) / (NIR + SWIR2)

    Bands: B08 (NIR, 10m) and B12 (SWIR2, resampled to 10m from 20m native)
    Resampling: bilinear (see raster_service.align_band_to_reference)

    High NBR: healthy vegetation
    Low NBR / negative NBR: burned areas, bare rock, post-fire
    dNBR (pre-fire NBR minus post-fire NBR) would indicate burn severity
    — but that requires two scenes (not implemented in this Phase).
    """
    numerator = nir.astype(np.float32) - swir2.astype(np.float32)
    denominator = nir.astype(np.float32) + swir2.astype(np.float32)
    return _safe_index(numerator, denominator)


# ─── Single-index processing ──────────────────────────────────────────────────

def _process_single_index(
    index_name: str,
    scene_id: str,
    assets: dict,
    output_dir: Path,
    force_reprocess: bool,
) -> IndexResult:
    """
    Process a single spectral index for one scene.

    Auth strategy (in priority order):
      1. CDSE S3 credentials (CDSE_S3_ACCESS_KEY + CDSE_S3_SECRET_KEY)
         -> GDAL reads via /vsis3/ using S3 URIs from STAC assets.
      2. OAuth2 Bearer token (COPERNICUS_CLIENT_ID + COPERNICUS_CLIENT_SECRET)
         -> GDAL reads via /vsicurl/ using HTTPS URLs (Bearer auth header).

    Returns an IndexResult (success or failure) — never raises.
    Errors are captured and returned in IndexResult.error_message.
    """
    start_time = time.monotonic()
    index_upper = index_name.upper()
    out_tiff = output_dir / f"{index_upper.lower()}.tif"
    out_png = output_dir / f"{index_upper.lower()}_preview.png"

    # -- Cache check ----------------------------------------------------------
    if not force_reprocess:
        cached = _load_cached_analysis(scene_id, index_upper)
        if cached:
            stats_dict = cached.get("stats") or {}
            stats = IndexStats(**stats_dict) if stats_dict else None
            return IndexResult(
                index=index_upper,
                status="completed",
                formula=_get_formula(index_upper),
                bands_used=INDEX_BANDS[index_upper],
                processing_resolution_m=10,
                resampling_method="bilinear" if index_upper == "NBR" else None,
                crs=cached.get("crs"),
                stats=stats,
                output_geotiff_path=str(cached.get("output_path") or out_tiff),
                preview_available=out_png.exists(),
                processing_time_seconds=None,
                error_message=None,
                cached=True,
                cloud_masking="nodata_only",
            )

    # -- Determine auth mode and configure GDAL -------------------------------
    use_s3 = settings.cdse_s3_configured

    if use_s3:
        # Preferred: GDAL /vsis3/ with S3 credentials
        logger.info("Using CDSE S3 credentials for band access (access_key prefix: %s****)",
                    settings.cdse_s3_access_key[:4] if settings.cdse_s3_access_key else "?")
        raster_service.configure_gdal_s3_auth(
            access_key=settings.cdse_s3_access_key,
            secret_key=settings.cdse_s3_secret_key,   # never logged
            endpoint_host=settings.cdse_s3_endpoint_host,
        )
        clear_fn = raster_service.clear_gdal_s3_auth
    else:
        # Fallback: GDAL /vsicurl/ with OAuth2 Bearer token
        if not settings.copernicus_configured:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Neither CDSE S3 credentials (CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY) "
                    "nor OAuth2 credentials (COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET) "
                    "are configured. Add them to your .env file."
                ),
            )
        try:
            token = cdse_auth.get_cdse_token()
        except HTTPException:
            raise
        except Exception as exc:
            return _failed_result(index_upper, f"CDSE authentication failed: {exc}", start_time)
        logger.info("Using CDSE OAuth2 Bearer token for band access (S3 not configured)")
        raster_service.configure_gdal_auth(token)
        clear_fn = raster_service.clear_gdal_auth

    try:
        result = _do_process_index(
            index_name=index_upper,
            assets=assets,
            out_tiff=out_tiff,
            out_png=out_png,
            start_time=start_time,
            prefer_s3_uri=use_s3,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Index %s processing failed: %s", index_upper, exc, exc_info=True)
        result = _failed_result(index_upper, str(exc), start_time)
    finally:
        clear_fn()

    return result


def _resolve_s3_uri(assets: dict, band_name: str) -> Optional[str]:
    """
    Extract the raw S3 URI (s3://eodata/...) for a specific band from
    the STAC asset dictionary.

    Used when CDSE S3 credentials are configured so GDAL accesses the
    band via /vsis3/ (faster, more reliable than Bearer-token /vsicurl/).

    Returns None if the band has no S3 URI (caller should fall back to HTTPS).
    Security: never logs or returns credential values.
    """
    band_upper = band_name.upper()
    for asset_name, asset_info in assets.items():
        if not hasattr(asset_info, "bands") or not asset_info.bands:
            continue
        if band_upper in (b.upper() for b in asset_info.bands):
            href = getattr(asset_info, "href", None) or ""
            if href.startswith("s3://"):
                return href
    return None


def _do_process_index(
    index_name: str,
    assets: dict,
    out_tiff: Path,
    out_png: Path,
    start_time: float,
    prefer_s3_uri: bool = False,
) -> IndexResult:
    """
    Execute the actual band reads and index calculation.

    prefer_s3_uri=True  -> pass raw s3:// URI to read_band_from_url (/vsis3/ mode)
    prefer_s3_uri=False -> resolve to HTTPS URL for /vsicurl/ Bearer-token mode
    """
    bands = INDEX_BANDS[index_name]
    resampling_method = None

    # -- Resolve band URLs ----------------------------------------------------
    # When prefer_s3_uri, we look for the raw S3 URI in each asset so GDAL
    # uses /vsis3/ with the configured AWS_* credentials.
    # Otherwise we use the HTTPS URL for /vsicurl/ Bearer-token access.
    band_urls = {}
    for band in bands:
        if prefer_s3_uri:
            url = _resolve_s3_uri(assets, band)
            if not url:
                # Fallback to HTTPS if no S3 URI found
                url = raster_service.resolve_band_url(assets, band)
                logger.warning("Band %s has no S3 URI -- falling back to HTTPS URL", band)
        else:
            url = raster_service.resolve_band_url(assets, band)

        if not url:
            return _failed_result(
                index_name,
                f"Band {band} not found in scene assets. "
                f"This band is required for {index_name}.",
                start_time,
            )
        band_urls[band] = url

    # ── Read bands ────────────────────────────────────────────────────────
    band_data: Dict[str, Tuple[np.ndarray, object, object]] = {}
    for band, url in band_urls.items():
        try:
            arr, transform, crs = raster_service.read_band_from_url(url, band)
            band_data[band] = (arr, transform, crs)
        except RuntimeError as exc:
            return _failed_result(index_name, str(exc), start_time)

    # ── Band alignment ────────────────────────────────────────────────────
    # For NBR: B12 (20m) must be resampled to B08 (10m) grid
    if index_name == "NBR":
        nir_arr, nir_transform, nir_crs = band_data["B08"]
        swir2_arr, swir2_transform, swir2_crs = band_data["B12"]

        logger.info(
            "NBR: resampling B12 (%s) → B08 grid (%s) using bilinear",
            swir2_arr.shape, nir_arr.shape,
        )
        try:
            swir2_aligned = raster_service.align_band_to_reference(
                source_array=swir2_arr,
                source_transform=swir2_transform,
                source_crs=swir2_crs,
                reference_array=nir_arr,
                reference_transform=nir_transform,
                reference_crs=nir_crs,
                resampling_method="bilinear",
            )
        except RuntimeError as exc:
            return _failed_result(index_name, str(exc), start_time)

        band_data["B12"] = (swir2_aligned, nir_transform, nir_crs)
        resampling_method = "bilinear"

    # Use CRS and transform from the first band (all must be same projection after align)
    primary_band = bands[0]
    _, primary_transform, primary_crs = band_data[primary_band]

    # ── Calculate index ───────────────────────────────────────────────────
    if index_name == "NDVI":
        red_arr = band_data["B04"][0]
        nir_arr = band_data["B08"][0]
        index_array = calculate_ndvi(nir=nir_arr, red=red_arr)

    elif index_name == "NDWI":
        green_arr = band_data["B03"][0]
        nir_arr = band_data["B08"][0]
        index_array = calculate_ndwi(green=green_arr, nir=nir_arr)

    elif index_name == "NBR":
        nir_arr = band_data["B08"][0]
        swir2_arr = band_data["B12"][0]
        index_array = calculate_nbr(nir=nir_arr, swir2=swir2_arr)

    else:
        return _failed_result(index_name, f"Unknown index: {index_name}", start_time)

    # ── Statistics ────────────────────────────────────────────────────────
    stats_dict = raster_service.compute_raster_stats(index_array)
    stats = IndexStats(**stats_dict)

    # ── Save GeoTIFF ──────────────────────────────────────────────────────
    out_tiff.parent.mkdir(parents=True, exist_ok=True)
    try:
        raster_service.save_geotiff(
            array=index_array,
            transform=primary_transform,
            crs=primary_crs,
            output_path=out_tiff,
        )
    except RuntimeError as exc:
        return _failed_result(index_name, f"GeoTIFF write failed: {exc}", start_time)

    # ── Generate PNG preview (non-fatal) ──────────────────────────────────
    preview_available = raster_service.generate_preview_png(
        array=index_array,
        index_name=index_name,
        output_path=out_png,
    )

    elapsed = time.monotonic() - start_time
    crs_str = str(primary_crs) if primary_crs else None

    logger.info(
        "%s completed in %.1fs: mean=%.4f valid_pixels=%d",
        index_name, elapsed,
        stats_dict.get("mean") or 0,
        stats_dict.get("valid_pixel_count") or 0,
    )

    return IndexResult(
        index=index_name,
        status="completed",
        formula=_get_formula(index_name),
        bands_used=bands,
        processing_resolution_m=10,
        resampling_method=resampling_method,
        crs=crs_str,
        stats=stats,
        output_geotiff_path=str(out_tiff),
        preview_available=preview_available,
        processing_time_seconds=round(elapsed, 2),
        error_message=None,
        cached=False,
        cloud_masking="nodata_only",
    )


def _failed_result(index_name: str, error_msg: str, start_time: float) -> IndexResult:
    elapsed = time.monotonic() - start_time
    return IndexResult(
        index=index_name,
        status="failed",
        formula=_get_formula(index_name),
        bands_used=INDEX_BANDS.get(index_name, []),
        processing_resolution_m=10,
        crs=None,
        stats=None,
        output_geotiff_path=None,
        preview_available=False,
        processing_time_seconds=round(elapsed, 2),
        error_message=error_msg,
        cached=False,
        cloud_masking="nodata_only",
    )


def _get_formula(index_name: str) -> str:
    formulas = {
        "NDVI": "(NIR - RED) / (NIR + RED)  [B08 - B04 / B08 + B04]",
        "NDWI": "(GREEN - NIR) / (GREEN + NIR)  [B03 - B08 / B03 + B08]  (McFeeters 1996)",
        "NBR":  "(NIR - SWIR2) / (NIR + SWIR2)  [B08 - B12 / B08 + B12]",
    }
    return formulas.get(index_name, f"Unknown formula for {index_name}")


# ─── Main public function ─────────────────────────────────────────────────────

def run_spectral_analysis(
    scene_id: str,
    indices: List[str],
    collection: str = "sentinel-2-l2a",
    force_reprocess: bool = False,
) -> SpectralAnalysisResponse:
    """
    Run real spectral analysis on a Sentinel-2 L2A scene.

    Steps:
      1. Load scene metadata from MongoDB / STAC API.
      2. Verify it is a Sentinel-2 L2A scene.
      3. For each requested index:
         a. Check cache.
         b. Authenticate with CDSE.
         c. Read real bands via /vsicurl/.
         d. Calculate index using NumPy.
         e. Write GeoTIFF.
         f. Generate PNG preview.
         g. Save MongoDB record.
      4. Return SpectralAnalysisResponse.

    Raises HTTP 404 if scene not found.
    Raises HTTP 503 if CDSE credentials not configured.
    Never returns mock data.
    """
    overall_start = time.monotonic()
    now = _now()

    # 1. Load scene metadata
    logger.info("Spectral analysis requested: scene_id=%s indices=%s", scene_id, indices)
    scene = get_scene_metadata(scene_id, collection)
    if scene is None:
        raise HTTPException(
            status_code=404,
            detail=f"Scene '{scene_id}' not found in collection '{collection}'. "
                   "Search for scenes first via POST /api/satellite/search.",
        )

    # 2. Verify scene type
    if scene.satellite != "sentinel-2" or scene.collection != "sentinel-2-l2a":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Spectral analysis requires a Sentinel-2 L2A scene. "
                f"Scene '{scene_id}' is '{scene.collection}' ({scene.satellite}). "
                f"Sentinel-1 SAR scenes do not have optical bands."
            ),
        )

    # 3. Prepare output directory
    output_dir = get_output_dir(scene_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Process each index
    processing_results: Dict[str, IndexResult] = {}
    completed_indices = []
    failed_indices = []

    for index_name in indices:
        logger.info("Processing index: %s for scene: %s", index_name, scene_id)
        try:
            result = _process_single_index(
                index_name=index_name,
                scene_id=scene_id,
                assets=scene.assets,
                output_dir=output_dir,
                force_reprocess=force_reprocess,
            )
        except HTTPException:
            raise  # 503 credentials error — propagate immediately
        except Exception as exc:
            logger.error("Unexpected error processing %s: %s", index_name, exc, exc_info=True)
            result = _failed_result(index_name, f"Unexpected error: {exc}", time.monotonic())

        processing_results[index_name.lower()] = result

        if result.status == "completed":
            completed_indices.append(index_name)
        else:
            failed_indices.append(index_name)

        # Save to MongoDB (non-fatal)
        if result.status == "completed":
            record = SpectralAnalysisRecord(
                record_id=_generate_record_id(),
                scene_id=scene_id,
                collection=collection,
                index_type=index_name,
                status="completed",
                source="Copernicus Data Space",
                created_at=now,
                processing_resolution_m=result.processing_resolution_m,
                crs=result.crs,
                stats=result.stats.model_dump() if result.stats else None,
                output_path=result.output_geotiff_path,
                preview_path=str(output_dir / f"{index_name.lower()}_preview.png")
                             if result.preview_available else None,
                error_message=None,
                cloud_masking=result.cloud_masking,
                resampling_method=result.resampling_method,
                bands_used=result.bands_used,
            )
            _save_analysis_record(record)
        else:
            _save_failed_record(scene_id, index_name, result.error_message or "Unknown error")

    # 5. Determine overall status
    if not failed_indices:
        overall_status = "completed"
    elif not completed_indices:
        overall_status = "failed"
    else:
        overall_status = "partial"

    total_time = round(time.monotonic() - overall_start, 2)
    all_cached = all(r.cached for r in processing_results.values())

    # ── Write to history (non-fatal, only for fresh results) ──────────────────
    if completed_indices and not all_cached:
        try:
            from app.services import history_service
            # Build compact stats summary (no large arrays)
            summary: dict = {"indices": completed_indices, "scene_id": scene_id[:20]}
            for idx_name, idx_result in processing_results.items():
                if idx_result.status == "completed" and idx_result.stats:
                    s = idx_result.stats
                    summary[f"{idx_name}_mean"] = round(s.mean, 4) if s.mean is not None else None
            history_service.create_history_item(
                record_type="spectral_analysis",
                operation_type="spectral_analysis",
                query=f"Spectral analysis ({', '.join(completed_indices)}) on {scene_id[:24]}",
                status="completed",
                scene_ids=[scene_id],
                satellite="sentinel-2",
                result_summary=summary,
            )
        except Exception as exc:
            logger.warning("Failed to write spectral history (non-fatal): %s", exc)

    return SpectralAnalysisResponse(
        scene_id=scene_id,
        collection=collection,
        satellite="sentinel-2",
        source="Copernicus Data Space Ecosystem",
        processing_backend="Rasterio + GDAL + NumPy",
        processing=processing_results,
        overall_status=overall_status,
        requested_indices=indices,
        completed_indices=completed_indices,
        failed_indices=failed_indices,
        total_processing_time_seconds=total_time,
        created_at=now,
        cached=all_cached,
        note=(
            "All spectral values are calculated from real Sentinel-2 L2A "
            "raster bands via Copernicus Data Space. No synthetic values are used."
        ),
    )



def get_cached_result(scene_id: str, index_type: str) -> Optional[dict]:
    """Retrieve a cached spectral analysis record from MongoDB."""
    return _load_cached_analysis(scene_id, index_type.upper())


def get_preview_path(scene_id: str, index_type: str) -> Optional[Path]:
    """Return the path to the PNG preview for a scene/index, if it exists."""
    output_dir = get_output_dir(scene_id)
    preview = output_dir / f"{index_type.lower()}_preview.png"
    return preview if preview.exists() else None

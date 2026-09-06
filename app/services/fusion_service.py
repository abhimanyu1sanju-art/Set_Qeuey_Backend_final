"""
SatQuery AI — Fusion Service (Phase 6)

Real SAR + Optical fusion pipeline. 
Combines Sentinel-1 (VV, VH) with Sentinel-2 L2A (NDVI, NDWI, NBR) deterministically.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def _cv2():
    import cv2
    return cv2

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _gen_result_id(s1_id: str, s2_id: str) -> str:
    slug = f"{s1_id[:8]}_{s2_id[:8]}"
    return f"fus_{slug}_{uuid.uuid4().hex[:6]}"

def _read_band(assets: dict, band: str, use_s3: bool) -> Optional[Tuple[np.ndarray, Any, Any]]:
    from app.services import raster_service
    url = raster_service.resolve_band_url(assets, band, prefer_s3=use_s3)
    if not url:
        return None
    try:
        return raster_service.read_band_from_url(url, band)
    except Exception as exc:
        logger.warning(f"Failed to read band {band}: {exc}")
        return None

def _norm_s2(arr: np.ndarray) -> np.ndarray:
    """Normalize S2 reflectance to [0,1] maintaining nodata as NaN."""
    a = arr.astype(np.float32)
    # S2_NODATA is 0
    nodata_mask = (a == 0)
    a = np.clip(a, 0, 10000) / 10000.0
    a[nodata_mask] = np.nan
    return a

def _safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = np.where(denom != 0, (a - b) / denom, np.nan)
    return idx.astype(np.float32)

def _calc_optical_index(mode: str, assets: dict, use_s3: bool) -> Tuple[np.ndarray, Any, Any, str]:
    if mode == "vegetation" or mode == "general":
        b08_res = _read_band(assets, "B08", use_s3)
        b04_res = _read_band(assets, "B04", use_s3)
        if not b08_res or not b04_res:
            raise RuntimeError("Missing required optical bands (B08, B04) for vegetation mode.")
        b08 = _norm_s2(b08_res[0])
        b04 = _norm_s2(b04_res[0])
        return _safe_index(b08, b04), b08_res[1], b08_res[2], "NDVI"
        
    elif mode == "water":
        b03_res = _read_band(assets, "B03", use_s3)
        b08_res = _read_band(assets, "B08", use_s3)
        if not b03_res or not b08_res:
            raise RuntimeError("Missing required optical bands (B03, B08) for water mode.")
        b03 = _norm_s2(b03_res[0])
        b08 = _norm_s2(b08_res[0])
        return _safe_index(b03, b08), b03_res[1], b03_res[2], "NDWI"
        
    elif mode == "burn":
        b08_res = _read_band(assets, "B08", use_s3)
        b12_res = _read_band(assets, "B12", use_s3)
        if not b08_res or not b12_res:
            raise RuntimeError("Missing required optical bands (B08, B12) for burn mode.")
        b08_arr, transform, crs = b08_res
        b12_arr, _, _ = b12_res
        
        # Upsample B12 (20m) to B08 (10m) grid
        cv2 = _cv2()
        h, w = b08_arr.shape
        if b12_arr.shape != (h, w):
            b12_arr = cv2.resize(b12_arr, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            
        b08 = _norm_s2(b08_arr)
        b12 = _norm_s2(b12_arr)
        return _safe_index(b08, b12), transform, crs, "NBR"
        
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")

def _calc_sar_db(arr: np.ndarray) -> np.ndarray:
    """Convert SAR linear intensity to decibels safely."""
    # S1 nodata is 0. Also clip negatives which are physically invalid for intensity.
    valid_mask = (arr > 0) & np.isfinite(arr)
    db = np.full_like(arr, np.nan, dtype=np.float32)
    db[valid_mask] = 10.0 * np.log10(arr[valid_mask])
    # Typical dB range for SAR backscatter is roughly -30 to 0
    return np.clip(db, -30, 10)

def _generate_preview(fusion_map: np.ndarray, out_path: Path) -> None:
    cv2 = _cv2()
    # Normalize map [0, 1] to [0, 255]
    fusion_norm = np.clip(np.nan_to_num(fusion_map, nan=0.0), 0, 1)
    u8 = (fusion_norm * 255).astype(np.uint8)
    
    # Use colormap for visualization (e.g. JET)
    heatmap = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    
    h, w = heatmap.shape[:2]
    if max(h, w) > 1024:
        scale = 1024.0 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        heatmap = cv2.resize(heatmap, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), heatmap)

def run_fusion_analysis(
    s1_scene_id: str,
    s2_scene_id: str,
    analysis_mode: str,
    confidence_threshold: float,
    force_reprocess: bool,
) -> dict:
    from app.core.config import settings
    from app.db.mongodb import get_satellite_scenes_collection, get_fusion_collection
    from app.services import raster_service, cdse_auth

    col = get_fusion_collection()
    if not force_reprocess:
        query = {
            "sentinel1_scene_id": s1_scene_id,
            "sentinel2_scene_id": s2_scene_id,
            "analysis_mode": analysis_mode,
            "status": "completed"
        }
        cached = col.find_one(query, {"_id": 0})
        if cached:
            cached["cached"] = True
            return cached

    start_time = time.time()
    result_id = _gen_result_id(s1_scene_id, s2_scene_id)

    scenes_col = get_satellite_scenes_collection()
    s1_doc = scenes_col.find_one({"scene_id": s1_scene_id}, {"_id": 0})
    s2_doc = scenes_col.find_one({"scene_id": s2_scene_id}, {"_id": 0})
    if not s1_doc or not s2_doc:
        raise ValueError("One or both scene IDs not found in local database. Please search for them first.")

    use_s3 = bool(settings.cdse_s3_access_key and settings.cdse_s3_secret_key)
    if use_s3:
        raster_service.configure_gdal_s3_auth(
            access_key=settings.cdse_s3_access_key,
            secret_key=settings.cdse_s3_secret_key,
            endpoint_host=settings.cdse_s3_endpoint_host,
        )
    else:
        token = cdse_auth.get_cdse_token()
        if token:
            raster_service.configure_gdal_auth(token)
        else:
            raise RuntimeError("No Copernicus credentials configured.")

    logger.info("Computing Optical Index...")
    optical_index, opt_transform, opt_crs, opt_name = _calc_optical_index(analysis_mode, s2_doc.get("assets", {}), use_s3)
    
    logger.info("Reading SAR VV...")
    # Pols are named 'vv', 'vh' in STAC assets typically, or 'VV', 'VH'
    s1_assets = s1_doc.get("assets", {})
    
    # helper to find asset by name case-insensitively
    # Also handles Copernicus S1 GRD keys like 's1a-iw-grd-vv-20250101...'
    def _find_pol(assets, p):
        p_up = p.upper()
        for k, v in assets.items():
            k_up = k.upper()
            # Exact match or key contains the polarisation as a word segment
            if (
                k_up == p_up
                or k_up.startswith(p_up + "_") or k_up.startswith(p_up + "-")
                or ("-" + p_up + "-") in k_up or ("_" + p_up + "_") in k_up
                or k_up.endswith("-" + p_up) or k_up.endswith("_" + p_up)
            ):
                return k
            if isinstance(v, dict):
                bands = v.get("bands") or []
                if p_up in [b.upper() for b in bands if isinstance(b, str)]:
                    return k
        return None

    vv_key = _find_pol(s1_assets, "VV")
    if not vv_key:
        raise RuntimeError("VV polarization not found in Sentinel-1 scene assets.")
        
    def _read_sar_band(assets: dict, pol_key: str, use_s3: bool):
        from app.services import raster_service
        asset = assets.get(pol_key)
        if not asset:
            return None
        
        # asset might be a dict or an AssetInfo object
        href = asset.get('href') if isinstance(asset, dict) else getattr(asset, 'href', None)
        https_href = asset.get('https_href') if isinstance(asset, dict) else getattr(asset, 'https_href', None)
        
        url = href if (use_s3 and href and href.startswith('s3://')) else (https_href or href)
        if not url:
            return None
            
        try:
            return raster_service.read_band_from_url(url, pol_key)
        except Exception as exc:
            logger.warning(f"Failed to read SAR band {pol_key}: {exc}")
            return None

    vv_res = _read_sar_band(s1_assets, vv_key, use_s3)
    if not vv_res:
        raise RuntimeError(f"Failed to read VV band. Asset key: {vv_key}")
    
    sar_vv, sar_transform, sar_crs = vv_res

    # Try VH
    vh_key = _find_pol(s1_assets, "VH")
    sar_vh = None
    available_pols = ["VV"]
    if vh_key:
        vh_res = _read_sar_band(s1_assets, vh_key, use_s3)
        if vh_res:
            sar_vh = vh_res[0]
            available_pols.append("VH")

    logger.info("Geospatially aligning SAR to Optical Grid...")
    # Resample SAR VV to Optical Grid
    aligned_vv = raster_service.align_band_to_reference(
        sar_vv, sar_transform, sar_crs,
        optical_index, opt_transform, opt_crs,
        resampling_method="bilinear"
    )
    
    aligned_vh = None
    if sar_vh is not None:
        aligned_vh = raster_service.align_band_to_reference(
            sar_vh, sar_transform, sar_crs,
            optical_index, opt_transform, opt_crs,
            resampling_method="bilinear"
        )
    
    logger.info("Computing Fusion Scores...")
    # Convert to dB
    vv_db = _calc_sar_db(aligned_vv)
    if aligned_vh is not None:
        vh_db = _calc_sar_db(aligned_vh)
        
    # Heuristic fusion algorithm based on analysis_mode
    fusion_map = np.full_like(optical_index, np.nan, dtype=np.float32)
    valid_mask = np.isfinite(optical_index) & np.isfinite(vv_db)
    
    if analysis_mode == "vegetation" or analysis_mode == "general":
        # NDVI high (1.0), VV dB high (-5) -> High vegetation structure
        # Normalize VV to [0,1] assuming -20 to 0 range
        vv_norm = np.clip((vv_db[valid_mask] + 20) / 20.0, 0, 1)
        opt_norm = np.clip(optical_index[valid_mask], 0, 1)
        fusion_map[valid_mask] = 0.5 * opt_norm + 0.5 * vv_norm
        
    elif analysis_mode == "water":
        # NDWI high (1.0), VV dB low (-20) -> Calm water
        vv_norm = np.clip((vv_db[valid_mask] + 25) / 20.0, 0, 1) # Low backscatter = water
        opt_norm = np.clip((optical_index[valid_mask] + 1) / 2.0, 0, 1) 
        fusion_map[valid_mask] = 0.6 * opt_norm + 0.4 * (1.0 - vv_norm)
        
    elif analysis_mode == "burn":
        # NBR low (-1.0), VH cross-pol decreases after burn
        vv_norm = np.clip((vv_db[valid_mask] + 20) / 20.0, 0, 1)
        opt_norm = np.clip((1.0 - optical_index[valid_mask]) / 2.0, 0, 1) # invert NBR so burn is high
        fusion_map[valid_mask] = 0.7 * opt_norm + 0.3 * (1.0 - vv_norm)

    # Thresholding
    fusion_map[fusion_map < confidence_threshold] = np.nan
    
    # Calculate stats
    stats = raster_service.compute_raster_stats(fusion_map)
    
    logger.info("Generating Fusion Preview...")
    preview_path = Path("outputs") / "fusion" / f"{result_id}.png"
    preview_image_b64 = None
    try:
        _generate_preview(fusion_map, preview_path)
        if preview_path.exists():
            import base64
            preview_image_b64 = base64.b64encode(preview_path.read_bytes()).decode('ascii')
    except Exception as exc:
        logger.warning("Fusion preview generation failed (non-fatal): %s", exc)
    
    processing_time = round(time.time() - start_time, 2)
    
    result_doc = {
        "result_id": result_id,
        "status": "completed",
        "sentinel1_scene_id": s1_scene_id,
        "sentinel2_scene_id": s2_scene_id,
        "analysis_mode": analysis_mode,
        "available_polarizations": available_pols,
        "optical_index_used": opt_name,
        "crs": str(opt_crs),
        "dimensions": list(optical_index.shape),
        "resolution_m": 10.0,
        "fusion_score_stats": stats,
        "created_at": _now().isoformat(),
        "error": None,
        "cached": False,
        "processing": {
            "duration_seconds": processing_time,
            "real_data": True,
            "resampling": "bilinear"
        },
        "preview_image_b64": preview_image_b64,
    }
    
    try:
        col.update_one(
            {
                "sentinel1_scene_id": s1_scene_id, 
                "sentinel2_scene_id": s2_scene_id, 
                "analysis_mode": analysis_mode
            },
            {"$set": result_doc},
            upsert=True,
        )
        logger.info("Fusion result saved: %s", result_id)
    except Exception as exc:
        logger.warning("Failed to save fusion result: %s", exc)

    # ── Write to history (non-fatal) ──────────────────────────────────────────
    try:
        from app.services import history_service
        stats = result_doc.get("fusion_score_stats") or {}
        history_service.create_history_item(
            record_type="sar_optical_fusion",
            operation_type="sar_optical_fusion",
            query=f"SAR+Optical fusion ({analysis_mode}) — S1: {s1_scene_id[:12]} / S2: {s2_scene_id[:12]}",
            status="completed",
            scene_ids=[s1_scene_id, s2_scene_id],
            result_id=result_id,
            satellite="sentinel-1+sentinel-2",
            result_summary={
                "analysis_mode": analysis_mode,
                "optical_index": result_doc.get("optical_index_used"),
                "fusion_mean": stats.get("mean"),
                "fusion_max": stats.get("max"),
                "duration_s": result_doc.get("processing", {}).get("duration_seconds"),
            },
        )
    except Exception as exc:
        logger.warning("Failed to write fusion history (non-fatal): %s", exc)

    return result_doc


def run_fusion_analysis_safe(
    s1_scene_id: str,
    s2_scene_id: str,
    analysis_mode: str,
    confidence_threshold: float,
    force_reprocess: bool,
) -> dict:
    import traceback
    try:
        return run_fusion_analysis(
            s1_scene_id, s2_scene_id, analysis_mode, confidence_threshold, force_reprocess
        )
    except Exception as exc:
        logger.error("Fusion analysis failed: %s", exc)
        logger.debug(traceback.format_exc())
        return {
            "result_id": _gen_result_id(s1_scene_id, s2_scene_id),
            "status": "failed",
            "sentinel1_scene_id": s1_scene_id,
            "sentinel2_scene_id": s2_scene_id,
            "analysis_mode": analysis_mode,
            "available_polarizations": [],
            "optical_index_used": "UNKNOWN",
            "error": str(exc),
            "cached": False,
            "created_at": _now().isoformat(),
            "processing": {
                "duration_seconds": 0.0,
                "real_data": True
            }
        }

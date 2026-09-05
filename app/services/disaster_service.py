"""
SatQuery AI — Disaster Detection Service (Phase 5)

Real satellite-based disaster detection pipeline using Copernicus Sentinel-2 L2A imagery.
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

# ─── Constants ────────────────────────────────────────────────────────────────
S2_NODATA = 0
S2_MAX_DN = 10_000

# Disaster thresholds
NDWI_THRESHOLD = 0.1      # Flood / Water
NBR_BURN_THRESHOLD = -0.1 # Burn area indication (often negative or decreased NBR)
NDVI_DROUGHT_THRESHOLD = 0.25 # Lower NDVI implies stress, but we'll use change if before scene is provided
MIN_REGION_PIXELS = 10

MODEL_INFO = {
    "name": "spectral-disaster-indicators-v1",
    "version": "1.0.0",
    "description": (
        "Phase 5 Disaster Detection using NDWI, NBR, and NDVI spectral "
        "indices on Sentinel-2 L2A data. Computes real pixel counts and areas."
    ),
}

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _gen_result_id(scene_id: str, dtype: str) -> str:
    slug = scene_id[:12].replace(" ", "_")
    return f"dis_{dtype}_{slug}_{uuid.uuid4().hex[:6]}"

# ─── Lazy imports ─────────────────────────────────────────────────────────────

def _cv2():
    import cv2
    return cv2

# ─── Band helpers ─────────────────────────────────────────────────────────────

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

def _resample_to_shape(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    cv2 = _cv2()
    if arr.shape == (target_h, target_w):
        return arr
    return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

def _norm(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    nodata_mask = (a == S2_NODATA)
    a = np.clip(a, 0, S2_MAX_DN) / S2_MAX_DN
    a[nodata_mask] = np.nan
    return a

# ─── Spectral indices ─────────────────────────────────────────────────────────

def _safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = np.where(denom != 0, (a - b) / denom, np.nan)
    return idx.astype(np.float32)

def compute_ndwi(b03: np.ndarray, b08: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR). Positive = water."""
    return _safe_index(b03, b08)

def compute_nbr(b08: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """NBR = (NIR - SWIR2) / (NIR + SWIR2). Burned areas have low NBR."""
    return _safe_index(b08, b12)

def compute_ndvi(b08: np.ndarray, b04: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). Positive = vegetation."""
    return _safe_index(b08, b04)

# ─── Region extraction ────────────────────────────────────────────────────────

def _extract_regions(
    binary_mask: np.ndarray,
    transform,
    confidence_map: np.ndarray,
    label: str,
    detection_method: str,
    min_pixels: int,
    pixel_area_m2: float,
    conf_threshold: float
) -> list:
    cv2 = _cv2()
    mask_u8 = (binary_mask.astype(np.uint8)) * 255
    n_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    detections = []
    
    if n_labels > 1:
        flat_labels = labels_img.ravel()
        flat_conf = confidence_map.ravel()
        valid_mask = np.isfinite(flat_conf)
        
        valid_labels = flat_labels[valid_mask]
        valid_conf = flat_conf[valid_mask]
        
        conf_sum = np.bincount(valid_labels, weights=valid_conf, minlength=n_labels)
        conf_count = np.bincount(valid_labels, minlength=n_labels)

    for component_id in range(1, n_labels):
        pixel_count = int(stats[component_id, cv2.CC_STAT_AREA])
        if pixel_count < min_pixels:
            continue

        left   = int(stats[component_id, cv2.CC_STAT_LEFT])
        top    = int(stats[component_id, cv2.CC_STAT_TOP])
        width  = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])

        min_x, min_y = transform * (left, top + height)
        max_x, max_y = transform * (left + width, top)

        cx_px, cy_px = centroids[component_id]
        cx_geo, cy_geo = transform * (cx_px, cy_px)

        count = conf_count[component_id]
        if count > 0:
            confidence = float(conf_sum[component_id] / count)
        else:
            confidence = 0.5
            
        confidence = float(np.clip(confidence, 0.0, 1.0))
        if confidence < conf_threshold:
            continue

        area_m2 = float(pixel_count * pixel_area_m2)

        detections.append({
            "label": label,
            "pixel_count": pixel_count,
            "area_m2": round(area_m2, 1),
            "confidence": round(confidence, 4),
            "bbox": [
                round(float(min_x), 2),
                round(float(min_y), 2),
                round(float(max_x), 2),
                round(float(max_y), 2),
            ],
            "centroid": [round(float(cx_geo), 2), round(float(cy_geo), 2)],
            "detection_method": detection_method,
        })
    return detections

def _generate_preview(
    base_heatmap: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    colormap_type=None
) -> None:
    cv2 = _cv2()
    base_display = np.nan_to_num(base_heatmap, nan=0.0)
    base_norm = np.clip((base_display + 1.0) / 2.0, 0, 1)
    base_u8 = (base_norm * 255).astype(np.uint8)
    
    if colormap_type == 'water':
        heatmap = cv2.applyColorMap(base_u8, cv2.COLORMAP_OCEAN)
        overlay_color = (255, 150, 0) # Cyan for water overlay
    elif colormap_type == 'fire':
        heatmap = cv2.applyColorMap(base_u8, cv2.COLORMAP_HOT)
        overlay_color = (0, 0, 255) # Red for burn overlay
    else:
        heatmap = cv2.applyColorMap(base_u8, cv2.COLORMAP_SUMMER)
        overlay_color = (0, 150, 255) # Orange for generic disaster

    overlay = heatmap.copy()
    overlay[mask > 0] = overlay_color

    blended = cv2.addWeighted(heatmap, 0.6, overlay, 0.4, 0)
    
    h, w = blended.shape[:2]
    if max(h, w) > 1024:
        scale = 1024.0 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        blended = cv2.resize(blended, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_disaster_analysis(
    scene_id: str,
    before_scene_id: Optional[str],
    disaster_type: str,
    confidence_threshold: float,
    force_reprocess: bool,
) -> dict:
    from app.core.config import settings
    from app.db.mongodb import get_satellite_scenes_collection, get_disaster_analyses_collection
    from app.services import raster_service, cdse_auth

    col = get_disaster_analyses_collection()
    if not force_reprocess:
        query = {"scene_id": scene_id, "disaster_type": disaster_type, "status": "completed"}
        if before_scene_id:
            query["before_scene_id"] = before_scene_id
        cached = col.find_one(query, {"_id": 0})
        if cached:
            cached["cached"] = True
            return cached

    start_time = time.time()
    result_id = _gen_result_id(scene_id, disaster_type)

    scenes_col = get_satellite_scenes_collection()
    scene_doc = scenes_col.find_one({"scene_id": scene_id}, {"_id": 0})
    if not scene_doc:
        raise ValueError(f"Scene {scene_id} not found in database.")

    assets = scene_doc.get("assets", {})
    
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

    # Variables to fill
    regions = []
    affected_area_m2 = 0.0
    evidence = {}
    bands_used = []
    severity = "Unknown"
    colormap_type = "default"
    base_map_for_preview = None
    mask_for_preview = None
    crs = None
    pixel_area_m2 = 100.0

    if disaster_type == "flood":
        # Need B03 (Green), B08 (NIR)
        res_b03 = _read_band(assets, "B03", use_s3)
        res_b08 = _read_band(assets, "B08", use_s3)
        if not res_b03 or not res_b08:
            raise RuntimeError("Required bands B03/B08 missing for Flood analysis")
        
        b03_arr, transform, crs = res_b03
        b08_arr, _, _ = res_b08
        h, w = b03_arr.shape
        bands_used = ["B03", "B08"]

        b03 = _norm(b03_arr)
        b08 = _norm(b08_arr)
        
        ndwi = compute_ndwi(b03, b08)
        valid = np.isfinite(ndwi)
        ndwi_clean = np.nan_to_num(ndwi, nan=0.0)

        mask = ((ndwi_clean > NDWI_THRESHOLD) & valid).astype(np.uint8)
        cv2 = _cv2()
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

        conf_map = np.clip((ndwi_clean - NDWI_THRESHOLD) / (1.0 - NDWI_THRESHOLD), 0, 1)

        pixel_area_m2 = abs(float(transform[0])) * abs(float(transform[4]))
        regions = _extract_regions(
            mask, transform, conf_map, "water_extent", "ndwi_segmentation",
            MIN_REGION_PIXELS, pixel_area_m2, confidence_threshold
        )
        
        affected_area_m2 = sum(r["area_m2"] for r in regions)
        valid_ndwi = ndwi[valid]
        evidence = {
            "spectral_metric": "NDWI",
            "spectral_mean": float(np.mean(valid_ndwi)) if len(valid_ndwi) > 0 else 0,
            "notes": "Flood analysis processed using NDWI indicator."
        }
        severity = "High" if affected_area_m2 > 1000000 else "Moderate"
        colormap_type = "water"
        base_map_for_preview = ndwi
        mask_for_preview = mask

    elif disaster_type in ["fire", "burn"]:
        res_b08 = _read_band(assets, "B08", use_s3)
        res_b12 = _read_band(assets, "B12", use_s3) # SWIR2 (20m)
        if not res_b08 or not res_b12:
            raise RuntimeError("Required bands B08/B12 missing for Fire/Burn analysis")

        b08_arr, transform, crs = res_b08
        b12_arr_20, _, _ = res_b12
        h, w = b08_arr.shape
        b12_arr = _resample_to_shape(b12_arr_20, h, w)
        bands_used = ["B08", "B12"]

        b08 = _norm(b08_arr)
        b12 = _norm(b12_arr)
        
        nbr = compute_nbr(b08, b12)
        valid = np.isfinite(nbr)
        nbr_clean = np.nan_to_num(nbr, nan=0.0)

        mask = ((nbr_clean < NBR_BURN_THRESHOLD) & valid).astype(np.uint8)
        cv2 = _cv2()
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

        # confidence is higher when NBR is lower
        conf_map = np.clip((NBR_BURN_THRESHOLD - nbr_clean) / 0.5, 0, 1)

        pixel_area_m2 = abs(float(transform[0])) * abs(float(transform[4]))
        regions = _extract_regions(
            mask, transform, conf_map, "burn_candidate", "nbr_segmentation",
            MIN_REGION_PIXELS, pixel_area_m2, confidence_threshold
        )
        
        affected_area_m2 = sum(r["area_m2"] for r in regions)
        valid_nbr = nbr[valid]
        evidence = {
            "spectral_metric": "NBR",
            "spectral_mean": float(np.mean(valid_nbr)) if len(valid_nbr) > 0 else 0,
        }
        severity = "High" if affected_area_m2 > 500000 else "Moderate"
        colormap_type = "fire"
        base_map_for_preview = -nbr # invert for heatmap so low NBR is hot
        mask_for_preview = mask

    elif disaster_type == "drought":
        res_b04 = _read_band(assets, "B04", use_s3)
        res_b08 = _read_band(assets, "B08", use_s3)
        if not res_b04 or not res_b08:
            raise RuntimeError("Required bands B04/B08 missing for Drought analysis")

        b04_arr, _, _ = res_b04
        b08_arr, transform, crs = res_b08
        h, w = b08_arr.shape
        bands_used = ["B04", "B08"]

        b04 = _norm(b04_arr)
        b08 = _norm(b08_arr)
        
        ndvi = compute_ndvi(b08, b04)
        valid = np.isfinite(ndvi)
        ndvi_clean = np.nan_to_num(ndvi, nan=0.0)

        # Vegetation stress: low NDVI
        mask = ((ndvi_clean < NDVI_DROUGHT_THRESHOLD) & (ndvi_clean > 0) & valid).astype(np.uint8)
        
        conf_map = np.clip((NDVI_DROUGHT_THRESHOLD - ndvi_clean) / NDVI_DROUGHT_THRESHOLD, 0, 1)

        pixel_area_m2 = abs(float(transform[0])) * abs(float(transform[4]))
        regions = _extract_regions(
            mask, transform, conf_map, "vegetation_stress", "ndvi_segmentation",
            MIN_REGION_PIXELS, pixel_area_m2, confidence_threshold
        )
        
        affected_area_m2 = sum(r["area_m2"] for r in regions)
        valid_ndvi = ndvi[valid]
        evidence = {
            "spectral_metric": "NDVI",
            "spectral_mean": float(np.mean(valid_ndvi)) if len(valid_ndvi) > 0 else 0,
            "notes": "Single-scene vegetation stress indicator."
        }
        severity = "Moderate"
        colormap_type = "default"
        base_map_for_preview = ndvi
        mask_for_preview = mask
        
    else:
        # Fallback for landslide, cyclone, infrastructure that need optical indicators
        # Just use NDVI change proxy or Phase 3 data for now in this skeleton
        raise ValueError(f"Disaster type {disaster_type} is implemented via Phase 3 change detection or not fully supported in this optical pass yet.")

    # Preview
    if base_map_for_preview is not None and mask_for_preview is not None:
        preview_dir = Path("outputs") / "disaster"
        preview_path = preview_dir / f"{result_id}.png"
        try:
            _generate_preview(base_map_for_preview, mask_for_preview, preview_path, colormap_type)
        except Exception as exc:
            logger.warning("Preview generation failed: %s", exc)

    processing_time = round(time.time() - start_time, 2)
    
    total_pixels = sum(r["pixel_count"] for r in regions)
    total_confidence = sum(r["confidence"] for r in regions)
    mean_confidence = total_confidence / len(regions) if regions else 0.0

    result_doc = {
        "result_id": result_id,
        "scene_id": scene_id,
        "before_scene_id": before_scene_id,
        "source": "Copernicus Data Space",
        "crs": str(crs) if crs else None,
        "resolution_m": 10.0,
        "disaster_type": disaster_type,
        "status": "completed",
        "affected_area_m2": round(affected_area_m2, 2),
        "affected_area_ha": round(affected_area_m2 / 10000, 2),
        "pixel_count": total_pixels,
        "region_count": len(regions),
        "change_percent": None,
        "confidence": round(mean_confidence, 4),
        "severity": severity,
        "regions": regions,
        "evidence": evidence,
        "model": MODEL_INFO,
        "processing": {
            "duration_seconds": processing_time,
            "real_data": True,
            "bands_used": bands_used,
            "resolution_m": 10.0,
        },
        "error": None,
        "cached": False,
        "created_at": _now().isoformat(),
    }

    try:
        col.update_one(
            {"scene_id": scene_id, "disaster_type": disaster_type},
            {"$set": result_doc},
            upsert=True,
        )
        logger.info("Disaster result saved: %s", result_id)
    except Exception as exc:
        logger.warning("Failed to save disaster result: %s", exc)

    # ── Write to history (non-fatal) ──────────────────────────────────────────
    try:
        from app.services import history_service
        scene_ids = [scene_id]
        if result_doc.get("before_scene_id"):
            scene_ids.append(result_doc["before_scene_id"])
        history_service.create_history_item(
            record_type="disaster_analysis",
            operation_type="disaster_analysis",
            query=f"{disaster_type.title()} detection on scene {scene_id}",
            status="completed",
            scene_ids=scene_ids,
            result_id=result_id,
            satellite="sentinel-2",
            result_summary={
                "disaster_type": disaster_type,
                "affected_area_ha": result_doc.get("affected_area_ha", 0),
                "pixel_count": result_doc.get("pixel_count", 0),
                "region_count": result_doc.get("region_count", 0),
                "confidence": result_doc.get("confidence"),
            },
        )
    except Exception as exc:
        logger.warning("Failed to write disaster history (non-fatal): %s", exc)

    return result_doc


def run_disaster_analysis_safe(
    scene_id: str,
    before_scene_id: Optional[str],
    disaster_type: str,
    confidence_threshold: float,
    force_reprocess: bool,
) -> dict:
    import traceback
    try:
        return run_disaster_analysis(
            scene_id, before_scene_id, disaster_type, confidence_threshold, force_reprocess
        )
    except Exception as exc:
        logger.error("Disaster analysis failed for %s: %s", scene_id, exc)
        logger.debug(traceback.format_exc())
        return {
            "result_id": _gen_result_id(scene_id, disaster_type),
            "scene_id": scene_id,
            "before_scene_id": before_scene_id,
            "disaster_type": disaster_type,
            "status": "failed",
            "error": str(exc),
            "affected_area_m2": 0,
            "affected_area_ha": 0,
            "pixel_count": 0,
            "region_count": 0,
            "confidence": 0,
            "regions": [],
            "evidence": {},
            "model": MODEL_INFO,
            "processing": {
                "duration_seconds": 0.0,
                "real_data": True,
                "bands_used": [],
                "resolution_m": 10.0,
            },
            "cached": False,
            "created_at": _now().isoformat(),
        }

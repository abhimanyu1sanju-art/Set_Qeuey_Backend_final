"""
SatQuery AI — Infrastructure Detection Service (Phase 4)

Real satellite-based infrastructure detection pipeline using spectral index
segmentation on Copernicus Sentinel-2 L2A imagery.

Algorithm:
    1. Fetch scene metadata from MongoDB (bands B04, B08, B11, B02, B03)
    2. Authenticate with Copernicus S3 (or HTTPS fallback)
    3. Read band arrays via existing raster_service (GDAL /vsis3/ or /vsicurl/)
    4. Resample B11 (20m) → 10m using OpenCV bilinear (no full rasterio dep needed)
    5. Compute spectral indices:
         NDBI  = (B11 − B08) / (B11 + B08)  → built-up/buildings
         NDVI  = (B08 − B04) / (B08 + B04)  → vegetation (inverted for road mask)
         BSI   = ((B11+B04)−(B08+B02)) / ((B11+B04)+(B08+B02))  → bare soil / roads
    6. Threshold + morphological operations (OpenCV) → binary masks
    7. Connected components → extract regions with area/bbox/centroid
    8. Map pixel coords → geographic coords using raster transform
    9. Tag construction candidates: new NDBI-positive regions overlapping Phase 3
       change map (if available); else use high-NDBI magnitude alone
    10. Generate preview PNG (colorized composite)
    11. Persist result to MongoDB 'infrastructure_results' collection
    12. Return structured InfraAnalysisResponse

Resolution note:
    Sentinel-2 10m resolution means each pixel ≈ 100 m².
    Individual buildings (typically 50–500 m²) often occupy 1–5 pixels.
    We detect built-up CLUSTERS, not individual building outlines — this is
    the correct, honest interpretation of 10m satellite imagery.

Never:
    - Returns mock/hardcoded detection results.
    - Invents pixel counts, areas, or geographic coordinates.
    - Claims individual building detections that exceed the data resolution.
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

# Thresholds calibrated for Sentinel-2 L2A surface reflectance (×10000 DN)
# NDBI > NDBI_THRESHOLD → built-up pixel
NDBI_THRESHOLD_DEFAULT = 0.05      # ~0.05 in normalized reflectance
# Road pixels: low NDVI, moderate BSI, linear structure
BSI_THRESHOLD_DEFAULT = 0.05
# Minimum cluster size in pixels to report (10px = ~1000 m²)
MIN_BUILDING_PIXELS = 10
MIN_ROAD_PIXELS = 5

MODEL_INFO = {
    "name": "spectral-index-segmentation-v1",
    "version": "1.0.0",
    "description": (
        "NDBI/BSI/NDVI spectral index segmentation for Sentinel-2 L2A at 10m. "
        "Appropriate for satellite urban/road mapping at this resolution. "
        "No COCO YOLO used — COCO classes are not suitable for 10m satellite imagery."
    ),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _gen_result_id(scene_id: str) -> str:
    slug = scene_id[:12].replace(" ", "_")
    return f"inf_{slug}_{uuid.uuid4().hex[:8]}"


# ─── Lazy imports ─────────────────────────────────────────────────────────────

def _cv2():
    import cv2
    return cv2


def _rasterio():
    import rasterio
    return rasterio


# ─── Band helpers ─────────────────────────────────────────────────────────────

def _read_band(assets: dict, band: str, use_s3: bool) -> Optional[Tuple[np.ndarray, Any, Any]]:
    """Read a band using existing raster_service utilities. Returns (arr, transform, crs)."""
    from app.services import raster_service
    url = raster_service.resolve_band_url(assets, band, prefer_s3=use_s3)
    if not url:
        logger.warning("Band %s not found in scene assets", band)
        return None
    try:
        arr, transform, crs = raster_service.read_band_from_url(url, band)
        return arr, transform, crs
    except Exception as exc:
        logger.warning("Failed to read band %s: %s", band, exc)
        return None


def _resample_to_shape(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Upsample or downsample a 2D array to (target_h, target_w) using bilinear interpolation."""
    cv2 = _cv2()
    if arr.shape == (target_h, target_w):
        return arr
    # OpenCV resize: (width, height) order
    resized = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


# ─── Spectral index computation ───────────────────────────────────────────────

def _safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b) with NaN where denominator is 0."""
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = np.where(denom != 0, (a - b) / denom, np.nan)
    return idx.astype(np.float32)


def compute_ndbi(b11: np.ndarray, b08: np.ndarray) -> np.ndarray:
    """NDBI = (SWIR1 − NIR) / (SWIR1 + NIR). Positive = built-up / bare."""
    return _safe_index(b11, b08)


def compute_ndvi(b08: np.ndarray, b04: np.ndarray) -> np.ndarray:
    """NDVI = (NIR − Red) / (NIR + Red). Positive = vegetation."""
    return _safe_index(b08, b04)


def compute_bsi(b11: np.ndarray, b04: np.ndarray, b08: np.ndarray, b02: np.ndarray) -> np.ndarray:
    """BSI = ((SWIR1+Red) − (NIR+Blue)) / ((SWIR1+Red) + (NIR+Blue)). Positive = bare soil/roads."""
    a = b11 + b04
    b_ = b08 + b02
    return _safe_index(a, b_)


# ─── Region extraction ────────────────────────────────────────────────────────

def _extract_regions(
    binary_mask: np.ndarray,
    transform,
    confidence_map: np.ndarray,
    label: str,
    detection_method: str,
    min_pixels: int,
    pixel_area_m2: float,
) -> list:
    """
    Use OpenCV connected components to extract detection regions from a binary mask.
    Returns list of dicts (will be serialized as InfraDetection).
    """
    cv2 = _cv2()
    mask_u8 = (binary_mask.astype(np.uint8)) * 255
    n_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    detections = []
    
    if n_labels > 1:
        # Optimize confidence calculation (avoid O(N*M) boolean mask creation)
        flat_labels = labels_img.ravel()
        flat_conf = confidence_map.ravel()
        valid_mask = np.isfinite(flat_conf)
        
        valid_labels = flat_labels[valid_mask]
        valid_conf = flat_conf[valid_mask]
        
        conf_sum = np.bincount(valid_labels, weights=valid_conf, minlength=n_labels)
        conf_count = np.bincount(valid_labels, minlength=n_labels)

    # stats shape: (n_labels, 5) — [left, top, width, height, area]
    for component_id in range(1, n_labels):  # skip background (0)
        pixel_count = int(stats[component_id, cv2.CC_STAT_AREA])
        if pixel_count < min_pixels:
            continue

        left   = int(stats[component_id, cv2.CC_STAT_LEFT])
        top    = int(stats[component_id, cv2.CC_STAT_TOP])
        width  = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])

        # Convert pixel bbox → geographic coords using affine transform
        # transform * (col, row) → (x, y)
        min_x, min_y = transform * (left, top + height)
        max_x, max_y = transform * (left + width, top)

        cx_px, cy_px = centroids[component_id]
        cx_geo, cy_geo = transform * (cx_px, cy_px)

        count = conf_count[component_id]
        if count > 0:
            confidence = float(conf_sum[component_id] / count)
        else:
            confidence = 0.5
            
        # Clip to 0–1
        confidence = float(np.clip(confidence, 0.0, 1.0))

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


# ─── Preview generation ───────────────────────────────────────────────────────

def _generate_preview(
    ndbi: np.ndarray,
    building_mask: np.ndarray,
    road_mask: np.ndarray,
    construction_mask: np.ndarray,
    out_path: Path,
) -> None:
    """Generate a colorized PNG showing index heatmap + masks."""
    cv2 = _cv2()
    # Normalize NDBI to 0–255 for heatmap background
    ndbi_display = np.nan_to_num(ndbi, nan=0.0)
    ndbi_norm = np.clip((ndbi_display + 1.0) / 2.0, 0, 1)
    ndbi_u8 = (ndbi_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(ndbi_u8, cv2.COLORMAP_MAGMA)

    # Overlay: buildings = cyan, roads = yellow, construction = red
    overlay = heatmap.copy()
    overlay[building_mask > 0] = (255, 200, 0)      # cyan-ish
    overlay[road_mask > 0] = (0, 220, 255)           # yellow
    overlay[construction_mask > 0] = (0, 50, 240)    # red (BGR)

    # Blend overlay with heatmap
    blended = cv2.addWeighted(heatmap, 0.6, overlay, 0.4, 0)

    # Resize for preview (max 1024 px on longest side)
    h, w = blended.shape[:2]
    if max(h, w) > 1024:
        scale = 1024.0 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        blended = cv2.resize(blended, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)
    logger.info("Infrastructure preview saved: %s", out_path)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_infrastructure_analysis(
    scene_id: str,
    detection_types: List[str],
    confidence_threshold: float,
    force_reprocess: bool,
) -> dict:
    """
    Full Phase 4 infrastructure detection pipeline.

    Args:
        scene_id: Sentinel-2 STAC scene ID (must exist in satellite_scenes MongoDB collection)
        detection_types: list of types to detect: 'buildings', 'roads', 'construction'
        confidence_threshold: minimum confidence to include a detection (0–1)
        force_reprocess: if True, ignore cached result

    Returns:
        dict matching InfraAnalysisResponse schema
    """
    from app.core.config import settings
    from app.db.mongodb import (
        get_satellite_scenes_collection,
        get_infrastructure_results_collection,
        get_change_analyses_collection,
    )
    from app.services import raster_service, cdse_auth

    # ── 0. Cache check ────────────────────────────────────────────────────────
    col = get_infrastructure_results_collection()
    if not force_reprocess:
        cached = col.find_one({"scene_id": scene_id, "status": "completed"}, {"_id": 0})
        if cached:
            cached["cached"] = True
            return cached

    start_time = time.time()

    result_id = _gen_result_id(scene_id)

    # ── 1. Load scene metadata ────────────────────────────────────────────────
    scenes_col = get_satellite_scenes_collection()
    scene_doc = scenes_col.find_one({"scene_id": scene_id}, {"_id": 0})
    if not scene_doc:
        raise ValueError(f"Scene {scene_id} not found in database. Run a satellite search first.")

    assets = scene_doc.get("assets", {})
    acquired_at = scene_doc.get("datetime") or scene_doc.get("date")

    # ── 2. Auth ───────────────────────────────────────────────────────────────
    use_s3 = bool(settings.cdse_s3_access_key and settings.cdse_s3_secret_key)
    if use_s3:
        raster_service.configure_gdal_s3_auth(
            access_key=settings.cdse_s3_access_key,
            secret_key=settings.cdse_s3_secret_key,
            endpoint_host=settings.cdse_s3_endpoint_host,
        )
        logger.info("Using CDSE S3 (/vsis3/) for band reads")
    else:
        token = cdse_auth.get_cdse_token()
        if token:
            raster_service.configure_gdal_auth(token)
            logger.info("Using CDSE Bearer token (/vsicurl/) for band reads")
        else:
            raise RuntimeError("No Copernicus credentials configured. Set CDSE_S3_* or COPERNICUS_CLIENT_* in .env")

    # ── 3. Read bands ─────────────────────────────────────────────────────────
    logger.info("Reading bands for scene %s", scene_id)

    # B08 NIR (10m) — required for NDBI, NDVI
    result_b08 = _read_band(assets, "B08", use_s3)
    if result_b08 is None:
        raise RuntimeError("Band B08 (NIR) unavailable — cannot compute spectral indices")
    b08_arr, transform, crs = result_b08
    h, w = b08_arr.shape
    logger.info("B08 shape: %dx%d  CRS: %s", h, w, crs)

    # B04 Red (10m) — for NDVI, BSI
    result_b04 = _read_band(assets, "B04", use_s3)
    if result_b04 is None:
        raise RuntimeError("Band B04 (Red) unavailable — cannot compute NDVI")
    b04_arr, _, _ = result_b04

    # B11 SWIR1 (20m) — for NDBI, BSI (must resample to 10m)
    result_b11 = _read_band(assets, "B11", use_s3)
    if result_b11 is None:
        raise RuntimeError("Band B11 (SWIR1) unavailable — cannot compute NDBI (required for building detection)")
    b11_arr_20m, _, _ = result_b11

    # B02 Blue (10m) — for BSI (road detection)
    result_b02 = _read_band(assets, "B02", use_s3)
    if result_b02 is None:
        logger.warning("B02 unavailable — road BSI will use simplified formula")
        b02_arr = np.zeros_like(b08_arr)
    else:
        b02_arr, _, _ = result_b02

    # ── 4. Resample B11 20m → 10m ────────────────────────────────────────────
    logger.info("Resampling B11 from 20m to 10m (bilinear, OpenCV)")
    b11_arr = _resample_to_shape(b11_arr_20m, h, w)

    # Normalize all bands from DN to reflectance (0–1 float)
    def _norm(arr: np.ndarray) -> np.ndarray:
        a = arr.astype(np.float32)
        nodata_mask = (a == S2_NODATA)
        a = np.clip(a, 0, S2_MAX_DN) / S2_MAX_DN
        a[nodata_mask] = np.nan
        return a

    b08 = _norm(b08_arr)
    b04 = _norm(b04_arr)
    b11 = _norm(b11_arr)
    b02 = _norm(b02_arr)

    # Valid pixel mask (all bands must be non-NaN)
    valid = np.isfinite(b08) & np.isfinite(b04) & np.isfinite(b11)

    # ── 5. Compute spectral indices ───────────────────────────────────────────
    logger.info("Computing NDBI, NDVI, BSI spectral indices")
    ndbi = compute_ndbi(b11, b08)
    ndvi = compute_ndvi(b08, b04)
    bsi  = compute_bsi(b11, b04, b08, b02)

    # Replace NaN with 0 for masking operations (NaN = invalid pixel)
    ndbi_clean = np.nan_to_num(ndbi, nan=0.0)
    ndvi_clean = np.nan_to_num(ndvi, nan=0.0)
    bsi_clean  = np.nan_to_num(bsi,  nan=0.0)

    # Pixel size from transform
    res_x = abs(float(transform[0]))
    res_y = abs(float(transform[4]))
    pixel_area_m2 = res_x * res_y
    logger.info("Pixel size: %.1fm x %.1fm = %.1f m²/pixel", res_x, res_y, pixel_area_m2)

    cv2 = _cv2()

    all_buildings: list = []
    all_roads: list = []
    all_construction: list = []

    # ── 6. Building / urban cluster detection ─────────────────────────────────
    if "buildings" in detection_types or "all" in detection_types:
        logger.info("Running building/urban-cluster detection (NDBI > %.3f, NDVI < 0.1)", NDBI_THRESHOLD_DEFAULT)

        # Built-up: high NDBI AND low NDVI (exclude vegetation)
        building_mask = (
            (ndbi_clean > NDBI_THRESHOLD_DEFAULT) &
            (ndvi_clean < 0.15) &
            valid
        ).astype(np.uint8)

        # Morphological close (fill small gaps within built-up blocks)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_CLOSE, kernel)
        building_mask = cv2.morphologyEx(building_mask, cv2.MORPH_OPEN,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # Confidence map: scale NDBI → 0–1 confidence
        building_conf = np.clip((ndbi_clean - NDBI_THRESHOLD_DEFAULT) / (0.5 - NDBI_THRESHOLD_DEFAULT), 0, 1)

        raw_buildings = _extract_regions(
            binary_mask=building_mask,
            transform=transform,
            confidence_map=building_conf,
            label="building_cluster",
            detection_method="ndbi_segmentation",
            min_pixels=MIN_BUILDING_PIXELS,
            pixel_area_m2=pixel_area_m2,
        )
        all_buildings = [d for d in raw_buildings if d["confidence"] >= confidence_threshold]
        logger.info("Buildings: %d clusters found (before threshold: %d)", len(all_buildings), len(raw_buildings))
    else:
        building_mask = np.zeros((h, w), dtype=np.uint8)
        building_conf = np.zeros((h, w), dtype=np.float32)

    # ── 7. Road / linear feature detection ───────────────────────────────────
    if "roads" in detection_types or "all" in detection_types:
        logger.info("Running road detection (BSI > %.3f, NDVI < 0.05)", BSI_THRESHOLD_DEFAULT)

        # Road pixels: high BSI (bare soil / impervious) + very low NDVI
        road_raw_mask = (
            (bsi_clean > BSI_THRESHOLD_DEFAULT) &
            (ndvi_clean < 0.05) &
            (ndbi_clean < 0.4) &   # Exclude very high-NDBI (buildings)
            valid
        ).astype(np.uint8)

        # Apply morphological skeletonization-like thinning
        # Thin: open to remove noise, then dilate slightly for connectivity
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        road_mask = cv2.morphologyEx(road_raw_mask, cv2.MORPH_OPEN, k3)
        k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))   # horizontal
        k5v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # vertical
        road_h = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k5)
        road_v = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k5v)
        road_mask = cv2.bitwise_or(road_h, road_v)

        road_conf = np.clip((bsi_clean - BSI_THRESHOLD_DEFAULT) / (0.4 - BSI_THRESHOLD_DEFAULT), 0, 1)

        raw_roads = _extract_regions(
            binary_mask=road_mask,
            transform=transform,
            confidence_map=road_conf,
            label="road_segment",
            detection_method="bsi_morphology",
            min_pixels=MIN_ROAD_PIXELS,
            pixel_area_m2=pixel_area_m2,
        )
        all_roads = [d for d in raw_roads if d["confidence"] >= confidence_threshold]
        logger.info("Roads: %d segments found (before threshold: %d)", len(all_roads), len(raw_roads))
    else:
        road_mask = np.zeros((h, w), dtype=np.uint8)

    # ── 8. Construction-change candidates ────────────────────────────────────
    construction_mask = np.zeros((h, w), dtype=np.uint8)
    if "construction" in detection_types or "all" in detection_types:
        logger.info("Running construction candidate detection")

        change_col = get_change_analyses_collection()
        # Look for any completed change analysis that used this scene as before or after
        change_doc = change_col.find_one(
            {
                "$or": [
                    {"before_scene_id": scene_id},
                    {"after_scene_id": scene_id},
                ],
                "status": "completed",
            },
            {"_id": 0, "regions": 1, "crs": 1},
        )

        if change_doc and change_doc.get("regions"):
            logger.info("Found Phase 3 change data — intersecting with NDBI mask for construction candidates")
            # Build a change overlay mask from Phase 3 region bboxes
            change_overlay = np.zeros((h, w), dtype=np.uint8)
            for region in change_doc["regions"]:
                bbox = region.get("bbox", [])
                if len(bbox) == 4:
                    # bbox is in CRS units → convert to pixel coords
                    minx_crs, miny_crs, maxx_crs, maxy_crs = bbox
                    # Invert transform: pixel = transform.inverse * (x, y)
                    try:
                        inv = ~transform
                        col_min, row_max = inv * (minx_crs, miny_crs)
                        col_max, row_min = inv * (maxx_crs, maxy_crs)
                        r0 = max(0, int(row_min))
                        r1 = min(h, int(row_max))
                        c0 = max(0, int(col_min))
                        c1 = min(w, int(col_max))
                        if r1 > r0 and c1 > c0:
                            change_overlay[r0:r1, c0:c1] = 1
                    except Exception:
                        pass

            # Construction candidate = change overlap + high NDBI (new built-up)
            construction_mask = (
                (change_overlay > 0) &
                (ndbi_clean > NDBI_THRESHOLD_DEFAULT) &
                valid
            ).astype(np.uint8)

            # Clean up
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            construction_mask = cv2.morphologyEx(construction_mask, cv2.MORPH_CLOSE, kern)

            construction_conf = np.clip(ndbi_clean * 2.0, 0, 1)  # scaled confidence

            raw_construction = _extract_regions(
                binary_mask=construction_mask,
                transform=transform,
                confidence_map=construction_conf,
                label="construction_change_candidate",
                detection_method="change_ndbi_overlap",
                min_pixels=MIN_BUILDING_PIXELS,
                pixel_area_m2=pixel_area_m2,
            )
            all_construction = [d for d in raw_construction if d["confidence"] >= confidence_threshold]
            logger.info("Construction candidates: %d (from Phase 3 + NDBI overlap)", len(all_construction))
        else:
            # No Phase 3 data — use high-NDBI alone as proxy, clearly labelled
            logger.info("No Phase 3 change data found — using NDBI > 0.2 high-magnitude as construction proxy")
            high_ndbi_mask = (
                (ndbi_clean > 0.20) &
                (ndvi_clean < 0.1) &
                valid
            ).astype(np.uint8)
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            construction_mask = cv2.morphologyEx(high_ndbi_mask, cv2.MORPH_CLOSE, kern)

            construction_conf = np.clip((ndbi_clean - 0.20) / 0.3, 0, 1)
            raw_construction = _extract_regions(
                binary_mask=construction_mask,
                transform=transform,
                confidence_map=construction_conf,
                label="construction_change_candidate",
                detection_method="high_ndbi_proxy_no_change_data",
                min_pixels=MIN_BUILDING_PIXELS,
                pixel_area_m2=pixel_area_m2,
            )
            all_construction = [d for d in raw_construction if d["confidence"] >= confidence_threshold]
            logger.info("Construction proxy candidates: %d", len(all_construction))

    # ── 9. Generate preview PNG ───────────────────────────────────────────────
    preview_dir = Path("outputs") / "infrastructure"
    preview_path = preview_dir / f"{result_id}.png"
    try:
        _generate_preview(ndbi, building_mask, road_mask, construction_mask, preview_path)
    except Exception as exc:
        logger.warning("Preview generation failed (non-fatal): %s", exc)

    # ── 10. Summary stats ─────────────────────────────────────────────────────
    total_built_area = sum(d["area_m2"] for d in all_buildings)
    total_road_area  = sum(d["area_m2"] for d in all_roads)

    valid_ndbi = ndbi[np.isfinite(ndbi)]
    valid_ndvi = ndvi[np.isfinite(ndvi)]

    summary = {
        "building_cluster_count": len(all_buildings),
        "road_segment_count": len(all_roads),
        "construction_candidate_count": len(all_construction),
        "total_built_up_area_m2": round(float(total_built_area), 1),
        "total_road_area_m2": round(float(total_road_area), 1),
        "ndbi_mean": round(float(np.mean(valid_ndbi)), 4) if len(valid_ndbi) > 0 else None,
        "ndvi_mean": round(float(np.mean(valid_ndvi)), 4) if len(valid_ndvi) > 0 else None,
    }

    processing_time = round(time.time() - start_time, 2)

    # ── 11. Build result document ─────────────────────────────────────────────
    result_doc = {
        "result_id": result_id,
        "scene_id": scene_id,
        "source": "Copernicus Data Space",
        "acquired_at": str(acquired_at) if acquired_at else None,
        "crs": str(crs) if crs else None,
        "resolution_m": float(res_x),
        "model": MODEL_INFO,
        "detections": {
            "buildings": all_buildings,
            "roads": all_roads,
            "construction_change_candidates": all_construction,
        },
        "summary": summary,
        "processing": {
            "duration_seconds": processing_time,
            "real_data": True,
            "bands_used": ["B08", "B04", "B11", "B02"],
            "resolution_m": float(res_x),
            "resampling_applied": "B11 20m→10m bilinear (OpenCV)" if b11_arr_20m.shape != (h, w) else None,
        },
        "status": "completed",
        "error": None,
        "cached": False,
        "created_at": _now().isoformat(),
    }

    # ── 12. Persist to MongoDB ────────────────────────────────────────────────
    try:
        col.update_one(
            {"scene_id": scene_id},
            {"$set": result_doc},
            upsert=True,
        )
        logger.info("Infrastructure result saved: %s (result_id=%s)", scene_id, result_id)
    except Exception as exc:
        logger.warning("Failed to save infrastructure result to MongoDB (non-fatal): %s", exc)

    # ── 13. Write to history (non-fatal) ─────────────────────────────────────
    try:
        from app.services import history_service
        history_service.create_history_item(
            record_type="infrastructure_analysis",
            operation_type="infrastructure_analysis",
            query=f"Infrastructure detection on scene {scene_id}",
            status="completed",
            scene_ids=[scene_id],
            result_id=result_id,
            satellite="sentinel-2",
            result_summary={
                "building_clusters": result_doc.get("summary", {}).get("building_cluster_count", 0),
                "road_segments": result_doc.get("summary", {}).get("road_segment_count", 0),
                "construction_candidates": result_doc.get("summary", {}).get("construction_candidate_count", 0),
                "built_up_area_m2": result_doc.get("summary", {}).get("total_built_up_area_m2"),
            },
        )
    except Exception as exc:
        logger.warning("Failed to write infrastructure history (non-fatal): %s", exc)

    return result_doc



def run_infrastructure_analysis_safe(
    scene_id: str,
    detection_types: List[str],
    confidence_threshold: float,
    force_reprocess: bool,
) -> dict:
    """Wrapper that catches exceptions and returns a failed result doc instead of raising."""
    import traceback
    try:
        return run_infrastructure_analysis(
            scene_id=scene_id,
            detection_types=detection_types,
            confidence_threshold=confidence_threshold,
            force_reprocess=force_reprocess,
        )
    except Exception as exc:
        logger.error("Infrastructure analysis failed for %s: %s", scene_id, exc)
        logger.debug(traceback.format_exc())
        from datetime import datetime, timezone
        return {
            "result_id": _gen_result_id(scene_id),
            "scene_id": scene_id,
            "source": "Copernicus Data Space",
            "acquired_at": None,
            "crs": None,
            "resolution_m": 10.0,
            "model": MODEL_INFO,
            "detections": {"buildings": [], "roads": [], "construction_change_candidates": []},
            "summary": {
                "building_cluster_count": 0, "road_segment_count": 0,
                "construction_candidate_count": 0,
                "total_built_up_area_m2": 0.0, "total_road_area_m2": 0.0,
                "ndbi_mean": None, "ndvi_mean": None,
            },
            "processing": {
                "duration_seconds": 0.0, "real_data": True,
                "bands_used": [], "resolution_m": 10.0, "resampling_applied": None,
            },
            "status": "failed",
            "error": str(exc),
            "cached": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

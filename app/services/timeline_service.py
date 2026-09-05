"""
SatQuery AI — Timeline Service (Phase 7)

Changes from initial implementation:
  - Early return when STAC search returns 0 scenes (no raster reads attempted)
  - Concurrent scene processing (max 3 workers) instead of sequential
  - Per-scene 60s timeout via concurrent.futures
  - Overall processing is bounded: min(req.limit, 10) scenes max
"""
import logging
import concurrent.futures
from datetime import datetime, timezone
import numpy as np
from typing import List, Dict, Optional, Any

from fastapi import HTTPException

from app.schemas.satellite import SatelliteSearchRequest
from app.schemas.timeline import (
    TimelineSearchRequest,
    TimelineObservation,
    TimelineChangeMetrics,
    TimelineResponse
)
from app.services.satellite_service import search_scenes, get_band_data
from app.services.raster_service import read_band_window_from_url, configure_gdal_auth, configure_gdal_s3_auth
from app.core.config import settings
from app.services.cdse_auth import get_cdse_token

logger = logging.getLogger(__name__)

# Per-scene processing timeout in seconds
SCENE_TIMEOUT_S = 90

# Maximum scenes to actually process (raster reads) regardless of limit param
MAX_PROCESS_SCENES = 10


def _configure_auth() -> bool:
    """Configure GDAL auth once; return True if using S3."""
    use_s3 = settings.cdse_s3_configured
    if use_s3:
        logger.info("Timeline: Using CDSE S3 credentials for band access")
        configure_gdal_s3_auth(
            access_key=settings.cdse_s3_access_key,
            secret_key=settings.cdse_s3_secret_key,
            endpoint_host=settings.cdse_s3_endpoint_host,
        )
    else:
        logger.info("Timeline: Using CDSE OAuth2 Bearer token")
        configure_gdal_auth(get_cdse_token())
    return use_s3


def _get_band_url(info: dict, use_s3: bool) -> str:
    """Pick the right URL from band asset info depending on auth mode."""
    href = info.get("href", "")
    https_href = info.get("https_href", "")
    if use_s3 and href and href.startswith("s3://"):
        return href
    return https_href or href or ""


def _process_scene(scene, collection: str, bbox: list, satellite: str, use_s3: bool) -> Optional[Dict[str, Any]]:
    """
    Process a single scene: read required bands and compute spectral metrics.
    Returns a dict with metrics, or None on failure.
    Runs in a thread pool — must be thread-safe (only reads, no shared writes).
    """
    metrics: Dict[str, Optional[float]] = {}

    try:
        if satellite == "sentinel-2":
            b4_info = get_band_data(scene.scene_id, collection, "B04")
            b8_info = get_band_data(scene.scene_id, collection, "B08")

            url_b4 = _get_band_url(b4_info, use_s3)
            url_b8 = _get_band_url(b8_info, use_s3)

            if not url_b4 or not url_b8:
                raise ValueError(f"Missing B04/B08 URLs for {scene.scene_id}")

            b4_arr, _, _ = read_band_window_from_url(url_b4, "B04", bbox)
            b8_arr, _, _ = read_band_window_from_url(url_b8, "B08", bbox)

            # NDVI = (NIR - Red) / (NIR + Red)
            denominator = b8_arr + b4_arr
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi_arr = np.where(denominator == 0, np.nan, (b8_arr - b4_arr) / denominator)

            valid_ndvi = ndvi_arr[~np.isnan(ndvi_arr)]
            metrics["ndvi"] = float(np.mean(valid_ndvi)) if valid_ndvi.size > 0 else None

            # NDWI = (Green - NIR) / (Green + NIR)
            try:
                b3_info = get_band_data(scene.scene_id, collection, "B03")
                url_b3 = _get_band_url(b3_info, use_s3)
                if url_b3:
                    b3_arr, _, _ = read_band_window_from_url(url_b3, "B03", bbox)
                    den_ndwi = b3_arr + b8_arr
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ndwi_arr = np.where(den_ndwi == 0, np.nan, (b3_arr - b8_arr) / den_ndwi)
                    valid_ndwi = ndwi_arr[~np.isnan(ndwi_arr)]
                    metrics["ndwi"] = float(np.mean(valid_ndwi)) if valid_ndwi.size > 0 else None
                else:
                    metrics["ndwi"] = None
            except Exception as e:
                logger.warning("Could not compute NDWI for %s: %s", scene.scene_id, e)
                metrics["ndwi"] = None

        elif satellite == "sentinel-1":
            try:
                vv_info = get_band_data(scene.scene_id, collection, "VV")
                url_vv = _get_band_url(vv_info, use_s3)
                if url_vv:
                    vv_arr, _, _ = read_band_window_from_url(url_vv, "VV", bbox)
                    valid_vv = vv_arr[~np.isnan(vv_arr)]
                    metrics["sar_vv"] = float(np.mean(valid_vv)) if valid_vv.size > 0 else None
                else:
                    metrics["sar_vv"] = None
            except Exception as e:
                logger.warning("Could not compute VV for %s: %s", scene.scene_id, e)
                metrics["sar_vv"] = None

    except Exception as e:
        logger.error("Error processing scene %s: %s", scene.scene_id, e)
        return None

    return metrics


def generate_timeline(req: TimelineSearchRequest) -> TimelineResponse:
    # 1. Configure GDAL auth once
    use_s3 = _configure_auth()

    # 2. Search for scenes via STAC
    search_req = SatelliteSearchRequest(
        bbox=req.bbox,
        start_date=req.start_date,
        end_date=req.end_date,
        satellite=req.satellite,
        max_cloud_cover=req.max_cloud_cover if req.satellite == "sentinel-2" else 100.0,
        limit=50  # Fetch more, then select evenly spaced subset
    )

    search_res = search_scenes(search_req)

    # 3. EARLY RETURN: no scenes found
    if not search_res.scenes:
        logger.info(
            "Timeline: 0 STAC scenes found for bbox=%s dates=%s/%s cloud<=%.0f. Returning empty.",
            req.bbox, req.start_date, req.end_date, req.max_cloud_cover
        )
        return TimelineResponse(
            observations=[],
            total_observations=0,
            satellite=req.satellite,
            bbox=req.bbox
        )

    # 4. Sort chronologically, then select evenly-spaced subset
    scenes = sorted(search_res.scenes, key=lambda s: s.acquired_at)
    n_process = min(req.limit, MAX_PROCESS_SCENES)

    if len(scenes) > n_process:
        step = len(scenes) / float(n_process)
        scenes = [scenes[int(round(i * step))] for i in range(n_process)]

    logger.info(
        "Timeline: processing %d/%d scenes (bbox=%s satellite=%s)",
        len(scenes), len(search_res.scenes), req.bbox, req.satellite
    )

    # 5. Process scenes concurrently (max 3 workers)
    observations: List[TimelineObservation] = []
    prev_metrics: Optional[Dict] = None
    prev_date: Optional[datetime] = None

    # Use ThreadPoolExecutor for concurrent S3 reads
    # We submit per-scene jobs and collect results in order
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_process_scene, scene, search_res.collection, req.bbox, req.satellite, use_s3): scene
            for scene in scenes
        }

        # Collect results maintaining chronological order
        scene_results: Dict = {}
        for future, scene in futures.items():
            try:
                metrics = future.result(timeout=SCENE_TIMEOUT_S)
                scene_results[scene.scene_id] = metrics
            except concurrent.futures.TimeoutError:
                logger.warning("Scene %s timed out after %ds", scene.scene_id, SCENE_TIMEOUT_S)
                scene_results[scene.scene_id] = None
            except Exception as e:
                logger.error("Scene %s processing failed: %s", scene.scene_id, e)
                scene_results[scene.scene_id] = None

    # 6. Build observations in chronological order (scenes list is already sorted)
    for scene in scenes:
        metrics = scene_results.get(scene.scene_id)
        if metrics is None:
            # Scene failed — skip but continue
            continue

        # Compute change deltas vs previous observation
        change_metrics = None
        if prev_metrics is not None and prev_date is not None:
            days_between = (scene.acquired_at - prev_date).days

            def _delta(key):
                cur = metrics.get(key)
                prv = prev_metrics.get(key)
                if cur is not None and prv is not None:
                    return cur - prv
                return None

            ndvi_delta = _delta("ndvi")
            ndwi_delta = _delta("ndwi")
            sar_delta  = _delta("sar_vv")

            if req.satellite == "sentinel-2":
                mag = abs(ndvi_delta) if ndvi_delta is not None else 0.0
                direction = "Increased" if (ndvi_delta and ndvi_delta > 0) else "Decreased"
            else:
                mag = abs(sar_delta) if sar_delta is not None else 0.0
                direction = "Increased" if (sar_delta and sar_delta > 0) else "Decreased"

            change_metrics = TimelineChangeMetrics(
                previous_date=prev_date.isoformat(),
                current_date=scene.acquired_at.isoformat(),
                days_between=days_between,
                ndvi_delta=ndvi_delta,
                ndwi_delta=ndwi_delta,
                sar_delta=sar_delta,
                magnitude=mag,
                direction=direction
            )

        prev_metrics = metrics
        prev_date = scene.acquired_at

        obs = TimelineObservation(
            observation_id=scene.scene_id,
            scene_id=scene.scene_id,
            acquired_at=scene.acquired_at,
            satellite=scene.satellite,
            sensor=scene.sensor,
            collection=scene.collection,
            cloud_cover=scene.cloud_cover,
            bbox=req.bbox,
            metrics=metrics,
            change_from_previous=change_metrics,
            preview_url=scene.preview_url
        )
        observations.append(obs)

    logger.info("Timeline: returning %d observations", len(observations))

    response = TimelineResponse(
        observations=observations,
        total_observations=len(observations),
        satellite=req.satellite,
        bbox=req.bbox
    )

    # ── Write to history (non-fatal) ──────────────────────────────────────────
    if len(observations) > 0:
        try:
            from app.services import history_service
            scene_ids = [obs.scene_id for obs in observations]
            history_service.create_history_item(
                record_type="timeline_analysis",
                operation_type="timeline_analysis",
                query=f"Timeline: {req.satellite} {req.start_date} to {req.end_date}",
                status="completed",
                scene_ids=scene_ids,
                satellite=req.satellite,
                bbox=list(req.bbox) if req.bbox else None,
                date_range={"start": str(req.start_date), "end": str(req.end_date)},
                result_summary={
                    "total_observations": len(observations),
                    "satellite": req.satellite,
                    "start_date": str(req.start_date),
                    "end_date": str(req.end_date),
                },
            )
        except Exception as exc:
            logger.warning("Failed to write timeline history (non-fatal): %s", exc)

    return response


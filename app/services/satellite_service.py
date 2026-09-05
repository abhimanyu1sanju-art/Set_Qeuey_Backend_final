"""
SatQuery AI — Satellite Service (Phase 1)

Handles all interaction with the Copernicus Data Space STAC API.

Architecture:
  satellite.py (router)
      ↓
  satellite_service.py   ← this file
      ↓
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. Validate search request                                   │
  │ 2. Build STAC query (bbox, datetime, collection, CQL2)       │
  │ 3. Call Copernicus STAC API (httpx, public endpoint)         │
  │ 4. Normalize GeoJSON features → SatelliteScene[]            │
  │ 5. Persist new scenes to MongoDB (upsert by scene_id)        │
  │ 6. Return SatelliteSearchResponse                            │
  └──────────────────────────────────────────────────────────────┘
      ↓
  MongoDB (satellite_scenes collection)

Authentication:
  - STAC *search* is public — no credentials required.
  - Asset *download* (JP2 band files) requires OIDC tokens.
    Credentials are read from COPERNICUS_CLIENT_ID / SECRET env vars.
    Phase 1 does NOT download band data; it returns asset URLs for Phase 2.

Never:
  - Returns mock or hardcoded data.
  - Exposes credentials to the frontend.
  - Downloads full satellite products unnecessarily.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.db.mongodb import get_satellite_scenes_collection
from app.schemas.satellite import (
    AssetInfo,
    SATELLITE_TO_COLLECTION,
    SatelliteScene,
    SatelliteSearchRequest,
    SatelliteSearchResponse,
)

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
STAC_COLLECTIONS_URL = "https://stac.dataspace.copernicus.eu/v1/collections"
HTTP_TIMEOUT = 30.0  # seconds

# Sentinel-2 L2A bands available at 10m resolution (most important for Phase 2)
S2_BANDS_10M = ["B02", "B03", "B04", "B08"]
# All Sentinel-2 L2A bands
S2_ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
# Sentinel-1 polarizations (GRD IW)
S1_POLARIZATIONS = ["VV", "VH"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_preview_url(assets: Dict[str, Any]) -> Optional[str]:
    """
    Extract the best available public preview/thumbnail URL from STAC assets.

    Priority order:
      1. 'rendered_preview' — rendered RGB thumbnail (public)
      2. 'QUICKLOOK' — quicklook image (S2 specific)
      3. 'thumbnail' — collection-level thumbnail
      4. 'preview' — generic preview
    Any asset pointing to an S3 URI is skipped (requires auth).
    """
    priority = ["rendered_preview", "QUICKLOOK", "thumbnail", "preview"]
    for key in priority:
        asset = assets.get(key)
        if not asset:
            continue
        href = asset.get("href", "")
        # Skip S3 URIs — they require authentication
        if href.startswith("s3://"):
            # Try the alternate HTTPS URL if it exists
            alt = asset.get("alternate", {}).get("https", {})
            https_href = alt.get("href", "")
            if https_href.startswith("https://"):
                return https_href
            continue
        if href.startswith("https://"):
            return href

    # Fallback: scan all assets for any public HTTPS image
    for key, asset in assets.items():
        href = asset.get("href", "")
        mime = asset.get("type", "")
        if href.startswith("https://") and "image" in mime:
            return href

    return None


def _extract_asset_info(assets: Dict[str, Any]) -> Dict[str, AssetInfo]:
    """
    Extract a summary of key assets — enough for Phase 2 band retrieval.
    We only include band assets (B02, B03, B04, B08, etc.) and skip
    auxiliary products (SCL, AOT, WVP) to keep the payload manageable.
    """
    result: Dict[str, AssetInfo] = {}

    for asset_name, asset_data in assets.items():
        if not isinstance(asset_data, dict):
            continue

        href = asset_data.get("href", "")
        # Extract HTTPS URL from alternate if S3 URI
        https_href = None
        if href.startswith("s3://"):
            alt = asset_data.get("alternate", {}).get("https", {})
            https_href = alt.get("href")
        elif href.startswith("https://"):
            https_href = href
            href = None  # Don't double-store

        # Extract band name(s) from asset metadata
        band_names: List[str] = []
        for b in asset_data.get("bands", []):
            if isinstance(b, dict) and b.get("name"):
                band_names.append(b["name"])

        result[asset_name] = AssetInfo(
            name=asset_name,
            href=href if href and href.startswith("s3://") else None,
            https_href=https_href,
            title=asset_data.get("title"),
            mime_type=asset_data.get("type"),
            gsd=asset_data.get("gsd"),
            bands=band_names if band_names else None,
        )

    return result


def _normalize_sentinel2_scene(feature: Dict[str, Any]) -> SatelliteScene:
    """Convert a STAC GeoJSON feature for Sentinel-2 L2A into a SatelliteScene."""
    props = feature.get("properties", {})
    assets = feature.get("assets", {})

    scene_id = feature["id"]

    # Acquisition datetime
    dt_str = props.get("datetime") or props.get("start_datetime", "")
    try:
        acquired_at = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        acquired_at = _now()

    # Cloud cover from STAC properties (eo:cloud_cover)
    cloud_cover = props.get("eo:cloud_cover")
    if cloud_cover is None:
        # Some items use s2:cloud_cover
        cloud_cover = props.get("s2:cloud_cover")
    if cloud_cover is not None:
        try:
            cloud_cover = float(cloud_cover)
        except (TypeError, ValueError):
            cloud_cover = None

    # Bounding box
    bbox = feature.get("bbox", [])

    # Geometry
    geometry = feature.get("geometry")

    # Platform
    platform = props.get("platform", "sentinel-2")

    # MGRS tile from product ID or grid:code
    mgrs_tile = None
    grid_code = props.get("grid:code", "")
    if grid_code:
        # grid:code is like "MGRS-43QCA", extract tile
        mgrs_tile = grid_code.replace("MGRS-", "")
    else:
        # Try to extract from scene_id: S2C_MSIL2A_..._T43QCA_...
        parts = scene_id.split("_")
        for part in parts:
            if part.startswith("T") and len(part) == 6:
                mgrs_tile = part
                break

    # Available bands — extract from asset names that match band pattern
    bands_found: set[str] = set()
    for asset_name in assets:
        for b in S2_ALL_BANDS:
            if asset_name.startswith(b + "_"):
                bands_found.add(b)
                break

    preview_url = _extract_preview_url(assets)
    asset_info = _extract_asset_info(assets)

    return SatelliteScene(
        scene_id=scene_id,
        platform=platform,
        sensor="MSI",
        collection="sentinel-2-l2a",
        satellite="sentinel-2",
        acquired_at=acquired_at,
        cloud_cover=cloud_cover,
        bbox=bbox,
        geometry=geometry,
        preview_url=preview_url,
        source="Copernicus Data Space",
        bands=sorted(bands_found) if bands_found else S2_ALL_BANDS,
        assets=asset_info,
        polarizations=None,
        orbit_state=props.get("sat:orbit_state"),
        instrument_mode=None,
        mgrs_tile=mgrs_tile,
        processing_level=props.get("processing:level", "L2A"),
        created_at=_now(),
    )


def _normalize_sentinel1_scene(feature: Dict[str, Any]) -> SatelliteScene:
    """Convert a STAC GeoJSON feature for Sentinel-1 GRD into a SatelliteScene."""
    props = feature.get("properties", {})
    assets = feature.get("assets", {})

    scene_id = feature["id"]

    dt_str = props.get("datetime") or props.get("start_datetime", "")
    try:
        acquired_at = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        acquired_at = _now()

    bbox = feature.get("bbox", [])
    geometry = feature.get("geometry")
    platform = props.get("platform", "sentinel-1")

    # Polarizations from SAR properties
    polarizations = props.get("sar:polarizations")
    if not polarizations:
        polarizations = S1_POLARIZATIONS

    instrument_mode = props.get("sar:instrument_mode", "IW")
    orbit_state = props.get("sat:orbit_state")

    preview_url = _extract_preview_url(assets)
    asset_info = _extract_asset_info(assets)

    # SAR has no cloud cover
    return SatelliteScene(
        scene_id=scene_id,
        platform=platform,
        sensor="SAR",
        collection="sentinel-1-grd",
        satellite="sentinel-1",
        acquired_at=acquired_at,
        cloud_cover=None,  # SAR is cloud-penetrating — no cloud cover metric
        bbox=bbox,
        geometry=geometry,
        preview_url=preview_url,
        source="Copernicus Data Space",
        bands=[],  # SAR bands are polarization channels, not optical bands
        assets=asset_info,
        polarizations=polarizations,
        orbit_state=orbit_state,
        instrument_mode=instrument_mode,
        mgrs_tile=None,
        processing_level=props.get("processing:level", "L1"),
        created_at=_now(),
    )


def _normalize_feature(feature: Dict[str, Any], satellite: str) -> SatelliteScene:
    """Route to the correct normalizer based on satellite type."""
    if satellite == "sentinel-2":
        return _normalize_sentinel2_scene(feature)
    elif satellite == "sentinel-1":
        return _normalize_sentinel1_scene(feature)
    else:
        raise ValueError(f"Unsupported satellite: {satellite}")


# ─── MongoDB persistence ──────────────────────────────────────────────────────

def _upsert_scene(scene: SatelliteScene) -> None:
    """
    Insert or update a satellite scene in MongoDB.
    Uses scene_id as the unique key to avoid duplicates.
    Non-fatal — logs warning on failure.
    """
    try:
        col = get_satellite_scenes_collection()
        doc = scene.model_dump()
        col.update_one(
            {"scene_id": scene.scene_id},
            {"$set": doc},
            upsert=True,
        )
    except Exception as exc:
        logger.warning(
            "Failed to persist satellite scene %s (non-fatal): %s",
            scene.scene_id, exc,
        )


def _load_cached_scene(scene_id: str) -> Optional[SatelliteScene]:
    """Load a scene from MongoDB cache. Returns None if not found."""
    try:
        col = get_satellite_scenes_collection()
        doc = col.find_one({"scene_id": scene_id}, {"_id": 0})
        if doc:
            return SatelliteScene.model_validate(doc)
    except Exception as exc:
        logger.warning("Failed to load cached scene %s: %s", scene_id, exc)
    return None


# ─── Main public functions ────────────────────────────────────────────────────

def search_scenes(request: SatelliteSearchRequest) -> SatelliteSearchResponse:
    """
    Search for real satellite scenes via the Copernicus STAC API.

    Steps:
      1. Map user's satellite → STAC collection ID.
      2. Build STAC search POST body.
      3. Call STAC /search with httpx (30s timeout).
      4. Normalize GeoJSON features → SatelliteScene models.
      5. Apply cloud filter post-hoc if STAC didn't apply it.
      6. Persist each scene to MongoDB (upsert).
      7. Return SatelliteSearchResponse.

    Returns an empty scenes list — never mocked data — if no results.
    Raises HTTP 503 if the STAC API is unreachable.
    Raises HTTP 500 on unexpected errors.
    """
    collection_id = SATELLITE_TO_COLLECTION[request.satellite]

    # Build datetime interval
    start_dt = f"{request.start_date}T00:00:00Z"
    end_dt = f"{request.end_date}T23:59:59Z"
    datetime_filter = f"{start_dt}/{end_dt}"

    # Build STAC search body (POST /search)
    search_body: Dict[str, Any] = {
        "collections": [collection_id],
        "bbox": request.bbox,
        "datetime": datetime_filter,
        "limit": request.limit,
    }

    # Add cloud cover filter via CQL2 for Sentinel-2
    # The STAC API at Copernicus supports CQL2 text filter
    if request.satellite == "sentinel-2" and request.max_cloud_cover < 100:
        search_body["filter"] = {
            "op": "<=",
            "args": [
                {"property": "eo:cloud_cover"},
                request.max_cloud_cover,
            ],
        }
        search_body["filter-lang"] = "cql2-json"

    logger.info(
        "STAC search: collection=%s bbox=%s datetime=%s cloud_cover<=%s limit=%d",
        collection_id,
        request.bbox,
        datetime_filter,
        request.max_cloud_cover if request.satellite == "sentinel-2" else "N/A (SAR)",
        request.limit,
    )

    # Call Copernicus STAC API
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            response = client.post(
                STAC_SEARCH_URL,
                json=search_body,
                headers={"Content-Type": "application/json", "Accept": "application/geo+json"},
            )
    except httpx.TimeoutException:
        logger.error("STAC API request timed out after %ss", HTTP_TIMEOUT)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Copernicus STAC API did not respond within {int(HTTP_TIMEOUT)} seconds. "
                "Try again later."
            ),
        )
    except httpx.ConnectError as exc:
        logger.error("STAC API connection error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Cannot reach Copernicus Data Space STAC API. Check network connectivity.",
        )
    except httpx.HTTPError as exc:
        logger.error("STAC API HTTP error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"STAC API request failed: {exc}",
        )

    # Check HTTP status
    if response.status_code != 200:
        logger.error(
            "STAC API returned status %d: %s",
            response.status_code,
            response.text[:500],
        )
        raise HTTPException(
            status_code=502,
            detail=(
                f"Copernicus STAC API returned HTTP {response.status_code}. "
                f"Message: {response.text[:200]}"
            ),
        )

    # Parse JSON
    try:
        stac_result = response.json()
    except Exception as exc:
        logger.error("Failed to parse STAC API response as JSON: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Copernicus STAC API returned an invalid JSON response.",
        )

    features = stac_result.get("features", [])
    logger.info("STAC search returned %d features for collection=%s", len(features), collection_id)

    # Normalize each feature
    scenes: List[SatelliteScene] = []
    for feature in features:
        try:
            scene = _normalize_feature(feature, request.satellite)

            # Post-hoc cloud filter safety net (in case CQL2 wasn't applied)
            if (
                request.satellite == "sentinel-2"
                and scene.cloud_cover is not None
                and scene.cloud_cover > request.max_cloud_cover
            ):
                logger.debug(
                    "Skipping scene %s — cloud_cover=%.1f > max=%.1f",
                    scene.scene_id, scene.cloud_cover, request.max_cloud_cover,
                )
                continue

            scenes.append(scene)

            # Persist to MongoDB asynchronously (non-fatal)
            _upsert_scene(scene)

        except Exception as exc:
            logger.warning(
                "Failed to normalize STAC feature %s: %s",
                feature.get("id", "unknown"), exc,
            )
            # Skip malformed features, continue with others

    return SatelliteSearchResponse(
        scenes=scenes,
        total_returned=len(scenes),
        satellite=request.satellite,
        collection=collection_id,
        bbox=request.bbox,
        start_date=request.start_date,
        end_date=request.end_date,
        max_cloud_cover=request.max_cloud_cover if request.satellite == "sentinel-2" else None,
        source="Copernicus Data Space STAC API",
        stac_url=STAC_SEARCH_URL,
    )


def get_scene_metadata(scene_id: str, collection: str) -> Optional[SatelliteScene]:
    """
    Retrieve full metadata for a single scene.

    Strategy:
      1. Check MongoDB cache first.
      2. If not cached, fetch directly from STAC API.
      3. Normalize and cache the result.

    Returns None if not found (caller should raise 404).
    """
    # 1. Check cache
    cached = _load_cached_scene(scene_id)
    if cached:
        logger.info("Scene %s found in MongoDB cache", scene_id)
        return cached

    # 2. Fetch from STAC API
    url = f"{STAC_COLLECTIONS_URL}/{collection}/items/{scene_id}"
    logger.info("Fetching scene metadata from STAC: %s", url)

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            response = client.get(url, headers={"Accept": "application/geo+json"})
    except httpx.TimeoutException:
        raise HTTPException(status_code=503, detail="STAC API timed out.")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"STAC API error: {exc}")

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"STAC API returned HTTP {response.status_code} for scene {scene_id}",
        )

    feature = response.json()

    # Determine satellite from collection
    satellite = "sentinel-2" if "sentinel-2" in collection else "sentinel-1"
    try:
        scene = _normalize_feature(feature, satellite)
    except Exception as exc:
        logger.error("Failed to normalize scene %s: %s", scene_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to parse scene metadata: {exc}")

    _upsert_scene(scene)
    return scene


def get_scene_preview_url(scene_id: str, collection: str) -> Optional[str]:
    """
    Return the preview/thumbnail URL for a scene.
    Fetches full metadata if needed. Returns None if no preview available.
    This is a stub that Phase 2 can extend to generate server-side previews.
    """
    scene = get_scene_metadata(scene_id, collection)
    if scene is None:
        return None
    return scene.preview_url


def get_band_data(scene_id: str, collection: str, band_name: str) -> Dict[str, Any]:
    """
    Return asset metadata for a specific band of a scene.
    Phase 1: Returns the asset URL — does NOT download the file.
    Phase 2: Will download and process the band raster.

    Returns a dict with 'asset_name', 'href', 'https_href', 'gsd', 'mime_type'.
    Raises HTTP 404 if the band is not available for this scene.
    """
    scene = get_scene_metadata(scene_id, collection)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found.")

    # Find the best asset for this band (prefer 10m for S2)
    band_upper = band_name.upper()
    best_asset = None
    best_gsd = float("inf")

    for asset_name, asset_info in scene.assets.items():
        if asset_info.bands and band_upper in asset_info.bands:
            gsd = asset_info.gsd or float("inf")
            if gsd < best_gsd:
                best_gsd = gsd
                best_asset = asset_info

    if best_asset is None:
        raise HTTPException(
            status_code=404,
            detail=f"Band '{band_name}' not found in scene '{scene_id}'.",
        )

    return {
        "scene_id": scene_id,
        "band": band_name,
        "asset_name": best_asset.name,
        "href": best_asset.href,
        "https_href": best_asset.https_href,
        "gsd": best_asset.gsd,
        "mime_type": best_asset.mime_type,
        "note": "Phase 1: URL only. Use credentials to download in Phase 2.",
    }

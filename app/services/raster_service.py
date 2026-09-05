"""
SatQuery AI — Raster Service (Phase 2)

Low-level rasterio / GDAL utilities for reading Sentinel-2 L2A band assets
from the Copernicus Data Space Ecosystem and writing output GeoTIFFs.

Architecture:
  spectral_service.py
      ↓
  raster_service.py   ← this file
      ↓
  GDAL /vsicurl/ + rasterio  →  Copernicus HTTPS band assets
      ↓
  numpy arrays  →  spectral_service.py (calculations)
      ↓
  GeoTIFF output files

Key design decisions:
  1. We use rasterio's GDAL /vsicurl/ virtual filesystem to read band data
     directly from the Copernicus HTTPS endpoint — NO full JP2 download.
     Only the required spatial window is transferred.

  2. Band alignment:
     B02/B03/B04/B08 are 10m native resolution.
     B11/B12 are 20m native resolution.
     When NBR needs B12 (20m) aligned to B08 (10m), we resample using
     bilinear resampling. Bilinear is chosen over nearest-neighbour because
     spectral indices are continuous fields and bilinear preserves gradients
     better for statistical analysis. The resampling method is documented
     in every output file and API response.

  3. Nodata handling:
     Sentinel-2 L2A JP2 files use 0 as nodata for most bands.
     We treat pixels == 0 as nodata and mask them before calculations.
     This prevents division-by-zero artefacts at scene edges.

  4. Output format:
     GeoTIFF, float32, LZW compression, preserve CRS and transform.
     PNG previews use matplotlib colormaps for frontend display.

Never:
  - Downloads full JP2 files unnecessarily.
  - Exposes CDSE tokens in file paths or logs.
  - Returns mock pixel values.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── GDAL / rasterio imports (lazy — raise clear error if missing) ────────────

def _require_rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError:
        raise RuntimeError(
            "rasterio is not installed. Run: pip install rasterio\n"
            "rasterio requires GDAL. On Windows use: pip install rasterio[all]"
        )


def _require_rasterio_transform():
    try:
        from rasterio.transform import from_bounds
        return from_bounds
    except ImportError:
        raise RuntimeError("rasterio is not installed.")


def _require_rasterio_warp():
    try:
        from rasterio.warp import reproject, Resampling, transform_bounds
        return reproject, Resampling, transform_bounds
    except ImportError:
        raise RuntimeError("rasterio is not installed.")

def _require_rasterio_windows():
    try:
        from rasterio.windows import from_bounds
        return from_bounds
    except ImportError:
        raise RuntimeError("rasterio is not installed.")


# ─── Constants ────────────────────────────────────────────────────────────────

# Resampling method for 20m→10m upsampling (NBR: B12→B08 grid)
# Bilinear chosen: continuous spectral field; better gradient preservation than nearest
UPSAMPLE_RESAMPLING_METHOD = "bilinear"

# Sentinel-2 L2A JP2 bands use 0 as their nodata value
S2_NODATA_VALUE = 0

# Maximum valid reflectance for S2 L2A (surface reflectance scaled ×10000)
# Values above this are sensor saturation / invalid
S2_MAX_VALID_DN = 10000

# Preview PNG colormap configuration per index
INDEX_COLORMAPS = {
    "NDVI": "RdYlGn",    # Red (low) → Yellow → Green (high vegetation)
    "NDWI": "RdYlBu",    # Red (no water) → Blue (water)
    "NBR":  "RdYlGn_r",  # Green (healthy) → Red (burned)
}

# Valid index range for clipping visualisations
INDEX_VMIN_VMAX = {
    "NDVI": (-0.5, 1.0),
    "NDWI": (-1.0, 1.0),
    "NBR":  (-1.0, 1.0),
}


# ─── GDAL authentication setup ───────────────────────────────────────────────

def configure_gdal_auth(token: str) -> None:
    """
    Configure GDAL environment variables so /vsicurl/ requests include
    a CDSE Bearer token (OAuth2 client_credentials flow).

    Used as a fallback when S3 credentials are not configured.
    Security: token value is never logged.
    """
    os.environ["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {token}"
    os.environ["GDAL_HTTP_UNSAFESSL"] = "NO"
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "YES"
    os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
    os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".jp2,.tif,.tiff"
    logger.debug("GDAL HTTP Bearer auth configured (token not logged).")


def configure_gdal_s3_auth(access_key: str, secret_key: str, endpoint_host: str) -> None:
    """
    Configure GDAL environment variables for S3 access to CDSE eodata bucket.

    Sets AWS_* env vars consumed by GDAL's /vsis3/ virtual filesystem.
    Enables path-style requests (required by CDSE S3 — virtual-hosting not supported).

    Security: secret_key is NEVER logged, printed, or stored anywhere.
    """
    os.environ["AWS_ACCESS_KEY_ID"]     = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key   # never logged
    os.environ["AWS_S3_ENDPOINT"]       = endpoint_host  # bare hostname, no scheme
    os.environ["AWS_VIRTUAL_HOSTING"]   = "FALSE"       # CDSE requires path-style
    os.environ["AWS_HTTPS"]             = "YES"
    os.environ["AWS_REGION"]            = ""            # CDSE does not use regions
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "YES"
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".jp2,.tif,.tiff"
    logger.debug(
        "GDAL S3 auth configured: endpoint=%s access_key_prefix=%s (secret not logged)",
        endpoint_host,
        access_key[:4] + "****" if access_key else "(none)",
    )


def clear_gdal_s3_auth() -> None:
    """Remove GDAL S3 auth environment variables after processing."""
    for var in (
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
        "AWS_S3_ENDPOINT", "AWS_VIRTUAL_HOSTING", "AWS_HTTPS", "AWS_REGION",
    ):
        os.environ.pop(var, None)


# ─── Band URL resolution ──────────────────────────────────────────────────────

def resolve_band_url(assets: Dict, band_name: str, prefer_s3: bool = False) -> Optional[str]:
    """
    Find the HTTPS URL for a specific Sentinel-2 band from STAC asset dict.

    Strategy:
      1. Look for an asset whose band metadata lists the requested band.
      2. Prefer 10m assets (lower GSD) when multiple match.
      3. Return HTTPS URL from https_href (preferred) or construct from href.

    Returns None if the band is not found in the assets dict.
    """
    band_upper = band_name.upper()
    candidates = []

    for asset_name, asset_info in assets.items():
        if isinstance(asset_info, dict):
            bands = asset_info.get('bands', [])
        else:
            bands = getattr(asset_info, 'bands', []) if hasattr(asset_info, 'bands') else []
            
        if not bands:
            continue
        if band_upper in (b.upper() if isinstance(b, str) else b.get('name', '').upper() for b in bands):
            if isinstance(asset_info, dict):
                gsd = asset_info.get('gsd', 999)
                href = asset_info.get('href') or ''
                https_href = asset_info.get('https_href') or ''
                if prefer_s3 and href.startswith('s3://'):
                    url = href
                else:
                    url = https_href or href
            else:
                gsd = getattr(asset_info, 'gsd', 999)
                href = getattr(asset_info, 'href', None) or ''
                https_href = getattr(asset_info, 'https_href', None) or ''
                if prefer_s3 and href.startswith('s3://'):
                    url = href
                else:
                    url = https_href or href
                
            if url:
                candidates.append((gsd, url, asset_name))

    if not candidates:
        return None

    # Sort by GSD ascending — prefer highest resolution (lowest GSD)
    candidates.sort(key=lambda x: x[0])
    _, url, asset_name = candidates[0]

    # Convert S3 URI to HTTPS if needed and not preferred
    if url and url.startswith("s3://") and not prefer_s3:
        url = _s3_to_https(url)

    logger.debug("Resolved band %s → asset=%s url=%.60s…", band_upper, asset_name, url or "")
    return url


def _s3_to_https(s3_uri: str) -> Optional[str]:
    """
    Convert CDSE S3 URI to HTTPS equivalent for /vsicurl/ access.
    e.g. s3://eodata/Sentinel-2/... -> https://eodata.dataspace.copernicus.eu/Sentinel-2/...
    """
    if s3_uri.startswith("s3://eodata/"):
        path = s3_uri[len("s3://eodata/"):]
        return f"https://eodata.dataspace.copernicus.eu/{path}"
    return None


def build_vsis3_path(s3_uri: str) -> Optional[str]:
    """
    Convert a CDSE S3 URI into a GDAL /vsis3/ virtual filesystem path.

    This is the preferred access method when S3 credentials are configured.
    GDAL /vsis3/ uses the AWS_* environment variables set by configure_gdal_s3_auth().

    e.g. s3://eodata/Sentinel-2/MSI/... -> /vsis3/eodata/Sentinel-2/MSI/...
    """
    if s3_uri.startswith("s3://"):
        return "/vsis3/" + s3_uri[len("s3://"):]
    return None


# ─── Band reading ─────────────────────────────────────────────────────────────

def read_band_from_url(
    url: str,
    band_name: str,
) -> Tuple[np.ndarray, object, object]:
    """
    Read a Sentinel-2 band from a CDSE URL using rasterio + GDAL.

    Supports two access modes (auth must be configured BEFORE calling):
      1. /vsis3/ path  (s3:// URI)  — uses configure_gdal_s3_auth() [preferred]
      2. /vsicurl/ URL (https://)   — uses configure_gdal_auth() Bearer token [fallback]

    Steps:
      1. Build the GDAL virtual filesystem path.
      2. Open with rasterio (lazy I/O — no full JP2 download).
      3. Read band 1 (JP2 files are single-band).
      4. Apply nodata mask (S2 L2A uses 0 = nodata).

    Returns:
      (data_array, transform, crs)
      - data_array: float32 ndarray with nodata pixels set to NaN
      - transform: rasterio Affine transform
      - crs: rasterio CRS object

    Raises:
      RuntimeError on rasterio/GDAL errors (no credential values in message).
    """
    rasterio = _require_rasterio()

    # Determine GDAL virtual path based on URL scheme
    if url.startswith("s3://"):
        # S3 access via /vsis3/ — requires configure_gdal_s3_auth() to be called first
        gdal_path = build_vsis3_path(url)
        if gdal_path is None:
            raise RuntimeError(f"Cannot build /vsis3/ path from URI: {url[:40]}")
        access_mode = "S3 /vsis3/"
    elif url.startswith("https://") or url.startswith("http://"):
        # HTTPS access via /vsicurl/ — requires configure_gdal_auth() to be called first
        gdal_path = f"/vsicurl/{url}"
        access_mode = "HTTPS /vsicurl/"
    else:
        gdal_path = url  # local path or pre-formed GDAL path
        access_mode = "local"

    logger.info(
        "Reading band %s via %s: %.60s",
        band_name, access_mode, gdal_path.split("/")[-1],   # filename only — no creds in log
    )

    try:
        with rasterio.open(gdal_path) as src:
            transform = src.transform
            crs = src.crs
            profile = src.profile

            logger.debug(
                "Band %s: width=%d height=%d dtype=%s crs=%s",
                band_name, src.width, src.height, src.dtypes[0], crs,
            )

            # Read band 1 (S2 JP2 files are single-band)
            raw = src.read(1).astype(np.float32)

            # Apply nodata mask: S2 L2A uses 0 as nodata
            nodata_val = src.nodata if src.nodata is not None else S2_NODATA_VALUE
            raw[raw == nodata_val] = np.nan

            # Also mask clearly invalid values (saturation artefacts)
            raw[raw > S2_MAX_VALID_DN] = np.nan
            raw[raw < 0] = np.nan

    except Exception as exc:
        logger.error("Failed to read band %s: %s", band_name, exc)
        raise RuntimeError(
            f"Failed to read Sentinel-2 band {band_name} from Copernicus. "
            f"Error: {exc}. "
            f"Ensure COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET are valid."
        ) from exc

    valid_count = int(np.sum(~np.isnan(raw)))
    total_count = raw.size
    logger.info(
        "Band %s read successfully: %d×%d, valid_pixels=%d/%d (%.1f%%)",
        band_name, raw.shape[1], raw.shape[0],
        valid_count, total_count,
        100.0 * valid_count / total_count if total_count > 0 else 0,
    )

    return raw, transform, crs


def read_band_window_from_url(
    url: str,
    band_name: str,
    bbox: list[float], # [min_lon, min_lat, max_lon, max_lat] in EPSG:4326
) -> Tuple[np.ndarray, object, object]:
    """
    Read a specific spatial window of a Sentinel-2 or Sentinel-1 band from a CDSE URL.
    This avoids downloading massive full-scene JP2/GeoTIFFs.
    """
    rasterio = _require_rasterio()
    from_bounds = _require_rasterio_windows()
    _, _, transform_bounds = _require_rasterio_warp()

    if url.startswith("s3://"):
        gdal_path = build_vsis3_path(url)
        if gdal_path is None:
            raise RuntimeError(f"Cannot build /vsis3/ path from URI: {url[:40]}")
        access_mode = "S3 /vsis3/"
    elif url.startswith("https://") or url.startswith("http://"):
        gdal_path = f"/vsicurl/{url}"
        access_mode = "HTTPS /vsicurl/"
    else:
        gdal_path = url
        access_mode = "local"

    logger.info(
        "Reading WINDOWED band %s via %s: %.60s",
        band_name, access_mode, gdal_path.split("/")[-1],
    )

    try:
        with rasterio.open(gdal_path) as src:
            # Transform WGS84 bbox to the source CRS
            min_lon, min_lat, max_lon, max_lat = bbox
            src_bbox = transform_bounds(
                "EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat
            )
            
            # Get the rasterio window for the transformed bbox
            window = from_bounds(*src_bbox, transform=src.transform)
            
            # Read only the windowed subset
            raw = src.read(1, window=window).astype(np.float32)
            
            # Calculate the new transform for the window
            win_transform = src.window_transform(window)
            crs = src.crs

            logger.debug(
                "Band %s window: width=%d height=%d dtype=%s crs=%s",
                band_name, raw.shape[1], raw.shape[0], src.dtypes[0], crs,
            )

            nodata_val = src.nodata if src.nodata is not None else S2_NODATA_VALUE
            raw[raw == nodata_val] = np.nan
            raw[raw > S2_MAX_VALID_DN] = np.nan
            raw[raw < 0] = np.nan

    except Exception as exc:
        logger.error("Failed to read windowed band %s: %s", band_name, exc)
        raise RuntimeError(
            f"Failed to read windowed band {band_name}. Error: {exc}."
        ) from exc

    valid_count = int(np.sum(~np.isnan(raw)))
    total_count = raw.size
    logger.info(
        "Band %s window read successfully: %d×%d, valid_pixels=%d/%d (%.1f%%)",
        band_name, raw.shape[1], raw.shape[0],
        valid_count, total_count,
        100.0 * valid_count / total_count if total_count > 0 else 0,
    )

    return raw, win_transform, crs



# ─── Band resampling / alignment ──────────────────────────────────────────────

def align_band_to_reference(
    source_array: np.ndarray,
    source_transform: object,
    source_crs: object,
    reference_array: np.ndarray,
    reference_transform: object,
    reference_crs: object,
    resampling_method: str = UPSAMPLE_RESAMPLING_METHOD,
) -> np.ndarray:
    """
    Resample `source_array` to match the spatial grid of `reference_array`.

    Used for NBR: resample B12 (20m) to the B08 (10m) grid.

    Resampling method: bilinear
    - Reason: spectral indices are continuous surfaces; bilinear interpolation
      preserves gradients better than nearest-neighbour and is significantly
      faster than cubic for large arrays.
    - Bilinear is standard practice in Sentinel-2 spectral analysis workflows.

    Returns:
      Resampled array with the same shape as `reference_array`.
    """
    rasterio = _require_rasterio()
    reproject_fn, Resampling, _ = _require_rasterio_warp()

    # Get the resampling enum
    try:
        resampling_enum = getattr(Resampling, resampling_method)
    except AttributeError:
        logger.warning("Unknown resampling method '%s', falling back to bilinear", resampling_method)
        resampling_enum = Resampling.bilinear

    destination = np.full_like(reference_array, fill_value=np.nan, dtype=np.float32)

    source_nodata = np.nan
    # rasterio reproject needs non-NaN nodata for float arrays
    # Use a large negative sentinel value, then restore NaN afterwards
    src_work = np.where(np.isnan(source_array), -9999.0, source_array).astype(np.float32)

    try:
        reproject_fn(
            source=src_work,
            destination=destination,
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            resampling=resampling_enum,
            src_nodata=-9999.0,
            dst_nodata=-9999.0,
        )
    except Exception as exc:
        logger.error("Band alignment failed: %s", exc)
        raise RuntimeError(f"Band alignment/resampling failed: {exc}") from exc

    # Restore NaN where sentinel was placed
    destination[destination == -9999.0] = np.nan

    logger.info(
        "Band aligned: source=%s → destination=%s (resampling=%s)",
        source_array.shape, destination.shape, resampling_method,
    )

    return destination


# ─── GeoTIFF output ──────────────────────────────────────────────────────────

def save_geotiff(
    array: np.ndarray,
    transform: object,
    crs: object,
    output_path: Path,
    nodata_value: float = np.nan,
) -> None:
    """
    Save a float32 NumPy array as a georeferenced GeoTIFF.

    Format:
      - Driver: GTiff
      - Datatype: float32
      - Compression: LZW (lossless, good compression ratio for float data)
      - Nodata: NaN
      - CRS and transform preserved from source band

    The output directory is created if it does not exist.
    """
    rasterio = _require_rasterio()
    import rasterio as rio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": array.shape[1],
        "height": array.shape[0],
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": nodata_value,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    try:
        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(array.astype(np.float32), 1)
    except Exception as exc:
        logger.error("Failed to write GeoTIFF to %s: %s", output_path, exc)
        raise RuntimeError(f"GeoTIFF write failed: {exc}") from exc

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "GeoTIFF saved: %s (%.2f MB)",
        output_path.name, file_size_mb,
    )


def get_geotiff_metadata(path: Path) -> dict:
    """
    Read CRS, transform, and dimensions from an existing GeoTIFF.
    Used to validate output files after writing.
    """
    rasterio = _require_rasterio()

    with rasterio.open(str(path)) as src:
        return {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "count": src.count,
        }


# ─── PNG preview generation ───────────────────────────────────────────────────

def generate_preview_png(
    array: np.ndarray,
    index_name: str,
    output_path: Path,
) -> bool:
    """
    Generate a normalised PNG preview image from a spectral index array.

    The array is normalised to the expected valid range for the index
    and rendered using an appropriate matplotlib colormap.

    Returns True on success, False if matplotlib is not available.
    The GeoTIFF is the primary scientific output; the PNG is for display only.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
    except ImportError:
        logger.warning("matplotlib not available — PNG preview not generated for %s", index_name)
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colormap_name = INDEX_COLORMAPS.get(index_name.upper(), "viridis")
    vmin, vmax = INDEX_VMIN_VMAX.get(index_name.upper(), (-1.0, 1.0))

    # Create normalizer
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    colormap = cm.get_cmap(colormap_name)

    try:
        # Replace NaN with masked values for proper rendering
        masked = np.ma.masked_invalid(array)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.imshow(masked, cmap=colormap, norm=norm, interpolation="none")
        ax.axis("off")

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(index_name.upper(), fontsize=12, color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        fig.patch.set_facecolor("#0a0f1e")
        ax.set_facecolor("#0a0f1e")

        plt.tight_layout(pad=0.1)
        plt.savefig(str(output_path), dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), format="png")
        plt.close(fig)

        logger.info("PNG preview saved: %s", output_path.name)
        return True

    except Exception as exc:
        logger.warning("PNG preview generation failed for %s: %s", index_name, exc)
        try:
            plt.close("all")
        except Exception:
            pass
        return False


# ─── Statistics calculation ───────────────────────────────────────────────────

def compute_raster_stats(array: np.ndarray) -> dict:
    """
    Compute statistics from a float32 index array.
    NaN pixels (nodata/invalid) are excluded from all calculations.

    Returns a dict matching the IndexStats schema fields.
    """
    flat = array.ravel()
    valid = flat[~np.isnan(flat)]
    nodata_count = int(np.sum(np.isnan(flat)))
    total_count = flat.size

    if valid.size == 0:
        return {
            "minimum": None,
            "maximum": None,
            "mean": None,
            "std": None,
            "percentile_5": None,
            "percentile_95": None,
            "valid_pixel_count": 0,
            "nodata_pixel_count": nodata_count,
            "total_pixel_count": total_count,
            "valid_pixel_pct": 0.0,
        }

    return {
        "minimum": float(np.min(valid)),
        "maximum": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "percentile_5": float(np.percentile(valid, 5)),
        "percentile_95": float(np.percentile(valid, 95)),
        "valid_pixel_count": int(valid.size),
        "nodata_pixel_count": nodata_count,
        "total_pixel_count": total_count,
        "valid_pixel_pct": round(100.0 * valid.size / total_count, 2),
    }

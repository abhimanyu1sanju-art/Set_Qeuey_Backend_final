"""
SatQuery AI — Image Utility Functions (Phase 2)

Responsibilities:
  - Image metadata extraction (JPEG, PNG, TIFF, GeoTIFF)
  - Image dimensions and format detection
  - Safe ULID-based image_id generation
  - Processed image creation (orientation correction, format-safe copy)
  - Decompression-safe image loading

No MongoDB code lives here.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Optional

from app.schemas.image import BoundsSchema, GeospatialMeta

logger = logging.getLogger(__name__)


# ─── Image ID Generation ──────────────────────────────────────────────────────

def generate_image_id() -> str:
    """
    Generate a collision-safe, lexicographically sortable image ID.
    Format: img_<ULID>  e.g. img_01JABC123DEFGHIJKLMNO
    Uses python-ulid for the ULID component.
    """
    try:
        from ulid import ULID
        return f"img_{ULID()}"
    except ImportError:
        import uuid
        logger.warning("python-ulid not available; falling back to UUID4")
        return f"img_{uuid.uuid4().hex}"


# ─── MIME type helpers ────────────────────────────────────────────────────────

def ext_to_mime(extension: str) -> str:
    """Map a file extension to a canonical MIME type."""
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    return mapping.get(extension.lower(), "application/octet-stream")


# ─── Standard Image Metadata ──────────────────────────────────────────────────

def extract_standard_metadata(data: bytes, filename: str) -> dict[str, Any]:
    """
    Extract width, height, format, and MIME type from image bytes using Pillow.
    Safe: opens with a pixel limit to prevent decompression bombs.
    Returns a dict with keys: width, height, format, mime_type.
    """
    from PIL import Image

    ext = Path(filename).suffix.lower()

    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
        fmt = img.format or "UNKNOWN"

    return {
        "width": width,
        "height": height,
        "format": fmt,
        "mime_type": ext_to_mime(ext),
    }


# ─── GeoTIFF Detection & Metadata ────────────────────────────────────────────

def is_geotiff(data: bytes) -> bool:
    """
    Return True if the given bytes appear to be a GeoTIFF.
    Checks for the presence of geospatial metadata via rasterio.
    Does NOT assume every TIFF is a GeoTIFF.
    """
    try:
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(data) as memfile:
            with memfile.open() as ds:
                # A TIFF is considered geospatial if it has a valid CRS
                # or a non-default geotransform.
                crs = ds.crs
                transform = ds.transform
                # Default transform has (1,0,0,0,1,0) — pure pixel grid
                has_crs = crs is not None
                has_transform = (
                    transform is not None
                    and not (
                        transform.a == 1.0
                        and transform.b == 0.0
                        and transform.c == 0.0
                        and transform.d == 0.0
                        and transform.e == 1.0
                        and transform.f == 0.0
                    )
                )
                return has_crs or has_transform
    except Exception:
        return False


def extract_geotiff_metadata(data: bytes) -> Optional[GeospatialMeta]:
    """
    Attempt to extract full geospatial metadata from a GeoTIFF.
    Returns a GeospatialMeta instance if geospatial, or a non-geospatial stub.
    Never fabricates metadata — only returns what is actually present.
    """
    try:
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(data) as memfile:
            with memfile.open() as ds:
                crs = ds.crs
                transform = ds.transform
                bounds = ds.bounds
                nodata = ds.nodata
                driver = ds.driver

                # Determine if this is truly geospatial
                has_geo = crs is not None or (
                    transform is not None and not (
                        transform.a == 1.0 and transform.b == 0.0
                        and transform.c == 0.0 and transform.d == 0.0
                        and transform.e == 1.0 and transform.f == 0.0
                    )
                )

                if not has_geo:
                    return GeospatialMeta(is_geospatial=False)

                # CRS string (e.g. "EPSG:4326")
                crs_str: Optional[str] = None
                if crs:
                    try:
                        crs_str = crs.to_epsg()
                        crs_str = f"EPSG:{crs_str}" if crs_str else crs.to_string()
                    except Exception:
                        crs_str = str(crs)

                # Resolution (pixel size in CRS units)
                resolution: Optional[list[float]] = None
                if transform:
                    try:
                        res_x = abs(float(transform.a))
                        res_y = abs(float(transform.e))
                        resolution = [res_x, res_y]
                    except Exception:
                        resolution = None

                # Bounding box
                bounds_schema: Optional[BoundsSchema] = None
                if bounds:
                    try:
                        bounds_schema = BoundsSchema(
                            left=float(bounds.left),
                            bottom=float(bounds.bottom),
                            right=float(bounds.right),
                            top=float(bounds.top),
                        )
                    except Exception:
                        bounds_schema = None

                # Affine transform as flat list [a, b, c, d, e, f]
                transform_list: Optional[list[float]] = None
                if transform:
                    try:
                        transform_list = [
                            float(transform.a), float(transform.b), float(transform.c),
                            float(transform.d), float(transform.e), float(transform.f),
                        ]
                    except Exception:
                        transform_list = None

                return GeospatialMeta(
                    is_geospatial=True,
                    crs=crs_str,
                    width=ds.width,
                    height=ds.height,
                    bands=ds.count,
                    resolution=resolution,
                    bounds=bounds_schema,
                    nodata=float(nodata) if nodata is not None else None,
                    transform=transform_list,
                    driver=driver,
                )

    except Exception as exc:
        logger.warning("GeoTIFF metadata extraction failed: %s", exc)
        return GeospatialMeta(is_geospatial=False)


# ─── Full Metadata Extraction ─────────────────────────────────────────────────

def extract_image_metadata(data: bytes, filename: str) -> dict[str, Any]:
    """
    Extract all image metadata from raw bytes.
    Handles JPEG, PNG, standard TIFF, and GeoTIFF.
    Returns a dict ready to feed into ImageDocument.
    """
    ext = Path(filename).suffix.lower()
    is_tiff = ext in {".tif", ".tiff"}

    std = extract_standard_metadata(data, filename)

    geospatial: Optional[GeospatialMeta] = None
    is_geo = False

    if is_tiff:
        geospatial = extract_geotiff_metadata(data)
        is_geo = geospatial.is_geospatial if geospatial else False
        # For GeoTIFF, prefer rasterio dimensions when available
        if geospatial and geospatial.is_geospatial and geospatial.width and geospatial.height:
            std["width"] = geospatial.width
            std["height"] = geospatial.height

    return {
        **std,
        "is_geospatial": is_geo,
        "geospatial": geospatial,
    }


# ─── Processed Image Creation ─────────────────────────────────────────────────

def create_processed_image(data: bytes, filename: str) -> bytes:
    """
    Create a processed copy of the image suitable for analysis.

    Rules:
      - JPEG/PNG: correct EXIF orientation, return optimised bytes
      - TIFF (non-GeoTIFF): same as above — orientation-corrected copy
      - GeoTIFF: return the original bytes UNCHANGED to preserve all bands,
        CRS, and geotransform. GeoTIFF data must never be blindly
        converted to RGB as that would destroy spectral information.

    Returns the processed image as bytes.
    """
    from PIL import Image, ImageOps
    import io as _io

    ext = Path(filename).suffix.lower()
    is_tiff = ext in {".tif", ".tiff"}

    # For GeoTIFF, preserve original bytes exactly
    if is_tiff and is_geotiff(data):
        logger.debug("GeoTIFF detected — preserving original bytes as processed copy")
        return data

    # For standard images, apply orientation correction
    try:
        with Image.open(_io.BytesIO(data)) as img:
            # Apply EXIF orientation (e.g. photos taken in portrait mode)
            img = ImageOps.exif_transpose(img)

            buf = _io.BytesIO()
            fmt = img.format or _ext_to_pil_format(ext)

            save_kwargs: dict = {}
            if fmt in ("JPEG", "JPG"):
                fmt = "JPEG"
                save_kwargs["quality"] = 95
                save_kwargs["optimize"] = True
            elif fmt == "PNG":
                save_kwargs["optimize"] = True

            img.save(buf, format=fmt, **save_kwargs)
            return buf.getvalue()

    except Exception as exc:
        logger.warning("Processed image creation failed (%s) — using original bytes", exc)
        return data


def _ext_to_pil_format(ext: str) -> str:
    mapping = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".tif": "TIFF",
        ".tiff": "TIFF",
    }
    return mapping.get(ext.lower(), "JPEG")

"""
SatQuery AI — Thumbnail Service (Phase 2)

Handles:
  - Thumbnail generation from image bytes
  - Aspect-ratio-preserving resize (max THUMBNAIL_SIZE × THUMBNAIL_SIZE)
  - Safe output as JPEG bytes
  - GeoTIFF-safe preview creation (RGB composite from first 3 bands)
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


def _clamp_thumbnail_size() -> tuple[int, int]:
    size = max(64, min(settings.thumbnail_size, 4096))
    return (size, size)


def generate_thumbnail(data: bytes, filename: str) -> bytes:
    """
    Generate a thumbnail from image bytes.

    - Standard images (JPEG, PNG, non-GeoTIFF TIFF):
        Resize with aspect-ratio preservation. Output: JPEG bytes.

    - GeoTIFF images:
        Extract an RGB preview from the first 3 bands (or 1 band as grayscale).
        Never destroys the original — this creates a visual derivative only.

    Returns JPEG bytes of the thumbnail.
    Raises RuntimeError on unrecoverable failure.
    """
    ext = Path(filename).suffix.lower()
    is_tiff = ext in {".tif", ".tiff"}
    max_size = _clamp_thumbnail_size()

    if is_tiff:
        return _thumbnail_from_tiff(data, max_size)
    else:
        return _thumbnail_from_standard(data, max_size)


def _thumbnail_from_standard(data: bytes, max_size: tuple[int, int]) -> bytes:
    """Generate thumbnail for JPEG / PNG / non-GeoTIFF TIFF."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as img:
        # Correct EXIF orientation before resizing
        img = ImageOps.exif_transpose(img)

        # Convert to RGB for JPEG output (handles RGBA, palette, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        img.thumbnail(max_size, Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()


def _thumbnail_from_tiff(data: bytes, max_size: tuple[int, int]) -> bytes:
    """
    Generate a visual thumbnail from a TIFF (including GeoTIFF).
    Tries rasterio first (for multi-band GeoTIFF), falls back to Pillow.
    """
    try:
        return _thumbnail_geotiff_rasterio(data, max_size)
    except Exception as exc:
        logger.debug("rasterio thumbnail failed (%s), falling back to Pillow", exc)
        return _thumbnail_from_standard(data, max_size)


def _thumbnail_geotiff_rasterio(data: bytes, max_size: tuple[int, int]) -> bytes:
    """
    Create a thumbnail from a GeoTIFF using rasterio.
    Extracts bands 1-3 as RGB (or band 1 as grayscale if < 3 bands).
    Normalises pixel values to 0–255 for display purposes only.
    """
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.io import MemoryFile

    with MemoryFile(data) as memfile:
        with memfile.open() as ds:
            band_count = ds.count
            width = ds.width
            height = ds.height

            # Determine overview level for efficient reading of large files
            overview_level = _choose_overview_level(width, height, max_size)

            if band_count >= 3:
                # Read first 3 bands as RGB
                arrays = []
                for band_idx in range(1, 4):
                    band_data = ds.read(band_idx, out_shape=(
                        1,
                        max(1, height >> overview_level),
                        max(1, width >> overview_level),
                    ))
                    arrays.append(_normalise_band(band_data[0]))
                rgb = np.stack(arrays, axis=-1).astype(np.uint8)
                img = Image.fromarray(rgb)
            else:
                # Single band — grayscale
                band_data = ds.read(1, out_shape=(
                    1,
                    max(1, height >> overview_level),
                    max(1, width >> overview_level),
                ))
                gray = _normalise_band(band_data[0]).astype(np.uint8)
                img = Image.fromarray(gray, mode="L").convert("RGB")

    img.thumbnail(max_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


def _choose_overview_level(width: int, height: int, max_size: tuple[int, int]) -> int:
    """
    Choose a downsampling power-of-2 to avoid reading huge images in full.
    Returns shift amount (0 = full resolution, 1 = half, 2 = quarter, …).
    """
    target = max(max_size)
    longest = max(width, height)
    level = 0
    while longest >> (level + 1) > target and level < 6:
        level += 1
    return level


def _normalise_band(array) -> "np.ndarray":
    """
    Linearly stretch a single-band array to 0–255 uint8.
    Handles NaN, inf, and all-zero arrays safely.
    """
    import numpy as np

    arr = array.astype(np.float32)
    # Replace non-finite values with 0
    arr = np.where(np.isfinite(arr), arr, 0.0)

    vmin = arr.min()
    vmax = arr.max()

    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.uint8)

    stretched = (arr - vmin) / (vmax - vmin) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)

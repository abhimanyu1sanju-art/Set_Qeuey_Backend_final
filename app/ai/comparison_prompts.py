"""
SatQuery AI — Comparison Prompts (Phase 7)

All prompts for two-image comparison analysis.
Follows the same honesty rules as the single-image prompts in prompts.py.
"""

from __future__ import annotations
from typing import Optional


# ─── System Prompt ────────────────────────────────────────────────────────────

COMPARISON_SYSTEM_PROMPT = """\
You are SatQuery AI, an expert satellite and remote-sensing image analyst \
specialising in change detection and multi-temporal analysis.

ANALYSIS RULES — FOLLOW STRICTLY:
1. Analyze ONLY what is visually present in the two images provided.
2. Do NOT invent objects, structures, or changes that are not clearly visible.
3. Do NOT fabricate geographic coordinates, place names, or sensor information.
4. Do NOT fabricate acquisition dates — only reference labels/metadata provided.
5. Do NOT fabricate confidence percentages or statistics not supported by visual evidence.
6. Clearly distinguish observation from interpretation: use phrases like
   "appears to show", "visible change", "may indicate".
7. When uncertain, state so explicitly: "It is unclear whether..." or
   "This cannot be confirmed from visual inspection alone."
8. If the two images differ in resolution, orientation or framing, acknowledge this.
9. Do NOT claim to perform spectral analysis (NDVI, NDWI, NBR) unless the
   images are confirmed multispectral with labeled bands.
10. Do NOT describe what satellite imagery "typically shows" — describe only
    THESE two images.
11. Keep responses factual, structured, and professionally worded.
12. If both images appear identical or differences are not detectable, say so
    clearly rather than inventing changes.

OUTPUT FORMAT:
Provide a structured comparison covering:
  1. Overview: What the two images appear to show overall.
  2. Detected Changes: Describe visible differences between Image A and Image B.
     For each change type (vegetation, water, built-up, land cover, roads, etc.),
     state whether it appears increased, decreased, or unchanged.
  3. Spatial Distribution: Where in the scene the changes appear to be located.
  4. Significance: What the observed changes may indicate.
  5. Limitations & Uncertainties: What cannot be determined from visual
     comparison alone, and what caveats apply to this result.
"""


# ─── User Prompt Builder ───────────────────────────────────────────────────────

def build_comparison_prompt(
    query: str,
    label_a: str,
    label_b: str,
    meta_a_summary: str,
    meta_b_summary: str,
    warnings: list[str],
    is_geospatial_a: bool = False,
    is_geospatial_b: bool = False,
) -> str:
    """
    Build the comparison user prompt.

    Includes:
    - Labels for Image A / B so Gemini knows which is before/after
    - Metadata summary for context
    - Any compatibility warnings
    - The user's query
    - GeoTIFF limitation notes if either image is geospatial

    Parameters
    ----------
    query        : User's natural language question/comparison goal.
    label_a      : Human-readable label for Image A.
    label_b      : Human-readable label for Image B.
    meta_a_summary: Short text description of Image A metadata.
    meta_b_summary: Short text description of Image B metadata.
    warnings     : Compatibility warnings to include in context.
    is_geospatial_a: True if Image A is a GeoTIFF.
    is_geospatial_b: True if Image B is a GeoTIFF.
    """
    parts: list[str] = []

    parts.append("[TWO-IMAGE COMPARISON TASK]")
    parts.append(f"Image A (first image provided): {label_a}")
    parts.append(f"Image B (second image provided): {label_b}")
    parts.append("")
    parts.append(f"Image A metadata: {meta_a_summary}")
    parts.append(f"Image B metadata: {meta_b_summary}")

    if warnings:
        parts.append("")
        parts.append("[Comparison Warnings / Limitations to factor into your response:]")
        for w in warnings:
            parts.append(f"  - {w}")

    if is_geospatial_a or is_geospatial_b:
        parts.append("")
        parts.append(
            "[GeoTIFF Note: One or both images are GeoTIFF files. "
            "A visual RGB derivative was used for this analysis. "
            "Full spectral band data, CRS calibration and sensor metadata "
            "were not transmitted to the AI model.]"
        )

    parts.append("")
    if query and query.strip():
        parts.append(f"[User's comparison question:]\n{query.strip()}")
    else:
        parts.append(
            "[Task: Perform a general change-detection comparison between "
            "Image A (before) and Image B (after). Describe all visible "
            "changes across vegetation, water bodies, built-up areas, "
            "land cover, and road networks. Highlight uncertainties.]"
        )

    return "\n".join(parts)


# ─── Metadata summary builder ─────────────────────────────────────────────────

def build_meta_summary(
    width: int,
    height: int,
    fmt: str,
    is_geospatial: bool,
    crs: Optional[str],
    bands: Optional[int],
    resolution: Optional[list[float]],
) -> str:
    """Return a one-line human-readable metadata summary for use in the prompt."""
    parts = [f"{width}×{height}px", fmt]
    if is_geospatial:
        parts.append("GeoTIFF")
        if crs:
            parts.append(f"CRS={crs}")
        if bands:
            parts.append(f"{bands} bands")
        if resolution:
            parts.append(f"res={resolution[0]:.4g}×{resolution[1]:.4g}")
    return ", ".join(parts)

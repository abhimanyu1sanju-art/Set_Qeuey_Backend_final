"""
SatQuery AI — Intent Detector (Phase 6)

Deterministic keyword/rule-based intent detection for satellite image queries.

Design rules:
  - No network calls, no AI needed for detection (fast, deterministic).
  - Each intent has a canonical name and a set of trigger keywords.
  - A query may resolve to 1..N distinct intents.
  - Duplicate intents are removed automatically.
  - Unknown or general queries fall back to the 'general' intent.
  - Detection is case-insensitive.

Supported intents (INTENT_LABELS):
  general       — catch-all / no specific intent detected
  vegetation    — plant cover, forests, crops, greenery
  water         — water bodies, rivers, lakes, floods
  land_cover    — land use classification, terrain types
  building      — structures, urban development, infrastructure
  road          — transportation network, roads, highways
  ndvi          — Normalized Difference Vegetation Index (explicit request)
  ndwi          — Normalized Difference Water Index (explicit request)
  nbr           — Normalized Burn Ratio (explicit request)
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

logger = logging.getLogger(__name__)


# ─── Supported Intent Labels ──────────────────────────────────────────────────

INTENT_LABELS: tuple[str, ...] = (
    "vegetation",
    "water",
    "land_cover",
    "building",
    "road",
    "ndvi",
    "ndwi",
    "nbr",
    "general",
)

# ─── Keyword Maps ─────────────────────────────────────────────────────────────
# Each entry is (intent_name, frozenset_of_trigger_keywords_or_phrases).
# Longer / more specific phrases take priority over short words because we
# search using word-boundary regex — so "land cover" won't accidentally
# trigger on "land" alone unless "land" is also a keyword.

_INTENT_KEYWORDS: list[tuple[str, frozenset[str]]] = [
    # ── Explicit spectral index requests (check before generic vegetation/water)
    ("ndvi", frozenset({
        "ndvi", "normalized difference vegetation index",
        "vegetation index", "chlorophyll index",
    })),
    ("ndwi", frozenset({
        "ndwi", "normalized difference water index",
        "water index", "mcfeeters",
    })),
    ("nbr", frozenset({
        "nbr", "normalized burn ratio",
        "burn ratio", "burn scar", "fire scar", "post-fire",
    })),

    # ── Vegetation
    ("vegetation", frozenset({
        "vegetation", "plant", "plants", "tree", "trees", "forest",
        "forests", "green", "greenery", "canopy", "crop", "crops",
        "agricultural", "agriculture", "grass", "grassland", "shrub",
        "shrubs", "foliage", "photosynthesis", "healthy vegetation",
        "stressed vegetation", "biomass", "botanical",
    })),

    # ── Water
    ("water", frozenset({
        "water", "lake", "river", "rivers", "pond", "ponds",
        "ocean", "sea", "flood", "flooding", "stream", "streams",
        "wetland", "wetlands", "irrigation", "reservoir", "reservoirs",
        "aquatic", "hydrological", "hydrology", "coastline", "beach",
        "waterway", "waterways", "dam",
    })),

    # ── Land cover / land use
    ("land_cover", frozenset({
        "land cover", "land use", "land-cover", "land-use",
        "classification", "classify", "terrain", "landscape",
        "urban", "rural", "suburban", "impervious", "bare soil",
        "bare ground", "open ground", "barren", "land types",
    })),

    # ── Buildings / infrastructure
    ("building", frozenset({
        "building", "buildings", "structure", "structures", "infrastructure",
        "roof", "rooftop", "settlement", "settlements", "urban development",
        "construction", "development", "footprint", "commercial",
        "residential", "industrial", "facility", "facilities",
    })),

    # ── Roads / transport
    ("road", frozenset({
        "road", "roads", "highway", "highways", "street", "streets",
        "path", "paths", "transportation", "transport", "network",
        "railway", "rail", "bridge", "bridges", "motorway",
    })),
]

# Display labels shown to the user
INTENT_DISPLAY_LABELS: dict[str, str] = {
    "vegetation":  "Vegetation Analysis",
    "water":       "Water Detection",
    "land_cover":  "Land Cover Classification",
    "building":    "Building & Infrastructure Detection",
    "road":        "Road Network Analysis",
    "ndvi":        "NDVI (Vegetation Index)",
    "ndwi":        "NDWI (Water Index)",
    "nbr":         "NBR (Burn Ratio)",
    "general":     "General Scene Analysis",
}

# Map each specific intent to its analysis_type hint used by the pipeline
INTENT_TO_ANALYSIS_TYPE: dict[str, str] = {
    "vegetation":  "vegetation_detection",
    "water":       "water_detection",
    "land_cover":  "land_cover",
    "building":    "building_detection",
    "road":        "road_detection",
    "ndvi":        "vegetation_detection",   # NDVI maps to vegetation pipeline
    "ndwi":        "water_detection",        # NDWI maps to water pipeline
    "nbr":         "general",                # NBR falls back to general analysis
    "general":     "general",
}


# ─── Detection Logic ──────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for keyword matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _contains_keyword(text: str, keyword: str) -> bool:
    """
    Check whether `keyword` appears as a whole word/phrase in `text`.
    Multi-word phrases are matched literally (after normalisation).
    Single words are matched with word boundaries to avoid partial matches
    (e.g. 'road' won't match 'abroad').
    """
    kw = _normalize(keyword)
    if " " in kw:
        # Multi-word phrase — simple substring match after normalisation
        return kw in text
    # Single word — word-boundary match
    return bool(re.search(r"\b" + re.escape(kw) + r"\b", text))


def detect_intents(query: str) -> list[str]:
    """
    Detect all analysis intents in `query` using deterministic keyword matching.

    Returns a deduplicated, ordered list of intent names.  Order reflects the
    order defined in ``_INTENT_KEYWORDS`` so results are deterministic.

    If no specific intent is detected, returns ``["general"]``.

    Parameters
    ----------
    query : str
        The user's natural-language question.

    Returns
    -------
    list[str]
        e.g. ["vegetation", "water"] or ["general"]
    """
    if not query or not query.strip():
        return ["general"]

    normalized_query = _normalize(query)
    detected: list[str] = []
    seen: set[str] = set()

    for intent_name, keywords in _INTENT_KEYWORDS:
        if intent_name in seen:
            continue
        for kw in keywords:
            if _contains_keyword(normalized_query, kw):
                detected.append(intent_name)
                seen.add(intent_name)
                logger.debug(
                    "Intent '%s' detected via keyword %r in query %r",
                    intent_name, kw, query[:80],
                )
                break  # One matching keyword is enough per intent

    if not detected:
        logger.debug("No specific intent detected; falling back to 'general'")
        return ["general"]

    logger.info("Detected intents for query %r: %s", query[:80], detected)
    return detected


def is_multi_intent(query: str) -> bool:
    """Return True if the query contains two or more distinct intents."""
    return len(detect_intents(query)) >= 2


def get_display_label(intent: str) -> str:
    """Return the human-readable display label for an intent."""
    return INTENT_DISPLAY_LABELS.get(intent, intent.replace("_", " ").title())


def get_analysis_type(intent: str) -> str:
    """Map an intent name to the analysis_type expected by the analysis pipeline."""
    return INTENT_TO_ANALYSIS_TYPE.get(intent, "general")


def build_intent_sub_query(intent: str, original_query: str) -> str:
    """
    Build a focused sub-query for a specific intent.

    For multi-intent queries the original query is broad, so we generate a
    targeted question to send to the AI for each individual intent.  The
    original query is also included so the AI has context.

    Parameters
    ----------
    intent        : The specific intent to focus on.
    original_query : The user's full original question.
    """
    templates: dict[str, str] = {
        "vegetation": (
            "Analyze the vegetation in this satellite image. "
            "Describe the type, density, health, and distribution of plant cover visible. "
            "Note any stressed, sparse, or absent vegetation. "
            "Do NOT fabricate NDVI scores — only describe what is visually apparent."
        ),
        "water": (
            "Identify and analyze all water bodies and water-related features in this satellite image. "
            "Describe their type (river, lake, reservoir, etc.), approximate extent, and condition. "
            "Do NOT fabricate NDWI scores — only describe what is visually apparent."
        ),
        "land_cover": (
            "Classify and describe the different land cover types visible in this satellite image. "
            "Identify built-up areas, vegetation, bare soil, water, and agricultural land. "
            "Estimate the approximate distribution if clearly visible."
        ),
        "building": (
            "Detect and analyze all buildings, structures, and infrastructure visible in this satellite image. "
            "Describe their type, density, arrangement, and any notable patterns."
        ),
        "road": (
            "Identify and analyze the road network and transportation infrastructure "
            "visible in this satellite image. Describe road types, patterns, and connectivity."
        ),
        "ndvi": (
            "Analyze the vegetation health and density visible in this satellite image. "
            "Describe what the spectral signature of the vegetation suggests about its health. "
            "Do NOT fabricate numerical NDVI values — only describe what is visually apparent."
        ),
        "ndwi": (
            "Analyze the water content and moisture levels visible in this satellite image. "
            "Describe any water bodies, wet surfaces, or moisture patterns. "
            "Do NOT fabricate numerical NDWI values — only describe what is visually apparent."
        ),
        "nbr": (
            "Analyze this satellite image for any burn scars, fire damage, or post-fire vegetation recovery. "
            "Describe the extent and patterns of any burned or recovering areas. "
            "Do NOT fabricate numerical NBR values — only describe what is visually apparent."
        ),
        "general": (
            "Provide a comprehensive general analysis of this satellite image. "
            "Describe the main visible features, land cover types, and any notable elements."
        ),
    }
    sub_query = templates.get(intent, templates["general"])

    # Append original context so the model understands the broader request
    original_trimmed = (original_query or "").strip()
    if original_trimmed:
        sub_query += (
            f"\n\n[Context — Original user question: \"{original_trimmed}\"]"
        )
    return sub_query

"""
SatQuery AI — Centralized Prompt Configuration (Phase 4)

ALL system and user prompts live here.
Route files, services, and providers must import from this module.
Do NOT scatter prompt strings across the application.

Design for satellite/remote-sensing honesty:
  - Analyze only what is visually present.
  - Never invent objects, coordinates, sensor names, or dates.
  - Clearly distinguish observation from uncertainty.
  - Never fabricate confidence scores.
"""

from __future__ import annotations

from typing import Optional


# ─── System Prompt ────────────────────────────────────────────────────────────

SATELLITE_ANALYSIS_SYSTEM_PROMPT = """\
You are SatQuery AI, an expert satellite and remote-sensing image analyst.

ANALYSIS RULES — FOLLOW STRICTLY:
1. Analyze ONLY what is visually present in the image provided.
2. Do NOT invent objects, structures, roads, bodies of water, or vegetation that are not clearly visible.
3. Do NOT fabricate geographic coordinates, place names, or sensor information.
4. Do NOT fabricate the satellite acquisition date or time.
5. Do NOT fabricate coordinate reference systems (CRS) or projection information.
6. Do NOT fabricate or estimate confidence scores — only report confidence if you are genuinely certain.
7. Clearly distinguish observation ("visible", "appears to show", "can be seen") from interpretation.
8. When uncertain, say so explicitly: "It is unclear whether..." or "This may indicate...".
9. If the image quality is low, blurred, or the content is ambiguous, state that clearly.
10. Do NOT claim to perform spectral analysis (e.g. NDVI, NDWI) unless the image is confirmed multispectral and the bands are clearly labeled.
11. Do NOT describe what a satellite image "typically shows" — describe only THIS image.
12. Keep responses factual, structured, and professionally worded.

OUTPUT FORMAT:
- Provide a clear, direct answer to the user's question.
- If multiple elements are visible, organize your response with brief sub-sections.
- End with a short note on any significant uncertainties or limitations of the visual analysis.
"""


# ─── User Prompt Builders ──────────────────────────────────────────────────────

def build_analysis_prompt(query: str, analysis_type: str = "general") -> str:
    """
    Build the user-facing prompt to send alongside the image.

    For Phase 4, all queries use the general vision path.
    Future phases will add specialized prompts per analysis_type.

    Parameters
    ----------
    query         : The user's natural-language question.
    analysis_type : Hint (e.g. 'general', 'building', 'vegetation').
    """
    if not query or not query.strip():
        return (
            "Provide a comprehensive general analysis of this satellite image. "
            "Describe the main visible features, land cover types, and any notable elements. "
            "Mention any significant uncertainties."
        )

    base = query.strip()

    # For Phase 4 we send the user's query directly.
    # Future phases can route to specialist prompts here.
    return base


# ─── TIFF/GeoTIFF guidance appended when needed ────────────────────────────────

GEOTIFF_LIMITATION_NOTE = (
    "\n\n[Image Note: The original file is a GeoTIFF or multispectral raster. "
    "A visual (RGB) derivative was used for this analysis. "
    "Spectral band information, geospatial metadata, and sensor calibration data "
    "were not transmitted to the AI model. "
    "Do not interpret this result as a full remote-sensing spectral analysis.]"
)


def build_final_user_message(query: str, analysis_type: str, is_geospatial: bool) -> str:
    """
    Assemble the complete user message, including any geospatial caveats.

    Parameters
    ----------
    query         : User question.
    analysis_type : Analysis hint.
    is_geospatial : Whether the image was identified as a GeoTIFF.
    """
    prompt = build_analysis_prompt(query, analysis_type)
    if is_geospatial:
        prompt += GEOTIFF_LIMITATION_NOTE
    return prompt


# ─── Phase 5 — Conversation Context ──────────────────────────────────────────

def build_conversation_context_message(
    query: str,
    analysis_type: str,
    is_geospatial: bool,
    conversation_history: list,
) -> str:
    """
    Build the user message that includes relevant previous conversation context
    so follow-up questions are answered with awareness of prior analysis.

    Parameters
    ----------
    query                : Current user question.
    analysis_type        : Analysis type hint.
    is_geospatial        : Whether the image is a GeoTIFF.
    conversation_history : List of dicts with 'question' and 'answer' keys,
                           ordered chronologically (oldest first).
                           Only completed Q&A turns are included.
    """
    # Build the base prompt (may include geospatial note)
    base_prompt = build_final_user_message(query, analysis_type, is_geospatial)

    if not conversation_history:
        return base_prompt

    # Format previous turns as readable context
    context_lines = ["[Previous conversation about this same satellite image:]\n"]
    for i, turn in enumerate(conversation_history, 1):
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        if q and a:
            context_lines.append(f"Q{i}: {q}")
            # Truncate very long answers to keep token count reasonable
            if len(a) > 800:
                a = a[:800] + "… [truncated]"
            context_lines.append(f"A{i}: {a}\n")

    context_lines.append("[End of previous conversation]\n")
    context_lines.append(
        "[Current question — answer it with awareness of the above context "
        "and the same satellite image:]\n"
    )
    context_lines.append(base_prompt)

    return "\n".join(context_lines)


"""
SatQuery AI — Comparison Pydantic Schemas (Phase 7)

Models for the two-image comparison pipeline.

Requests:  ComparisonRequest
Responses: ComparisonResponse, ComparisonSummary
Internal:  ComparisonDocument
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Request ──────────────────────────────────────────────────────────────────

class ComparisonRequest(BaseModel):
    """Body for POST /api/comparisons."""

    image_id_a: str = Field(
        ...,
        description="image_id of Image A (Before). Must already be uploaded.",
    )
    image_id_b: str = Field(
        ...,
        description="image_id of Image B (After). Must already be uploaded.",
    )
    query: str = Field(
        default="",
        max_length=5000,
        description=(
            "Natural-language question about the comparison. "
            "Leave blank for a general change-detection analysis."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional session_id. When provided both images must be linked to "
            "the session. The result is associated with the session."
        ),
    )
    label_a: str = Field(
        default="Image A — Before",
        max_length=60,
        description="Human-readable label for Image A (e.g. 'Before', '2020-01').",
    )
    label_b: str = Field(
        default="Image B — After",
        max_length=60,
        description="Human-readable label for Image B (e.g. 'After', '2024-06').",
    )


# ─── Metadata compatibility ───────────────────────────────────────────────────

class ImageMetaSummary(BaseModel):
    """Compact metadata extracted from each image for comparison context."""

    image_id: str
    filename: str
    width: int
    height: int
    format: str
    mime_type: str
    is_geospatial: bool
    crs: Optional[str] = None
    bands: Optional[int] = None
    resolution: Optional[list[float]] = None


class CompatibilityResult(BaseModel):
    """
    Result of the pre-comparison metadata compatibility check.

    compatible=True means the comparison can proceed.
    compatible=False means the images cannot be meaningfully compared.
    warnings are non-fatal differences (e.g. different resolutions).
    """

    compatible: bool = Field(
        description="True when comparison can proceed; False when it is blocked.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Reasons for incompatibility (non-empty only when compatible=False).",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings (e.g. dimension mismatch, different CRS).",
    )


# ─── Preprocessing / Alignment ───────────────────────────────────────────────

class PreprocessingResult(BaseModel):
    """
    Records what preprocessing was applied to prepare the images for comparison.
    No preprocessing is invented — only what actually happened is recorded.
    """

    image_a_prepared: bool = False
    image_b_prepared: bool = False
    alignment_attempted: bool = False
    alignment_succeeded: bool = False
    alignment_method: Optional[str] = None  # e.g. "geospatial_crs_match", "none"
    alignment_note: Optional[str] = None
    limitations: list[str] = Field(default_factory=list)


# ─── Detected Change ──────────────────────────────────────────────────────────

class DetectedChange(BaseModel):
    """One detected change item extracted from the AI comparison response."""

    feature: str = Field(description="Feature name, e.g. 'Vegetation Cover'.")
    direction: str = Field(
        description="Change direction: 'up' | 'down' | 'stable' | 'unknown'.",
    )
    label: str = Field(
        description="Human-readable change label, e.g. 'Appears Decreased'.",
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional detail from AI analysis.",
    )


# ─── Response ─────────────────────────────────────────────────────────────────

class ComparisonResponse(BaseModel):
    """Full comparison result returned by POST /api/comparisons."""

    comparison_id: str
    session_id: Optional[str] = None

    # Image information
    image_id_a: str
    image_id_b: str
    label_a: str
    label_b: str
    meta_a: ImageMetaSummary
    meta_b: ImageMetaSummary

    # Pipeline status
    compatibility: CompatibilityResult
    preprocessing: PreprocessingResult
    status: str = Field(
        description="'completed' | 'failed' | 'incompatible'",
    )

    # Results
    query: str
    ai_explanation: Optional[str] = Field(
        default=None,
        description="AI-generated explanation of the detected changes.",
    )
    detected_changes: list[DetectedChange] = Field(
        default_factory=list,
        description="Structured list of detected change items.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations of this comparison result.",
    )
    error_message: Optional[str] = None

    # Provider info
    provider: str
    model: str

    # Timestamps
    created_at: datetime
    completed_at: Optional[datetime] = None


# ─── Compact list item ────────────────────────────────────────────────────────

class ComparisonSummary(BaseModel):
    """Compact comparison summary for listing endpoints (future)."""

    comparison_id: str
    image_id_a: str
    image_id_b: str
    label_a: str
    label_b: str
    status: str
    created_at: datetime


# ─── Internal MongoDB Document ────────────────────────────────────────────────

class ComparisonDocument(BaseModel):
    """
    Represents a comparison document as stored in MongoDB.
    Used internally — not directly serialised to API responses.
    """

    comparison_id: str
    session_id: Optional[str] = None
    image_id_a: str
    image_id_b: str
    label_a: str
    label_b: str
    query: str
    status: str
    meta_a: dict = Field(default_factory=dict)
    meta_b: dict = Field(default_factory=dict)
    compatibility: dict = Field(default_factory=dict)
    preprocessing: dict = Field(default_factory=dict)
    ai_explanation: Optional[str] = None
    detected_changes: list[dict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    provider: str
    model: str
    created_at: datetime
    completed_at: Optional[datetime] = None

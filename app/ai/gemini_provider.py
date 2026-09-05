"""
SatQuery AI — Gemini AI Provider (Phase 4)

Implements BaseAIProvider for Google Gemini via the google-genai SDK.

Responsibilities:
  - Authenticate using AI_API_KEY from settings (never from frontend).
  - Send image bytes + prompt to the Gemini Vision API.
  - Apply timeout from settings.ai_timeout_seconds.
  - Return AIRawResponse — never raise exceptions.
  - Never touch MongoDB, sessions, or question logic.

Image handling:
  - For standard RGB images (JPEG, PNG, WebP): send file bytes directly.
  - For TIFF/GeoTIFF: use the processed (RGB-converted) derivative stored in
    uploads/processed/ so the model receives a usable image.
    The prompts.py module appends a limitation note automatically for GeoTIFF.

Retry policy:
  - One retry on transient network/5xx errors only.
  - No retry on 400, 401, 403 (bad request / auth failure).
  - No aggressive retry to avoid duplicate charges.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

# Top-level SDK imports — required for Pyright/Pylance to resolve the module.
# Ensure google-genai is installed in the project venv:
#   venv/Scripts/pip install -U google-genai
from google import genai
from google.genai import types as genai_types

from app.ai.base import AIRawResponse, AIRequest, BaseAIProvider
from app.core.config import settings

logger = logging.getLogger(__name__)

# MIME types Gemini Vision API accepts natively
_GEMINI_SUPPORTED_MIME = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/heic",
    "image/heif",
}

# Error substrings that must NOT trigger a retry.
# 404/NOT_FOUND: obsolete or invalid model — never retryable.
_NON_RETRYABLE = (
    "400",
    "401",
    "403",
    "404",
    "PERMISSION_DENIED",
    "INVALID_ARGUMENT",
    "API_KEY_INVALID",
    "UNAUTHENTICATED",
    "NOT_FOUND",
)


class GeminiProvider(BaseAIProvider):
    """
    Google Gemini Vision provider using the google-genai SDK.

    Usage: instantiated once by provider.py get_provider().
    """

    def __init__(self) -> None:
        self._client: genai.Client | None = None  # Lazy-initialised on first call

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return settings.ai_model or "gemini-3.6-flash"

    def _get_client(self) -> genai.Client:
        """Lazy-initialise the google-genai client."""
        if self._client is None:
            self._client = genai.Client(api_key=settings.ai_api_key)
        return self._client

    def _resolve_image_path(self, request: AIRequest) -> tuple[Path, str]:
        """
        Choose the best image file to send.

        For TIFF/GeoTIFF: use the processed JPEG derivative (uploads/processed/).
        For supported MIME types: use the file as provided.

        Returns (path, effective_mime_type).
        """
        original_path = request.image_path
        mime = (request.mime_type or "").lower()

        # TIFF/GeoTIFF — use processed derivative
        is_tiff = (
            mime in ("image/tiff", "image/geotiff")
            or original_path.suffix.lower() in (".tif", ".tiff")
        )
        if is_tiff:
            # Look for the processed version (JPEG) in uploads/processed/
            stem = original_path.stem  # e.g. img_01ABC...
            processed_dir = settings.processed_path
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = processed_dir / f"{stem}{ext}"
                if candidate.exists():
                    logger.info(
                        "TIFF detected: using processed derivative %s", candidate
                    )
                    return candidate, "image/jpeg"
            # Fallback: try to read the original TIFF directly
            logger.warning(
                "TIFF processed derivative not found for %s; will attempt direct send.",
                original_path,
            )

        # Standard supported image — send as-is
        effective_mime = mime if mime in _GEMINI_SUPPORTED_MIME else "image/jpeg"
        return original_path, effective_mime

    def analyze(self, request: AIRequest) -> AIRawResponse:
        """
        Send image + prompt to Gemini and return the raw response.
        Never raises — returns AIRawResponse(success=False) on any error.
        """
        start = time.monotonic()

        try:
            client = self._get_client()
        except Exception as exc:
            return AIRawResponse(
                success=False,
                error_message=f"Gemini client initialisation failed: {exc}",
                provider=self.provider_name,
                model=self.model_name,
            )

        # Resolve image file to send
        try:
            image_path, effective_mime = self._resolve_image_path(request)
            image_bytes = image_path.read_bytes()
        except Exception as exc:
            return AIRawResponse(
                success=False,
                error_message=f"Failed to read image file: {exc}",
                provider=self.provider_name,
                model=self.model_name,
            )

        # Build prompt — use context-aware builder when history is present
        system_prompt = request.system_prompt or ""
        if request.conversation_history:
            from app.ai.prompts import build_conversation_context_message
            # is_geospatial is not available here; the service already baked it into query
            user_message = request.query  # query already contains context from service layer
        else:
            user_message = request.query

        # Attempt with one retry on transient error
        for attempt in range(2):
            try:
                logger.info(
                    "Gemini request [attempt %d]: model=%s mime=%s size=%d bytes query=%r",
                    attempt + 1,
                    self.model_name,
                    effective_mime,
                    len(image_bytes),
                    user_message[:80],
                )

                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        genai_types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=effective_mime,
                        ),
                        user_message,
                    ],
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=4096,
                        # temperature intentionally omitted — not compatible with
                        # gemini-3.6-flash and causes 400 INVALID_ARGUMENT errors.
                    ),
                )

                elapsed = time.monotonic() - start
                logger.info("Gemini response received in %.2fs", elapsed)

                # Extract text
                text = ""
                try:
                    text = response.text or ""
                except Exception:
                    # Some response types require iterating candidates
                    try:
                        text = response.candidates[0].content.parts[0].text or ""
                    except Exception:
                        text = ""

                return AIRawResponse(
                    text=text,
                    success=True,
                    provider=self.provider_name,
                    model=self.model_name,
                    raw_data={"elapsed_seconds": round(elapsed, 3)},
                )

            except Exception as exc:
                error_str = str(exc)
                elapsed = time.monotonic() - start
                logger.warning(
                    "Gemini attempt %d failed after %.2fs: %s",
                    attempt + 1,
                    elapsed,
                    error_str[:200],
                )

                # Do NOT retry auth/bad-request/not-found errors
                if any(code in error_str for code in _NON_RETRYABLE):
                    return AIRawResponse(
                        success=False,
                        error_message=f"Gemini non-retryable error: {error_str}",
                        provider=self.provider_name,
                        model=self.model_name,
                    )

                if attempt == 0:
                    # Wait briefly before retry (transient error)
                    time.sleep(1.5)
                    continue

                # Final failure after retry
                return AIRawResponse(
                    success=False,
                    error_message=f"Gemini provider error: {error_str}",
                    provider=self.provider_name,
                    model=self.model_name,
                )

        # Should not be reached
        return AIRawResponse(
            success=False,
            error_message="Unexpected provider loop exit.",
            provider=self.provider_name,
            model=self.model_name,
        )

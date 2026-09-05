"""
SatQuery AI — AI Provider Factory (Phase 4)

Resolves the configured provider from settings.ai_provider.
Adding a new provider: register it in _PROVIDER_REGISTRY.

Design:
  - get_provider() is called by the analysis service, not by routes.
  - Returns a BaseAIProvider instance or raises HTTP 503.
  - Provider instances are not cached at module level so the API key is always
    read from the current settings (useful when tests patch settings).
"""

from __future__ import annotations

import logging
from typing import Type

from fastapi import HTTPException

from app.ai.base import BaseAIProvider
from app.core.config import settings

logger = logging.getLogger(__name__)


def _load_gemini() -> Type[BaseAIProvider]:
    from app.ai.gemini_provider import GeminiProvider
    return GeminiProvider


# ─── Provider registry ────────────────────────────────────────────────────────
# Map provider name → callable that returns the concrete class.
# Keep imports lazy so importing provider.py never triggers heavy SDK imports.

_PROVIDER_REGISTRY: dict[str, callable] = {
    "gemini": _load_gemini,
}


def get_provider() -> BaseAIProvider:
    """
    Return a ready-to-use AI provider instance.

    Raises HTTP 503 if:
      - AI_API_KEY is not configured.
      - The configured AI_PROVIDER is not supported.
    """
    if not settings.ai_configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI provider is not configured. "
                "Please set AI_API_KEY in your .env file."
            ),
        )

    provider_name = (settings.ai_provider or "gemini").lower().strip()
    loader = _PROVIDER_REGISTRY.get(provider_name)

    if loader is None:
        supported = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        logger.error(
            "Unsupported AI provider: %r. Supported: %s", provider_name, supported
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"AI provider '{provider_name}' is not supported. "
                f"Supported providers: {supported}"
            ),
        )

    provider_class = loader()
    return provider_class()


def list_supported_providers() -> list[str]:
    """Return names of all registered providers."""
    return sorted(_PROVIDER_REGISTRY.keys())

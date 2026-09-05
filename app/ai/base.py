"""
SatQuery AI — Abstract AI Provider Base (Phase 4)

Defines the contract that every AI provider must fulfil.
Adding a new provider = subclass BaseAIProvider + register in provider.py.

Design rules:
  - Provider classes must be stateless between calls.
  - Provider classes must NOT touch MongoDB.
  - Provider classes must NOT contain session or question logic.
  - Provider classes return a raw dict; response_parser.py normalises it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Request / Response data classes ─────────────────────────────────────────

@dataclass
class AIRequest:
    """
    Input to any AI provider.

    image_path            - Absolute path to the image file on disk.
    mime_type             - MIME type of the image (e.g. 'image/jpeg').
    query                 - The user's natural-language question.
    system_prompt         - Optional override for the system instruction.
    analysis_type         - Hint for the provider (currently used for prompt selection).
    conversation_history  - Previous Q&A turns for follow-up context.
                            Each entry is {"question": str, "answer": str}.
                            Limited to last N turns by the service layer.
    """
    image_path: Path
    mime_type: str
    query: str
    system_prompt: Optional[str] = None
    analysis_type: str = "general"
    conversation_history: list = field(default_factory=list)


@dataclass
class AIRawResponse:
    """
    Raw response from the provider, before parsing.

    text           - The provider's text output (may be empty on error).
    success        - False when the provider returned an error.
    error_message  - Human-readable error string (populated when success=False).
    raw_data       - Full provider response payload for debugging.
    provider       - Which provider produced this response.
    model          - Model identifier used.
    """
    text: str = ""
    success: bool = True
    error_message: Optional[str] = None
    raw_data: Optional[dict] = field(default_factory=dict)
    provider: str = "unknown"
    model: str = "unknown"


# ─── Abstract Provider ────────────────────────────────────────────────────────

class BaseAIProvider(abc.ABC):
    """
    Abstract base class for AI vision providers.

    Concrete implementations:
      - GeminiProvider  (app/ai/gemini_provider.py)

    Future providers:
      - OpenAIProvider
      - AnthropicProvider
    """

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'gemini', 'openai')."""
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Model identifier as understood by the provider's API."""
        ...

    @abc.abstractmethod
    def analyze(self, request: AIRequest) -> AIRawResponse:
        """
        Send image + query to the provider and return the raw response.

        Must be synchronous (run in thread pool via FastAPI if needed).
        Must NOT raise exceptions — return AIRawResponse(success=False) instead.
        Must NOT write to MongoDB.
        Must NOT contain session or question logic.
        Must respect the timeout configured in settings.ai_timeout_seconds.
        """
        ...

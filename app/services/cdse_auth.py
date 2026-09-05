"""
SatQuery AI — CDSE OAuth2 Authentication (Phase 2)

Provides Bearer tokens required to download Sentinel-2 L2A band assets
from the Copernicus Data Space Ecosystem (CDSE).

Architecture:
  spectral_service.py / raster_service.py
      ↓
  cdse_auth.py   ← this file
      ↓
  POST https://identity.dataspace.copernicus.eu/auth/realms/CDSE/
       protocol/openid-connect/token
      ↓
  Bearer token → added to GDAL_HTTP_HEADERS for /vsicurl/ reads

Security rules:
  - Credentials are ONLY read from environment variables (via config.settings).
  - Tokens are NEVER logged (even at DEBUG level).
  - Tokens are NEVER returned to the frontend.
  - If credentials are missing, raises HTTP 503 — never falls back to mock data.

Token caching:
  - Tokens are cached in-process until 60 seconds before expiry.
  - Thread-safe via threading.Lock.
  - Cache is cleared on credential change (module reload).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import httpx
from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── CDSE token endpoint ──────────────────────────────────────────────────────

CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)

# Safety buffer: refresh token 60 seconds before it expires
TOKEN_REFRESH_BUFFER_SECONDS = 60

# ─── In-process token cache ───────────────────────────────────────────────────

_token_lock = threading.Lock()
_cached_token: Optional[str] = None
_token_expires_at: float = 0.0  # Unix timestamp


# ─── Public API ───────────────────────────────────────────────────────────────

def get_cdse_token() -> str:
    """
    Return a valid CDSE Bearer token.

    Steps:
      1. Check in-process cache — return cached token if still valid.
      2. Validate credentials exist (raises HTTP 503 if missing).
      3. Request a new token from the CDSE identity server.
      4. Cache the token with expiry.
      5. Return the token.

    Raises:
      HTTPException 503 — credentials not configured.
      HTTPException 503 — token endpoint unreachable / timed out.
      HTTPException 502 — token endpoint returned non-200 response.
    """
    global _cached_token, _token_expires_at

    with _token_lock:
        now = time.monotonic()

        # 1. Return cached token if still valid (with safety buffer)
        if _cached_token and now < (_token_expires_at - TOKEN_REFRESH_BUFFER_SECONDS):
            logger.debug("Using cached CDSE token (expires in %.0fs)", _token_expires_at - now)
            return _cached_token

        # 2. Validate credentials
        client_id = settings.copernicus_client_id
        client_secret = settings.copernicus_client_secret

        if not client_id or not client_secret:
            logger.error(
                "CDSE credentials not configured. "
                "Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in your .env file. "
                "Register at https://dataspace.copernicus.eu/"
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Copernicus Data Space credentials are not configured. "
                    "Real spectral band analysis requires COPERNICUS_CLIENT_ID and "
                    "COPERNICUS_CLIENT_SECRET in your .env file. "
                    "Register at: https://dataspace.copernicus.eu/ "
                    "No mock data will be returned."
                ),
            )

        # 3. Request new token
        logger.info("Requesting new CDSE OAuth2 token (client_id=%s…)", client_id[:6])

        payload = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        }

        try:
            response = httpx.post(
                CDSE_TOKEN_URL,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0,
            )
        except httpx.TimeoutException:
            logger.error("CDSE token endpoint timed out after 30s")
            raise HTTPException(
                status_code=503,
                detail="CDSE identity server did not respond in 30 seconds. Try again later.",
            )
        except httpx.ConnectError as exc:
            logger.error("Cannot reach CDSE token endpoint: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Cannot reach Copernicus identity server. Check network connectivity.",
            )
        except httpx.HTTPError as exc:
            logger.error("CDSE token request HTTP error: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=f"CDSE token request failed: {exc}",
            )

        if response.status_code != 200:
            # Do NOT log response body — may contain credential hints
            logger.error(
                "CDSE token endpoint returned HTTP %d (credentials may be invalid)",
                response.status_code,
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"CDSE identity server returned HTTP {response.status_code}. "
                    "Check that COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET are correct."
                ),
            )

        try:
            token_data = response.json()
        except Exception as exc:
            logger.error("Failed to parse CDSE token response: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="CDSE identity server returned an invalid JSON response.",
            )

        access_token = token_data.get("access_token")
        if not access_token:
            logger.error("CDSE token response missing 'access_token' field")
            raise HTTPException(
                status_code=502,
                detail="CDSE identity server did not return an access_token.",
            )

        expires_in = token_data.get("expires_in", 600)  # default 10 minutes

        # 4. Cache the token
        _cached_token = access_token
        _token_expires_at = time.monotonic() + float(expires_in)

        logger.info(
            "CDSE token acquired successfully (expires_in=%ss)", expires_in
        )
        # NEVER log the token itself

    return _cached_token


def is_cdse_configured() -> bool:
    """Return True if CDSE credentials are present in settings."""
    return bool(
        settings.copernicus_client_id
        and settings.copernicus_client_secret
    )


def clear_token_cache() -> None:
    """
    Clear the in-process token cache.
    Useful when credentials change or for testing.
    """
    global _cached_token, _token_expires_at
    with _token_lock:
        _cached_token = None
        _token_expires_at = 0.0
    logger.info("CDSE token cache cleared.")

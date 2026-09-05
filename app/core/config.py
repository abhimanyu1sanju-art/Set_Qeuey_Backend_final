"""
SatQuery AI — Application Configuration
Loaded once at startup via Pydantic Settings.
All values come from environment variables / .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── General ──────────────────────────────────────────────────────────────
    environment: str = "development"
    api_prefix: str = "/api"

    # ── MongoDB ───────────────────────────────────────────────────────────────
    mongodb_uri: str = "mongodb://localhost:27017"
    database_name: str = "satquery"

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Stored as a comma-separated string in .env; parsed into a list below.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Optional regex for dynamic CORS origins (e.g. Vercel preview deployments).
    # When set, FastAPI CORSMiddleware also accepts any origin matching this pattern.
    # Covers both project spellings (set-queqy / set-qeuery) and all preview hashes.
    # Example: https://set-q[a-z]+-frontend-3a22[a-z0-9-]*.vercel.app
    cors_origin_regex: str = ""

    # ── Phase 2 — Image Storage ───────────────────────────────────────────────
    upload_dir: str = "uploads"           # Root upload directory (relative to cwd)
    max_file_size_mb: int = 20            # Maximum upload size in megabytes
    thumbnail_size: int = 512             # Max thumbnail dimension (width & height)

    # ── Phase 4 — AI Provider ────────────────────────────────────────────────
    # Set AI_PROVIDER, AI_API_KEY and AI_MODEL in your .env file.
    # The application starts without these values but returns HTTP 503 on
    # analysis requests if AI_API_KEY is not configured.
    ai_provider: str = "gemini"           # Provider name: 'gemini' | 'openai' | ...
    ai_api_key: Optional[str] = None      # Secret key — NEVER send to frontend or logs
    ai_model: str = "gemini-3.6-flash"    # Model identifier for the provider
    ai_timeout_seconds: int = 60          # Max seconds to wait for an AI response

    # ── Phase 1 — Copernicus Data Space (Satellite) ───────────────────────────
    # STAC search is public — no credentials required.
    # Credentials are only needed for asset/band download (Phase 2+).
    copernicus_stac_url: str = "https://stac.dataspace.copernicus.eu/v1"
    copernicus_client_id: Optional[str] = None
    copernicus_client_secret: Optional[str] = None

    # ── Phase 2 — Spectral Band Analysis ─────────────────────────────────────
    # CDSE OAuth2 token endpoint (standard — should not need to change).
    cdse_token_url: str = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/protocol/openid-connect/token"
    )
    # Directory where spectral GeoTIFF outputs are written.
    # Relative to CWD when the backend is started.
    spectral_output_dir: str = "outputs/spectral"

    # ── Phase 2 — Copernicus S3 Band Access ──────────────────────────────────
    # Used by raster_service to read Sentinel-2 band JP2 files via
    # GDAL /vsis3/ virtual filesystem.
    # SECURITY: CDSE_S3_SECRET_KEY must NEVER appear in logs, API responses,
    # frontend code, error messages, or Git history.
    # Generate at: dataspace.copernicus.eu → Profile → S3 Credentials
    cdse_s3_access_key: Optional[str] = None
    cdse_s3_secret_key: Optional[str] = None          # SECRET — never log
    cdse_s3_endpoint: str = "https://eodata.dataspace.copernicus.eu"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a clean list (handles trailing whitespace)."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def cors_origin_regex_or_none(self) -> Optional[str]:
        """Return the CORS origin regex, or None if not configured."""
        return self.cors_origin_regex.strip() or None

    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"

    @property
    def max_file_size_bytes(self) -> int:
        """Maximum upload size expressed in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def ai_configured(self) -> bool:
        """True only when a non-empty AI_API_KEY has been set."""
        return bool(self.ai_api_key and self.ai_api_key.strip())

    @property
    def copernicus_configured(self) -> bool:
        """True when CDSE OAuth2 credentials are present for bearer-token band download."""
        return bool(
            self.copernicus_client_id
            and self.copernicus_client_secret
        )

    @property
    def cdse_s3_configured(self) -> bool:
        """True when CDSE S3 credentials are present for /vsis3/ band access."""
        return bool(
            self.cdse_s3_access_key
            and self.cdse_s3_secret_key
        )

    @property
    def cdse_s3_endpoint_host(self) -> str:
        """
        Return the S3 endpoint as a bare hostname (no scheme) for GDAL.
        GDAL's AWS_S3_ENDPOINT expects 'eodata.dataspace.copernicus.eu',
        not 'https://eodata.dataspace.copernicus.eu'.
        """
        host = self.cdse_s3_endpoint
        for prefix in ("https://", "http://"):
            if host.startswith(prefix):
                host = host[len(prefix):]
        return host.rstrip("/")

    @property
    def upload_path(self) -> Path:
        """Absolute Path object for the upload root directory."""
        return Path(self.upload_dir).resolve()

    @property
    def original_path(self) -> Path:
        return self.upload_path / "original"

    @property
    def processed_path(self) -> Path:
        return self.upload_path / "processed"

    @property
    def thumbnails_path(self) -> Path:
        return self.upload_path / "thumbnails"


# Single application-wide settings instance.
# Import this in any module that needs configuration:
#   from app.core.config import settings
settings = Settings()

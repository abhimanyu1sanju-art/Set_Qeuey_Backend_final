"""
SatQuery AI — FastAPI Application Entry Point

Architecture:
  Phase 1:  GET /api/health                                        — service health + MongoDB status
  Phase 2:  POST /api/images/upload                               — image upload, validation, storage
            GET  /api/images/{id}                                 — image metadata retrieval
            GET  /api/images/{id}/thumbnail                      — serve thumbnail JPEG
            DELETE /api/images/{id}                               — image deletion
  Phase 3:  POST   /api/sessions                                  — create session
            GET    /api/sessions                                  — list sessions
            GET    /api/sessions/{id}                             — get session
            DELETE /api/sessions/{id}                             — delete session
            POST   /api/sessions/{id}/images/{img_id}            — link image to session
            GET    /api/sessions/{id}/images                     — session images
            POST   /api/sessions/{id}/questions                  — add question
            GET    /api/sessions/{id}/questions                  — list questions
            PATCH  /api/sessions/{id}/questions/{q_id}           — update question
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health
from app.api.routes import images
from app.api.routes import sessions
from app.api.routes import analysis
from app.api.routes import comparisons
from app.api.routes import history
from app.api.routes import satellite
from app.api.routes import spectral  # Phase 2: Real spectral analysis
from app.api.routes import change    # Phase 3: Change detection
from app.api.routes import infrastructure  # Phase 4: Infrastructure detection
from app.api.routes import disaster      # Phase 5: Disaster detection
from app.api.routes import fusion        # Phase 6: Fusion
from app.api.routes import timeline
from app.core.config import settings
from app.db.mongodb import (
    close_client, ensure_image_indexes, ensure_session_indexes,
    ensure_analysis_indexes, ensure_comparison_indexes, ensure_history_indexes,
    ensure_satellite_indexes, ensure_spectral_analysis_indexes,
    ensure_change_analysis_indexes, ensure_infrastructure_indexes,
    verify_connection,
)
from app.storage.local_storage import ensure_upload_dirs

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.is_development else logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Run startup tasks before yielding control to the application,
    then run shutdown tasks when the application exits.
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("SatQuery AI backend starting — environment: %s", settings.environment)
    logger.info("CORS origins: %s", settings.cors_origins_list)
    logger.info("CORS origin regex: %s", settings.cors_origin_regex_or_none or "(none)")

    # Phase 1: MongoDB connectivity check
    db_ok = verify_connection()
    if db_ok:
        logger.info("MongoDB ready: db=%s", settings.database_name)
        # Phase 2: Ensure image indexes exist
        ensure_image_indexes()
        # Phase 3: Ensure session indexes exist
        ensure_session_indexes()
        # Phase 4: Ensure analysis indexes exist
        ensure_analysis_indexes()
        # Phase 7: Ensure comparison indexes exist
        ensure_comparison_indexes()
        # Phase 8: Ensure history indexes exist
        ensure_history_indexes()
        # Phase 1 (Satellite): Ensure satellite_scenes indexes exist
        ensure_satellite_indexes()
        # Phase 2 (Spectral): Ensure spectral_analyses indexes exist
        ensure_spectral_analysis_indexes()
        # Phase 3 (Change): Ensure change_analyses indexes exist
        ensure_change_analysis_indexes()
        # Phase 4 (Infrastructure): Ensure infrastructure_results indexes exist
        ensure_infrastructure_indexes()
    else:
        logger.warning(
            "MongoDB is NOT reachable — "
            "the API will start anyway; /api/health will report 'unavailable'."
        )

    # Phase 2: Ensure upload directories exist on startup
    try:
        ensure_upload_dirs()
        logger.info(
            "Upload directories ready: %s/{original,processed,thumbnails}",
            settings.upload_dir,
        )
    except Exception as exc:
        logger.error("Failed to create upload directories: %s", exc)

    # Phase 2 (Spectral): Ensure spectral output directory exists
    try:
        from app.services.spectral_service import ensure_output_dirs
        ensure_output_dirs()
    except Exception as exc:
        logger.error("Failed to create spectral output directories: %s", exc)

    yield  # Application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("SatQuery AI backend shutting down…")
    close_client()


# ─── Application factory ─────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="SatQuery AI",
        description=(
            "**SatQuery AI** — Satellite and Remote-Sensing Image Analysis Backend.\n\n"
            "**Phase 2 — Image Management System**\n\n"
            "Supports upload, validation, storage, metadata extraction, and GeoTIFF "
            "geospatial metadata for satellite and remote-sensing images.\n\n"
            "**Phase 3 — Session System**\n\n"
            "Manages analysis sessions: link images, store questions and answers, "
            "retrieve history from MongoDB.\n\n"
            "**Phase 4 — Real AI Analysis**\n\n"
            "AI-powered image analysis via configurable provider (Gemini). "
            "Requires `AI_API_KEY` in environment.\n\n"
            "**Phase 6 — Multi-Intent Analysis**\n\n"
            "Detects multiple analysis intents in a single query and executes each separately.\n\n"
            "**Phase 7 — Two-Image Comparison**\n\n"
            "AI-powered change-detection comparison between two uploaded satellite images. "
            "Sends both images to Gemini in a single request, validates metadata "
            "compatibility, and returns structured change-detection results.\n\n"
            "**Supported formats:** JPG, JPEG, PNG, TIFF, GeoTIFF\n\n"
        ),
        version="0.8.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_origin_regex=settings.cors_origin_regex_or_none,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────
    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(images.router, prefix=settings.api_prefix)       # Phase 2
    app.include_router(sessions.router, prefix=settings.api_prefix)     # Phase 3
    app.include_router(analysis.router, prefix=settings.api_prefix)     # Phase 4
    app.include_router(comparisons.router, prefix=settings.api_prefix)  # Phase 7
    app.include_router(history.router, prefix=settings.api_prefix)      # Phase 8
    app.include_router(satellite.router, prefix=settings.api_prefix)    # Phase 1 (Satellite)
    app.include_router(spectral.router, prefix=settings.api_prefix)     # Phase 2 (Spectral)
    app.include_router(change.router, prefix=settings.api_prefix)       # Phase 3 (Change)
    app.include_router(infrastructure.router, prefix=settings.api_prefix)  # Phase 4 (Infrastructure)
    app.include_router(disaster.router, prefix=settings.api_prefix)     # Phase 5 (Disaster)
    app.include_router(fusion.router, prefix=settings.api_prefix)       # Phase 6 (Fusion)
    app.include_router(timeline.router, prefix=f"{settings.api_prefix}/timeline")

    logger.info("Routers registered. API prefix: %s", settings.api_prefix)
    return app


# ─── Module-level app instance ───────────────────────────────────────────────
# Entry point: uvicorn app.main:app --reload
app = create_app()

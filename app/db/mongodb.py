"""
SatQuery AI — MongoDB Connection Layer

Provides:
  - A lazily-initialised PyMongo client
  - A helper to access the configured database
  - A connection test used during startup (failure is non-fatal)
  - Phase 2: images collection helper + unique index on image_id
  - Phase 3: sessions collection helper + indexes
  - Phase 4: analyses collection helper + indexes
  - Phase 7: comparisons collection helper + indexes
  - Phase 8: history collection helper + indexes
  - Phase 1 (satellite): satellite_scenes collection helper + indexes
"""

from __future__ import annotations

import logging
from typing import Optional

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ConfigurationError, ServerSelectionTimeoutError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level client reference; initialised on first use.
_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    """Return (or create) the shared MongoClient instance."""
    global _client
    if _client is None:
        _client = MongoClient(
            settings.mongodb_uri,
            # Atlas needs more time than a local server on first connection.
            serverSelectionTimeoutMS=15_000,
            connectTimeoutMS=15_000,
            socketTimeoutMS=20_000,
            # Force TLS for Atlas (mongodb+srv already implies it, but explicit is safer).
            tls=True,
        )
    return _client


def get_database() -> Database:
    """Return the configured database from the shared client."""
    return get_client()[settings.database_name]


def verify_connection() -> bool:
    """
    Ping MongoDB to confirm reachability.
    Returns True on success, False on failure.
    Designed for startup health checks — never raises.
    """
    global _client
    try:
        get_client().admin.command("ping")
        logger.info("MongoDB connection successful (db: %s)", settings.database_name)
        return True
    except (ConnectionFailure, ConfigurationError, ServerSelectionTimeoutError, Exception) as exc:
        # Reset the client so the next verify_connection() call creates a fresh one.
        _client = None
        # Log the error type only — never log the URI or credentials.
        logger.warning("MongoDB connection failed [%s]: %s", type(exc).__name__, exc)
        return False


def close_client() -> None:
    """Close the shared client (called on application shutdown)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB client closed.")


# ─── Phase 2 — Collection helpers ────────────────────────────────────────────

def get_images_collection() -> Collection:
    """Return the 'images' collection from the configured database."""
    return get_database()["images"]


def ensure_image_indexes() -> None:
    """
    Create required indexes on the images collection.
    Called once during application startup.
    A unique index on image_id makes future lookups fast and collision-safe.
    """
    try:
        col = get_images_collection()
        col.create_index([("image_id", ASCENDING)], unique=True, name="image_id_unique")
        logger.info("MongoDB indexes ensured on 'images' collection.")
    except Exception as exc:
        # Non-fatal: log and continue. The index may already exist.
        logger.warning("Could not ensure image indexes [%s]: %s", type(exc).__name__, exc)


# ─── Phase 3 — Session collection helpers ─────────────────────────────────────

def get_sessions_collection() -> Collection:
    """Return the 'sessions' collection from the configured database."""
    return get_database()["sessions"]


def ensure_session_indexes() -> None:
    """
    Create required indexes on the sessions collection.
    Called once during application startup.
      - Unique index on session_id for fast lookups
      - Descending index on updated_at for list sorting
    """
    try:
        col = get_sessions_collection()
        col.create_index([("session_id", ASCENDING)], unique=True, name="session_id_unique")
        col.create_index([("updated_at", DESCENDING)], name="updated_at_desc")
        col.create_index([("created_at", DESCENDING)], name="created_at_desc")
        logger.info("MongoDB indexes ensured on 'sessions' collection.")
    except Exception as exc:
        # Non-fatal: log and continue. The index may already exist.
        logger.warning("Could not ensure session indexes [%s]: %s", type(exc).__name__, exc)


# ─── Phase 4 — Analyses collection helpers ────────────────────────────────────

def get_analyses_collection() -> Collection:
    """Return the 'analyses' collection from the configured database."""
    return get_database()["analyses"]


def ensure_analysis_indexes() -> None:
    """
    Create required indexes on the analyses collection.
    Called once during application startup.
      - Unique index on analysis_id for fast lookups
      - Index on session_id for session-scoped queries
      - Index on image_id for image-scoped queries
      - Descending index on created_at for history ordering
    """
    try:
        col = get_analyses_collection()
        col.create_index([("analysis_id", ASCENDING)], unique=True, name="analysis_id_unique")
        col.create_index([("session_id", ASCENDING)], name="session_id_idx")
        col.create_index([("image_id", ASCENDING)], name="image_id_idx")
        col.create_index([("created_at", DESCENDING)], name="created_at_desc")
        logger.info("MongoDB indexes ensured on 'analyses' collection.")
    except Exception as exc:
        # Non-fatal: log and continue. The index may already exist.
        logger.warning("Could not ensure analysis indexes [%s]: %s", type(exc).__name__, exc)

# ─── Phase 7 — Comparisons collection helpers ───────────────────────────────────

def get_comparisons_collection() -> Collection:
    """Return the 'comparisons' collection from the configured database."""
    return get_database()["comparisons"]


def ensure_comparison_indexes() -> None:
    """
    Create required indexes on the comparisons collection.
    Called once during application startup.
      - Unique index on comparison_id for fast lookups
      - Index on session_id for session-scoped queries
      - Index on image_id_a / image_id_b for image-scoped queries
      - Descending index on created_at for history ordering
    """
    try:
        col = get_comparisons_collection()
        col.create_index([("comparison_id", ASCENDING)], unique=True, name="comparison_id_unique")
        col.create_index([("session_id", ASCENDING)], name="comparison_session_idx")
        col.create_index([("image_id_a", ASCENDING)], name="comparison_image_a_idx")
        col.create_index([("image_id_b", ASCENDING)], name="comparison_image_b_idx")
        col.create_index([("created_at", DESCENDING)], name="comparison_created_at_desc")
        logger.info("MongoDB indexes ensured on 'comparisons' collection.")
    except Exception as exc:
        logger.warning("Could not ensure comparison indexes [%s]: %s", type(exc).__name__, exc)


# ─── Phase 8 — History collection helpers ────────────────────────────────────

def get_history_collection() -> Collection:
    """Return the 'history' collection from the configured database."""
    return get_database()["history"]


# ─── Phase 1 (Satellite) — Satellite scenes collection helpers ────────────────

def get_satellite_scenes_collection() -> Collection:
    """Return the 'satellite_scenes' collection from the configured database."""
    return get_database()["satellite_scenes"]


def ensure_history_indexes() -> None:
    """
    Create required indexes on the history collection.
    Called once during application startup.
      - Unique index on history_id for fast lookups
      - Index on session_id for session-scoped queries
      - Index on analysis_type for type filtering
      - Descending index on created_at for history ordering / pagination
      - Text index on query for full-text search
    """
    try:
        col = get_history_collection()
        col.create_index([("history_id", ASCENDING)], unique=True, name="history_id_unique")
        col.create_index([("session_id", ASCENDING)], name="history_session_idx")
        col.create_index([("analysis_type", ASCENDING)], name="history_type_idx")
        col.create_index([("created_at", DESCENDING)], name="history_created_at_desc")
        col.create_index([("query", "text")], name="history_query_text")
        logger.info("MongoDB indexes ensured on 'history' collection.")
    except Exception as exc:
        logger.warning("Could not ensure history indexes [%s]: %s", type(exc).__name__, exc)


def ensure_satellite_indexes() -> None:
    """
    Create required indexes on the satellite_scenes collection.
    Called once during application startup.
      - Unique index on scene_id (STAC item ID)
      - Index on acquired_at for date-range queries
      - Index on platform for satellite filtering
      - Index on collection for collection-scoped queries
    """
    try:
        col = get_satellite_scenes_collection()
        col.create_index([("scene_id", ASCENDING)], unique=True, name="scene_id_unique")
        col.create_index([("acquired_at", DESCENDING)], name="satellite_acquired_at_desc")
        col.create_index([("platform", ASCENDING)], name="satellite_platform_idx")
        col.create_index([("collection", ASCENDING)], name="satellite_collection_idx")
        logger.info("MongoDB indexes ensured on 'satellite_scenes' collection.")
    except Exception as exc:
        logger.warning("Could not ensure satellite indexes [%s]: %s", type(exc).__name__, exc)


# ─── Phase 2 (Spectral) — Spectral analyses collection helpers ───────────────

def get_spectral_analyses_collection() -> Collection:
    """Return the 'spectral_analyses' collection from the configured database."""
    return get_database()["spectral_analyses"]


def ensure_spectral_analysis_indexes() -> None:
    """
    Create required indexes on the spectral_analyses collection.
    Called once during application startup.
      - Compound unique index on (scene_id, index_type) — one record per scene+index
      - Index on created_at for date ordering
      - Index on status for filtering completed/failed records
    """
    try:
        col = get_spectral_analyses_collection()
        col.create_index(
            [("scene_id", ASCENDING), ("index_type", ASCENDING)],
            unique=True,
            name="spectral_scene_index_unique",
        )
        col.create_index([("created_at", DESCENDING)], name="spectral_created_at_desc")
        col.create_index([("status", ASCENDING)], name="spectral_status_idx")
        logger.info("MongoDB indexes ensured on 'spectral_analyses' collection.")
    except Exception as exc:
        logger.warning(
            "Could not ensure spectral analysis indexes [%s]: %s",
            type(exc).__name__, exc,
        )

# ─── Phase 3 — Change Detection analyses collection helpers ───────────────

def get_change_analyses_collection() -> Collection:
    """Return the 'change_analyses' collection from the configured database."""
    return get_database()["change_analyses"]

def ensure_change_analysis_indexes() -> None:
    """
    Create required indexes on the change_analyses collection.
    Called once during application startup.
    """
    try:
        col = get_change_analyses_collection()
        col.create_index([("analysis_id", ASCENDING)], unique=True, name="change_analysis_id_unique")
        col.create_index([("created_at", DESCENDING)], name="change_created_at_desc")
        logger.info("MongoDB indexes ensured on 'change_analyses' collection.")
    except Exception as exc:
        logger.warning("Could not ensure change analysis indexes [%s]: %s", type(exc).__name__, exc)


# ─── Phase 4 — Infrastructure Detection collection helpers ────────────────────

def get_infrastructure_collection() -> Collection:
    return get_database()["infrastructure_analyses"]

def get_disaster_collection() -> Collection:
    return get_database()["disaster_analyses"]

def get_fusion_collection() -> Collection:
    return get_database()["fusion_analyses"]


def get_satellite_timeline_collection() -> Collection:
    return get_database()["satellite_timeline"]


def get_infrastructure_results_collection() -> Collection:
    """Return the 'infrastructure_results' collection from the configured database."""
    return get_database()["infrastructure_results"]


def ensure_infrastructure_indexes() -> None:
    """
    Create required indexes on the infrastructure_results collection.
    Called once during application startup.
      - Unique index on scene_id (one result per scene, updated on reprocess)
      - Index on result_id for preview lookups
      - Index on created_at for ordering
      - Index on status for filtering
    """
    try:
        col = get_infrastructure_results_collection()
        col.create_index([("scene_id", ASCENDING)], unique=True, name="infra_scene_id_unique")
        col.create_index([("result_id", ASCENDING)], name="infra_result_id_idx")
        col.create_index([(("created_at", DESCENDING))], name="infra_created_at_desc")
        col.create_index([(("status", ASCENDING))], name="infra_status_idx")
        logger.info("MongoDB indexes ensured on 'infrastructure_results' collection.")
    except Exception as exc:
        logger.warning("Could not ensure infrastructure indexes [%s]: %s", type(exc).__name__, exc)


# ─── Phase 5 — Disaster Analysis collection helpers ───────────────────────────

def get_disaster_analyses_collection() -> Collection:
    """Return the 'disaster_analyses' collection from the configured database."""
    return get_database()["disaster_analyses"]


def ensure_disaster_analysis_indexes() -> None:
    """
    Create required indexes on the disaster_analyses collection.
    Called once during application startup.
    """
    try:
        col = get_disaster_analyses_collection()
        col.create_index([("result_id", ASCENDING)], unique=True, name="disaster_result_id_unique")
        col.create_index([("scene_id", ASCENDING)], name="disaster_scene_id_idx")
        col.create_index([("created_at", DESCENDING)], name="disaster_created_at_desc")
        col.create_index([("status", ASCENDING)], name="disaster_status_idx")
        logger.info("MongoDB indexes ensured on 'disaster_analyses' collection.")
    except Exception as exc:
        logger.warning("Could not ensure disaster indexes [%s]: %s", type(exc).__name__, exc)

"""
conftest.py — SatQuery AI Test Configuration

Provides:
  - autouse session fixture that clears all collections before the test
    session starts, so each pytest invocation begins with a clean slate.
  - The fixture runs ONCE per full pytest session (not per test), avoiding
    the overhead of clearing between individual tests while still preventing
    cross-module state leakage.

WHY THIS IS NEEDED:
  Tests create real documents in MongoDB (or mongomock when Atlas is
  unavailable). When all test files run together in the same process,
  documents from one file pollute assertions in another.  This conftest
  clears the relevant collections once before any tests run.

IMPORTANT:
  - Never runs against a production database; the database_name from
    settings is always the test database name defined in .env.
  - Does NOT import secrets or print connection strings.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)


# ─── Collections to clear before each test session ────────────────────────────

_TEST_COLLECTIONS = ["images", "sessions", "analyses", "comparisons", "history"]


def _clear_all_collections() -> None:
    """Drop all documents from test collections. Non-fatal if DB is offline."""
    try:
        from app.db.mongodb import get_database
        db = get_database()
        for col_name in _TEST_COLLECTIONS:
            result = db[col_name].delete_many({})
            logger.debug("Cleared %d docs from '%s'", result.deleted_count, col_name)
        logger.info("Test collections cleared successfully.")
    except Exception as exc:
        # If MongoDB is unavailable the tests will use the in-process fallback;
        # just log and continue — do NOT raise.
        logger.warning("Could not clear test collections [%s]: %s", type(exc).__name__, exc)


# ─── Session-scoped autouse fixture ──────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def clean_db_before_session():
    """Clear all test collections once before the pytest session begins."""
    _clear_all_collections()
    yield
    # Optional: clear again after session to leave DB tidy
    _clear_all_collections()


# ─── Module-scoped cleanup fixture (opt-in) ───────────────────────────────────

@pytest.fixture(scope="module")
def clean_db():
    """
    Optional: explicit per-module cleanup.
    Tests that need a completely isolated DB state can request this fixture:

        def test_something(clean_db):
            ...
    """
    _clear_all_collections()
    yield
    _clear_all_collections()

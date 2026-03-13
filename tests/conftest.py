"""
Pytest configuration and shared fixtures.

External dependencies (PGVector, OpenAI) are stubbed out via sys.modules
BEFORE any application module is imported, preventing connection attempts
during test collection.
"""
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out modules that attempt I/O at import time
# ---------------------------------------------------------------------------
_mock_agent = MagicMock()

sys.modules["app.rag.retriever"] = MagicMock(retriever=MagicMock())
sys.modules["agent.graph"] = MagicMock(agent=_mock_agent)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def app():
    from app.main import app as _app

    return _app


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture()
def mock_agent():
    """Return the shared agent mock, reset before each test."""
    _mock_agent.reset_mock()
    return _mock_agent

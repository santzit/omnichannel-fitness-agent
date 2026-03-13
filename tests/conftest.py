"""
Pytest configuration and shared fixtures.
"""
import os

import pytest

# ---------------------------------------------------------------------------
# Reusable skip marker for tests that require real OpenAI credentials.
# Tests decorated with @needs_openai are skipped when OPENAI_API_KEY is not
# set, so the suite stays green in environments without credentials while
# exercising real OpenAI in CI where the secret is injected.
# ---------------------------------------------------------------------------
needs_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not configured",
)


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

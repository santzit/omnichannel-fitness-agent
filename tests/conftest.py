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


@pytest.fixture(scope="session")
def rag_store():
    """Enable pgvector and populate the vector store once for the whole session.

    Connects directly to Postgres, enables the vector extension, then calls
    ingest() to embed and store all documents from /docs.  Any test that calls
    the real agent without patching the retriever must declare this fixture as
    a dependency so the store is ready before the first retrieval attempt.

    The fixture does NOT skip when credentials are absent — it fails loudly,
    which is the intended behaviour (tests must fail, not silently pass as
    skipped, when OPENAI_API_KEY or POSTGRES_URL is misconfigured).
    """
    import psycopg

    from app.rag.ingest import ingest

    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError(
            "POSTGRES_URL is not set. "
            "Configure it as a repository variable (vars.POSTGRES_URL) so the "
            "test job and the application use the same connection string."
        )
    # psycopg.connect uses a plain postgresql:// DSN — strip the driver tag.
    dsn = postgres_url.replace("postgresql+psycopg://", "postgresql://")

    with psycopg.connect(dsn) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()

    ingest()

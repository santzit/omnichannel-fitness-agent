"""
RAG integration tests — exercises the full pipeline end-to-end:
real embeddings → real PGVector store → real LLM.

The ``rag_store`` session fixture (conftest.py) enables the pgvector extension
and ingests /docs/santz_academy_qa.md once before the suite runs.  Every agent
test then calls the real agent with no retriever patching so the test validates
the entire path:

    user question → PGVector similarity search → LLM answer

Tests FAIL (not skip) when OPENAI_API_KEY or POSTGRES_URL is misconfigured.

Two additional tests require no credentials:

* ``test_rag_qa_document_exists`` — sanity-checks the source document.
* ``test_rag_retriever_default_connection_is_not_none`` /
  ``test_rag_ingest_default_connection_is_not_none`` — regression tests that
  guard against the production bug where connection=None was passed to PGVector.
"""

import os
import pathlib
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Pre-load the QA document for the sanity-check test.
# ---------------------------------------------------------------------------

_QA_DOC_PATH = pathlib.Path(__file__).parent.parent / "docs" / "santz_academy_qa.md"
_QA_CONTENT = _QA_DOC_PATH.read_text(encoding="utf-8")

# Phrases the agent uses when it has NO context and falls back to a human.
# Any of these appearing in an answer means RAG retrieval was not used.
_HUMAN_REDIRECT_PHRASES = [
    "atendente humano",
    "encaminhar",
    "não está disponível nos documentos",
    "não tenho essa informação",
]


def _ask_agent(question: str) -> str:
    """Invoke the real agent end-to-end — no retriever patching."""
    from agent.graph import agent

    result = agent.invoke({"user_message": question})
    return result["response"]


def _assert_no_human_redirect(answer: str) -> None:
    """Fail if the answer contains any human-redirect fallback phrase."""
    lower = answer.lower()
    for phrase in _HUMAN_REDIRECT_PHRASES:
        assert phrase not in lower, (
            f"Agent fell back to human redirect (phrase: {phrase!r}) "
            f"instead of answering from RAG context.\nAnswer: {answer!r}"
        )


# ---------------------------------------------------------------------------
# Agent-answer tests — require OPENAI_API_KEY + running Postgres with pgvector
# The rag_store fixture populates the store before any of these tests run.
# ---------------------------------------------------------------------------


def test_rag_academy_name(rag_store):
    """Agent should reply with the academy name 'SANTZ Academy'."""
    answer = _ask_agent("Qual o nome da academia?")
    _assert_no_human_redirect(answer)
    assert "SANTZ" in answer, f"Expected 'SANTZ' in answer, got: {answer!r}"


def test_rag_academy_address(rag_store):
    """Agent should reply with the street address from the QA document."""
    answer = _ask_agent("Qual o endereço da academia?")
    _assert_no_human_redirect(answer)
    assert "Palmeiras" in answer or "450" in answer, (
        f"Expected address details in answer, got: {answer!r}"
    )


def test_rag_wednesday_hours(rag_store):
    """Agent should reply with the weekday opening hours (Wednesday = Mon-Fri)."""
    answer = _ask_agent("Qual o horário de atendimento na quarta-feira?")
    _assert_no_human_redirect(answer)
    assert "06h" in answer or "23h" in answer, (
        f"Expected weekday hours in answer, got: {answer!r}"
    )


def test_rag_parking(rag_store):
    """Agent should confirm the academy has free parking."""
    answer = _ask_agent("Vocês possuem estacionamento?")
    _assert_no_human_redirect(answer)
    assert "estacionamento" in answer.lower(), (
        f"Expected parking info in answer, got: {answer!r}"
    )


# ---------------------------------------------------------------------------
# Document sanity check — no credentials needed
# ---------------------------------------------------------------------------


def test_rag_qa_document_exists():
    """The QA document must exist and contain the expected key facts."""
    assert _QA_DOC_PATH.exists(), f"QA document not found at {_QA_DOC_PATH}"
    assert "SANTZ" in _QA_CONTENT, "QA document missing academy name"
    assert "Palmeiras" in _QA_CONTENT, "QA document missing street address"
    assert "estacionamento" in _QA_CONTENT.lower(), "QA document missing parking info"
    assert "06h" in _QA_CONTENT, "QA document missing weekday opening hour"


# ---------------------------------------------------------------------------
# Infrastructure regression tests — no credentials needed
# Guard against connection=None being passed to PGVector (production bug).
# ---------------------------------------------------------------------------


def test_rag_retriever_default_connection_is_not_none():
    """Regression: retriever must NOT pass connection=None to PGVector.

    When POSTGRES_URL is absent from the environment, retriever.py must fall
    back to a hard-coded default rather than None — otherwise PGVector raises:
        'connection should be a connection string or an instance of
         sqlalchemy.engine.Engine or sqlalchemy.ext.asyncio.engine.AsyncEngine'

    The default must also use the psycopg3 driver scheme
    (postgresql+psycopg://) that langchain_postgres requires.
    """
    from app.rag.retriever import _LazyRetriever

    captured = {}

    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = MagicMock(invoke=MagicMock(return_value=[]))

    def spy_pgvector(**kwargs):
        captured.update(kwargs)
        return mock_vs

    # Remove POSTGRES_URL so the fallback default is the only option.
    # Patch at app.rag._embeddings so both the Azure and the standard-OpenAI
    # code paths are intercepted regardless of which one the environment selects.
    env_without_postgres = {k: v for k, v in os.environ.items() if k != "POSTGRES_URL"}
    with patch.dict(os.environ, env_without_postgres, clear=True):
        with patch("app.rag._embeddings.OpenAIEmbeddings", return_value=MagicMock()):
            with patch("app.rag._embeddings.AzureOpenAIEmbeddings", return_value=MagicMock()):
                with patch("langchain_postgres.PGVector", side_effect=spy_pgvector):
                    _LazyRetriever().invoke("test")

    connection = captured.get("connection")
    assert connection is not None, (
        "retriever.py passed connection=None to PGVector — POSTGRES_URL default is missing"
    )
    assert connection.startswith("postgresql+psycopg://"), (
        f"Default must use the psycopg3 driver scheme (postgresql+psycopg://), got: {connection!r}"
    )


def test_rag_ingest_default_connection_is_not_none():
    """Regression: ingest must NOT pass connection=None to PGVector.from_documents.

    Mirrors test_rag_retriever_default_connection_is_not_none for the ingest
    path: both must agree on a valid default so that containers without an
    explicit POSTGRES_URL in .env still work.
    """
    from app.rag.ingest import ingest

    captured = {}

    mock_loader = MagicMock()
    mock_loader.load.return_value = []

    def spy_from_documents(**kwargs):
        captured.update(kwargs)

    # Remove POSTGRES_URL so the fallback default is the only option.
    env_without_postgres = {k: v for k, v in os.environ.items() if k != "POSTGRES_URL"}
    with patch.dict(os.environ, env_without_postgres, clear=True):
        with patch(
            "langchain_community.document_loaders.DirectoryLoader",
            return_value=mock_loader,
        ):
            with patch("app.rag._embeddings.OpenAIEmbeddings", return_value=MagicMock()):
                with patch("app.rag._embeddings.AzureOpenAIEmbeddings", return_value=MagicMock()):
                    with patch(
                        "langchain_postgres.PGVector.from_documents",
                        side_effect=spy_from_documents,
                    ):
                        ingest()

    connection = captured.get("connection")
    assert connection is not None, (
        "ingest.py passed connection=None to PGVector — POSTGRES_URL default is missing"
    )
    assert connection.startswith("postgresql+psycopg://"), (
        f"Default must use the psycopg3 driver scheme (postgresql+psycopg://), got: {connection!r}"
    )


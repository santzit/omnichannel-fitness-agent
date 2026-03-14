"""
RAG integration tests — exercises the full pipeline end-to-end:
real embeddings → real PGVector store → real LLM.

The ``rag_store`` session fixture (conftest.py) enables the pgvector extension
and ingests /docs/santz_academy_qa.md once before the suite runs.  Every agent
test then calls the real agent with no retriever patching so the test validates
the entire path:

    user question → PGVector similarity search → LLM answer

Tests FAIL (not skip) when OPENAI_API_KEY or POSTGRES_URL is misconfigured.

No mocks are used so that a misconfigured embedding model (e.g. EMBEDDING_MODEL
set to a chat model instead of an embedding model) causes these tests to fail
exactly as it would on VPS.
"""

import pathlib

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


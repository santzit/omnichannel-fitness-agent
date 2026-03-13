"""
RAG integration tests — verifies that the agent uses context retrieved from the
vector store to answer factual questions about SANTZ Academy.

The retriever is patched to return the full QA document so the test is not
coupled to a live PGVector instance.  The LLM call is real, so these tests are
skipped when OPENAI_API_KEY is not configured (they run in CI where the secret
is injected).
"""

import pathlib
from unittest.mock import MagicMock, patch

from tests.conftest import needs_openai

# ---------------------------------------------------------------------------
# Pre-load the QA document once so every test gets the same context.
# ---------------------------------------------------------------------------

_QA_DOC_PATH = pathlib.Path(__file__).parent.parent / "docs" / "santz_academy_qa.md"
_QA_CONTENT = _QA_DOC_PATH.read_text(encoding="utf-8")


def _make_doc_mock(content: str):
    """Return a mock LangChain Document whose .page_content is *content*."""
    doc = MagicMock()
    doc.page_content = content
    return doc


def _ask_agent_with_rag(question: str) -> str:
    """Invoke the agent with the QA document injected as retrieval context."""
    from agent.graph import agent

    mock_doc = _make_doc_mock(_QA_CONTENT)

    with patch("agent.graph.retriever") as mock_retriever:
        mock_retriever.invoke.return_value = [mock_doc]
        result = agent.invoke({"user_message": question})

    return result["response"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@needs_openai
def test_rag_academy_name():
    """Agent should reply with the academy name 'SANTZ Academy'."""
    answer = _ask_agent_with_rag("Qual o nome da academia?")
    assert "SANTZ" in answer, f"Expected 'SANTZ' in answer, got: {answer!r}"


@needs_openai
def test_rag_academy_address():
    """Agent should reply with the street address from the QA document."""
    answer = _ask_agent_with_rag("Qual o endereço da academia?")
    assert "Palmeiras" in answer or "450" in answer, (
        f"Expected address details in answer, got: {answer!r}"
    )


@needs_openai
def test_rag_wednesday_hours():
    """Agent should reply with the weekday opening hours (Wednesday = Mon-Fri)."""
    answer = _ask_agent_with_rag("Qual o horário de atendimento na quarta-feira?")
    # Weekday hours are 06h às 23h; check for the opening or closing hour including the 'h' suffix.
    assert "06h" in answer or "23h" in answer, f"Expected weekday hours in answer, got: {answer!r}"


@needs_openai
def test_rag_parking():
    """Agent should confirm the academy has free parking."""
    answer = _ask_agent_with_rag("Vocês possuem estacionamento?")
    assert "estacionamento" in answer.lower(), f"Expected parking info in answer, got: {answer!r}"

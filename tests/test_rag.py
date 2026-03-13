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

# Phrases the agent uses when it has NO context and falls back to a human.
# Any of these appearing in an answer means RAG was not used.
_HUMAN_REDIRECT_PHRASES = [
    "atendente humano",
    "encaminhar",
    "não está disponível nos documentos",
    "não tenho essa informação",
]


def _make_doc_mock(content: str):
    """Return a mock LangChain Document whose .page_content is *content*."""
    doc = MagicMock()
    doc.page_content = content
    return doc


def _ask_agent_with_rag(question: str) -> str:
    """Invoke the agent with the QA document injected as retrieval context.

    Also asserts that the retriever was actually called with the question,
    so a broken pipeline (retriever skipped) fails here rather than in the
    individual test assertions.
    """
    from agent.graph import agent

    mock_doc = _make_doc_mock(_QA_CONTENT)

    with patch("agent.graph.retriever") as mock_retriever:
        mock_retriever.invoke.return_value = [mock_doc]
        result = agent.invoke({"user_message": question})

    # Verify the retrieval step was reached — catches wiring bugs.
    mock_retriever.invoke.assert_called_once_with(question)

    return result["response"]


def _assert_no_human_redirect(answer: str) -> None:
    """Fail if the answer contains any human-redirect fallback phrase.

    This is the key guard against the original bug where the agent said
    'Vou encaminhar sua dúvida para um atendente humano' because context
    was empty (retriever not wired up / vector store not populated).
    """
    lower = answer.lower()
    for phrase in _HUMAN_REDIRECT_PHRASES:
        assert phrase not in lower, (
            f"Agent fell back to human redirect (phrase: {phrase!r}) "
            f"instead of answering from RAG context.\nAnswer: {answer!r}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@needs_openai
def test_rag_academy_name():
    """Agent should reply with the academy name 'SANTZ Academy'."""
    answer = _ask_agent_with_rag("Qual o nome da academia?")
    _assert_no_human_redirect(answer)
    assert "SANTZ" in answer, f"Expected 'SANTZ' in answer, got: {answer!r}"


@needs_openai
def test_rag_academy_address():
    """Agent should reply with the street address from the QA document."""
    answer = _ask_agent_with_rag("Qual o endereço da academia?")
    _assert_no_human_redirect(answer)
    assert "Palmeiras" in answer or "450" in answer, (
        f"Expected address details in answer, got: {answer!r}"
    )


@needs_openai
def test_rag_wednesday_hours():
    """Agent should reply with the weekday opening hours (Wednesday = Mon-Fri)."""
    answer = _ask_agent_with_rag("Qual o horário de atendimento na quarta-feira?")
    _assert_no_human_redirect(answer)
    # Weekday hours are 06h às 23h; check for the opening or closing hour.
    assert "06h" in answer or "23h" in answer, f"Expected weekday hours in answer, got: {answer!r}"


@needs_openai
def test_rag_parking():
    """Agent should confirm the academy has free parking."""
    answer = _ask_agent_with_rag("Vocês possuem estacionamento?")
    _assert_no_human_redirect(answer)
    assert "estacionamento" in answer.lower(), f"Expected parking info in answer, got: {answer!r}"


@needs_openai
def test_rag_empty_context_falls_back_gracefully():
    """When retrieval returns nothing the agent should respond politely.

    This is the expected behaviour when the vector store is empty or the
    retriever fails — the agent must NOT crash and must NOT invent answers.
    It should tell the user it will redirect them to a human attendant.
    """
    from agent.graph import agent

    with patch("agent.graph.retriever") as mock_retriever:
        mock_retriever.invoke.return_value = []
        result = agent.invoke({"user_message": "Quais são os planos disponíveis?"})

    answer = result["response"]
    assert answer and answer.strip(), "Agent returned an empty response"

    lower = answer.lower()
    has_redirect = any(phrase in lower for phrase in _HUMAN_REDIRECT_PHRASES)
    has_caveat = "não" in lower or "desculpe" in lower
    assert has_redirect or has_caveat, (
        f"Expected agent to acknowledge missing context, got: {answer!r}"
    )


def test_rag_qa_document_exists():
    """The QA document must exist and contain the expected key facts."""
    assert _QA_DOC_PATH.exists(), f"QA document not found at {_QA_DOC_PATH}"
    assert "SANTZ" in _QA_CONTENT, "QA document missing academy name"
    assert "Palmeiras" in _QA_CONTENT, "QA document missing street address"
    assert "estacionamento" in _QA_CONTENT.lower(), "QA document missing parking info"
    assert "06h" in _QA_CONTENT, "QA document missing weekday opening hour"

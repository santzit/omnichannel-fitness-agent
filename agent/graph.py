import logging
import os

_log = logging.getLogger(__name__)

from langgraph.graph import StateGraph

from .prompts import SYSTEM_PROMPT
from .state import AgentState
from app.rag.retriever import retriever  # _LazyRetriever — safe at import time


class _LazyAgent:
    """Defers OpenAI client creation and graph compilation until first invoke().

    This allows the module to be imported without OPENAI_API_KEY being set
    (e.g. for tests that patch agent.invoke or don't exercise the LLM path).
    The real graph is built on the first invoke() call.
    """

    def __init__(self) -> None:
        self._compiled = None

    def _build(self):
        from openai import OpenAI

        _openai_kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
        _endpoint = os.getenv("OPENAI_ENDPOINT")
        if _endpoint:
            _openai_kwargs["base_url"] = _endpoint

        client = OpenAI(**_openai_kwargs)
        _LLM_MODEL = os.getenv("LLM_MODEL")
        if not _LLM_MODEL:
            raise ValueError(
                "LLM_MODEL is not set. "
                "Configure it as a repository variable (vars.LLM_MODEL) "
                "with the model or deployment name to use (e.g. gpt-4.1)."
            )

        def retrieve_docs(state):
            try:
                docs = retriever.invoke(state["user_message"])
                return {"context": [doc.page_content for doc in docs]}
            except Exception as exc:
                _log.warning("RAG retrieval failed, answering without context: %s", exc)
                return {"context": []}

        def generate_answer(state):
            context = "\n\n".join(state["context"])
            prompt = SYSTEM_PROMPT.format(context=context)
            response = client.chat.completions.create(
                model=_LLM_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": state["user_message"]},
                ],
            )
            return {"response": response.choices[0].message.content}

        graph = StateGraph(AgentState)
        graph.add_node("retrieve", retrieve_docs)
        graph.add_node("generate", generate_answer)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        return graph.compile()

    def invoke(self, state, *args, **kwargs):
        if self._compiled is None:
            self._compiled = self._build()
        return self._compiled.invoke(state, *args, **kwargs)


agent = _LazyAgent()

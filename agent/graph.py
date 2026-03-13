import logging
import os

_log = logging.getLogger(__name__)

if os.getenv("MOCK_AGENT") == "true":
    class _MockCompiledGraph:
        def invoke(self, state, *args, **kwargs):
            return {"response": "Mock fitness agent response."}

    agent = _MockCompiledGraph()
else:
    from langgraph.graph import StateGraph
    from .state import AgentState
    from .prompts import SYSTEM_PROMPT
    from app.rag.retriever import retriever
    from openai import OpenAI, OpenAIError

    _openai_kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
    _endpoint = os.getenv("OPENAI_ENDPOINT")
    if _endpoint:
        _openai_kwargs["base_url"] = _endpoint

    client = OpenAI(**_openai_kwargs)

    _LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

    def retrieve_docs(state):
        try:
            docs = retriever.invoke(state["user_message"])
            return {"context": [doc.page_content for doc in docs]}
        except (OpenAIError, Exception) as exc:
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

    agent = graph.compile()
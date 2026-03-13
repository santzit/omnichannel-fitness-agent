import logging
import os

_log = logging.getLogger(__name__)


class _LazyRetriever:
    """Defers PGVector connection until the first invoke() call.

    This prevents import-time failures when POSTGRES_URL is not configured
    (e.g. during unit/integration tests without a live database).  Retrieval
    errors are caught by the agent graph so the LLM still answers from its own
    knowledge when PGVector is unavailable.
    """

    def __init__(self) -> None:
        self._retriever = None

    def invoke(self, text: str):
        if self._retriever is None:
            from langchain_openai import OpenAIEmbeddings
            from langchain_postgres import PGVector

            connection = os.getenv("POSTGRES_URL")

            embeddings_kwargs = {}
            endpoint = os.getenv("OPENAI_ENDPOINT")
            if endpoint:
                embeddings_kwargs["openai_api_base"] = endpoint

            embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
            embeddings = OpenAIEmbeddings(model=embedding_model, **embeddings_kwargs)

            vector_store = PGVector(
                connection=connection,
                collection_name="fitness_docs",
                embeddings=embeddings,
            )
            self._retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        return self._retriever.invoke(text)


retriever = _LazyRetriever()
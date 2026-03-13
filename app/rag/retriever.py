import os

if os.getenv("MOCK_AGENT") == "true":
    class _MockRetriever:
        def invoke(self, text):
            return []

    retriever = _MockRetriever()
else:
    from langchain_postgres import PGVector
    from langchain_openai import OpenAIEmbeddings

    connection = os.getenv("POSTGRES_URL")

    _embeddings_kwargs = {}
    endpoint = os.getenv("OPENAI_ENDPOINT")
    if endpoint:
        _embeddings_kwargs["openai_api_base"] = endpoint

    embeddings = OpenAIEmbeddings(**_embeddings_kwargs)

    vector_store = PGVector(
        connection=connection,
        collection_name="fitness_docs",
        embeddings=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )
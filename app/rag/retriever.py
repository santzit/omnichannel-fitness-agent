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

    embeddings = OpenAIEmbeddings()

    vector_store = PGVector(
        connection=connection,
        collection_name="fitness_docs",
        embeddings=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )
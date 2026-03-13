import logging
import os

_log = logging.getLogger(__name__)


def ingest() -> None:
    """Load docs from ./docs, split them, and upsert into the PGVector store."""
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_postgres import PGVector
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = DirectoryLoader("./docs")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings_kwargs = {}
    endpoint = os.getenv("OPENAI_ENDPOINT")
    if endpoint:
        embeddings_kwargs["openai_api_base"] = endpoint

    embeddings = OpenAIEmbeddings(**embeddings_kwargs)

    connection = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@postgres:5432/fitness")

    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection=connection,
        collection_name="fitness_docs",
    )
    _log.info("Ingested %d chunks into fitness_docs collection.", len(chunks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()

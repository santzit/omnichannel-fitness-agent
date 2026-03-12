from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
import os

loader = DirectoryLoader("./docs")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

connection = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@postgres:5432/fitness")

vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection=connection,
    collection_name="fitness_docs",
)
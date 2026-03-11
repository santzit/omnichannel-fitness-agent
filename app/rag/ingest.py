from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from embeddings import embeddings

loader = DirectoryLoader("./docs")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

vector_store = PGVector.from_documents(
    chunks,
    embeddings,
    connection_string="postgresql://postgres:postgres@db:5432/fitness"
)
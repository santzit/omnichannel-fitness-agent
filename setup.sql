CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536),
    metadata JSONB
);

CREATE INDEX idx_documents_embedding
ON documents
USING ivfflat (embedding vector_cosine_ops);
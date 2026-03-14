#!/bin/bash
set -e

echo "==> Ingesting RAG documents into PGVector..."
python -m app.rag.ingest
echo "==> Ingest completed successfully."

echo "==> Starting API server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000

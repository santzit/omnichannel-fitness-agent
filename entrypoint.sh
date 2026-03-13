#!/bin/bash

echo "==> Ingesting RAG documents into PGVector..."
if python -m app.rag.ingest; then
    echo "==> Ingest completed successfully."
else
    echo "==> WARNING: Ingest failed. The server will start without pre-loaded RAG context." >&2
fi

echo "==> Starting API server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000

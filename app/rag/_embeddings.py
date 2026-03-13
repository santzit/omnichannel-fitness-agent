"""Shared factory for LangChain embeddings clients.

Uses :class:`~langchain_openai.OpenAIEmbeddings`, which is compatible with
both standard OpenAI and OpenAI-compatible Azure AI Services endpoints
(e.g. ``https://<resource>.cognitiveservices.azure.com/openai/v1/``).

This is the same pattern used by ``agent/graph.py`` for the LLM: set
``OPENAI_ENDPOINT`` as the base URL and ``OPENAI_API_KEY`` as the API key —
no Azure-specific client is needed for OpenAI-compatible endpoints.

Environment variables
---------------------
OPENAI_ENDPOINT
    Base URL for the embeddings API.  For Azure AI Services use the full
    endpoint URL including the ``/openai/v1/`` path, e.g.
    ``https://<resource>.cognitiveservices.azure.com/openai/v1/``.
    When absent, the standard OpenAI endpoint is used.
EMBEDDING_MODEL
    Model / deployment name for embeddings.  Must be set explicitly — no
    default is provided so that misconfiguration fails loudly rather than
    silently using a model that may not be deployed on the resource.
"""

import os

from langchain_openai import OpenAIEmbeddings


def build_embeddings():
    """Return the configured :class:`~langchain_openai.OpenAIEmbeddings` client.

    Raises ``ValueError`` if ``EMBEDDING_MODEL`` is not set.
    """
    embedding_model = os.getenv("EMBEDDING_MODEL")
    if not embedding_model:
        raise ValueError(
            "EMBEDDING_MODEL is not set. "
            "Configure it as a repository variable (vars.EMBEDDING_MODEL) "
            "with the exact model or deployment name to use for embeddings "
            "(e.g. text-embedding-ada-002)."
        )

    kwargs = {}
    endpoint = os.getenv("OPENAI_ENDPOINT")
    if endpoint:
        kwargs["openai_api_base"] = endpoint

    return OpenAIEmbeddings(model=embedding_model, **kwargs)


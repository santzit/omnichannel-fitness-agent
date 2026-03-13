"""Shared factory for LangChain embeddings clients.

Auto-detects Azure OpenAI vs. standard OpenAI based on the OPENAI_ENDPOINT
environment variable so that both ingest.py and retriever.py use the correct
client without duplicating the detection logic.

Azure vs. standard OpenAI differences that cause the production 400 error when
the wrong client is used:

* **URL format** — Azure requires the deployment name in the path and an
  ``api-version`` query parameter; the standard client just appends ``/v1/``.
* **Auth header** — Azure uses ``api-key: <key>``; OpenAI uses
  ``Authorization: Bearer <key>``.

Environment variables
---------------------
OPENAI_ENDPOINT
    Azure Cognitive Services endpoint (e.g.
    ``https://<resource>.cognitiveservices.azure.com``) *or* any
    OpenAI-compatible base URL for non-Azure deployments.
EMBEDDING_MODEL
    Model name (standard OpenAI) **or** deployment name (Azure).
    Defaults to ``text-embedding-ada-002``.
OPENAI_API_VERSION
    Azure API version string (e.g. ``2024-02-01``).
    Ignored for standard OpenAI. Defaults to ``2024-02-01``.
OPENAI_API_TYPE
    Set to ``azure`` to force Azure client even when OPENAI_ENDPOINT does not
    contain an Azure domain.
"""

import os
from urllib.parse import urlparse

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings


def build_embeddings():
    """Return the configured LangChain embeddings client.

    Uses :class:`~langchain_openai.AzureOpenAIEmbeddings` when the endpoint
    looks like an Azure Cognitive Services URL (or ``OPENAI_API_TYPE=azure``),
    and :class:`~langchain_openai.OpenAIEmbeddings` otherwise.

    Azure notes
    -----------
    ``OPENAI_ENDPOINT`` may include a path such as ``/openai/v1/`` (common in
    Azure AI Services resources).  ``AzureOpenAIEmbeddings`` constructs its own
    deployment URL from the base resource endpoint and must receive only the
    scheme + host — any extra path component would result in a doubled path and
    a 404.  We strip the path with ``urllib.parse.urlparse`` before passing the
    value to ``azure_endpoint``.
    """
    endpoint = os.getenv("OPENAI_ENDPOINT", "")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

    is_azure = (
        "azure.com" in endpoint
        or "cognitiveservices" in endpoint
        or os.getenv("OPENAI_API_TYPE", "").lower() == "azure"
    )

    if is_azure and endpoint:
        parsed = urlparse(endpoint)
        azure_base = f"{parsed.scheme}://{parsed.netloc}"
        return AzureOpenAIEmbeddings(
            azure_endpoint=azure_base,
            azure_deployment=embedding_model,
            openai_api_version=os.getenv("OPENAI_API_VERSION", "2024-02-01"),
        )

    if is_azure and not endpoint:
        raise ValueError(
            "OPENAI_API_TYPE is set to 'azure' but OPENAI_ENDPOINT is not configured. "
            "Set OPENAI_ENDPOINT to your Azure Cognitive Services resource URL."
        )

    kwargs = {}
    if endpoint:
        kwargs["openai_api_base"] = endpoint
    return OpenAIEmbeddings(model=embedding_model, **kwargs)

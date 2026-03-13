"""
Integration test — calls the real Azure / OpenAI endpoint.

Runs only when OPENAI_API_KEY is set in the environment, so the test is
automatically skipped during local development without credentials while
running in CI when the repository secret is injected.
"""
import os

import pytest
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not configured")
def test_agent_hello_world_real_llm():
    """Send 'hello world' to the configured LLM and assert a non-empty reply."""
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_ENDPOINT:
        kwargs["base_url"] = OPENAI_ENDPOINT

    client = OpenAI(**kwargs)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful fitness assistant."},
            {"role": "user", "content": "hello world"},
        ],
    )

    reply = response.choices[0].message.content
    assert reply is not None, "LLM returned None response"
    assert reply.strip(), "LLM returned a blank response"

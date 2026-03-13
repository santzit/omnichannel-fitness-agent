from unittest.mock import patch

from tests.conftest import needs_openai


@needs_openai
def test_chat_completions_returns_agent_response(client):
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a fitness assistant."},
            {"role": "user", "content": "Give me a workout tip."},
        ],
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    content = body["choices"][0]["message"]["content"]
    assert content and content.strip(), "Agent returned an empty reply"


def test_chat_completions_uses_last_user_message(client):
    """The endpoint must extract the last user message and pass it to the agent.
    The agent is patched locally to capture what it receives — this tests the
    endpoint's routing logic, not the LLM."""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "I want to lose weight."},
            {"role": "assistant", "content": "I can help with that."},
            {"role": "user", "content": "Suggest a diet plan."},
        ],
    }

    with patch("app.api.openai_compat.agent") as mock:
        mock.invoke.return_value = {"response": "Here is a diet plan."}
        client.post("/v1/chat/completions", json=payload)

    mock.invoke.assert_called_once_with({"user_message": "Suggest a diet plan."})


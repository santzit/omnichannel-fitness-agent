def test_chat_completions_returns_agent_response(client, mock_agent):
    mock_agent.invoke.return_value = {"response": "Focus on compound movements."}

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
    assert body["choices"][0]["message"]["content"] == "Focus on compound movements."
    mock_agent.invoke.assert_called_once_with({"user_message": "Give me a workout tip."})


def test_chat_completions_uses_last_user_message(client, mock_agent):
    mock_agent.invoke.return_value = {"response": "Great choice!"}

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "I want to lose weight."},
            {"role": "assistant", "content": "I can help with that."},
            {"role": "user", "content": "Suggest a diet plan."},
        ],
    }

    client.post("/v1/chat/completions", json=payload)

    mock_agent.invoke.assert_called_once_with({"user_message": "Suggest a diet plan."})

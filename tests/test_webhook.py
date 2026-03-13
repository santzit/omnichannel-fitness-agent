from unittest.mock import AsyncMock, patch

from tests.conftest import needs_openai

_WHATSAPP_PATH = "app.api.meta_webhook.send_whatsapp_message"

_SAMPLE_PAYLOAD = {
    "entry": [
        {
            "changes": [
                {
                    "value": {
                        "messages": [
                            {
                                "from": "5511999999999",
                                "text": {"body": "What workout should I do?"},
                            }
                        ]
                    }
                }
            ]
        }
    ]
}


# ---------------------------------------------------------------------------
# Webhook verification (GET)
# ---------------------------------------------------------------------------


def test_verify_webhook_returns_challenge(client):
    response = client.get(
        "/webhooks/meta",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "fitness_agent_token",
            "hub.challenge": "1234567890",
        },
    )
    assert response.status_code == 200
    assert response.text == "1234567890"


def test_verify_webhook_wrong_token_returns_forbidden(client):
    response = client.get(
        "/webhooks/meta",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_token",
            "hub.challenge": "1234567890",
        },
    )
    assert response.status_code == 403
    assert response.json() == {"status": "forbidden"}


def test_verify_webhook_wrong_mode_returns_forbidden(client):
    response = client.get(
        "/webhooks/meta",
        params={
            "hub.mode": "unsubscribe",
            "hub.verify_token": "fitness_agent_token",
            "hub.challenge": "1234567890",
        },
    )
    assert response.status_code == 403
    assert response.json() == {"status": "forbidden"}


# ---------------------------------------------------------------------------
# Incoming WhatsApp message (POST)
# ---------------------------------------------------------------------------


@needs_openai
def test_post_message_invokes_agent_and_sends_reply(client):
    with patch(_WHATSAPP_PATH, new_callable=AsyncMock) as mock_send:
        response = client.post("/webhooks/meta", json=_SAMPLE_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_send.assert_awaited_once()
    _to, reply = mock_send.call_args.args
    assert _to == "5511999999999"
    assert reply and reply.strip(), "Agent returned an empty reply"

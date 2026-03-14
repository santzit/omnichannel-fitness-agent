from xml.etree import ElementTree

from tests.conftest import needs_openai


# ---------------------------------------------------------------------------
# End-to-end: simulate the Twilio POST; the real agent/OpenAI handles the rest
# ---------------------------------------------------------------------------


@needs_openai
def test_twilio_hello_world_invokes_agent_and_returns_twiml(client):
    """POST 'hello world' to /webhooks/twilio — the real agent is invoked and
    the reply is wrapped in a valid TwiML <Response><Message> envelope."""
    response = client.post(
        "/webhooks/twilio",
        data={"From": "+15550001234", "Body": "hello world"},
    )

    assert response.status_code == 200
    assert "application/xml" in response.headers["content-type"]

    root = ElementTree.fromstring(response.text)
    assert root.tag == "Response"
    message_elem = root.find("Message")
    assert message_elem is not None
    assert message_elem.text and message_elem.text.strip(), "Agent returned an empty reply"


@needs_openai
def test_twilio_webhook_reads_body_field(client):
    """The endpoint reads the Body field correctly regardless of the From number."""
    response = client.post(
        "/webhooks/twilio",
        data={"From": "+5511988887777", "Body": "What is today's workout?"},
    )

    assert response.status_code == 200
    root = ElementTree.fromstring(response.text)
    message_elem = root.find("Message")
    assert message_elem is not None
    assert message_elem.text and message_elem.text.strip(), "Agent returned an empty reply"


def test_twiml_reply_escapes_xml_special_chars():
    """_twiml_reply must escape XML special characters so the TwiML document
    remains valid and parseable regardless of what the LLM returns."""
    from app.api.twilio_webhook import _twiml_reply

    response = _twiml_reply("Use weights < 10kg & stay safe > always")
    root = ElementTree.fromstring(response.body.decode())
    assert root.find("Message").text == "Use weights < 10kg & stay safe > always"


from xml.etree import ElementTree


def test_twilio_hello_world_invokes_agent_and_returns_twiml(client, mock_agent):
    """POST 'hello world' to /webhooks/twilio — agent is called and the reply
    is wrapped in a valid TwiML <Response><Message> envelope."""
    mock_agent.invoke.return_value = {"response": "Hi! How can I help with your fitness goals?"}

    response = client.post(
        "/webhooks/twilio",
        data={"From": "+15550001234", "Body": "hello world"},
    )

    assert response.status_code == 200
    assert "application/xml" in response.headers["content-type"]

    # Verify agent was called with the correct message
    mock_agent.invoke.assert_called_once_with({"user_message": "hello world"})

    # Verify the TwiML envelope is valid and contains the agent reply
    root = ElementTree.fromstring(response.text)
    assert root.tag == "Response"
    message_elem = root.find("Message")
    assert message_elem is not None
    assert message_elem.text == "Hi! How can I help with your fitness goals?"


def test_twilio_webhook_reads_body_field(client, mock_agent):
    """The endpoint reads the Body field correctly regardless of the From number."""
    mock_agent.invoke.return_value = {"response": "Keep pushing!"}

    response = client.post(
        "/webhooks/twilio",
        data={"From": "+5511988887777", "Body": "What is today's workout?"},
    )

    assert response.status_code == 200
    mock_agent.invoke.assert_called_once_with({"user_message": "What is today's workout?"})


def test_twilio_webhook_escapes_xml_special_chars(client, mock_agent):
    """Agent replies containing XML special characters must be escaped so the
    TwiML document remains valid and parseable."""
    mock_agent.invoke.return_value = {"response": "Use weights < 10kg & stay safe > always"}

    response = client.post(
        "/webhooks/twilio",
        data={"From": "+15550001234", "Body": "Any tips?"},
    )

    assert response.status_code == 200
    root = ElementTree.fromstring(response.text)
    assert root.find("Message").text == "Use weights < 10kg & stay safe > always"

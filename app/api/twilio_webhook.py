from fastapi import APIRouter, Form
from fastapi.responses import Response
from agent.graph import agent
import logging
from xml.sax.saxutils import escape

router = APIRouter()
logger = logging.getLogger(__name__)


def _twiml_reply(message: str) -> Response:
    """Wrap *message* in a minimal TwiML response."""
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Message>{escape(message)}</Message>"
        "</Response>"
    )
    return Response(content=xml, media_type="application/xml")


@router.post("/webhooks/twilio")
async def twilio_webhook(
    From: str = Form(...),
    Body: str = Form(...),
):
    logger.info("Received Twilio message from %s", From)
    result = agent.invoke({"user_message": Body})
    reply = result["response"]
    return _twiml_reply(reply)

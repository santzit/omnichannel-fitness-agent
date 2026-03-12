from fastapi import APIRouter, Request, Query
from fastapi.responses import PlainTextResponse
from agent.graph import agent
from services.whatsapp import send_whatsapp_message
import os

router = APIRouter()

VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "fitness_agent_token")


@router.get("/webhooks/meta")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(content=hub_challenge)
    return {"status": "forbidden"}


@router.post("/webhooks/meta")
async def meta_webhook(request: Request):

    data = await request.json()

    message = data["entry"][0]["changes"][0]["value"]["messages"][0]

    user_text = message["text"]["body"]

    phone = message["from"]

    result = agent.invoke({
        "user_message": user_text
    })

    reply = result["response"]

    await send_whatsapp_message(phone, reply)

    return {"status": "ok"}
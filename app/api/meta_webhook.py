from fastapi import APIRouter, Request
from agent.graph import agent
from services.whatsapp import send_whatsapp_message

router = APIRouter()

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
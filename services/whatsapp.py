import httpx
import os

TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_ID = os.getenv("PHONE_NUMBER_ID")

async def send_whatsapp_message(phone, text):

    url = f"https://graph.facebook.com/v19.0/{PHONE_ID}/messages"

    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "text",
        "text": {"body": text}
    }

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, headers=headers)
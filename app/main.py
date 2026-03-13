from fastapi import FastAPI
from app.api.meta_webhook import router as webhook_router
from app.api.openai_compat import router as openai_router
from app.api.twilio_webhook import router as twilio_router

app = FastAPI(title="Omnichannel Fitness Agent")


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(webhook_router)
app.include_router(openai_router)
app.include_router(twilio_router)

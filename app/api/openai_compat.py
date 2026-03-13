from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from agent.graph import agent

router = APIRouter(prefix="/v1")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False


@router.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    user_message = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), ""
    )

    result = agent.invoke({"user_message": user_message})

    return {
        "id": "chat-fitness-agent",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["response"]},
                "finish_reason": "stop",
            }
        ],
    }

from fastapi import APIRouter
from schemas.chat_request import ChatRequest
from services.chatbot_logic import chatbot_logic

router = APIRouter()

@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    return {"response": chatbot_logic(request)}

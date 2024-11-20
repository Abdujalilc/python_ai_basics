from fastapi import APIRouter
from models import ChatRequest
from chatbot_response import chatbot_logic

router = APIRouter()

@router.post("/chat")
def chat(request: ChatRequest):
    response = chatbot_logic(request.question)
    return {"response": response}

from fastapi import APIRouter
from services.add_knowledge import add_to_knowledge

router = APIRouter()

@router.post("/add_knowledge")
def add_knowledge(content: str):
    return {"message": add_to_knowledge(content)}

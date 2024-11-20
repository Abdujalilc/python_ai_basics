from fastapi import APIRouter
from models import KnowledgeRequest
from add_knowledge import add_to_knowledge

router = APIRouter()

@router.post("/add_knowledge")
def add_knowledge(request: KnowledgeRequest):
    status = add_to_knowledge(request.content)
    return {"status": status}

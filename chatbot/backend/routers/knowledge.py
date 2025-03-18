from fastapi import APIRouter
from schemas.knowledge_request import KnowledgeRequest
from services.add_knowledge import add_to_knowledge

router = APIRouter()

@router.post("/add_knowledge")
def add_knowledge(newKnowledge: KnowledgeRequest):
    return {"message": add_to_knowledge(newKnowledge)}

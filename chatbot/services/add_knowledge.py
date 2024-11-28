from schemas.knowledge_request import KnowledgeRequest
from database.database_utils import insert_knowledge
from custom_faiss.faiss_manager_singleton import get_faiss_manager

def add_to_knowledge(req: KnowledgeRequest) -> str:
    insert_knowledge(req.content)
    get_faiss_manager(req.embedder_model,req.distance_metric).refresh_index()
    return "Knowledge added successfully!"

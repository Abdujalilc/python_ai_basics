from database.database_utils import insert_knowledge
from custom_faiss.faiss_manager_singleton import faiss_manager

def add_to_knowledge(content: str) -> str:
    insert_knowledge(content)
    faiss_manager.refresh_index()
    return "Knowledge added successfully!"

from database.database_utils import insert_knowledge
from custom_faiss.faiss_manager_singleton import faiss_manager
from models.faiss_embedder import embedder

def add_to_knowledge(content: str) -> str:
    """Add knowledge and refresh FAISS index."""
    insert_knowledge(content)       # Add to database
    faiss_manager.refresh_index()   # Refresh FAISS index using the manager
    return "Knowledge added successfully!"

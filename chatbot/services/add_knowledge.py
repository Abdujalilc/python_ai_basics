# add_knowledge.py
from database.database_utils import insert_knowledge
from custom_faiss.faiss_utils import refresh_faiss_index

def add_to_knowledge(content: str) -> str:
    """Add knowledge and refresh FAISS index."""
    insert_knowledge(content)       # Add to database
    refresh_faiss_index()           # Update FAISS index
    return "Knowledge added successfully!"

from faiss.faiss_index import prepare_faiss
from database.database_utils import fetch_all_knowledge

# Global FAISS index and knowledge base
index = None
knowledge_base = []

def refresh_faiss_index():
    """Refresh the FAISS index with current knowledge."""
    global index, knowledge_base
    knowledge_base = fetch_all_knowledge()
    index = prepare_faiss(knowledge_base)

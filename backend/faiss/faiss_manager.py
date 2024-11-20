from faiss.faiss_index import prepare_faiss
from database.database_utils import fetch_all_knowledge

class FaissManager:
    """Manages FAISS index and knowledge base."""
    def __init__(self):
        self.index = None
        self.knowledge_base = []

    def refresh_index(self):
        """Refresh the FAISS index with current knowledge."""
        self.knowledge_base = fetch_all_knowledge()
        self.index = prepare_faiss(self.knowledge_base)

    def search(self, query, k=1):
        """Search the FAISS index."""
        if not self.index or not self.knowledge_base:
            raise ValueError("FAISS index or knowledge base is not initialized.")
        query_embedding = embedder.encode([query])
        distances, top_ids = self.index.search(query_embedding, k)
        return distances, top_ids

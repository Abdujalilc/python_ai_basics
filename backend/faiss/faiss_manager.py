from faiss.faiss_index import prepare_faiss
from database.database_utils import fetch_all_knowledge

class FaissManager:
    def __init__(self, embedder):
        self.index = None
        self.knowledge_base = []
        self.embedder = embedder

    def refresh_index(self):
        """Refresh the FAISS index with current knowledge."""
        self.knowledge_base = fetch_all_knowledge()
        self.index = prepare_faiss(self.knowledge_base)

    def search(self, query, k=5):
        """Search the FAISS index."""
        if not self.index:
            raise ValueError("FAISS index is not initialized.")
        query_embedding = self.embedder.encode([query])
        distances, top_ids = self.index.search(query_embedding, k)
        return distances, top_ids

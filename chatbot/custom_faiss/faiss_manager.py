import faiss
from models.faiss_embedder import embedder
from database.database_utils import fetch_all_knowledge


class FaissManager:
    def __init__(self, embedder):
        """Initialize FAISS manager with an embedder."""
        self.index = None
        self.knowledge_base = []
        self.embedder = embedder

    def refresh_index(self):
        """Refresh the FAISS index with current knowledge."""
        self.knowledge_base = fetch_all_knowledge()
        self.index = self.prepare_faiss(self.knowledge_base)

    def prepare_faiss(self, contents):
        """Create and prepare FAISS index."""
        embeddings = self.embedder.encode(contents) if contents else []
        index = faiss.IndexFlatL2(self.embedder.get_sentence_embedding_dimension())
        if embeddings:
            index.add(embeddings)
        return index

    def search(self, query, k=5):
        """Search the FAISS index."""
        if not self.index:
            raise ValueError("FAISS index is not initialized.")
        query_embedding = self.embedder.encode([query])
        distances, top_ids = self.index.search(query_embedding, k)
        return distances, top_ids
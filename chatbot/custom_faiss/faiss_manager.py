import faiss
from models.faiss_embedder import embedder
from database.database_utils import fetch_all_knowledge


class FaissManager:
    def __init__(self, embedder):
        self.faiss_data  = None
        self.raw_knowledge = []
        self.embedder = embedder

    def refresh_index(self):
        self.raw_knowledge = fetch_all_knowledge()
        self.faiss_data  = self.prepare_faiss(self.raw_knowledge)

    def prepare_faiss(self, contents):
        embeddings = self.embedder.encode(contents) if contents else []
        index = faiss.IndexFlatL2(self.embedder.get_sentence_embedding_dimension())##dimensionality of embedding vector 
        if len(embeddings) > 0:
            index.add(embeddings)
        return index

    def search(self, query, nearest_neighbor_number=5):
        if not self.faiss_data :
            raise ValueError("FAISS index is not initialized.")
        query_embedding = self.embedder.encode([query])
        neighbor_distances, top_nearest_neighbor_ids = self.faiss_data.search(query_embedding, nearest_neighbor_number)
        return neighbor_distances, top_nearest_neighbor_ids
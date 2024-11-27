import faiss
import numpy as np
from database.database_utils import fetch_all_knowledge


class FaissManager:
    def __init__(self, embedder, distance_metric="euclidean"):
        self.faiss_data = None
        self.raw_knowledge = []
        self.embedder = embedder
        self.distance_metric = distance_metric  # Store the distance metric

    def refresh_index(self):
        self.raw_knowledge = fetch_all_knowledge()
        self.faiss_data = self.prepare_faiss(self.raw_knowledge)

    def prepare_faiss(self, contents):
        embeddings = self.embedder.encode(contents) if contents else []
        
        # Handle cosine similarity normalization
        if self.distance_metric == "cosine" and embeddings:
            embeddings = np.array(embeddings)
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
            index = faiss.IndexFlatIP(self.embedder.get_sentence_embedding_dimension())
        else:  # Default to Euclidean distance
            index = faiss.IndexFlatL2(self.embedder.get_sentence_embedding_dimension())
        
        if len(embeddings) > 0:
            index.add(embeddings)
        return index

    def search(self, query, nearest_neighbor_number=5):
        if not self.faiss_data:
            raise ValueError("FAISS index is not initialized.")
        
        query_embedding = self.embedder.encode([query])
        
        # Normalize query embedding for cosine similarity
        if self.distance_metric == "cosine":
            query_embedding = np.array(query_embedding)
            query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        neighbor_distances, top_nearest_neighbor_ids = self.faiss_data.search(query_embedding, nearest_neighbor_number)
        return neighbor_distances, top_nearest_neighbor_ids

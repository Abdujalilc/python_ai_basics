from custom_faiss.faiss_manager import FaissManager
from models.faiss_embedder import embedder

# Create a single shared instance of FaissManager
faiss_manager = FaissManager(embedder)
from custom_faiss.faiss_manager import FaissManager
from models.faiss_embedder import create_embedder


def create_faiss_manager(model_name: str):
    embedder = create_embedder(model_name)
    return FaissManager(embedder)

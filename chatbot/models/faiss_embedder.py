from sentence_transformers import SentenceTransformer

SentenceTransformer("all-MiniLM-L12-v2")  # Downloads and caches the model
SentenceTransformer("all-MiniLM-L6-v2")   # Cache other models if needed
SentenceTransformer("all-distilroberta-v1")

def create_embedder(model_name: str = "all-MiniLM-L12-v2"):
    return SentenceTransformer(model_name)
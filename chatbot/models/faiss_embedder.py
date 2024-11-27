from sentence_transformers import SentenceTransformer

def create_embedder(model_name: str):
    return SentenceTransformer(model_name)
import faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def prepare_faiss(contents):
    """Create and prepare FAISS index."""
    embeddings = embedder.encode(contents) if contents else []
    index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
    if len(embeddings) > 0:
        index.add(embeddings)
    return index

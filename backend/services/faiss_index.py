import faiss
from sentence_transformers import SentenceTransformer
from database_utils import fetch_all_knowledge

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def prepare_faiss():
    contents = fetch_all_knowledge()  # Fetch data from database
    embeddings = embedder.encode(contents) if contents else []

    index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
    if len(embeddings) > 0:  # Check if there are embeddings
        index.add(embeddings)

    return index, contents

index, knowledge_base = prepare_faiss()

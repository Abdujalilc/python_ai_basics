import faiss
from sentence_transformers import SentenceTransformer
from database import cursor

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def prepare_faiss():
    cursor.execute("SELECT content FROM knowledge")
    rows = cursor.fetchall()
    contents = [row[0] for row in rows]
    embeddings = embedder.encode(contents) if contents else []

    index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
    if embeddings:
        index.add(embeddings)

    return index, contents

index, knowledge_base = prepare_faiss()

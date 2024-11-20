from faiss_index import index, knowledge_base, embedder
from database import cursor, db

def add_to_knowledge(content: str):
    cursor.execute("INSERT INTO knowledge (content) VALUES (?)", (content,))
    db.commit()
    global index, knowledge_base
    from faiss_index import prepare_faiss  # Re-import to refresh the index
    index, knowledge_base = prepare_faiss()
    return "Knowledge added successfully!"
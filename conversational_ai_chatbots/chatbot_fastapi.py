from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss

# Step 1: Load Pretrained Models
pretrained_model = "distilgpt2"  # You can replace with "bigscience/bloom-560m" or "t5-small"
lm_model = AutoModelForCausalLM.from_pretrained(pretrained_model)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight content embedding model

# Step 2: Set Up SQLite Database
db = sqlite3.connect("knowledge.db", check_same_thread=False)
cursor = db.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY, content TEXT);
""")
db.commit()

# Step 3: Prepare FAISS Index
def prepare_faiss():
    cursor.execute("SELECT content FROM knowledge")
    rows = cursor.fetchall()
    contents = [row[0] for row in rows]
    embeddings = embedder.encode(contents) if contents else []
    
    index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
    if len(embeddings) > 0:  # Check if embeddings array is not empty
        index.add(embeddings)
    
    return index, contents

index, knowledge_base = prepare_faiss()

# Step 4: Define Chatbot Logic
def chatbot_logic(user_input: str):
    if not knowledge_base or not index:
        return "Knowledge base is empty. Add content first."

    # Search Knowledge Base
    query_embedding = embedder.encode([user_input])
    _, top_ids = index.search(query_embedding, k=1)
    context = knowledge_base[top_ids[0][0]] if top_ids[0][0] < len(knowledge_base) else ""

    # Generate Response
    input_text = f"{context}\n\nUser: {user_input}\nBot:"
    inputs = tokenizer(input_text, return_tensors="pt")
    output = lm_model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
    raw_response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove excessive newlines
    cleaned_response = "\n".join(line for line in raw_response.splitlines() if line.strip())
    return cleaned_response


def add_to_knowledge(content: str):
    cursor.execute("INSERT INTO knowledge (content) VALUES (?)", (content,))
    db.commit()
    global index, knowledge_base
    index, knowledge_base = prepare_faiss()
    return "Knowledge added successfully!"

# Step 5: Define FastAPI App
app = FastAPI()

# Catch-all route to redirect all requests to the docs
@app.get("/{path:path}")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# Step 6: Define Request Models
class ChatRequest(BaseModel):
    question: str

class KnowledgeRequest(BaseModel):
    content: str

# Step 7: Define API Endpoints
@app.post("/chat")
def chat(request: ChatRequest):
    response = chatbot_logic(request.question)
    return {"response": response}

@app.post("/add_knowledge")
def add_knowledge(request: KnowledgeRequest):
    status = add_to_knowledge(request.content)
    return {"status": status}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot_fastapi:app", host="127.0.0.1", port=8092, reload=True)

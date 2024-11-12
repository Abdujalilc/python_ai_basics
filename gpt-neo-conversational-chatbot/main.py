from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import sqlite3
import uvicorn
import json

app = FastAPI()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
gpt_neo = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# SQLite setup
conn = sqlite3.connect('chatbot.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS faqs (id INTEGER PRIMARY KEY, question TEXT, answer TEXT, embedding BLOB)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS conversations (user_id TEXT, question TEXT, answer TEXT)''')
conn.commit()

class UserQuery(BaseModel):
    user_id: str
    question: str

def embed_text(text):
    return embedder.encode(text).tolist()

@app.on_event("startup")
async def load_embeddings():
    cursor.execute("SELECT question FROM faqs")
    if not cursor.fetchall():  # Populate FAQs if empty
        faqs = [("What is AI?", "AI stands for Artificial Intelligence."),
                ("What is FastAPI?", "FastAPI is a web framework.")]
        cursor.executemany("INSERT INTO faqs (question, answer, embedding) VALUES (?, ?, ?)", 
                           [(q, a, json.dumps(embed_text(q))) for q, a in faqs])
        conn.commit()

@app.post("/ask")
async def ask(request: UserQuery):
    user_query = request.question
    user_id = request.user_id

    # Step 2: Embed user question
    query_embedding = embed_text(user_query)

    # Step 2: Find similar questions from FAQs
    cursor.execute("SELECT question, answer, embedding FROM faqs")
    faqs = cursor.fetchall()
    similar_faqs = []
    for q, a, e in faqs:
        embedding = json.loads(e)
        similarity = util.cos_sim(query_embedding, embedding).item()
        similar_faqs.append((q, a, similarity))
    similar_faqs = sorted(similar_faqs, key=lambda x: x[2], reverse=True)[:3]

    # Step 3: Retrieve previous user conversations
    cursor.execute("SELECT question, answer FROM conversations WHERE user_id=?", (user_id,))
    conversation_history = cursor.fetchall()

    # Step 4: Create context with similar answers and conversation history
    context = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in similar_faqs] + 
                        [f"Q: {q}\nA: {a}" for q, a in conversation_history])
    
    # Step 5: Generate GPT-Neo response
    gpt_response = gpt_neo(f"{context}\nQ: {user_query}\nA:", max_length=50)[0]["generated_text"].split("A:")[-1]

    # Step 6: Save response to conversations
    cursor.execute("INSERT INTO conversations (user_id, question, answer) VALUES (?, ?, ?)", (user_id, user_query, gpt_response))
    conn.commit()

    # Step 7: Return the response
    return {"response": gpt_response}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)

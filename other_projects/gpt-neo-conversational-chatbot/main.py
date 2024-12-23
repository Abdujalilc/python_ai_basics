from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import sqlite3
import uvicorn
import json
import datetime
from fastapi.responses import RedirectResponse

app = FastAPI()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
gpt_neo = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

# SQLite setup
conn = sqlite3.connect('db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS faqs (id INTEGER PRIMARY KEY, question TEXT, answer TEXT, embedding BLOB)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS conversations (user_id TEXT, question TEXT, answer TEXT, timestamp DATETIME DEFAULT (DATETIME('now', 'localtime')))''')
conn.commit()

class UserQuery(BaseModel):
    user_id: str
    question: str

def embed_text(text):
    return embedder.encode(text).tolist()

@app.on_event("startup")
async def load_embeddings():
    cursor.execute("SELECT question FROM faqs")
    if not cursor.fetchall():
        with open("faqs.json") as faqs_json:
            faqs = json.load(faqs_json)
        cursor.executemany("INSERT INTO faqs (question, answer, embedding) VALUES (?, ?, ?)", 
                           [(row["question"], row["answer"], json.dumps(embed_text(row["question"]))) for row in faqs])
        conn.commit()
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.post("/ask")
async def ask(request: UserQuery):
    user_query = request.question
    user_id = request.user_id

    # Step 2: Embed user question
    current_query_embedded = embed_text(user_query)

    # Step 2: Find similar questions from FAQs
    cursor.execute("SELECT question, answer, embedding FROM faqs")
    faqs = cursor.fetchall()
    similar_faqs = []
    for question, answer, embedded_question in faqs:
        embedded_faq_questions = json.loads(embedded_question)
        similarity = util.cos_sim(current_query_embedded, embedded_faq_questions).item()
        similar_faqs.append((question, answer, similarity))
    similar_faqs = [faq for faq in similar_faqs if faq[2] >= 0.7]
    similar_faqs = sorted(similar_faqs, key=lambda x: x[2], reverse=True)[:3]

    if not similar_faqs:
        similar_faqs = [("No FAQs available", "No answers available", 0.0)]

    # Step 3: Retrieve previous user conversations
    thirty_minutes_ago = datetime.datetime.now() - datetime.timedelta(minutes=30)

    cursor.execute("SELECT question, answer FROM conversations WHERE user_id=? AND timestamp >= ? LIMIT 3", (user_id, thirty_minutes_ago))
    conversation_history = cursor.fetchall()

    # Step 4: Create context with similar answers and conversation history
    context = "\n".join([f"Current - Q: {q}\nA: {a} (Similarity: {similarity})" for q, a, similarity in similar_faqs] + 
                    [f"History - Q: {q}\nA: {a}" for q, a in conversation_history])

    # Step 5: Generate a tailored, context-aware response using GPT-Neo
    gpt_response = gpt_neo(f"{context}\nQ: {user_query}\nA:", max_new_tokens=200, temperature=0.7)[0]["generated_text"]
    """answer_start = gpt_response.find("A:")
    if answer_start != -1:
        gpt_response = gpt_response[answer_start + 2:].strip()"""



    # Step 6: Save response to conversations
    cursor.execute("DELETE FROM conversations WHERE timestamp < ?", (thirty_minutes_ago,))
    cursor.execute("INSERT INTO conversations (user_id, question, answer) VALUES (?, ?, ?)", (user_id, user_query, gpt_response))
    conn.commit()

    # Step 7: Return the response
    return {"response": gpt_response}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
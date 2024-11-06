import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import struct

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")


def encode_question(question: str) -> bytes:
    embedding = model.encode(question, convert_to_tensor=True)
    return embedding.numpy().tobytes()


class FAQRequest(BaseModel):
    question: str
    answer: str


class QueryRequest(BaseModel):
    question: str


def create_table():
    with sqlite3.connect("faqs.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS faqs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                question_embedding BLOB
            )
        """
        )
        conn.commit()


create_table()


@app.post("/faq/add")
def add_faq(faq: FAQRequest):
    question_embedding = encode_question(faq.question)
    with sqlite3.connect("faqs.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO faqs (question, answer, question_embedding) VALUES (?, ?, ?)",
            (faq.question, faq.answer, question_embedding),
        )
        conn.commit()
    return {"message": "FAQ added successfully"}


@app.post("/faq/query")
def query_faq(query: QueryRequest):
    user_question = query.question
    user_embedding = model.encode(user_question, convert_to_tensor=True)

    with sqlite3.connect("faqs.db") as conn:
        cursor = conn.cursor()
        """cursor.execute("SELECT id, question, answer, question_embedding FROM faqs")"""
        cursor.execute(
            "SELECT id, question, answer, question_embedding FROM faqs WHERE question MATCH ?",
            (user_question,),
        )
        faqs = cursor.fetchall()

    if not faqs:
        raise HTTPException(status_code=404, detail="No FAQs available")

    similarities = []
    for faq_id, question, answer, question_embedding in faqs:
        stored_embedding = torch.tensor(
            list(
                struct.unpack(
                    "%sf" % (len(question_embedding) // 4), question_embedding
                )
            )
        )
        similarity = util.pytorch_cos_sim(user_embedding, stored_embedding).item()
        similarities.append((faq_id, question, answer, similarity))

    similarities.sort(key=lambda x: x[3], reverse=True)

    best_match = similarities[0]
    if best_match[3] > 0.7:
        return {
            "best_answer": best_match[2],
            "similar_questions": [
                {"question": faq[1], "similarity": faq[3]} for faq in similarities[1:4]
            ],
        }
    else:
        return {
            "message": "No exact match found. Here are some similar questions.",
            "similar_questions": [
                {"question": faq[1], "answer": faq[2], "similarity": faq[3]}
                for faq in similarities[:3]
            ],
        }


@app.get("/faq/all")
def get_all_faqs():
    with sqlite3.connect("faqs.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer FROM faqs")
        faqs = cursor.fetchall()
    return {"faqs": faqs}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8091)

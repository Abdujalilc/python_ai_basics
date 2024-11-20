from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlite3 import connect

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

# Define a class to store user input
class QueryRequest(BaseModel):
    question: str

# Simulated authentication function
def get_user_id(token: str):
    # Validate the token and retrieve the user ID
    # For simplicity, assume token is just user ID here
    return int(token)

# Define a route for querying the FAQ and storing conversation
@app.post("/faq/query")
async def query_faq(query: QueryRequest, token: str = Depends(oauth2_scheme)):
    user_id = get_user_id(token)
    user_question = query.question

    # Find the best matching FAQ
    # (This part can be enhanced with your SentenceTransformer code)
    best_answer = "This is the most relevant answer."

    # Store the conversation
    with connect("faqs.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (user_id, user_input, bot_response, timestamp) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (user_id, user_question, best_answer)
        )
        conn.commit()

    # Respond to the user
    return {"response": best_answer}


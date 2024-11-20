from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPTNeoForCausalLM, AutoTokenizer
import sqlite3
import pdb

# Initialize FastAPI app
app = FastAPI()

# SQLite setup
conn = sqlite3.connect('chat_history.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (id INTEGER PRIMARY KEY, user_input TEXT, bot_response TEXT)''')
conn.commit()

# Load GPT-Neo model
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Define input model for the API
class UserInput(BaseModel):
    user_input: str

@app.post("/chat/")
async def chat(data: UserInput):
    user_input = data.user_input
    ##pdb.set_trace()
    
    # Tokenize input with return_tensors="pt"
    inputs = tokenizer(user_input, return_tensors="pt")
    
    # Generate response
    outputs = model.generate(inputs["input_ids"], max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Store conversation in SQLite
    cursor.execute("INSERT INTO conversations (user_input, bot_response) VALUES (?, ?)",
                   (user_input, response))
    conn.commit()

    return {"user_input": user_input, "bot_response": response}


@app.get("/history/")
async def get_history():
    cursor.execute("SELECT user_input, bot_response FROM conversations")
    rows = cursor.fetchall()
    return [{"user_input": row[0], "bot_response": row[1]} for row in rows]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8092)
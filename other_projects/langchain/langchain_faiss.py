import os
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.question_answering import load_qa_chain
import uvicorn

# Set environment variables for Hugging Face cache
os.environ['HF_HOME'] = "D:\\huggingface_cache"

# Initialize SQLite connection and create table if it does not exist
def init_db():
    conn = sqlite3.connect("chatbot_responses.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            response TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Call the function to initialize the database
init_db()

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sample documents for initialization
sample_texts = ["Hello world", "How are you?", "This is a sample document."]
# Initialize FAISS vector store from texts and embeddings
vector_store = FAISS.from_texts(sample_texts, embedding=embeddings)

# Initialize FastAPI app
app = FastAPI()

# Define memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history")

# Set up the conversational retrieval chain with a simple QA chain
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 similar entries
qa_chain = load_qa_chain("openai-gpt")  # Loads a simple question-answering chain

# Create the full retrieval chain
conversation_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=qa_chain)

# Define the request body model
class ChatRequest(BaseModel):
    input: str

# Endpoint for interacting with the chatbot
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.input

    # Generate a response
    try:
        # Passing in the history separately as the API no longer uses 'memory' here
        conversation_history = memory.load_memory_variables({}).get("chat_history", "")
        response = conversation_chain({"input": user_input, "history": conversation_history})["response"]
        memory.save_context({"input": user_input}, {"response": response})  # Update memory manually
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server when the script is executed directly
if __name__ == "__main__":
    uvicorn.run("basic_langchain:app", host="127.0.0.1", port=8009, reload=True)

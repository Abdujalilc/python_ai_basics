import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Define conversation memory
memory = ConversationBufferMemory()

# Define a prompt template for the assistant
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    You are a helpful assistant.
    Here is the conversation history:
    {history}
    User: {input}
    Assistant:"""
)

# Initialize the Hugging Face model pipeline (using GPT-2 as an example)
llm_pipeline = pipeline("text-generation", model="gpt2")

# Define a function to generate responses
def generate_response(inputs):
    question = inputs["input"]
    history = inputs["history"]
    # Generate a response using the local model
    response = llm_pipeline(f"Answer the question: {question}. Here is some context: {history}", max_new_tokens=50)
    return response[0]["generated_text"]

# Define the request body model
class ChatRequest(BaseModel):
    input: str

# Endpoint for interacting with the chatbot
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.input
    conversation_history = memory.load_memory_variables({}).get("history", "")

    # Generate a response
    try:
        response = generate_response({"input": user_input, "history": conversation_history})
        # Update memory with new input and response
        memory.save_context({"input": user_input}, {"response": response})
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server when the script is executed directly
if __name__ == "__main__":
    uvicorn.run("basic_langchain:app", host="127.0.0.1", port=8009, reload=True)

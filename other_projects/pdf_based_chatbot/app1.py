import os
import uvicorn
import pdfplumber
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize FastAPI app
app = FastAPI()

# Load the local model and tokenizer using pipeline
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
llm_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=50,  # Only limits generated tokens
    truncation=True     # Ensures inputs are truncated to fit model max length
)

# Define request model
class ChatRequest(BaseModel):
    input: str

# Load the embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"  # Add newline to split later
    return text

# Initialize FAISS vector store with texts and embeddings
def initialize_vector_store(texts, embeddings):
    return FAISS.from_texts(texts, embeddings)

# Load PDF data and convert to embeddings
pdf_text = extract_text_from_pdf("sample_test_document.pdf")
texts = pdf_text.split("\n")  # Split by newlines for context separation
vector_store = initialize_vector_store(texts, embedding)

# Set up conversational memory
memory = ConversationBufferMemory()

# Define a prompt template for the assistant
prompt_template = PromptTemplate(
    input_variables=["history", "retrieved_content", "input"],
    template="""
    You are a helpful assistant.
    Here is the conversation history:
    {history}
    Based on the following information from the document:
    {retrieved_content}
    User: {input}
    Assistant:"""
)

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.input
    conversation_history = memory.load_memory_variables({}).get("history", "")

    # Retrieve relevant content from the PDF
    docs = vector_store.similarity_search(user_input, k=3)  # Retrieve top 3 most similar chunks
    retrieved_content = "\n".join([doc.page_content for doc in docs])

    # Generate the prompt with retrieved content
    prompt = prompt_template.format(
        history=conversation_history,
        retrieved_content=retrieved_content,
        input=user_input
    )

    # Generate response using the llm_pipeline
    response = llm_pipeline(prompt)[0]["generated_text"]

    # Save conversation to memory
    memory.save_context({"input": user_input}, {"response": response})
    return {"response": response}

# Run FastAPI app
if __name__ == "__main__":
    uvicorn.run("app1:app", host="127.0.0.1", port=8000, reload=True)

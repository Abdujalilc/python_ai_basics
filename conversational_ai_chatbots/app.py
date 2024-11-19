from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from models import ChatRequest, KnowledgeRequest
from chatbot_response import chatbot_logic
from add_knowledge import add_to_knowledge

app = FastAPI()

@app.get("/{path:path}")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/chat")
def chat(request: ChatRequest):
    response = chatbot_logic(request.question)
    return {"response": response}

@app.post("/add_knowledge")
def add_knowledge(request: KnowledgeRequest):
    status = add_to_knowledge(request.content)
    return {"status": status}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8092, reload=True)
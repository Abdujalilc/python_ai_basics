from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str

class KnowledgeRequest(BaseModel):
    content: str

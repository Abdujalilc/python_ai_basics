from pydantic import BaseModel

class KnowledgeRequest(BaseModel):
    content: str
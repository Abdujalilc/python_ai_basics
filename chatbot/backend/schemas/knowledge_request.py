from pydantic import BaseModel

class KnowledgeRequest(BaseModel):
    content: str    
    distance_metric: str = "cosine"
    embedder_model : str = "all-MiniLM-L12-v2"
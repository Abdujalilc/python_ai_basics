from pydantic import BaseModel

class FAISSSettings(BaseModel):
    nearest_neighbor: int = 5
    similarity_threshold: float = 1.2
    distance_metric: str = "cosine"

class TransformerSettings(BaseModel):
    language_model: str = ""
    max_new_tokens: int = 50
    temperature: float = 0.3
    top_k: int = 20
    top_p: float = 0.85
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0
    stop_sequence: str = ""
    seed: int = 42
    beam_width: int = 3

class ChatRequest(BaseModel):
    question: str
    faiss_settings: FAISSSettings
    transformer_settings: TransformerSettings

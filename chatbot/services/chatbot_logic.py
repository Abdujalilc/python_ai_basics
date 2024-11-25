from custom_faiss.faiss_manager import FaissManager
from models.language_model import lm_model, tokenizer
from models.faiss_embedder import embedder

faiss_manager = FaissManager(embedder)

def chatbot_logic(user_input: str):
    """Core chatbot logic."""
    # Refresh FAISS index if empty
    if not faiss_manager.knowledge_base or not faiss_manager.index:
        return "The knowledge base is empty. Please add content first."

    # Search FAISS index
    try:
        distances, top_ids = faiss_manager.search(user_input, k=1)
    except ValueError:
        return "I couldn't find any relevant information in the knowledge base."

    similarity_threshold = 0.5
    if distances[0][0] > similarity_threshold:
        return "I couldn't find any relevant information in the knowledge base."

    # Retrieve context
    context = faiss_manager.knowledge_base[top_ids[0][0]]

    # Prepare prompt and generate response
    input_text = f"Answer briefly: {user_input}\nContext: {context}"
    inputs = tokenizer(input_text, return_tensors="pt")
    output = lm_model.generate(
        inputs["input_ids"], max_new_tokens=50, temperature=0.3, top_k=20, top_p=0.85
    )

    # Process the response
    raw_response = tokenizer.decode(output[0], skip_special_tokens=True)
    return raw_response.split(".")[0].strip()

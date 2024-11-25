from custom_faiss.faiss_manager_singleton import faiss_manager
from models.language_model import lm_model, tokenizer
from models.faiss_embedder import embedder

def chatbot_logic(user_input: str):
    """Core chatbot logic with relaxed filtering for closer matches."""
    # Ensure FAISS index and knowledge base are loaded
    if not faiss_manager.knowledge_base or not faiss_manager.index:
        return "The knowledge base is empty. Please add content first."

    # Search FAISS index
    try:
        distances, top_ids = faiss_manager.search(user_input, k=3)  # Retrieve top 3 matches
    except ValueError:
        return "I couldn't find any relevant information in the knowledge base."

    # Lower similarity threshold to allow closer matches
    similarity_threshold = 1.2
    valid_matches = [
        (distances[0][i], faiss_manager.knowledge_base[top_ids[0][i]])
        for i in range(len(distances[0]))
        if distances[0][i] <= similarity_threshold
    ]

    # If no valid matches are found
    if not valid_matches:
        return "I couldn't find any relevant information in the knowledge base."

    # Use the closest match
    best_match = valid_matches[0][1]

    # Prepare prompt and generate response
    input_text = f"Answer briefly: {user_input}\nContext: {best_match}"
    inputs = tokenizer(input_text, return_tensors="pt")
    output = lm_model.generate(
        inputs["input_ids"], max_new_tokens=50, temperature=0.3, top_k=20, top_p=0.85
    )

    # Process the response
    raw_response = tokenizer.decode(output[0], skip_special_tokens=True)
    return raw_response.split(".")[0].strip()

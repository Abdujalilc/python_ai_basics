from custom_faiss.faiss_manager_singleton import faiss_manager
from models.language_model import lm_model, tokenizer
from models.faiss_embedder import embedder

def chatbot_logic(user_input: str):
    if not faiss_manager.raw_knowledge or not faiss_manager.faiss_data:
        return "Knowledge base is empty. Add content first."

    try:
        neighbor_distances, top_nearest_neighbor_ids = faiss_manager.search(user_input, nearest_neighbor_number=3)
    except ValueError:
        return "No relevant information found."

    similarity_threshold = 1.2
    valid_matches = [
        faiss_manager.raw_knowledge[top_nearest_neighbor_ids[0][i]]
        for i in range(len(neighbor_distances[0]))
        if neighbor_distances[0][i] <= similarity_threshold
    ]

    if not valid_matches:
        return "No relevant information found."

    inputs = tokenizer(f"Answer briefly: {user_input}\nContext: {valid_matches[0]}", return_tensors="pt")##pt-PyTorch
    output = lm_model.generate(inputs["input_ids"], max_new_tokens=50, temperature=0.3, top_k=20, top_p=0.85)
    return tokenizer.decode(output[0], skip_special_tokens=True).split(".")[0].strip()

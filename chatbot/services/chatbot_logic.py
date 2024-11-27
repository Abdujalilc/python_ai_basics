from schemas.chat_request import ChatRequest
from custom_faiss.faiss_manager_singleton import create_faiss_manager
from models.language_model import lm_model, tokenizer

def chatbot_logic(request: ChatRequest):
    faiss_manager = create_faiss_manager(request.faiss_settings.embedder_model,request.faiss_settings.distance_metric)
    if not faiss_manager.raw_knowledge or not faiss_manager.faiss_data:
        return "Knowledge base is empty. Add content first."

    try:
        neighbor_distances, top_nearest_neighbor_ids = faiss_manager.search(request.question, nearest_neighbor_number=request.faiss_settings.nearest_neighbor)
    except ValueError:
        return "No relevant information found."

    similarity_threshold = request.faiss_settings.similarity_threshold
    valid_matches = [
        faiss_manager.raw_knowledge[top_nearest_neighbor_ids[0][i]]
        for i in range(len(neighbor_distances[0]))
        if neighbor_distances[0][i] <= similarity_threshold
    ]

    if not valid_matches:
        return "No relevant information found."

    inputs = tokenizer(f"Answer briefly: {request.question}\nContext: {valid_matches[0]}", return_tensors="pt")##pt-PyTorch
    output = lm_model.generate(inputs["input_ids"], max_new_tokens=request.transformer_settings.max_new_tokens, temperature=request.transformer_settings.temperature, top_k=request.transformer_settings.top_k, top_p=request.transformer_settings.top_p)
    return tokenizer.decode(output[0], skip_special_tokens=request.transformer_settings.skip_special_tokens).split(".")[0].strip()

from schemas.chat_request import ChatRequest
from custom_faiss.faiss_manager_singleton import create_faiss_manager
from models.language_model import load_language_model
import torch

def chatbot_logic(req: ChatRequest):
    faiss_manager = create_faiss_manager(req.faiss_settings.embedder_model,req.faiss_settings.distance_metric)
    if not faiss_manager.raw_knowledge or not faiss_manager.faiss_data:
        return "Knowledge base is empty. Add content first."

    try:
        neighbor_distances, top_nearest_neighbor_ids = faiss_manager.search(req.question, nearest_neighbor_number=req.faiss_settings.nearest_neighbor)
    except ValueError:
        return "No relevant information found."

    similarity_threshold = req.faiss_settings.similarity_threshold
    valid_matches = [
        faiss_manager.raw_knowledge[top_nearest_neighbor_ids[0][i]]
        for i in range(len(neighbor_distances[0]))
        if neighbor_distances[0][i] <= similarity_threshold
    ]

    if not valid_matches:
        return "No relevant information found."
    lm_model, tokenizer = load_language_model(req.transformer_settings.language_model)
    inputs = tokenizer(f"Answer briefly: {req.question}\nContext: {valid_matches[0]}", return_tensors="pt")##pt-PyTorch
    
    torch.manual_seed(req.transformer_settings.seed)
    output = lm_model.generate(
        inputs["input_ids"], 
        max_new_tokens=req.transformer_settings.max_new_tokens, 
        temperature=req.transformer_settings.temperature, 
        top_k=req.transformer_settings.top_k, 
        top_p=req.transformer_settings.top_p,
        repetition_penalty=req.transformer_settings.repetition_penalty,
        length_penalty=req.transformer_settings.length_penalty,
        num_beams=req.transformer_settings.beam_width
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=req.transformer_settings.skip_special_tokens)

    # Stop at stop_sequence if it exists
    stop_sequence = req.transformer_settings.stop_sequence
    if stop_sequence in generated_text:
        generated_text = generated_text.split(stop_sequence)[0]

    return generated_text

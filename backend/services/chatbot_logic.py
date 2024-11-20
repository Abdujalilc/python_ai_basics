from faiss.faiss_index import index, knowledge_base, embedder
from models.language_model import lm_model, tokenizer

def chatbot_logic(user_input: str):
    """Core chatbot logic to handle user input."""
    # Validate knowledge base
    if not knowledge_base or not index:
        return "The knowledge base is empty. Please add content first."

    # Search FAISS for relevant context
    query_embedding = embedder.encode([user_input])
    distances, top_ids = index.search(query_embedding, k=1)
    similarity_threshold = 0.5

    if distances[0][0] > similarity_threshold:
        return "I couldn't find any relevant information in the knowledge base."

    # Retrieve context from knowledge base
    context = knowledge_base[top_ids[0][0]] if top_ids[0][0] < len(knowledge_base) else ""

    # Prepare prompt and generate response
    input_text = f"Answer briefly: {user_input}\nContext: {context}"
    inputs = tokenizer(input_text, return_tensors="pt")
    output = lm_model.generate(inputs["input_ids"], max_new_tokens=50, temperature=0.3, top_k=20, top_p=0.85, no_repeat_ngram_size=3)
    
    # Extract and return response
    response = tokenizer.decode(output[0], skip_special_tokens=True).split(".")[0].strip()
    return response

from transformers import AutoModelForCausalLM, AutoTokenizer
from faiss_index import index, knowledge_base, embedder
from database import cursor, db

pretrained_model = "distilgpt2"
lm_model = AutoModelForCausalLM.from_pretrained(pretrained_model)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

def chatbot_logic(user_input: str):
    if not knowledge_base or not index:
        return "The knowledge base is empty. Please add content first."

    # Search Knowledge Base
    query_embedding = embedder.encode([user_input])
    distances, top_ids = index.search(query_embedding, k=1)
    similarity_threshold = 0.5
    if distances[0][0] > similarity_threshold:
        return "I couldn't find any relevant information in the knowledge base."

    # Fetch Context
    context = knowledge_base[top_ids[0][0]] if top_ids[0][0] < len(knowledge_base) else ""

    # Simplified Input Prompt
    input_text = f"Answer briefly: Who was Nicola Tesla?\nContext: {context}"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate Response
    output = lm_model.generate(
        inputs["input_ids"],
        max_new_tokens=50,  # Restrict length
        temperature=0.3,  # Reduce randomness
        top_k=20,  # Narrow down top choices
        top_p=0.85,  # Focus on high-probability tokens
        no_repeat_ngram_size=3  # Avoid repetition
    )

    # Process the Response
    raw_response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the first sentence or phrase
    response = raw_response.split(".")[0].strip()  # Stops at the first period
    return response

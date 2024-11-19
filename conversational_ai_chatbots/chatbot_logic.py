from transformers import AutoModelForCausalLM, AutoTokenizer
from faiss_index import index, knowledge_base, embedder
from database import cursor, db

pretrained_model = "distilgpt2"
lm_model = AutoModelForCausalLM.from_pretrained(pretrained_model)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

def chatbot_logic(user_input: str):
    if not knowledge_base or not index:
        return "Knowledge base is empty. Add content first."

    query_embedding = embedder.encode([user_input])
    _, top_ids = index.search(query_embedding, k=1)
    context = knowledge_base[top_ids[0][0]] if top_ids[0][0] < len(knowledge_base) else ""

    input_text = f"{context}\n\nUser: {user_input}\nBot:"
    inputs = tokenizer(input_text, return_tensors="pt")
    output = lm_model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
    raw_response = tokenizer.decode(output[0], skip_special_tokens=True)

    cleaned_response = "\n".join(line for line in raw_response.splitlines() if line.strip())
    return cleaned_response

def add_to_knowledge(content: str):
    cursor.execute("INSERT INTO knowledge (content) VALUES (?)", (content,))
    db.commit()
    global index, knowledge_base
    from faiss_index import prepare_faiss  # Re-import to refresh the index
    index, knowledge_base = prepare_faiss()
    return "Knowledge added successfully!"

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import sqlite3
import torch
import struct
from sentence_transformers import SentenceTransformer

# Initialize models
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
gpt_neo_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

@app.post("/faq/query")
def query_faq(query: QueryRequest):
    user_question = query.question
    user_embedding = model.encode(user_question, convert_to_tensor=True)

    with sqlite3.connect("faqs.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, question, answer, question_embedding FROM faqs")
        faqs = cursor.fetchall()

    if not faqs:
        raise HTTPException(status_code=404, detail="No FAQs available")

    similarities = []
    for faq_id, question, answer, question_embedding in faqs:
        stored_embedding = torch.tensor(
            list(struct.unpack("%sf" % (len(question_embedding) // 4), question_embedding))
        )
        similarity = util.pytorch_cos_sim(user_embedding, stored_embedding).item()
        similarities.append((faq_id, question, answer, similarity))

    similarities.sort(key=lambda x: x[3], reverse=True)
    best_match = similarities[0]

    if best_match[3] > 0.7:
        # Build conversation context
        context = "User: " + user_question + "\nFAQ: " + best_match[2] + "\nPrevious: " + "\n".join([faq[1] for faq in similarities[:3]])
        
        # Generate response using GPT-Neo
        inputs = tokenizer(context, return_tensors="pt")
        outputs = gpt_neo_model.generate(inputs["input_ids"], max_length=200)
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"generated_answer": generated_answer.strip()}

    else:
        return {"message": "No exact match found. Here are some similar questions.", "similar_questions": [{"question": faq[1], "similarity": faq[3]} for faq in similarities[:3]]}

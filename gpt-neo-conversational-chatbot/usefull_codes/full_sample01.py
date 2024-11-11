import sqlite3
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import numpy as np
import struct
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize model and tokenizer
gpt_neo_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Used for generating sentence embeddings

# SQLite connection to fetch data
def get_faq_data():
    with sqlite3.connect("faqs.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, question, answer, question_embedding FROM faqs")
        faqs = cursor.fetchall()
    return faqs

# Store user input and bot response in conversations table
def store_conversation(user_input, bot_response):
    with sqlite3.connect("conversations.db") as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO conversations (user_input, bot_response) VALUES (?, ?)",
                       (user_input, bot_response))
        conn.commit()

# Query the FAQ table to find the most relevant answers
def query_faq_and_generate_response(user_question):
    faqs = get_faq_data()  # Fetch FAQ data from database
    
    # Generate the embedding for the user question
    user_embedding = embedding_model.encode(user_question)
    
    # Calculate cosine similarity between user question and FAQ questions
    similarities = []
    for faq_id, question, answer, question_embedding in faqs:
        # Convert question embedding from BLOB to a numpy array
        stored_embedding = np.array(struct.unpack(f"{len(question_embedding)//4}f", question_embedding))
        
        # Compute cosine similarity between user question and stored FAQ question
        sim = cosine_similarity([user_embedding], [stored_embedding])[0][0]
        similarities.append((faq_id, question, answer, sim))

    # Sort FAQs by similarity
    similarities.sort(key=lambda x: x[3], reverse=True)
    
    # Use the most similar FAQ (if similarity threshold met) for GPT-Neo input
    if similarities[0][3] > 0.7:
        context = f"Q: {similarities[0][1]}\nA: {similarities[0][2]}\nUser: {user_question}\nBot:"
    else:
        context = f"User: {user_question}\nBot: (No close FAQ match, bot will answer)"

    # Tokenize the input context for GPT-Neo model
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=1024)
    
    # Generate response from GPT-Neo
    outputs = gpt_neo_model.generate(inputs["input_ids"], max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Store the conversation (user input and bot response)
    store_conversation(user_question, response)

    return response

# Example usage
user_question = "How can I reset my password?"
bot_response = query_faq_and_generate_response(user_question)
print(f"Bot Response: {bot_response}")

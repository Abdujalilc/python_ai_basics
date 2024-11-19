import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st

# Load Models
pretrained_model = "distilgpt2"
lm_model = AutoModelForCausalLM.from_pretrained(pretrained_model)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Set Up SQLite DB
db = sqlite3.connect("knowledge.db", check_same_thread=False)
cursor = db.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY, content TEXT);")
db.commit()

# Prepare FAISS Index
def prepare_faiss():
    cursor.execute("SELECT content FROM knowledge")
    rows = cursor.fetchall()
    contents = [row[0] for row in rows]
    embeddings = embedder.encode(contents)
    index = faiss.IndexFlatL2(embeddings.shape[1]) if contents else None
    if index:
        index.add(embeddings)
    return index, contents

index, knowledge_base = prepare_faiss()

# Chatbot Logic
def chatbot(user_input):
    if not knowledge_base or not index:
        return "Knowledge base is empty. Add content first."
    query_embedding = embedder.encode([user_input])
    _, top_ids = index.search(query_embedding, k=1)
    context = knowledge_base[top_ids[0][0]] if top_ids[0][0] < len(knowledge_base) else ""
    input_text = f"{context}\n\nUser: {user_input}\nBot:"
    inputs = tokenizer(input_text, return_tensors="pt")
    output = lm_model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Add Knowledge
def add_to_db(content):
    cursor.execute("INSERT INTO knowledge (content) VALUES (?)", (content,))
    db.commit()
    global index, knowledge_base
    index, knowledge_base = prepare_faiss()
    return "Knowledge added successfully!"

# Streamlit Interface
st.title("Local Content-Aware Chatbot")

# Add Knowledge
with st.expander("Add Knowledge"):
    knowledge_input = st.text_area("Add Knowledge Base Content")
    if st.button("Add Content"):
        status = add_to_db(knowledge_input)
        st.success(status)

# Chatbot Interaction
user_input = st.text_input("Ask a Question")
if st.button("Submit"):
    response = chatbot(user_input)
    st.text_area("Response", value=response, height=150)

# Run this app using the following command: 
# streamlit run chatbot_streamlit.py 

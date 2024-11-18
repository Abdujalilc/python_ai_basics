import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr

# Step 1: Load Pretrained Models
pretrained_model = "distilgpt2"  # You can replace with "bigscience/bloom-560m" or "t5-small"
lm_model = AutoModelForCausalLM.from_pretrained(pretrained_model)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight content embedding model

# Step 2: Set Up SQLite Database
db = sqlite3.connect("knowledge.db", check_same_thread=False)
cursor = db.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY, content TEXT);
""")
db.commit()

# Step 3: Prepare FAISS Index
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

# Step 4: Define Chatbot Logic
def chatbot(user_input):
    if not knowledge_base or not index:
        return "Knowledge base is empty. Add content first."

    # Step 4.1: Search Knowledge Base
    query_embedding = embedder.encode([user_input])
    _, top_ids = index.search(query_embedding, k=1)
    context = knowledge_base[top_ids[0][0]] if top_ids[0][0] < len(knowledge_base) else ""

    # Step 4.2: Generate Response Using Pretrained Model
    input_text = f"{context}\n\nUser: {user_input}\nBot:"
    inputs = tokenizer(input_text, return_tensors="pt")
    output = lm_model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Step 5: Add Content to Knowledge Base
def add_to_db(content):
    cursor.execute("INSERT INTO knowledge (content) VALUES (?)", (content,))
    db.commit()
    global index, knowledge_base
    index, knowledge_base = prepare_faiss()
    return "Knowledge added successfully!"

# Step 6: Build Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("## Local Content-Aware Chatbot")
    
    # Chatbot Interface
    user_input = gr.Textbox(label="Ask a Question")
    bot_response = gr.Textbox(label="Response", interactive=False)
    submit_btn = gr.Button("Submit")
    
    # Knowledge Management
    knowledge_input = gr.Textbox(label="Add Knowledge Base Content")
    add_btn = gr.Button("Add Content")
    add_status = gr.Textbox(label="Status", interactive=False)
    
    # Linking Functions
    submit_btn.click(chatbot, inputs=[user_input], outputs=[bot_response])
    add_btn.click(add_to_db, inputs=[knowledge_input], outputs=[add_status])

# Step 7: Run Locally
app.launch(share=True)

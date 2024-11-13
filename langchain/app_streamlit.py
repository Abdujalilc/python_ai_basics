import sqlite3
import streamlit as st
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, create_structured_chat_agent, AgentExecutor

# Connect to SQLite database
conn = sqlite3.connect('faq.db')
cursor = conn.cursor()

# Create a simple FAQ table (if it doesn't exist)
cursor.execute('''
CREATE TABLE IF NOT EXISTS faq (
    question TEXT,
    answer TEXT
)
''')

# Insert sample data into the table (avoid duplicates)
cursor.execute("INSERT OR IGNORE INTO faq (question, answer) VALUES ('What is LangChain?', 'LangChain is a framework for building language model applications')")
cursor.execute("INSERT OR IGNORE INTO faq (question, answer) VALUES ('How do I use GPT-Neo?', 'You can use GPT-Neo by installing Hugging Face and loading the model')")
conn.commit()

# Query function to search the database
def get_answer_from_db(query):
    cursor.execute("SELECT answer FROM faq WHERE question LIKE ?", ('%' + query + '%',))
    result = cursor.fetchone()
    return result[0] if result else "Sorry, I couldn't find an answer to that."

# Load a local language model using Hugging Face Transformers
llm_pipeline = pipeline("text-generation", model="gpt2")  # Change model if needed

# Define a function to generate responses with the local model
def generate_response(question, context=""):
    response = llm_pipeline(f"Answer the question: {question}. Here is some context: {context}", max_length=100)
    return response[0]["generated_text"]

# Define the prompt template with required variables
prompt = PromptTemplate(
    input_variables=["question", "context", "tool_names", "tools", "agent_scratchpad"],
    template="""You are a helpful assistant.
Answer the question: {question}.
Here is some context: {context}.
Available tools: {tool_names}.
{tools}
{agent_scratchpad}"""
)

# Create a tool for the database lookup
db_tool = Tool(
    name="Database Query",
    func=get_answer_from_db,
    description="Use this tool to query the SQLite database for answers."
)

# Initialize the agent without LLMChain
tools = [db_tool]
agent = create_structured_chat_agent(
    llm=lambda inputs: generate_response(inputs["question"], inputs.get("context", "")),
    tools=tools,
    prompt=prompt
)

# Create an AgentExecutor to handle the conversation
agent_executor = AgentExecutor(agent=agent, verbose=True)

# Streamlit UI for the chatbot
st.title("FAQ Chatbot")
st.write("Ask your question, and I will fetch an answer from the database.")

# User input and response handling
user_input = st.text_input("You: ")

if user_input:
    # Use the agent to get a response
    response = agent_executor.run(input=user_input)
    st.write(f"Chatbot: {response}")

# Close the connection after use
conn.close()

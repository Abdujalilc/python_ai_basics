import sqlite3
import streamlit as st
from langchain.chains import ConversationalChain
from langchain.llms import GPTNeo
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain.tools import Tool

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

# Insert some sample data into the table
cursor.execute("INSERT INTO faq (question, answer) VALUES ('What is LangChain?', 'LangChain is a framework for building language model applications')")
cursor.execute("INSERT INTO faq (question, answer) VALUES ('How do I use GPT-Neo?', 'You can use GPT-Neo by installing Hugging Face and loading the model')")
conn.commit()

# Query function to search the database
def get_answer_from_db(query):
    cursor.execute("SELECT answer FROM faq WHERE question LIKE ?", ('%' + query + '%',))
    result = cursor.fetchone()
    return result[0] if result else "Sorry, I couldn't find an answer to that."

# Define your LLM and memory
llm = GPTNeo(model="EleutherAI/gpt-neo-2.7B")
memory = ConversationBufferMemory(memory_key="chat_history")

# Define the prompt template
prompt = PromptTemplate(input_variables=["question", "context"], template="Answer the question: {question}. Here is some context: {context}")

# Create a tool for the database lookup
db_tool = Tool(
    name="Database Query",
    func=get_answer_from_db,
    description="Use this tool to query the SQLite database for answers."
)

# Initialize the agent
tools = [db_tool]
agent = initialize_agent(tools, llm, memory=memory, verbose=True)

# Streamlit UI
st.set_option('server.port', 8502)
st.title("FAQ Chatbot")
st.write("Ask your question, and I will fetch an answer from the database.")

# User input and response
user_input = st.text_input("You: ")

if user_input:
    response = agent.run(input=user_input)
    st.write(f"Chatbot: {response}")

# Close the connection after use
conn.close()

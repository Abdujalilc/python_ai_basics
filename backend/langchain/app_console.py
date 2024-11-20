import sqlite3
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType

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

# Insert sample data into the table (avoid duplicate entries)
cursor.execute("INSERT OR IGNORE INTO faq (question, answer) VALUES ('What is LangChain?', 'LangChain is a framework for building language model applications')")
cursor.execute("INSERT OR IGNORE INTO faq (question, answer) VALUES ('How do I use GPT-Neo?', 'You can use GPT-Neo by installing Hugging Face and loading the model')")
conn.commit()

# Query function to search the database
def get_answer_from_db(query):
    cursor.execute("SELECT answer FROM faq WHERE question LIKE ?", ('%' + query + '%',))
    result = cursor.fetchone()
    return result[0] if result else "Sorry, I couldn't find an answer to that."

# Define LLM and memory
llm = HuggingFaceHub(repo_id="gpt2")  # Replace "gpt2" with a suitable model from Hugging Face
memory = ConversationBufferMemory(memory_key="chat_history")

# Define the prompt template
prompt = PromptTemplate(input_variables=["question", "context"], template="Answer the question: {question}. Here is some context: {context}")

# Define LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Create a tool for the database lookup
db_tool = Tool(
    name="Database Query",
    func=get_answer_from_db,
    description="Use this tool to query the SQLite database for answers."
)

# Initialize the agent
tools = [db_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

# Main conversation loop
def chat():
    print("Chatbot: How can I assist you?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        # Run the conversation with the agent
        response = agent.run(input=user_input)
        print(f"Chatbot: {response}")

# Start the chat
chat()

# Close the connection after use
conn.close()

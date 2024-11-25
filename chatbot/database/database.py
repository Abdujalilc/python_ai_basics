import sqlite3
import os

# Resolve the absolute path to the correct 'data' folder
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/knowledge.db"))
print("Database Path:", db_path)  # Debugging: Verify the resolved path

# Ensure the directory exists
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Connect to the database
db = sqlite3.connect(db_path, check_same_thread=False)

# Initialize the database
cursor = db.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY, content TEXT);
""")
db.commit()

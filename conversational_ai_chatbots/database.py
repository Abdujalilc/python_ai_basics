import sqlite3

db = sqlite3.connect("knowledge.db", check_same_thread=False)
cursor = db.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS knowledge (id INTEGER PRIMARY KEY, content TEXT);
""")
db.commit()

# database_utils.py
from database import cursor, db

def insert_knowledge(content: str):
    """Insert new knowledge into the database."""
    cursor.execute("INSERT INTO knowledge (content) VALUES (?)", (content,))
    db.commit()

def fetch_all_knowledge():
    """Fetch all knowledge content from the database."""
    cursor.execute("SELECT content FROM knowledge")
    rows = cursor.fetchall()
    return [row[0] for row in rows] if rows else []

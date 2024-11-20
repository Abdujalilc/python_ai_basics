from database.database import cursor, db

def fetch_all_knowledge():
    """Fetch all knowledge content from the database."""
    cursor.execute("SELECT content FROM knowledge")
    rows = cursor.fetchall()
    return [row[0] for row in rows] if rows else []

def insert_knowledge(content):
    """Insert new knowledge into the database."""
    cursor.execute("INSERT INTO knowledge (content) VALUES (?)", (content,))
    db.commit()

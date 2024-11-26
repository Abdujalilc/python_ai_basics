from database.database import cursor, db

def fetch_all_knowledge():
    cursor.execute("SELECT content FROM knowledge")
    rows = cursor.fetchall()
    return [row[0] for row in rows] if rows else []

def insert_knowledge(content):
    cursor.execute("INSERT INTO knowledge (content) VALUES (?)", (content,))
    db.commit()
# database_utils.py
from database import cursor

def fetch_all_knowledge():
    cursor.execute("SELECT content FROM knowledge")
    rows = cursor.fetchall()
    return [row[0] for row in rows] if rows else []

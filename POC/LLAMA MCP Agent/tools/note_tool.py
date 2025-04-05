from langchain_core.tools import tool
import sqlite3, os

DB_PATH = "db/notes.db"

@tool
def create_note(title: str, content: str) -> str:
    """Save a note with a title and content to SQLite."""
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS notes (title TEXT, content TEXT)''')
    c.execute("INSERT INTO notes (title, content) VALUES (?, ?)", (title, content))
    conn.commit()
    conn.close()
    return f"Note '{title}' saved successfully."
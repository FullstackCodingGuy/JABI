import autogen
import sqlite3
import requests
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"
DEFAULT_MODEL = "llama3.2:3b"

config_list = [
    {
        "model": DEFAULT_MODEL,
        "base_url": "http://localhost:11434/v1",
        'api_key': 'ollama',
    },
]

class PersonalAssistant:
    def __init__(self):
        self.model = DEFAULT_MODEL  # Replace with "ollama/mistral" for local models
        self.db_path = "memory.db"
        self.init_db()

        print("Loading model...", self.model)
     
        self.assistant = autogen.AssistantAgent(name="Personal_Assistant",
                       max_consecutive_auto_reply=10,
                       system_message="You should act as a student!",
                       llm_config={
                           "config_list": config_list,
                           "temperature": 1,
                       })


    def init_db(self):
        """Initialize SQLite memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                ai_response TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save_interaction(self, user_input, ai_response):
        """Save conversation to SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO memory (user_input, ai_response) VALUES (?, ?)", 
                       (user_input, ai_response))
        conn.commit()
        conn.close()

    def get_recent_context(self, limit=5):
        """Retrieve chat history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT user_input, ai_response FROM memory ORDER BY id DESC LIMIT ?", (limit,))
        history = cursor.fetchall()
        conn.close()
        return "\n".join([f"User: {u}\nAI: {a}" for u, a in reversed(history)])

    def chat(self, user_input):
        """AI Conversation with Memory"""
        context = self.get_recent_context()
        prompt = f"{context}\nUser: {user_input}\nAI:"
        
        response = self.assistant.generate_reply(messages=[{"role": "user", "content": prompt}])

        print("AI Response:", response)

        ai_response = response
        
        self.save_interaction(user_input, ai_response)
        return ai_response

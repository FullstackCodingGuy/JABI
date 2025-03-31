
Got it! We'll build a **corporate-themed AI agent** with:  
✅ **FastAPI** as the backend  
✅ **SQLite** for context memory  
✅ **Tools integration** (Send Email, WhatsApp, Schedule Meetings)  
✅ **Minimal corporate UI**  

---

# **🚀 Step 1: Install Dependencies**  
Run the following:  
```sh
pip install fastapi uvicorn ollama sqlite3 requests
```

---

# **🚀 Step 2: Create the AI Agent with SQLite Memory**
📄 **`agent.py`**  
```python
import ollama
import sqlite3
from datetime import datetime

class AIAgent:
    def __init__(self, model="mistral"):
        self.model = model
        self.db_path = "memory.db"
        self.init_db()

    def init_db(self):
        """Initialize SQLite database for memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_input TEXT,
                ai_response TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save_interaction(self, user_input, ai_response):
        """Save conversation history to SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO memory (timestamp, user_input, ai_response) VALUES (?, ?, ?)", 
                       (datetime.now(), user_input, ai_response))
        conn.commit()
        conn.close()

    def get_context(self, limit=5):
        """Retrieve recent chat history for context."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT user_input, ai_response FROM memory ORDER BY id DESC LIMIT ?", (limit,))
        history = cursor.fetchall()
        conn.close()
        return "\n".join([f"User: {u}\nAI: {a}" for u, a in reversed(history)])

    def chat(self, user_input):
        """Generate AI response with memory context."""
        context = self.get_context()
        prompt = f"{context}\nUser: {user_input}\nAI:"
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])

        ai_response = response["message"]["content"]
        self.save_interaction(user_input, ai_response)
        return ai_response
```

---

# **🚀 Step 3: Build FastAPI Backend**
📄 **`main.py`**  
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import AIAgent
import smtplib

app = FastAPI(title="AI Agent", description="AI Agent with SQLite & Task Automation", version="1.0")

agent = AIAgent()

class ChatRequest(BaseModel):
    message: str

class EmailRequest(BaseModel):
    to_email: str
    subject: str
    body: str

@app.post("/chat")
def chat(request: ChatRequest):
    response = agent.chat(request.message)
    return {"response": response}

@app.post("/send-email")
def send_email(request: EmailRequest):
    """Send an email (Replace with your SMTP settings)."""
    try:
        sender_email = "your-email@example.com"
        password = "your-password"
        server = smtplib.SMTP("smtp.example.com", 587)
        server.starttls()
        server.login(sender_email, password)
        message = f"Subject: {request.subject}\n\n{request.body}"
        server.sendmail(sender_email, request.to_email, message)
        server.quit()
        return {"status": "Email Sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "AI Agent API Running"}
```
---

# **🚀 Step 4: Run the Backend**
```sh
uvicorn main:app --reload
```
This starts the **FastAPI server** on `http://127.0.0.1:8000/`.

---

# **🚀 Step 5: Create Minimal Corporate UI**
📄 **`ui.py`** (Streamlit-based UI)
```python
import streamlit as st
import requests

st.set_page_config(page_title="AI Agent", layout="centered")

st.title("💼 AI Assistant")
st.write("Corporate AI Assistant for Productivity")

user_input = st.text_input("You:", "")

if st.button("Ask AI"):
    response = requests.post("http://127.0.0.1:8000/chat", json={"message": user_input}).json()
    st.text_area("AI:", response["response"], height=200)

st.sidebar.title("Tasks")
email_to = st.sidebar.text_input("Email To")
email_subject = st.sidebar.text_input("Subject")
email_body = st.sidebar.text_area("Email Body")

if st.sidebar.button("Send Email"):
    email_response = requests.post("http://127.0.0.1:8000/send-email", json={
        "to_email": email_to,
        "subject": email_subject,
        "body": email_body
    }).json()
    st.sidebar.success(email_response.get("status", "Failed"))
```

---

# **🚀 Step 6: Run the UI**
```sh
streamlit run ui.py
```
This will launch a **minimal corporate UI** with a chat interface & task automation panel.

---

## **📌 Next Enhancements**
- 📅 **Integrate Google Calendar** for scheduling meetings  
- 📲 **WhatsApp API (Twilio) for messaging**  
- 📊 **Corporate Dashboard UI**  





# References

- https://www.digitalocean.com/resources/articles/agentic-ai
- https://microsoft.github.io/autogen/stable/
- https://microsoft.github.io/autogen/0.2/docs/autogen-studio/getting-started/
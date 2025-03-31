### **🚀 Building a Personal Assistant with AutoGen**  

**AutoGen** is a powerful framework for building multi-agent AI systems that can collaborate, use tools, and automate workflows.  

---

## **🔹 Features of Your AI Personal Assistant**
✅ **Conversational AI** (Chat with natural language)  
✅ **Task Automation** (Email, Meetings, WhatsApp, Web Search)  
✅ **Multi-Agent Collaboration** (Planner, Researcher, Coder Agents)  
✅ **Memory & Context Awareness** (Uses SQLite)  
✅ **FastAPI for Web Interface**  

---

## **1️⃣ Install AutoGen & Dependencies**  
Run the following to install AutoGen & required libraries:  
```sh
pip install pyautogen openai fastapi uvicorn sqlite3 requests
```

---

## **2️⃣ Define Your AutoGen Personal Assistant**
📄 **`assistant.py`**  
```python
import autogen
import sqlite3
import requests

class PersonalAssistant:
    def __init__(self):
        self.model = "gpt-4"  # Replace with "ollama/mistral" for local models
        self.db_path = "memory.db"
        self.init_db()

        self.assistant = autogen.AssistantAgent(
            name="Personal_Assistant",
            llm_config={"model": self.model}
        )

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
        
        response = self.assistant.chat(messages=[{"role": "user", "content": prompt}])
        ai_response = response["content"]
        
        self.save_interaction(user_input, ai_response)
        return ai_response
```

---

## **3️⃣ Build FastAPI Backend**
📄 **`main.py`**  
```python
from fastapi import FastAPI
from pydantic import BaseModel
from assistant import PersonalAssistant

app = FastAPI()
assistant = PersonalAssistant()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    response = assistant.chat(request.message)
    return {"response": response}

@app.get("/")
def root():
    return {"message": "Personal Assistant API is running!"}
```

---

## **4️⃣ Run the API Server**
```sh
uvicorn main:app --reload
```
✅ **Your AI Assistant API is now running on** `http://127.0.0.1:8000/`  

---

## **5️⃣ Extend the Assistant with Task Automation**
**🔹 Add Tool Integration (Send Emails, Schedule Meetings, WhatsApp Messaging)**  

### **📧 Send Emails**
```python
import smtplib

def send_email(to_email, subject, body):
    sender_email = "your-email@example.com"
    password = "your-password"
    
    server = smtplib.SMTP("smtp.example.com", 587)
    server.starttls()
    server.login(sender_email, password)
    
    message = f"Subject: {subject}\n\n{body}"
    server.sendmail(sender_email, to_email, message)
    server.quit()
    return "Email Sent Successfully!"
```

---

## **6️⃣ Create a Minimal UI for the Assistant**
📄 **`ui.py`** (Streamlit-based UI)
```python
import streamlit as st
import requests

st.set_page_config(page_title="AI Personal Assistant", layout="centered")

st.title("💼 Personal AI Assistant")
st.write("Your AI-powered productivity assistant")

user_input = st.text_input("You:", "")

if st.button("Ask AI"):
    response = requests.post("http://127.0.0.1:8000/chat", json={"message": user_input}).json()
    st.text_area("AI:", response["response"], height=200)

st.sidebar.title("Tools")
email_to = st.sidebar.text_input("Recipient Email")
email_subject = st.sidebar.text_input("Email Subject")
email_body = st.sidebar.text_area("Email Content")

if st.sidebar.button("Send Email"):
    email_response = requests.post("http://127.0.0.1:8000/send-email", json={
        "to_email": email_to,
        "subject": email_subject,
        "body": email_body
    }).json()
    st.sidebar.success(email_response.get("status", "Failed"))
```

---

## **7️⃣ Run the UI**
```sh
streamlit run ui.py
```
✅ **You now have a full AI personal assistant with a web UI!**

---

## **📌 Next Enhancements**
- 📅 **Integrate Google Calendar** for meetings  
- 📲 **WhatsApp API (Twilio) for messaging**  
- 📊 **Add Dashboard & Multi-Agent Capabilities**  

Do you need **WhatsApp integration or Google Calendar scheduling** next? 🚀
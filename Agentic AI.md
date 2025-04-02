**Agentic AI** systems involve multiple components that enable autonomy, reasoning, decision-making, and interaction with the environment. Here’s a **breakdown of the key components**:  

<img width="903" alt="image" src="https://github.com/user-attachments/assets/38b131cc-ed69-42cb-805c-bfde739f6d81" />

<img width="599" alt="image" src="https://github.com/user-attachments/assets/33ecd62a-ac96-464b-8553-28da9399f342" />

---

### **🚀 Core Components of Agentic AI**  

#### **1️⃣ Large Language Model (LLM)**
   - The **brain** of the agent.  
   - Generates responses, makes decisions, and processes natural language.  
   - Examples: **GPT-4, Llama 3, Mistral, Claude, Gemini**.

#### **2️⃣ Memory & Context Management**
   - Stores past interactions, facts, or decisions to maintain coherence.  
   - Types:
     - **Short-term memory** (session-level context)
     - **Long-term memory** (vector databases like FAISS, ChromaDB)
     - **Episodic memory** (stores user interactions in time sequences)

#### **3️⃣ Planning & Reasoning Module**
   - Helps the agent **break down complex tasks** into actionable steps.  
   - Approaches:
     - **ReAct (Reasoning + Acting)**
     - **Chain-of-Thought (CoT)**
     - **Tree of Thoughts (ToT)**
     - **Monte Carlo Tree Search (MCTS)** (for decision-based AI)
     - **Self-Reflective Agents** (learn from past mistakes)

#### **4️⃣ Tools & Plugins (External APIs)**
   - Connects to external tools for enhanced capabilities.  
   - Examples:
     - **Web search** (Google API, SerpAPI)
     - **Retrieval-Augmented Generation (RAG)** (FAISS, ChromaDB, Weaviate)
     - **Databases** (PostgreSQL, MongoDB)
     - **Python Execution** (code interpreters for calculations)
     - **Browser Automation** (Selenium, Playwright)

#### **5️⃣ Multi-Agent Collaboration**
   - Some agentic AI systems involve multiple agents with specialized roles.  
   - **Examples**:
     - **CrewAI** → Agents work together (e.g., Researcher, Coder, Tester).
     - **AutoGen** → Enables multiple agents to interact & solve tasks.  
     - **LangGraph** → Graph-based agent workflows.

#### **6️⃣ State Management & Workflow Engine**
   - Helps in **tracking agent states**, tasks, and decision-making.  
   - **LangGraph**: Builds AI workflows with persistent states.  
   - **AutoGen**: Handles multi-agent conversation memory.  

#### **7️⃣ Execution Environment**
   - Defines how the agent runs & interacts with APIs, databases, and other tools.  
   - Example frameworks:
     - **FastAPI / Flask** → API-based agents  
     - **LangChain / LangGraph** → AI agents & tool integration  
     - **AutoGen / CrewAI** → Multi-agent orchestration  
     - **Ray / Dask** → Parallel execution for scalable agents  

#### **8️⃣ Observability & Logging**
   - Tracks agent behavior for debugging & optimization.  
   - **Tools**:
     - OpenTelemetry → Tracing agent actions  
     - LangSmith → Debugging LLM-based agents  
     - Weights & Biases → Monitoring AI agent performance  

---

### **🔥 Example: AI Assistant Architecture**
Imagine a **RAG-based AI Assistant** with **web search & execution tools**:

1️⃣ User asks: _"Summarize the latest AI research on self-supervised learning."_  
2️⃣ **Memory module** checks previous interactions.  
3️⃣ **Retrieval module** (FAISS) fetches relevant research papers.  
4️⃣ **LLM processes the retrieved data** and generates a summary.  
5️⃣ If needed, the **agent calls a web search API** for recent updates.  
6️⃣ Response is refined using **reasoning techniques (ReAct, CoT, ToT)**.  
7️⃣ **Logs & telemetry** capture execution traces for debugging.  

---

Let's build a **LangGraph-based Agentic AI Chatbot** with **Retrieval-Augmented Generation (RAG)** using **LLama3, FAISS, and Google Search API**. 🚀  

---

## **🛠️ Tech Stack**
- **LangGraph** – Graph-based workflow for AI agents  
- **Ollama (Llama3)** – Local LLM for reasoning  
- **FAISS** – Vector database for retrieval caching  
- **Google Search API** – Live web search  
- **FastAPI** – Web interface for chat API  

---

## **📌 Features**
✅ **Conversational AI with Memory** (LangGraph)  
✅ **Retrieval-Augmented Generation (RAG)** (FAISS for context-aware retrieval)  
✅ **Web Search API Integration** (Google Search API)  
✅ **Tool-Calling Agents** (AI can decide whether to use search or memory)  
✅ **FastAPI API Server** (Exposes endpoints for querying the chatbot)  

---

### **📂 Project Structure**
```
agentic-ai-chatbot/
│── models/
│   ├── faiss_store.pkl  # FAISS vector storage
│── agent_chatbot.py     # LangGraph AI agent
│── search_tool.py       # Web search tool (Google API)
│── app.py               # FastAPI server
│── requirements.txt
```

---

## **1️⃣ Install Dependencies**
```bash
pip install langchain langgraph faiss-cpu fastapi uvicorn ollama sentence-transformers requests
```

---

## **2️⃣ Implement FAISS for RAG**
📌 **`faiss_store.py` – Vector database for knowledge retrieval**
```python
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
d = 384  # Embedding size
faiss_index = faiss.IndexFlatL2(d)
query_cache = []  # Stores (query, result) pairs

def get_embedding(text):
    return np.array([embedding_model.encode(text)])

def search_faiss(query):
    """Check if query exists in FAISS cache."""
    if len(query_cache) == 0:
        return None
    query_vector = get_embedding(query)
    D, I = faiss_index.search(query_vector, 1)  # Find closest match
    if D[0][0] < 0.5:  # Similarity threshold
        return query_cache[I[0][0]][1]  # Return cached result
    return None

def store_in_faiss(query, result):
    """Store new query and result in FAISS cache."""
    query_vector = get_embedding(query)
    faiss_index.add(query_vector)
    query_cache.append((query, result))
    with open("models/faiss_store.pkl", "wb") as f:
        pickle.dump(query_cache, f)  # Save cache
```

---

## **3️⃣ Implement Google Search API**
📌 **`search_tool.py` – Fetch live data from Google**
```python
import requests
from faiss_store import search_faiss, store_in_faiss

GOOGLE_API_KEY = "your_google_api_key"
GOOGLE_CSE_ID = "your_google_cse_id"

def web_search(query):
    """Perform web search with FAISS caching."""
    cached_result = search_faiss(query)
    if cached_result:
        print("Returning cached result.")
        return cached_result

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "num": 1}
    response = requests.get(url, params=params)
    data = response.json()

    if "items" in data:
        result = data["items"][0]["snippet"]
        store_in_faiss(query, result)
        return result
    return "No relevant information found."
```

---

## **4️⃣ Build LangGraph AI Agent**
📌 **`agent_chatbot.py` – AI agent with RAG & Web Search**
```python
from langgraph.graph import StateGraph
from langchain.llms import Ollama
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from search_tool import web_search
from faiss_store import search_faiss

class AgentState:
    def __init__(self):
        self.memory = []

def handle_chat(state: AgentState, message: str):
    """Handles AI chat with memory and retrieval"""
    cached_answer = search_faiss(message)
    if cached_answer:
        return AIMessage(content=cached_answer)

    llm = Ollama(model="llama3")
    response = llm.invoke([SystemMessage(content="You are a helpful assistant."),
                           HumanMessage(content=message)])
    state.memory.append((message, response.content))
    return AIMessage(content=response.content)

def handle_search(state: AgentState, message: str):
    """Handles web search"""
    search_result = web_search(message)
    return AIMessage(content=search_result)

# Define LangGraph Workflow
graph = StateGraph(AgentState)
graph.add_node("chat", handle_chat)
graph.add_node("search", handle_search)

graph.set_entry_point("chat")
graph.add_edge("chat", "search")  # If no answer, fetch from search
agent_executor = graph.compile()
```

---

## **5️⃣ Create API Server**
📌 **`app.py` – FastAPI Server**
```python
from fastapi import FastAPI
from agent_chatbot import agent_executor, AgentState

app = FastAPI()

@app.get("/chat")
async def chat(query: str):
    state = AgentState()
    response = agent_executor.invoke(state, query)
    return {"response": response.content}
```

---

## **6️⃣ Run the Chatbot**
```bash
uvicorn app:app --reload
```
API available at: **http://localhost:8000/chat?query=your+question**

---

## **🛠️ Features Breakdown**
- **🔥 LangGraph for AI Workflow** (chat + web search if needed)  
- **📚 FAISS for RAG** (stores retrieved data for context-aware responses)  
- **🔍 Google Search API** (fetches live web results if no cached answer)  
- **🧠 Local Llama3 Model** (fast reasoning without API costs)  
- **🚀 FastAPI Web Server** (for chatbot API access)  

---

### **🚀 Next Steps?**
1️⃣ **Add Multi-Agent Support** (CrewAI for specialized agents)  
2️⃣ **Improve Context Handling** (LangGraph's state persistence)  
3️⃣ **Integrate AutoGen** (for more advanced AI collaboration)  

Let's enhance your **AI chatbot** with **WebSocket streaming** for real-time responses and **Voice-to-Text** for hands-free conversation! 🚀  

---

# **🔥 Enhancements**
✅ **🔴 Streaming Responses** → WebSockets for real-time AI chat  
✅ **🎙️ Voice-to-Text** → Use **Whisper API** for speech recognition  

---

## **1️⃣ Update FastAPI for WebSocket Streaming**
📌 **Modify `app.py` to Support Streaming**
```python
from fastapi import FastAPI, WebSocket
from agent_chatbot import chat_with_ai
import asyncio

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            query = await websocket.receive_text()
            for word in chat_with_ai(query).split():
                await websocket.send_text(word + " ")
                await asyncio.sleep(0.1)  # Simulate real-time response
    except Exception:
        await websocket.close()
```
---

## **2️⃣ Update React Frontend for WebSockets**
📌 **Modify `App.tsx` to Handle Streaming**
```tsx
import React, { useState, useEffect, useRef } from "react";

const App: React.FC = () => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket("ws://localhost:8000/ws");
    ws.current.onmessage = (event) => setResponse((prev) => prev + event.data);
    return () => ws.current?.close();
  }, []);

  const handleChat = () => {
    setResponse(""); // Clear previous response
    ws.current?.send(query);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-900 text-white">
      <h1 className="text-2xl font-bold mb-4">AI Chatbot</h1>
      <input
        className="border p-2 rounded w-96 text-black"
        placeholder="Ask me anything..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button onClick={handleChat} className="mt-2 bg-blue-500 p-2 rounded">
        Send
      </button>
      <div className="mt-4 p-4 bg-gray-800 rounded w-96">{response}</div>
    </div>
  );
};

export default App;
```
---

## **3️⃣ Add Voice-to-Text with Whisper**
📌 **Install OpenAI Whisper API**
```bash
pip install openai
```

📌 **Modify `app.py` to Handle Voice Input**
```python
from fastapi import UploadFile, File
import openai

OPENAI_API_KEY = "your_openai_api_key"

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    audio_data = file.file.read()
    response = openai.Audio.transcribe("whisper-1", audio_data, api_key=OPENAI_API_KEY)
    return {"text": response["text"]}
```
---

## **4️⃣ Add Voice Input in React**
📌 **Modify `App.tsx` to Use Microphone**
```tsx
import React, { useState } from "react";
import axios from "axios";

const App: React.FC = () => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [isRecording, setIsRecording] = useState(false);

  const handleTranscription = async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append("file", audioBlob, "audio.webm");
    const res = await axios.post("http://localhost:8000/transcribe/", formData);
    setQuery(res.data.text);
  };

  const handleVoice = async () => {
    setIsRecording(true);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    const audioChunks: Blob[] = [];

    mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
    mediaRecorder.onstop = () => handleTranscription(new Blob(audioChunks, { type: "audio/webm" }));

    mediaRecorder.start();
    setTimeout(() => {
      mediaRecorder.stop();
      setIsRecording(false);
    }, 5000); // 5 seconds recording
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-900 text-white">
      <h1 className="text-2xl font-bold mb-4">AI Chatbot</h1>
      <input
        className="border p-2 rounded w-96 text-black"
        placeholder="Ask me anything..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button onClick={handleVoice} className="mt-2 bg-green-500 p-2 rounded">
        {isRecording ? "Recording..." : "🎤 Speak"}
      </button>
      <button onClick={() => ws.current?.send(query)} className="mt-2 bg-blue-500 p-2 rounded">
        Send
      </button>
      <div className="mt-4 p-4 bg-gray-800 rounded w-96">{response}</div>
    </div>
  );
};

export default App;
```
---

## **5️⃣ Run Everything**
📌 **Start FastAPI Backend**
```bash
uvicorn app:app --reload
```
📌 **Start React Frontend**
```bash
cd ai-chatbot
npm start
```
🎉 **Visit:** `http://localhost:3000` to chat and use voice input!

---

# **🚀 Final Features**
✅ **Real-Time Streaming Chat with WebSockets**  
✅ **Voice-to-Text Support via OpenAI Whisper**  
✅ **Multi-Agent CrewAI for AI Task Coordination**  
✅ **React Chat UI with Microphone Input**  

---

# **🔥 Next Steps?**
1️⃣ **Deploy to Cloud (AWS, Vercel, Firebase)**  
2️⃣ **Text-to-Speech (TTS) for AI Responses**  
3️⃣ **Mobile App Version (React Native)**  

-----
We'll build a **corporate-themed AI agent** with:  
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

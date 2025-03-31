Great! Let's **extend your AI personal assistant** using **Ollama** for fast and reliable automation. 🚀  

---

### **🛠️ Features for Your AI Personal Assistant**
✅ **Chat & Reasoning** – Fast, conversational responses  
✅ **Task Automation** – Automate daily tasks via commands  
✅ **Code Assistance** – AI-powered coding helper  
✅ **Intent Matching** – Understands and processes user requests  
✅ **Local Execution** – Runs entirely on your MacBook Pro (Intel i9, 16GB RAM)  

---

### **📌 Best Tech Stack**
🔹 **Backend:** FastAPI (lightweight, high-performance)  
🔹 **LLM:** Ollama with **Mistral 7B** (fast, efficient)  
🔹 **Automation:** Python subprocess, APIs, and shell commands  
🔹 **Voice Input (Optional):** `speechrecognition` + `pyaudio`  
🔹 **Local Database (Optional):** SQLite or JSON for storing tasks  

---

## **🚀 Step 1: Install Dependencies**
Run this in **Terminal**:
```bash
pip install fastapi uvicorn ollama openai speechrecognition pyaudio
```

---

## **🚀 Step 2: Create the AI Assistant Server**
Create a new Python file: **`assistant.py`**
```python
from fastapi import FastAPI
import ollama
import subprocess
import speech_recognition as sr

app = FastAPI()

# Load AI Model
MODEL = "mistral"  # Fastest option on CPU (or use 'gemma:2b')

@app.get("/chat")
def chat(query: str):
    """Chat with AI Assistant"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": query}])
    return {"response": response["message"]}

@app.get("/run-command")
def run_command(command: str):
    """Execute system commands (for automation)"""
    try:
        output = subprocess.check_output(command, shell=True, text=True)
        return {"status": "success", "output": output}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/voice-command")
def voice_command():
    """Recognize voice input and respond"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        return chat(command)  # Send voice query to AI chat
    except sr.UnknownValueError:
        return {"response": "Could not understand audio"}
    except sr.RequestError:
        return {"response": "Error with voice recognition service"}

# Run the API Server
# Use: uvicorn assistant:app --host 0.0.0.0 --port 8000
```

---

## **🚀 Step 3: Run the AI Assistant**
Start the API server:
```bash
uvicorn assistant:app --host 0.0.0.0 --port 8000
```

Now, open your **browser** and test the API:
```
http://localhost:8000/chat?query=Hello
```
or  
```
http://localhost:8000/run-command?command=ls
```
or (for voice input)  
```
http://localhost:8000/voice-command
```

---

## **🎯 Next Steps**
1️⃣ Add **task scheduling** (e.g., reminders, calendar events).  
2️⃣ Add **file management** (e.g., read/write files via AI).  
3️⃣ Add **integrations** (e.g., control music, send emails, control smart devices).  
4️⃣ Convert into a **full app with UI** (React Native for mobile).  


---


# Troubleshooting

The error indicates that **`portaudio.h`** is missing, which is required to build `pyaudio`. You need to install **PortAudio** first.

### **🔹 Fix PyAudio Installation on macOS**
#### **1️⃣ Install PortAudio**
Run this command in your terminal:
```bash
brew install portaudio
```

#### **2️⃣ Install PyAudio with Correct Path**
```bash
pip install pyaudio --global-option="build_ext" --global-option="-I$(brew --prefix portaudio)/include" --global-option="-L$(brew --prefix portaudio)/lib"
```
If that fails, try:
```bash
pip install --no-cache-dir pyaudio
```

#### **3️⃣ Verify PyAudio Installation**
```bash
python -c "import pyaudio; print(pyaudio.PyAudio())"
```


---
Killing and 
Running Ollama in different port

The error **"listen tcp 127.0.0.1:11434: bind: address already in use"** means that **another process is already using port 11434**, which is the default port for Ollama.  

### **✅ Fix the Issue**
#### **1️⃣ Check if Ollama is already running**
Run this command to see if an Ollama process is running:
```bash
ps aux | grep ollama
```
If you see an **active Ollama process**, try restarting it:
```bash
ollama stop
ollama start
```

#### **2️⃣ Kill the Process Using Port 11434**
If the port is still occupied, find and kill the process manually:
```bash
lsof -i :11434  # List processes using port 11434
```
You'll see an output like:
```
COMMAND   PID   USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
ollama   12345  user   23u  IPv4  123456  0t0  TCP 127.0.0.1:11434 (LISTEN)
```
Now, kill the process using its PID:
```bash
kill -9 12345  # Replace 12345 with the actual PID
```
Then, restart Ollama:
```bash
ollama start
```

#### **3️⃣ Change the Ollama API Port**
If you want to use a different port, modify the Ollama startup command:
```bash
OLLAMA_HOST=127.0.0.1:12345 ollama serve
```
This runs Ollama on **port 12345** instead of the default **11434**.

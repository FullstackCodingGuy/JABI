# to run this assistant
# uvicorn 2RunAssistant:app --host 0.0.0.0 --port 8000


from fastapi import FastAPI
import ollama
import subprocess
import speech_recognition as sr

app = FastAPI()

# Reuse the Ollama client globally
ollama_client = ollama.Client()

# Load AI Model
MODEL = "mistral"  # Fastest option on CPU (or use 'gemma:2b')

@app.get("/chat")
def chat(query: str):
    """Chat with AI Assistant"""
    response = ollama_client.chat(model=MODEL, 
                           messages=[{"role": "user", "content": query}],
                           options={"num_threads": 4},
                           stream=False
                           )
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

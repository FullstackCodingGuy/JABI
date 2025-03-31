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

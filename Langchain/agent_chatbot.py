import ollama
import requests
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from langgraph.graph import StateGraph, END

# Define request model
class ChatRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, str]] = []

# Define a tool for web search
def web_search(query):
    response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
    return response.json().get("Abstract", "No relevant information found.")

# Define a tool for running Python code
def run_python(code):
    try:
        result = subprocess.run(["python", "-c", code], capture_output=True, text=True)
        return result.stdout.strip() if result.stdout else result.stderr.strip()
    except Exception as e:
        return str(e)

# Ollama-based LLM using Llama3
def query_llama3(prompt):
    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Define the agent logic
def agent_logic(state):
    query = state["query"].lower()
    chat_history = state.get("chat_history", [])

    if "search" in query:
        response = web_search(query.replace("search", "").strip())
    elif "calculate" in query or "run python" in query:
        code = query.replace("calculate", "").replace("run python", "").strip()
        response = run_python(code)
    else:
        response = query_llama3(query)

    chat_history.append({"user": query, "bot": response})  # Store conversation history

    return {"query": query, "response": response, "chat_history": chat_history}  # ✅ Returning dict

# Build the LangGraph (with state schema)
graph = StateGraph("agent_chatbot")
graph.add_node("agent_logic", agent_logic)
graph.set_entry_point("agent_logic")
graph.add_edge("agent_logic", END)
graph_func = graph.compile()

# FastAPI Setup
app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    state = {"query": request.message, "chat_history": request.chat_history}
    final_state = graph_func.invoke(state)
    return {"response": final_state["response"], "chat_history": final_state["chat_history"]}

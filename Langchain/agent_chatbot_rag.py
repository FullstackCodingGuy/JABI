import ollama
import requests
import subprocess
import faiss
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from langgraph.graph import StateGraph, END
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize FastAPI
app = FastAPI()

# Define request model
class ChatRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, str]] = []

# Define tools
def web_search(query):
    response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
    return response.json().get("Abstract", "No relevant information found.")

def run_python(code):
    try:
        result = subprocess.run(["python", "-c", code], capture_output=True, text=True)
        return result.stdout.strip() if result.stdout else result.stderr.strip()
    except Exception as e:
        return str(e)

def query_llama3(prompt):
    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Initialize FAISS vector store for RAG
vector_db_file = "vectorstore.pkl"

def load_vectorstore():
    try:
        with open(vector_db_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def save_vectorstore(vectorstore):
    with open(vector_db_file, "wb") as f:
        pickle.dump(vectorstore, f)

def create_vectorstore():
    """ Create FAISS vector store from sample text documents. """
    docs = [
        "The FIFA World Cup 2022 was won by Argentina, defeating France in the final.",
        "Elon Musk is the CEO of Tesla and SpaceX.",
        "Python is a high-level programming language known for its simplicity and readability.",
    ]
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.create_documents(docs)
    
    # Generate embeddings
    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Save to file
    save_vectorstore(vectorstore)
    return vectorstore

# Load or create vector database
vectorstore = load_vectorstore() or create_vectorstore()

# RAG Retrieval function
def retrieve_context(query):
    """ Retrieve relevant document from the vector database """
    if vectorstore:
        results = vectorstore.similarity_search(query, k=1)
        return results[0].page_content if results else "No relevant information found."
    return "No vector database available."

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
        # Retrieve context using FAISS and use it in the LLM prompt
        context = retrieve_context(query)
        full_prompt = f"Use the following context to answer:\n\n{context}\n\nUser: {query}"
        response = query_llama3(full_prompt)

    chat_history.append({"user": query, "bot": response})

    return {"query": query, "response": response, "chat_history": chat_history}

# Build the LangGraph
graph = StateGraph("agent_chatbot_rag")
graph.add_node("agent_logic", agent_logic)
graph.set_entry_point("agent_logic")
graph.add_edge("agent_logic", END)
graph_func = graph.compile()

@app.post("/chat")
async def chat(request: ChatRequest):
    state = {"query": request.message, "chat_history": request.chat_history}
    final_state = graph_func.invoke(state)
    return {"response": final_state["response"], "chat_history": final_state["chat_history"]}


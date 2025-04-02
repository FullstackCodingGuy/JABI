# Deprecated: The code below is deprecated and will be removed in a future version.
# from langchain_community.llms import Ollama


from langchain_ollama import OllamaLLM

# Load LLaMA 3.2 model via Ollama
llm = OllamaLLM(model="llama3.2")

# Test response
print(llm.invoke("What is Agentic AI?"))

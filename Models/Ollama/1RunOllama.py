# ensure you have already installed ollama if not
# pip install ollama


import ollama

response = ollama.chat(model='mistral', messages=[{"role": "user", "content": "Hello!"}])
print(response["message"])

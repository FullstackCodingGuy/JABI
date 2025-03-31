To run the ollama (in phone)

cd ollama directory (where it is downloaded)

```
./ollama serve &
```

To pull 
```
ollama pull mistral
```

To run
```
ollama run mistral
```
This will automatically if the model not exists locally

To list existing models
```
ollama list
```

### Use Ollama in Python for Automation
If you need Python integration, use Ollama’s Python API:

```
import ollama

response = ollama.chat(model='mistral', messages=[{"role": "user", "content": "Hello!"}])
print(response["message"])

```
✅ Avoids slow Hugging Face runtime and gives fast responses.



### Run Ollama in a Fast API Server (Better than Flask)
To serve responses via API, use FastAPI + Ollama:
```
pip install fastapi uvicorn
```

then create a python server file
```
from fastapi import FastAPI
import ollama

app = FastAPI()

@app.get("/chat")
def chat(query: str):
    response = ollama.chat(model='mistral', messages=[{"role": "user", "content": query}])
    return {"response": response["message"]}

# Run the API
# Command: uvicorn server:app --host 0.0.0.0 --port 8000
```
✅ This runs 4x faster than Flask! 🚀


### Conclusion:
✅ Ollama is the best choice for your MacBook Pro (Intel i9, 16GB RAM).
✅ Mistral 7B runs faster than Gemma on CPU.
✅ Use FastAPI + Ollama Python API for a fast AI assistant.
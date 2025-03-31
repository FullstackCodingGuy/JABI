To deploy the Gemma model locally and serve it via a Flask API, follow these steps:

---

## **1. Set Up Your Environment**
Ensure you have the necessary dependencies installed:

```bash
pip install flask torch transformers sentencepiece
```

---

## **2. Download and Load the Gemma Model**
Use Hugging Face's `transformers` library to download and load the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "google/gemma-2b"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Send input to GPU
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## **3. Create a Flask API**
Create a Flask API to serve inferences:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    response = generate_text(prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

---

## **4. Run the Flask Server**
Save the above Flask code as `app.py` and start the server:

```bash
python app.py
```

---

## **5. Test the API**
Use `curl` or Postman to test the API:

```bash
curl -X POST "http://127.0.0.1:5000/generate" -H "Content-Type: application/json" -d '{"prompt": "Tell me a joke."}'
```

**Expected Response:**
```json
{
  "response": "Why did the chicken cross the road? To get to the other side!"
}
```

---

To ensure that **Gemma-2B** is not downloaded every time you run the program, you can take the following steps:

---

## **1. Use Hugging Face’s Default Cache Directory**
By default, Hugging Face stores models in:
- **Linux/Mac**: `~/.cache/huggingface/`
- **Windows**: `C:\Users\YourUsername\.cache\huggingface\`

If the model is already downloaded, `from_pretrained()` will **load it from cache** instead of re-downloading.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2b"

# Load tokenizer and model (it will use cache if already downloaded)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
```

---

## **2. Set a Custom Cache Directory (Optional)**
If you want to store the model in a custom directory, set the `HF_HOME` environment variable:

```python
import os

# Set a custom cache directory
os.environ["HF_HOME"] = "/your/custom/path"

# Load the model (it will now store in the custom path)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
```

---

## **3. Manually Download & Store the Model**
You can **pre-download** the model and specify a local directory:

```python
# First, download and save the model
MODEL_PATH = "./gemma_model"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", cache_dir=MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", cache_dir=MODEL_PATH)
```

Next time you run the script, it will **load from `./gemma_model/` instead of downloading again**.

---

## **4. Verify Model is Cached**
To check if the model is stored locally, run:

```bash
ls ~/.cache/huggingface/hub
```
or
```bash
ls /your/custom/path
```

---

### **Final Takeaway**
- By default, `from_pretrained()` **does not re-download** if the model is cached.
- If you move machines, **download manually** and specify `cache_dir` or `HF_HOME`.
- To update the model, **delete the cache** and rerun:

```bash
rm -rf ~/.cache/huggingface/hub
```

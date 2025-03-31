from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load the model & tokenizer
MODEL_NAME = "google/gemma-2b"
# MODEL_NAME = "google/gemma-1.1-2b-it" - slow
# MODEL_NAME = "google/gemma-1.1b"



device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
                                          torch_dtype=torch.float32,  # Float32 is better for CPU
                                            device_map="cpu").to(device)

model = torch.compile(model) # Optional: Compile the model for better performance

def generate_response(prompt):
    print("Generating response for prompt:", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=512, num_return_sequences=1, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

print(generate_response("Hello, Gemma!"))

@app.route("/")
def home():
    return "Hello! This is the Gemma API. Use /generate to interact."

@app.route("/generate", methods=["POST"])
def handle_prompt():
    data = request.get_json()
    prompt = data.get("prompt", "")

    print(' Prompt received:', prompt)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    response = generate_response(prompt)
    if not response:
        return jsonify({"error": "Failed to generate response"}), 500

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

model_name = "google/gemma-2b"

# Load the Gemma model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Export to ONNX
onnx_model_path = "gemma-2b.onnx"
ORTModelForCausalLM.from_pretrained(model, save_directory=onnx_model_path, export=True)

print(f"Gemma 2B converted to ONNX at {onnx_model_path}")

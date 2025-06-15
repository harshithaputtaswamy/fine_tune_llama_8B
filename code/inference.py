# code/inference.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

def model_fn(model_dir):
    """
    Loads the model and tokenizer from the model_dir.
    """
    print(f"Loading model from: {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer (saved with transformers==4.40.1, should be compatible)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Load model (make sure it's compatible with your instance type, e.g., bfloat16 for G5)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()

    # Match training behavior for pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model and tokenizer loaded successfully.")
    return {"model": model, "tokenizer": tokenizer, "device": device}

def predict_fn(input_data, model_dict):
    """
    Generates text based on input_data.
    """
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]

    # input_data is typically a dictionary, e.g., {'text': 'Your prompt'}
    # You might need to adjust this based on how you send requests.
    text = input_data.get("text", "")
    if not text:
        raise ValueError("Input data must contain a 'text' key with a non-empty string.")

    # You can add generation parameters from input_data if you want
    max_new_tokens = input_data.get("max_new_tokens", 100)
    do_sample = input_data.get("do_sample", True)
    temperature = input_data.get("temperature", 0.9)
    top_k = input_data.get("top_k", 50)

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id # Important for generation
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {decoded_output}")
    return {"generated_text": decoded_output}

# You can add input_fn and output_fn if you need custom request/response handling,
# but the default JSON handling for text input often works with model_fn/predict_fn.
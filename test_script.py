import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import boto3
import sagemaker

# --- Configuration ---
# Path to your fine-tuned model's local directory
# This should be the directory that contains model.safetensors, config.json, tokenizer.json, etc.
local_model_path = "./model_data" # Replace with your actual local path
# prefix = "huggingface-qlora-mistralai-Mistral-7B--2025-06-13-22-34-50-126/output/model/"  # with trailing slash

# sess = sagemaker.Session()
# s3 = boto3.resource('s3')
# bucket = s3.Bucket(sess.default_bucket())
# for obj in bucket.objects.filter(Prefix=prefix):
#     target = os.path.join(local_model_path, os.path.relpath(obj.key, prefix))
#     os.makedirs(os.path.dirname(target), exist_ok=True)
#     bucket.download_file(obj.key, target)
# print("Download complete")

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load Tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Ensure the tokenizer files are in the specified path and compatible with your transformers version.")
    exit() # Stop if tokenizer fails to load

# --- Load Model ---
try:
    # Configuration for BitsAndBytes (if you trained with quantization)
    bnb_config = None
    if "4bit" in local_model_path or "8bit" in local_model_path: # Adjust based on your model's naming
        print("Assuming quantized model due to path name. Setting up BitsAndBytesConfig.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # Set to True for 4-bit, False for 8-bit or no quantization
            # bnb_4bit_quant_type="nf4", # For 4-bit
            # bnb_4bit_compute_dtype=torch.bfloat16, # Or torch.float16
            # bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, # Use bfloat16 for G5, float32 for CPU/other GPUs
        trust_remote_code=True,
        quantization_config=bnb_config if bnb_config else None,
        # token="", # Only needed if model is private on Hugging Face Hub
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure model weights are in the specified path, compatible with your torch_dtype/quantization, and your libraries are correctly installed.")
    exit() # Stop if model fails to load

# --- Run Inference ---
def wrap_with_legal_prompt(user_question: str) -> str:
    prompt = f"""
        You are a legal expert trained in U.S. immigration law. Given the following legal scenario, provide a detailed and statute-backed response.

        Scenario:
        {user_question.strip()}

        Please cite applicable provisions of the Immigration and Nationality Act (INA), the U.S. Code (8 U.S.C.), or USCIS policy manuals. Explain the legal consequences step by step, including removal eligibility, eligibility for immigration benefits, and any applicable exceptions or discretionary relief.
    """
    return prompt.strip()

prompt = ("If an immigrant is granted Temporary Protected Status (TPS) but later commits a non-violent felony, "
    "what are the legal implications for their TPS eligibility and potential deportation proceedings under current U.S. immigration law?")
print(f"\n--- Generating text for prompt: '{wrap_with_legal_prompt(prompt)}' ---")

inputs = tokenizer(wrap_with_legal_prompt(prompt), return_tensors="pt").to(device)

# Adjust generation parameters as needed
generation_config = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "pad_token_id": tokenizer.eos_token_id,
}

try:
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Generated Text ---")
    print(generated_text)

except Exception as e:
    print(f"Error during text generation: {e}")
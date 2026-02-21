# smoke_test_7b_offload.py
# Smoke test for Qwen2.5-Coder-7B-Instruct using GPU + CPU offload.
# This is intended for low-VRAM GPUs (e.g., 6GB) where full 7B placement on GPU fails.

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Keep a margin below total 6GB VRAM to reduce OOM risk.
MAX_MEMORY = {
    0: "5GiB",
    "cpu": "24GiB",
}

# Offload folder stores temporary CPU/disk offload tensors.
OFFLOAD_DIR = "offload_7b"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# 4-bit quantization + CPU offload allowance for modules that cannot stay on GPU.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True,
)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Loading model with CPU offload...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=MAX_MEMORY,
    offload_folder=OFFLOAD_DIR,
    torch_dtype="auto",
    trust_remote_code=True,
)

print("Model main device:", model.device)
if hasattr(model, "hf_device_map"):
    print("Device map:", model.hf_device_map)

prompt = "Write a short Python function that checks whether a string is a palindrome."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

print("\n=== Generated text ===\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))

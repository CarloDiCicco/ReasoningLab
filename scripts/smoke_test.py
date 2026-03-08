# run_qwen_smoke_test.py
# Minimal local inference smoke test for Qwen2.5-Coder.
# Uses 4-bit quantization so the model can fit on 6GB VRAM.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from reasoninglab.models.qwen_local import _PresencePenaltyLogitsProcessor
from transformers.generation.logits_process import LogitsProcessorList

# Model ID on the Hugging Face Hub (smaller 3B model for 6GB VRAM)
# Qwen/Qwen2.5-Coder-3B-Instruct
# Qwen/Qwen3-4B-Instruct-2507
MODEL_ID = "Qwen/Qwen3.5-4B"

# 4-bit quantization config (requires bitsandbytes)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype="auto",
    trust_remote_code=True,
)

# Explicit GPU verification
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Model device:", model.device)
else:
    print("Model device:", model.device)
    print("WARNING: CUDA not available. The model will run on CPU.")

# Simple prompt — apply chat template with thinking disabled.
prompt = "Write a Python function that checks if a string is a palindrome."
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

# Tokenize and move tensors to the right device
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

# Generate a short response
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
        logits_processor=LogitsProcessorList([_PresencePenaltyLogitsProcessor(1.0, inputs["input_ids"].shape[1])]),
    )

# Decode and print result
print(tokenizer.decode(output[0], skip_special_tokens=True))

"""Inspect LCB/3384: print full prompt, then generate with Qwen3.5 and print full output.

Usage:
    python scripts/debug_thinking.py
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3.5-4B" # Qwen/Qwen3-4B-Instruct-2507
MAX_NEW_TOKENS = 4096
TASK_ID = "LCB/3384"

# ── Load prompt ───────────────────────────────────────────────────────────────
with open("data/lcb_leetcode.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        if record["task_id"] == TASK_ID:
            prompt = record["prompt"]
            test_code = record["test_code"]
            break
    else:
        raise RuntimeError(f"{TASK_ID} not found in dataset")

print("=" * 70)
print(f"TASK: {TASK_ID}")
print("=" * 70)
print("PROMPT SENT TO MODEL:")
print("-" * 70)
print(prompt)
print("-" * 70)
print(f"TEST CODE:\n{test_code}")
print("=" * 70)
print()

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype="auto",
    trust_remote_code=True,
)
model.eval()

# ── Generate ──────────────────────────────────────────────────────────────────
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
prompt_tokens = inputs["input_ids"].shape[1]

print(f"prompt_tokens={prompt_tokens}, max_new_tokens={MAX_NEW_TOKENS}")
print("Generating...")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

new_ids = output_ids[0][prompt_tokens:]
completion_tokens = new_ids.shape[0]
full_output = tokenizer.decode(new_ids, skip_special_tokens=True)

print()
print("=" * 70)
print(f"MODEL OUTPUT  (completion_tokens={completion_tokens}, hit_cap={completion_tokens >= MAX_NEW_TOKENS})")
print("=" * 70)
print(full_output)
print("=" * 70)

# ── Analysis ──────────────────────────────────────────────────────────────────
lines = full_output.splitlines()
comment_lines = [l for l in lines if l.strip().startswith("#")]
code_lines    = [l for l in lines if l.strip() and not l.strip().startswith("#")
                 and not l.strip().startswith("```")]
print(f"\nSTATISTICS:")
print(f"  total lines     : {len(lines)}")
print(f"  comment lines   : {len(comment_lines)}")
print(f"  code lines      : {len(code_lines)}")
print(f"  comment ratio   : {len(comment_lines)/max(len(lines),1):.1%}")
print(f"  truncated       : {completion_tokens >= MAX_NEW_TOKENS}")

"""
scripts/smoke_qwen_local.py
────────────────────────────
Manual smoke test for models/qwen_local.py.

Verifies that QwenModel loads correctly, generates non-empty text, and
returns sensible token counts and timing.  Also checks that hidden_states
are None for a normal H1 call and are a list of arrays when requested.

Run from the repo root:
    python scripts/smoke_qwen_local.py

Change MODEL_ID below to test a different model variant.
"""

import sys
from pathlib import Path

import torch

# Allow running directly from the repo root without pip install.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reasoninglab.config.schema import GenerationConfig
from reasoninglab.models.qwen_local import QwenModel

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"  # swap to test 4B or 7B

GEN_CONFIG = GenerationConfig(
    max_new_tokens=128,
    do_sample=False,  # greedy decoding: deterministic, reproducible
)

# A simple code-completion prompt similar to what the policies will construct.
PROMPT = (
    "Write a Python function that checks if a string is a palindrome.\n"
    "Return True if it is, False otherwise.\n\n"
    "def is_palindrome(s: str) -> bool:\n"
)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Model          : {MODEL_ID}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading model (first run downloads weights; subsequent runs are cached)...")
    model = QwenModel(MODEL_ID, GEN_CONFIG)
    print(f"Device map     : {model.hf_device_map}\n")

    # ── H1 generate (no hidden states) ───────────────────────────────────────
    print("Running generate() [H1 mode, return_hidden_states=False]...")
    out = model.generate(PROMPT, return_hidden_states=False)

    print("\n--- GenerationOutput ---")
    print(f"  prompt_tokens    : {out.prompt_tokens}")
    print(f"  completion_tokens: {out.completion_tokens}")
    print(f"  elapsed_s        : {out.elapsed_s:.3f}s")
    print(f"  hidden_states    : {out.hidden_states}")
    print("\n--- Generated text ---")
    print(out.text)
    print("----------------------\n")

    # Basic H1 sanity checks
    assert out.text.strip(),         "FAIL: generated text is empty"
    assert out.prompt_tokens > 0,    "FAIL: prompt_tokens is 0"
    assert out.completion_tokens > 0, "FAIL: completion_tokens is 0"
    assert out.elapsed_s > 0,        "FAIL: elapsed_s is 0"
    assert out.hidden_states is None, "FAIL: hidden_states should be None for H1"
    print("H1 checks passed.\n")

    # ── H2 generate (with hidden states) ─────────────────────────────────────
    print("Running generate() [H2 mode, return_hidden_states=True]...")
    out_h2 = model.generate(PROMPT, return_hidden_states=True)

    assert out_h2.hidden_states is not None,   "FAIL: hidden_states is None for H2"
    assert len(out_h2.hidden_states) > 0,      "FAIL: hidden_states list is empty"
    assert out_h2.hidden_states[0].ndim == 1,  "FAIL: expected 1D array per layer"
    print(f"H2 hidden_states : {len(out_h2.hidden_states)} layers, "
          f"dim={out_h2.hidden_states[0].shape[0]}")
    print("H2 checks passed.\n")

    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
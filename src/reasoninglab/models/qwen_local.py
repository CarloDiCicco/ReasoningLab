"""
models/qwen_local.py
────────────────────
Local Qwen inference adapter.

Loads a Qwen model once in __init__ and runs inference through generate().
All supported models are loaded with 4-bit NF4 quantization (bitsandbytes)
to fit within the RTX 4050 6GB VRAM budget.

Supported models (from README):
  Qwen/Qwen2.5-Coder-3B-Instruct  — 3B, device_map="auto", no offload
  Qwen/Qwen3.5-4B                 — 4B, device_map="auto", no offload  [default]
  Qwen/Qwen3-4B-Instruct-2507     — 4B, device_map="auto", no offload  [previous default]
  Qwen/Qwen2.5-Coder-7B-Instruct  — 7B, device_map="auto" + max_memory + offload

Public API:
  GenerationOutput — result of one model.generate() call
  QwenModel        — model wrapper: loads once, generates many times

H2 preparation:
  generate(return_hidden_states=True) runs an additional forward pass over the
  prompt and returns the last-token hidden state for each transformer layer.
  This is a no-op for H1 (default: False) and adds no overhead in normal runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from reasoninglab.config.schema import GenerationConfig


# ─────────────────────────────────────────────────────────────────────────────
# Custom logits processor: presence penalty
# ─────────────────────────────────────────────────────────────────────────────

class _PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Flat additive penalty on tokens already seen anywhere in the sequence.

    Implements the same semantics as OpenAI's / vLLM's `presence_penalty`:
    for every token that appears at least once in input_ids, subtract `penalty`
    from its logit.  This discourages the model from repeating tokens it has
    already used, regardless of how many times they appeared (unlike the
    multiplicative `repetition_penalty` which scales with frequency).

    penalty=0.0 is a no-op.  Typical effective range: 0.5–2.0.
    Qwen3.5-4B model card recommends 1.5 for non-thinking general tasks.
    """

    def __init__(self, penalty: float, prompt_length: int) -> None:
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i, seq in enumerate(input_ids):
            # Only penalize tokens that appeared in the *generated* portion.
            # Penalizing prompt tokens would suppress unavoidable Python keywords
            # (def, return, for, if, self...) from the very first generated token.
            generated_ids = seq[self.prompt_length:]
            unique_tokens = generated_ids.unique()
            scores[i, unique_tokens] -= self.penalty
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Lowercase substrings that identify a 7B model from its model_id.
# 7B models need a different BitsAndBytesConfig and offload-capable load kwargs.
_7B_SUBSTRINGS: tuple[str, ...] = ("7b",)

# Offload folder for 7B weight chunks spilled from VRAM to disk.
# Must match the offload_7b/ folder already present in the repo root.
_OFFLOAD_FOLDER = "offload_7b"

# Memory caps for 7B loading — matches the tested settings in smoke_test_7b_offload.py.
# Leaving ~1 GiB headroom below 6 GiB prevents OOM during generation.
_7B_GPU_MEMORY = "5GiB"
_7B_CPU_MEMORY = "24GiB"


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GenerationOutput:
    """Result of one QwenModel.generate() call.

    frozen=True: immutable after creation, consistent with ExecutionResult
    and AttemptRecord.
    """

    # Generated text with the input prompt stripped.
    # This is the candidate code string passed to executor.execute_candidate().
    text: str

    # Number of tokens in the input prompt (before generation).
    prompt_tokens: int

    # Number of newly generated tokens (completion only, prompt not counted).
    # The runner computes AttemptRecord.tokens as prompt_tokens + completion_tokens.
    completion_tokens: int

    # Wall-clock seconds from the first token generation to the last.
    # Measures GPU inference time only (tokenization excluded).
    # The runner combines this with executor.elapsed_s to get the full
    # attempt latency stored in AttemptRecord.elapsed_s.
    elapsed_s: float

    # H2 preparation: last-token hidden state for selected transformer layers.
    # Dict mapping layer index (int) to numpy array of shape (hidden_dim,).
    # Contains layer 0 (embedding) + last 8 transformer layers.
    # None when generate() is called with return_hidden_states=False (H1 default).
    hidden_states: dict | None = None  # dict[int, np.ndarray] when populated


# ─────────────────────────────────────────────────────────────────────────────
# Model class
# ─────────────────────────────────────────────────────────────────────────────

class QwenModel:
    """Local Qwen inference wrapper.

    The tokenizer and model are loaded once in __init__ and reused across all
    generate() calls.  This avoids paying the ~20 second model-load cost on
    every inference call — critical when running 50+ tasks × B attempts each.

    Args:
        model_id:          HuggingFace Hub model ID.
        generation_config: Decoding parameters from ExperimentConfig.generation.
    """

    def __init__(self, model_id: str, generation_config: GenerationConfig) -> None:
        self.model_id = model_id
        self.generation_config = generation_config

        # ── Tokenizer ────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

        # ── 4-bit quantization config ─────────────────────────────────────────
        # NF4 (Normal Float 4-bit) reduces each weight from 16-bit to ~4 bits.
        # 3B model: ~6 GiB float16 → ~1.5 GiB; 4B: ~8 GiB → ~2 GiB.
        # Both fit comfortably within the RTX 4050's 6 GiB VRAM.
        # double_quant compresses the quantization constants themselves for
        # a small additional memory saving.
        #
        # 7B models use the same quantization base plus one extra flag:
        # llm_int8_enable_fp32_cpu_offload=True allows Accelerate to offload
        # modules that do not fit on GPU to CPU RAM in float32.  Without this
        # flag the 7B model may fail to load on 6 GiB VRAM even with 4-bit.
        if self._is_7b(model_id):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,  # required for 7B offload
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        # ── Model load kwargs — 7B needs offload settings ─────────────────────
        load_kwargs: dict = {
            "device_map":          "auto",
            "quantization_config": bnb_config,
            "torch_dtype":         "auto",
            "trust_remote_code":   True,
        }
        if self._is_7b(model_id):
            # max_memory tells device_map="auto" the VRAM and RAM budget.
            # Modules that do not fit on GPU are offloaded to CPU RAM, and
            # then to offload_folder on disk if CPU RAM is also exhausted.
            load_kwargs["max_memory"]     = {0: _7B_GPU_MEMORY, "cpu": _7B_CPU_MEMORY}
            load_kwargs["offload_folder"] = _OFFLOAD_FOLDER

        # ── Load model ────────────────────────────────────────────────────────
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        # eval() disables dropout layers — required for deterministic greedy
        # decoding and for reproducible hidden-state captures in H2.
        self.model.eval()

        # ── Store device map for experiment logging ────────────────────────────
        # RuntimeConfig.capture_device_map=True means the runner logs this dict
        # to verify whether the model ran fully on GPU or used offload.
        # {"": 0} means all weights on GPU 0 (no offload used).
        # A mixed dict means some layers were placed on CPU or disk.
        self.hf_device_map: dict = dict(
            getattr(self.model, "hf_device_map", None) or {}
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        return_hidden_states: bool = False,
    ) -> GenerationOutput:
        """Generate a completion for the given prompt.

        Args:
            prompt:               Raw input string (task description / code prefix).
                                  The caller (policy) is responsible for formatting.
            return_hidden_states: If True, also run a forward pass to capture
                                  per-layer last-token hidden states for H2 probing.
                                  Keep False for all H1 runs — adds no overhead.

        Returns:
            GenerationOutput with text, token counts, latency, and optionally
            hidden states.
        """
        # Format as a chat message and apply the model's chat template.
        # enable_thinking=False suppresses Qwen3/3.5 <think>...</think> output,
        # which would otherwise consume most of max_new_tokens and truncate
        # the actual code.  For Qwen2.5 models the flag is silently ignored.
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_tokens: int = inputs["input_ids"].shape[1]

        # Build generation kwargs.
        # temperature and top_p are only passed when do_sample=True.
        # Passing temperature=0.0 with do_sample=False triggers a warning in
        # some transformers versions and has no effect on greedy decoding.
        # Build presence_penalty logits processor if penalty > 0.
        # HF generate() does not natively support presence_penalty (only vLLM/SGLang do),
        # so we implement it as a custom LogitsProcessor passed via logits_processor.
        logits_processors = LogitsProcessorList()
        if self.generation_config.presence_penalty > 0.0:
            logits_processors.append(
                _PresencePenaltyLogitsProcessor(self.generation_config.presence_penalty, prompt_tokens)
            )

        gen_kwargs: dict = {
            "max_new_tokens":     self.generation_config.max_new_tokens,
            "do_sample":          self.generation_config.do_sample,
            # Qwen models may not have pad_token_id set; eos_token_id is the
            # standard fallback to suppress a padding-related warning.
            "pad_token_id":       self.tokenizer.eos_token_id,
            # Multiplicative penalty (HF native). 1.0 = no penalty.
            "repetition_penalty": self.generation_config.repetition_penalty,
            # Additive flat penalty via custom processor. No-op when list is empty.
            "logits_processor":   logits_processors,
        }
        if self.generation_config.do_sample:
            gen_kwargs["temperature"] = self.generation_config.temperature
            gen_kwargs["top_p"]       = self.generation_config.top_p
            gen_kwargs["top_k"]       = self.generation_config.top_k
            gen_kwargs["min_p"]       = self.generation_config.min_p

        # Run generation and measure wall-clock inference time.
        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - start

        # output_ids contains prompt tokens followed by completion tokens.
        # Strip the prompt portion so we return only the newly generated text.
        new_ids = output_ids[0][prompt_tokens:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        completion_tokens: int = new_ids.shape[0]

        # H2: capture per-layer hidden states via a separate prompt forward pass.
        # We do NOT use output_hidden_states inside model.generate() because
        # that returns states for every generated token (very large, unnecessary).
        # A single forward pass over the prompt captures the model's internal
        # representation of the problem before generation begins — exactly what
        # H2's linear probe will classify as pass vs fail.
        hidden_states = None
        if return_hidden_states:
            hidden_states = self._get_prompt_hidden_states(inputs)

        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_s=elapsed,
            hidden_states=hidden_states,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_prompt_hidden_states(self, inputs: dict) -> dict:
        """Run one forward pass over the prompt and extract last-token hidden states.

        Called only when return_hidden_states=True (H2 runs).

        We save only layer 0 (embedding output) and the last 8 transformer
        layers to avoid storing 37 × 2560-dim vectors per datapoint.  Late
        layers carry the strongest task-level signal for linear probing.

        Returns:
            dict mapping layer index (int) to np.ndarray of shape (hidden_dim,).
            Keys: {0, N-7, N-6, ..., N} where N = num_hidden_layers.
        """
        import numpy as np  # deferred import — only needed for H2 runs

        with torch.no_grad():
            forward_out = self.model(**inputs, output_hidden_states=True)

        # forward_out.hidden_states: tuple of (num_layers + 1) tensors.
        # Index 0 = embedding output, indices 1..N = transformer block outputs.
        # Each tensor has shape (batch=1, seq_len, hidden_dim).
        all_hs = forward_out.hidden_states
        num_layers = len(all_hs)  # num_hidden_layers + 1 (for embedding)

        # Select layer 0 (embedding) + last 8 transformer layers.
        selected_indices = [0] + list(range(max(1, num_layers - 8), num_layers))

        result: dict = {}
        for idx in selected_indices:
            # .float() converts bfloat16/float16 to float32 for numpy compat.
            vec: np.ndarray = all_hs[idx][0, -1, :].float().cpu().numpy()
            result[idx] = vec

        return result

    @staticmethod
    def _is_7b(model_id: str) -> bool:
        """Return True if model_id identifies a 7B model requiring offload settings."""
        return any(kw in model_id.lower() for kw in _7B_SUBSTRINGS)

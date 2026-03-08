# ReasoningLab

## Goal
Build a reproducible framework to improve, measure and study LLM “reasoning” capabilities.

## Why
Show ML Research Engineer-grade capability:
- controlled experimentation (ablations, deterministic runs)
- objective evaluation (unit tests / checkers)
- budget-aware tradeoffs (accuracy vs tokens/latency)
- lightweight interpretability (internals-backed hypothesis tests)

## Deliverables
- CLI to run experiments with configs
- results folder with plots + tables
- short technical writeup explaining findings + failure modes

## Computational Resoruces
- Personal Computer (Laptop) with:
    - 13th Gen Intel(R) Core(TM) i9-13900H
    - NVIDIA GeForce RTX 4050 6GB VRAM GPU

## Qwen Runtime Settings (3B vs 7B)
- `Qwen2.5-Coder-3B-Instruct`, `Qwen/Qwen3-4B-Instruct-2507`, and `Qwen/Qwen3.5-4B`: run normally with `device_map="auto"` on this machine (RTX 4050 6GB), without CPU offload settings.
- `Qwen2.5-Coder-7B-Instruct`: should be loaded with offload-capable settings (`device_map="auto"` + `max_memory` + offload enabled), even when CPU offload is not always used at runtime.
- Reason: `device_map="auto"` uses a conservative placement policy. For 7B, it may decide to place some modules on CPU/disk depending on current VRAM state; if offload is not enabled, loading can fail.
- If the printed device map is `{"": 0}`, all model weights are on GPU for that run (offload was allowed, but not used).
- For experiment stability, keep 7B in offload-capable mode and always log `hf_device_map` for each run.

## Loop 1 (L1): Budgeted Repair vs Best-of + Mechanistic Probing

### Purpose
Build a **rigorous, budget-controlled evaluation harness** for code reasoning (objective unit tests), then use it to:
1) establish a **minimal efficiency frontier** for fixed inference-time policies under a call budget (**H1**),
2) probe whether internal activations carry a **decodable correctness signal** (**H2**), and
3) use that signal to implement a **minimal adaptive controller** that allocates compute online (stop/branch/repair) (**H3**).

---

### Hypotheses

**H1 (baseline systems, minimal).**  
Under a fixed call budget **B**, **best-of-B** and **repair-B** define different efficiency frontiers (pass@1 vs calls/tokens/latency) and produce different failure-mode profiles (syntax/runtime/assertion/timeout).  
*Note: H1 is intentionally “thin”: it exists to validate the harness + generate trajectories for H2/H3.*

**H2 (predictive / mechanistic).**  
There exists a **linearly decodable correctness signal** in frozen model activations that predicts **pass vs fail** above surface-only signals (logprob/entropy). The strongest signal is captured by a **small, well-defined representation** in late layers (e.g., last-token vs pooled-token representations) and should generalize across **prompt paraphrases** and task variants.

**H3 (policy, minimal controller).**  
A **rule-based controller** that uses **external verifier feedback** + **H2 probe score** to allocate compute online (stop/branch/repair) achieves a better budgeted frontier than fixed policies (best-of-B, repair-B) under the same **B**.  
*Learning/RL for the controller is deferred to a later loop.*

---

### Experimental Design (shared for H1–H3)

#### Task setup
- Domain: **coding problems with deterministic unit tests** (objective verification).
- Each run logs full trajectories: prompts, code, test results, errors, tokens, latency, and (when enabled) probe scores.
- **Benchmark**: LiveCodeBench release_v6, LeetCode easy/medium, filtered to problems after 2024-05-01 (161 tasks). Note: LiveCodeBench v6 only covers up to ~Apr 2025; Qwen3.5-4B's training cutoff (~Sep 2025, estimated) post-dates the entire dataset, so contamination-free filtering is not feasible. All policy arms are equally affected, so relative comparisons remain valid.

#### Budget definition
- **B = number of model calls per task** (primary budget for V1).
- Secondary accounting: total tokens and latency per task (reported, not optimized first).

#### Policy arms (all matched to the same B)
1) **Baseline (1×):** single attempt → unit tests.  
2) **Best-of-B:** generate B independent candidates → run unit tests → select first passing candidate (or best available).  
3) **Repair-B:** 1 initial attempt → on failure, iterate (B−1) repair steps using verifier feedback → final output.  
4) **Controller-B (H3):** adaptive policy that at each step chooses among {stop, branch, repair} based on verifier feedback + probe score, while never exceeding B calls.

#### Metrics
- **pass@1** (primary)  
- Tokens/task, latency/task  
- Failure taxonomy distribution: syntax, runtime error, assertion fail, timeout  
- Controller diagnostics (H3): avg calls used; branch vs repair frequency; early-stop rate

---

### Mechanistic Probe (H2)

#### Labels
- Ground truth label per attempt: **pass vs fail** from unit tests.

#### Features (minimal representations to compare)
- **Last-token** hidden state (at a chosen layer)  
- **Pooled tokens** hidden state (e.g., mean over last K code tokens; optionally mean over all code tokens)

#### Probe model
- **Linear probe** (logistic regression) predicting pass vs fail.  
- Report: AUC / accuracy (+ optional calibration).

#### Key tests (what makes H2 non-trivial)
- **Trajectory test:** for repair trajectories that end in a pass, probe score should **increase across repair steps** (step 0 → step 1 → …) relative to trajectories that never pass.  
- **Generalization test:** probe trained on one prompt format should retain lift on **paraphrased prompts** and mild task variants.

---

### Controller Policy (H3): Minimal Implementation

- Inputs per step:
  - latest verifier outcome (error type + failing tests, if available)
  - probe score on the current attempt (and optionally its change vs previous step)
- Actions:
  - **Stop** early if probe score is high (emit current solution)
  - **Branch** (sample another candidate) if uncertain and budget remains
  - **Repair** immediately if likely fail and verifier feedback is informative
- Constraint: never exceed **B** calls per task.

*Optional stretch:* tune thresholds on a held-out split (no RL; just threshold selection).

---

### Optional (Post-V1) Mechanistic Causal Check
Ablation-based causal checks are **not required for L1 shipping**, but can be added after H3:
- Select signal-rich layer(s)/head(s) from H2.  
- Apply minimal ablation during generation (e.g., zero a small head set or a layer output).  
- Compare Δpass@1 under ablation for **Controller-B / Repair-B / Best-of-B**.

---

### L1 Deliverables
- **Reproducible Runner:** CLI tool with config/seed support producing structured JSONL logs.  
- **Baseline Frontier (H1):** pass@1 vs B (and tokens/latency) for Baseline, Best-of-B, Repair-B + failure taxonomy.  
- **Probe Report (H2):** best layers/representations + AUC/accuracy + trajectory and paraphrase generalization plots.  
- **Controller Report (H3):** Controller-B vs fixed policies under equal B, including compute allocation diagnostics.  
- **Qualitative Error Analysis:** case studies of stubborn failures (non-converging repairs, misleading verifier feedback, brittle “test-passing” hacks).

---

## Loop 2 (L2): Verifier-Driven Post-Training (SFT + DPO) on Small Qwen3

### Purpose
Extend the same harness into a **true post-training loop** (data → train → eval) by using the verifier to generate **high-signal supervision**:
1) create SFT and preference datasets from **policy trajectories** (best-of/repair/controller),  
2) run **parameter-efficient fine-tuning** (QLoRA/LoRA) on a **small model** (e.g., Qwen3-0.6B or Qwen3-1.7B or Qwen3 4B by renting some GPUs), and  
3) measure whether post-training improves **budgeted reliability** and/or reduces the need for expensive inference-time search.

---

### Hypotheses IDEAs (to reviewed and refined)

**H4 (post-training lift).**  
Verifier-driven **SFT** improves pass@1 on the held-out test set; adding **DPO** further improves pass@1 and reduces failure modes that persist under pure SFT.

**H5 (compute substitution).**  
A post-trained small model shifts the frontier so that **lower B** (fewer calls) achieves comparable pass@1 to the base model with higher B (i.e., training substitutes for inference-time search/repair).

**H6 (interaction with H2/H3).**  
Post-training changes the internal correctness signal: probe separability and/or trajectory monotonicity improves, enabling more reliable controller decisions.

---

## Current Repo Structure

```text
src/
  reasoninglab/
    __init__.py                      
    cli.py                           # CLI entrypoint to run/evaluate H1 experiments.
    config/
      __init__.py                    
      schema.py                      # Pydantic schemas for validating experiment configuration.
    models/
      __init__.py                    .
      qwen_local.py                  # Local Qwen inference adapter (generation API).
    tasks/
      __init__.py                    
      schema.py                      # Task data contracts (task_id, prompt, tests, timeout, etc.).
      loaders.py                     # Task file loading/parsing into validated task objects.
    verify/
      __init__.py                    
      executor.py                    # Executes candidate code/tests with timeout and returns raw outcomes.
      taxonomy.py                    # Maps raw failures to normalized labels (syntax/runtime/assertion/timeout).
    policies/
      __init__.py                    
      baseline.py                    # Single-attempt policy (1 call).
      best_of_b.py                   # Best-of-B independent sampling policy.
      repair_b.py                    # Iterative repair policy under budget B.
    eval/
      __init__.py                    
      runner.py                      # Main orchestration loop across tasks/policies/budget.
      metrics.py                     # Computes pass-rate/cost/failure metrics from run outputs.
    io/
      __init__.py                    # Declares I/O utilities subpackage.
      jsonl_logger.py                # Writes structured run logs (attempt-level JSONL + summaries).

configs/
  h1/
    smoke.yaml                       # Minimal config for quick end-to-end smoke validation.

scripts/
  smoke_test.py                      # Direct local smoke test for Qwen3.5-4B runtime.
  smoke_test_7b_offload.py          # Direct local smoke test for Qwen2.5-7B offload runtime.

tests/
  test_budget_accounting.py          # Verifies policies never exceed call budget B.
  test_metrics.py                    # Verifies metric computations and edge cases.
  test_policy_parity.py              # Verifies fairness/parity invariants across policies.
  test_runner_integration.py         # End-to-end wiring test of runner with fake components.
  test_taxonomy.py                   # Verifies failure-type classification logic.
```

## Minimal Run-Loop Skeleton (File Call Flow)

```text
cli.py
  -> config/schema.py (validate config)
  -> tasks/loaders.py + tasks/schema.py (load + validate tasks)
  -> eval/runner.py (orchestrate experiment)
      -> policies/{baseline|best_of_b|repair_b}.py (choose policy behavior)
          -> models/qwen_local.py (generate candidate code)
          -> verify/executor.py (run tests)
          -> verify/taxonomy.py (classify failure type)
          -> io/jsonl_logger.py (append attempt record)
      -> eval/metrics.py (aggregate run metrics)
      -> io/jsonl_logger.py (write final summary)
```
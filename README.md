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
- `Qwen2.5-Coder-3B-Instruct` and `Qwen/Qwen3-4B-Instruct-2507`: run normally with `device_map="auto"` on this machine (RTX 4050 6GB), without CPU offload settings.
- `Qwen2.5-Coder-7B-Instruct`: should be loaded with offload-capable settings (`device_map="auto"` + `max_memory` + offload enabled), even when CPU offload is not always used at runtime.
- Reason: `device_map="auto"` uses a conservative placement policy. For 7B, it may decide to place some modules on CPU/disk depending on current VRAM state; if offload is not enabled, loading can fail.
- If the printed device map is `{"": 0}`, all model weights are on GPU for that run (offload was allowed, but not used).
- For experiment stability, keep 7B in offload-capable mode and always log `hf_device_map` for each run.

## Loop 1 (L1): Budgeted Repair vs Best-of + Mechanistic Probing

### Purpose
Build a **rigorous, budget-controlled evaluation harness** for code reasoning (objective unit tests), then use it to:
1) compare two inference-time policies under a fixed call budget (**H1**),  
2) probe whether internal activations carry a **decodable correctness signal** (**H2**), and  
3) run a minimal **causal check** via ablations tied to the probe (**H3**).

---

### Hypotheses

**H1 (policy, fixed budget).**  
Under a fixed call budget **B**, an iterative **repair loop** (generate → test → patch → retest) achieves higher **pass@1** than **best-of-B** sampling, and shows a different failure-mode profile (syntax/runtime/assertion/timeout).

**H2 (predictive / mechanistic).**  
There exists a **linearly decodable correctness signal** in model activations that predicts **pass vs fail** above chance. The strongest signal is captured by a **small, well-defined representation** in late layers (e.g., last-token vs pooled-token representations).

**H3 (causal follow-up, minimal).**  
Ablating the **layer/heads** identified as most “signal-rich” by **H2** reduces performance **more for repair than for best-of**, indicating repair relies on specific internal computation.

---

### Experimental Design (shared for H1–H3)

#### Task setup
- Domain: **coding problems with deterministic unit tests** (objective verification).
- Each run logs full trajectories: prompts, code, test results, errors, tokens, latency.

#### Budget definition
- **B = number of model calls per task** (primary budget for V1).
- Optional secondary accounting: total tokens and latency per task.

#### Policy arms (all matched to the same B)
1) **Baseline (1×):** single attempt → unit tests.  
2) **Best-of-B:** generate B independent candidates → run unit tests → select first passing candidate (or best available).  
3) **Repair-B:** 1 initial attempt → on failure, iterate (B−1) repair steps using verifier feedback → final output.

#### Metrics
- **pass@1** (primary)
- **pass@B** (secondary, policy-dependent meaning)
- Tokens/task, latency/task
- Failure taxonomy distribution: syntax, runtime error, assertion fail, timeout

---

### Mechanistic Probe (H2)

#### Labels
- Ground truth label per attempt: **pass vs fail** from unit tests.

#### Features (minimal representations to compare)
- **Last-token** hidden state (at a chosen layer)
- **Pooled tokens** hidden state (e.g., mean over last K code tokens; optionally mean over all code tokens)

#### Probe model
- **Linear probe** (logistic regression) predicting pass vs fail.
- Report: AUC / accuracy + calibration (optional).

#### Key test (link to H1)
- For repair trajectories that end in a pass, test whether the probe score tends to **increase across repair steps** (step 0 → step 1 → …), compared to trajectories that never pass.

---

### Causal Check via Ablation (H3)

- Select the layer/heads with highest predictive signal from **H2**.
- Apply a minimal intervention during generation:
  - ablate (zero) a small subset of heads, or ablate one layer output (small scope).
- Compare Δpass@1 under ablation for:
  - **Repair-B** vs **Best-of-B**
- H3 expectation: ablation causes a **larger drop** for repair than for best-of.

---

### L1 Deliverables
- **Reproducible Runner:** CLI tool with config/seed support producing structured JSONL logs.
- **Quantitative Summary:** Tables and plots for pass@1 vs Budget (B), and Compute Cost (tokens/latency).
- **Failure Taxonomy:** Distribution analysis of error types (Syntax vs. Logic) across policy arms.
- **Probe Report (H2):** Identification of signal-rich layers/representations + trajectory shift plots.
- **Ablation Report (H3):** Measured Δpass@1 showing the causal necessity of identified circuits.
- **Qualitative Error Analysis:** Case studies of "stubborn" failures where the model enters repair loops but fails to converge.

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
  smoke_test.py                      # Direct local smoke test for Qwen3-4B runtime.
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
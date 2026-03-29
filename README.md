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

## Qwen models Settings 
- `Qwen2.5-Coder-3B-Instruct`, `Qwen/Qwen3-4B-Instruct-2507`, and `Qwen/Qwen3.5-4B`: run normally with `device_map="auto"` on this machine (RTX 4050 6GB), without CPU offload settings.
- `Qwen2.5-Coder-7B-Instruct`: should be loaded with offload-capable settings (`device_map="auto"` + `max_memory` + offload enabled), even when CPU offload is not always used at runtime.
- Reason: `device_map="auto"` uses a conservative placement policy. For 7B, it may decide to place some modules on CPU/disk depending on current VRAM state; if offload is not enabled, loading can fail.
- If the printed device map is `{"": 0}`, all model weights are on GPU for that run (offload was allowed, but not used).
- For experiment stability, keep 7B in offload-capable mode and always log `hf_device_map` for each run.
- `Qwen/Qwen3.5-4B` has showed some "endless repetitions" problems on some LCB problems (Qwen Developers and users around the word encountered in different situations this problem too), with the implementation of the presence_penalty parameter, the problem decreased but did not solve completely.

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

## Experiment History

### Experiment 1: H1 Baseline — Best-of-B vs Repair-B (2026-03-14)

**Goal**: validate the harness and get an initial comparison between fixed policies.

**Setup**: 20 LeetCode easy/medium tasks from LiveCodeBench, Qwen3-4B, B=3, temperature=0.7. This was NOT a proper experiment — small sample, easy tasks, just a sanity check to confirm the harness works and get a first intuition.

**Results**:

| Policy | Pass Rate | Total Attempts | Mean Time (s) |
|--------|-----------|----------------|---------------|
| best_of_b | 35% (7/20) | 48 | 83.7 |
| repair_b | 90% (18/20) | 35 | 91.5 |

**Takeaway**: Repair-B dominates Best-of-B on easy tasks. Error feedback from the verifier is far more valuable than sampling diversity. This is consistent with the literature. Not a surprising result, but it validated that the harness works end-to-end.

---

### Experiment 2: H2 Probe — Logistic Classifier on Hidden States (2026-03-14)

**Goal**: test whether a linear classifier on frozen hidden states can predict pass vs fail before the model generates code.

**Setup**: 388 samples (310 train / 78 test) from Repair-B runs on a **mixed dataset of LiveCodeBench (LeetCode) and HumanEval+ tasks**. Both sources were mixed into train, validation, and test sets. Extracted last-token hidden states at layers 29-36 of Qwen3-4B. Trained StandardScaler -> PCA -> LogisticRegression (L2-regularized, C tuned via 5-fold CV).

**Results (best layers)**:

| Layer | Test AUC | Test Acc | Test F1 | CV AUC |
|-------|----------|----------|---------|--------|
| 35 | 0.964 | 91.0% | 0.896 | 0.904 |
| avg_35-36 | 0.964 | 91.0% | 0.896 | 0.904 |
| 34 | 0.961 | 91.0% | 0.896 | 0.902 |
| concat (all layers) | 0.955 | 89.7% | 0.886 | 0.903 |

**Conclusion**: **This worked.** A simple logistic classifier on layer-35 hidden states separates pass from fail with AUC > 0.96. The signal is concentrated in the last 2-3 decoder layers. The model's internal state, at the moment it finishes reading the prompt and before it generates code, already encodes whether it will produce a correct solution. This is a decodable correctness signal.

---

### Experiment 3: H3 Probe-Guided Policy — Routing with Probe Score (2026-03-21/22)

**Goal**: use the probe score from Experiment 2 to build an adaptive policy. Compute P(pass) from the hidden state on the first attempt, then route to Repair-B (if score is above threshold) or Best-of-B (if below). The idea: don't waste repair attempts on tasks the model has no chance of fixing.

**Setup**: 196 tasks from LiveCodeBench, Qwen3-4B, B=3. Multiple thresholds were tried.

**Results**: pass rate = **30.1%** (59/196 tasks). **Failed.**

**Why it failed — the conceptual problem**: the probe score is computed on the hidden state of the **initial prompt** (attempt 0). But Repair-B uses an **enriched prompt** that includes the previous wrong code, the error message, and repair instructions. The routing decision is made at a moment when the model has NOT yet seen the repair information. So we are predicting "will repair work?" based on a state that doesn't account for the additional information that repair provides. This is a fundamental conceptual flaw, not a tuning problem — no threshold would fix it.

In retrospect, Repair-B is almost always the better strategy (as Experiment 1 showed), so any routing that diverts tasks away from Repair-B just hurts performance.

---

### Experiment 4: H2 Trajectory Analysis — Hidden States Across Repair Attempts (2026-03-28/29)

**Goal**: go deeper than Experiment 2 — study how the model's hidden state evolves across multiple repair attempts. Does the representation converge toward a "success region"? Is there a geometric "repair-success direction" in activation space?

**Setup**: 102 **hard** tasks from LiveCodeBench (significantly harder than Experiments 1-3), Qwen3-4B, B=5 (up to 5 attempts per task), temperature=0.7. Captured layer-35 hidden states at every attempt. 26/102 tasks were "repetition loops" (model stuck generating the same broken output, hitting 768 token limit every time).

**Results**: pass rate 34.3% (35/102 eventually pass), but only **30 successful repair transitions total** spread across 4 transition steps.

**Five analyses were run. Most did not produce reliable results:**

1. **Per-attempt classifier**: attempt 1 showed AUC=0.84 (22 passes out of 97 samples). All other attempts: F1=0.0 (classifier predicted "fail" for everything — too few positive samples). Consistent with Experiment 2 but adds no new information.

2. **Repair-success direction (RepE-style)**: cosine similarity 0.82 between per-transition directions (0->1 vs 1->2), suggesting a consistent direction exists. But computed from only 5 and 3 success deltas respectively. Too few to trust.

3. **Convergence**: both pass and fail trajectories oscillate equally (cos ~ -0.22). No signal.

4. **PCA visualization**: shows structure (three visual clusters, trajectories as paths) but confounded by prompt-structure differences between attempt 0 and 1+. Hard to interpret.

5. **Distance to success centroid**: centroid computed from 5 tasks that passed on attempt 0 (first try). Dominated by prompt-structure confound. Slight divergence at later steps (pass tasks trending closer, fail trending farther) but n=3-8 — not reliable.

**Root cause of failure: data starvation.** 30 successful repair transitions spread across 4 transition steps = 3-22 positive samples per step, in 2560-dimensional space. Every analysis requires splitting by (attempt index) x (pass/fail outcome), creating cells with single-digit samples. No statistical method can extract reliable signal from this.

**Secondary issue**: the prompt changes substantially between attempt 0 (original problem) and attempts 1+ (original + previous code + error), creating a dominant confound in any analysis that mixes attempt indices.

---

### Summary of What Worked and What Did Not

| Experiment | Worked? | Key Result |
|------------|---------|------------|
| H1 baseline (best-of-B vs repair-B) | Sanity check | Repair-B 90% vs Best-of-B 35% (easy tasks, not a proper experiment) |
| H2 probe (logistic classifier) | **Yes** | AUC=0.964 at layer 35, pass/fail is linearly decodable |
| H3 probe-guided policy | **No** | Conceptual flaw: attempt-0 score can't predict repair success |
| H2 trajectory (5 sub-analyses) | **Mostly no** | Data starvation: 30 success transitions for 5 analyses in 2560 dims |

**The main blocker across failed experiments is insufficient data**, specifically too few successful repair events. The probe itself works well (Experiment 2), but using it operationally (Experiment 3) or extending it to trajectory analysis (Experiment 4) requires either more data or a fundamentally different approach.

---

## Current Repo Structure

```text
src/
  reasoninglab/
    __init__.py
    cli.py                           # CLI entrypoint to run/evaluate experiments.
    config/
      __init__.py
      schema.py                      # Pydantic schemas for validating experiment configuration.
    models/
      __init__.py
      qwen_local.py                  # Local Qwen inference adapter (generation + hidden state capture).
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
      probe_guided.py                # Probe-guided adaptive policy (routes repair vs retry based on probe score).
    probing/
      __init__.py
      data.py                        # Data loading utilities for probe training (npz + jsonl joining).
      probe.py                       # Probe training pipeline (StandardScaler -> PCA -> LogisticRegression).
    eval/
      __init__.py
      runner.py                      # Main orchestration loop across tasks/policies/budget.
      metrics.py                     # Computes pass-rate/cost/failure metrics from run outputs.
    io/
      __init__.py
      jsonl_logger.py                # Writes structured run logs (attempt-level JSONL + summaries).

configs/
  h1/
    smoke.yaml                       # Minimal config for quick end-to-end smoke validation.
    smoke-best_of_b.yaml             # Best-of-B experiment config (B=3, LeetCode tasks).
    smoke-repair_b.yaml              # Repair-B experiment config (B=3, LeetCode tasks).
  h2/
    eval-probe_guided.yaml           # Probe-guided policy experiment config.

scripts/
  train_probe.py                     # Train H2 logistic probe on hidden states across layers.
  analyze_trajectories.py            # Trajectory analysis (5 sub-analyses on multi-attempt hidden states).
  debug_thinking.py                  # Inspect LCB prompts and Qwen generation for debugging.
  prepare_humaneval_plus.py          # Dataset preparation for HumanEval+.
  prepare_livecodebench.py           # Dataset preparation for LiveCodeBench.
  smoke_test.py                      # Direct local smoke test for Qwen3.5-4B runtime.
  smoke_test_7b_offload.py           # Direct local smoke test for Qwen2.5-7B offload runtime.

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
      -> policies/{baseline|best_of_b|repair_b|probe_guided}.py (choose policy behavior)
          -> models/qwen_local.py (generate candidate code + optional hidden state capture)
          -> verify/executor.py (run tests)
          -> verify/taxonomy.py (classify failure type)
          -> io/jsonl_logger.py (append attempt record)
      -> eval/metrics.py (aggregate run metrics)
      -> io/jsonl_logger.py (write final summary)
```
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

### Experiment 5: H2 Trajectory Analysis v2 — Scaled Dataset + Deeper Geometric Probes (2026-03-28 / 2026-04-06)

**Goal**: fix the root cause of Experiment 4 (data starvation) by scaling the dataset roughly 4x, then re-run trajectory analyses with proper statistical controls (permutation testing), cleaner coordinate systems (PCA fitted on attempt-0 only), and confound checks (prompt-length correlation, attempt-0 residualized probe). The deeper question: is there a stable geometric signature of "the model is about to produce correct code," or is the entire signal an artifact of prompt structure and task difficulty?

#### 5.1 Dataset construction (2026-03-28 / 2026-04-02)

Rather than running one large job, the dataset was built incrementally from four separate Repair-B runs that were later merged:

| Source run | Tasks | Difficulty | B |
|---|---|---|---|
| `h2-trajectory-repair_b_20260328_123657` | 102 | hard | 5 |
| `h2-trajectory-repair_b-easy_medium_20260329_193826` | 98 | easy/medium | 5 |
| `h2-trajectory-repair_b-easy_medium_remaining_20260401_103759` | 156 | easy/medium | 5 |
| `h2-trajectory-repair_b-easy_medium_remaining_20260402_104451` | 88 | easy/medium | 5 |
| **Merged (`runs/h2-trajectory-all/`)** | **444** | **mixed (all LCB)** | **5** |

Several runs crashed mid-execution, requiring the "remaining-tasks" strategy (regenerate the list of tasks that had not yet produced hidden states and re-run only those). A dedicated `scripts/merge_trajectory_runs.py` joins the runs, deduplicates by `task_id`, verifies no attempt-index collisions, and recomputes `summary.jsonl` using the repository's canonical `compute_metrics()` so that the merged numbers are comparable to any single run.

**Final merged dataset**: 444 tasks, 1494 attempts, 444 `.npz` hidden-state files. Pass rate 52.0% (231/444 eventually pass). 47/444 (10.6%) are "repetition loops" (same failure_type and `completion_tokens == 768` at every attempt), tagged but not removed. Trajectory length distribution: 16% of tasks pass on first try (length 1), 31% take 2 attempts, 49% exhaust the B=5 budget.

This is ~4.4x more data than Experiment 4 and covers the full LCB difficulty range.

#### 5.2 The 13 analyses

The trajectory analysis script (`scripts/analyze_trajectories.py`) was extended from 5 analyses (Experiment 4) to 13 analyses. The first 5 were kept untouched so the two experiments can be compared on the same code paths; the 6 new analyses (numbered 6-13, skipping 7 which is a helper) address the two main weaknesses of Experiment 4: (a) no statistical significance testing, and (b) confounding by prompt structure. All numerical results are in `results/h2/trajectory_analysis/metrics.json`. The script can be re-run with `python scripts/analyze_trajectories.py`.

**Theoretical framing**: these analyses are inspired by Representation Engineering (RepE) and activation-addition work. The core idea is that if the model's internal state encodes something meaningful about "the upcoming generation will be correct," then the *contrastive direction* `mean(delta | next_attempt_passes) - mean(delta | next_attempt_fails)` should be a stable, reproducible vector in activation space. Each individual delta is noisy; the contrastive mean cancels out the shared component (everything common to both pass and fail transitions — including prompt-length shifts) and isolates the component that actually differs between pass and fail.

##### Analysis 1 — Per-attempt pass/fail classifier (pooled + per-attempt-index)

*What*: Train a logistic probe (the same `StandardScaler -> PCA -> LogisticRegression` pipeline used in Experiment 2) on layer-35 hidden states. Two modes: pooled across all attempts (1494 samples), and one classifier per attempt index (0, 1, 2, 3, 4).

*Results*:

| Classifier | N | Pass rate | CV AUC | Test AUC | Accuracy | F1 |
|---|---|---|---|---|---|---|
| Pooled (all attempts) | 1494 | 15.5% | 0.919 | 0.936 | 0.893 | 0.610 |
| Attempt 0 only | 444 | 15.8% | 0.923 | 0.927 | 0.854 | 0.480 |
| Attempt 1 only | 374 | 36.4% | 0.895 | 0.831 | 0.773 | 0.679 |
| Attempt 2 only | 238 | 6.7% | 0.709 | 0.741 | 0.938 | 0.000 |
| Attempt 3 only | 222 | 2.7% | 0.305 | 0.977 | 0.978 | 0.000 |
| Attempt 4 only | 216 | 1.4% | NaN | 0.953 | 0.977 | 0.000 |

**Why F1 drops from Experiment 2 (~0.88) to here (~0.48-0.61)**: Experiment 2 had ~50% positive rate (balanced dataset). Here the positive rate is 15.8% at attempt 0. At the default threshold of 0.5, the probe predicts "fail" for nearly everything — high accuracy (85%) but F1 crashes. This is a threshold artifact, not a signal loss. AUC (threshold-independent) remains at 0.927, matching Experiment 2. With optimal threshold tuning, F1 would recover substantially.

*Interpretation*: The attempt-0 probe replicates Experiment 2 at essentially the same AUC (~0.92), on a larger and more diverse dataset. The pooled probe reaches AUC 0.936 because it has 3x more data. Attempt 1 still has real signal (CV AUC 0.895). Attempts 2-4 are unreliable — the CV AUC collapses or goes NaN because the positive class drops to 2-7% and cross-validation folds end up with zero positives. F1 is 0.0 at attempts 2-4: the classifier just predicts "fail" for everything, which is trivially ~97% accurate. **This is data starvation at the positive-class level, not a model-state change.**

##### Analysis 2 — Repair-success direction (RepE-style, per transition + pooled)

*What*: For each transition `k -> k+1` where attempt `k` failed, compute `delta_k = h_{k+1} - h_k` (both captured at layer 35, immediately after reading the prompt and before generation). Group deltas by transition index. For each group, compute `direction = mean(deltas | next passes) - mean(deltas | next fails)`. Measure the L2 norm of each direction, and the cosine similarity between directions at different transitions (the "consistency check": is the repair-success direction a stable geometric property or does it change at every step?).

*Results*:

| Transition | N success | N failure | ‖direction‖ |
|---|---|---|---|
| 0 -> 1 | 128 | 108 | 23.78 |
| 1 -> 2 | 13 | 137 | 55.65 |
| 2 -> 3 | 4 | 150 | 38.52 |
| 3 -> 4 | 2 | 154 | 85.79 |

**Note**: All direction analyses exclude tasks where attempt 0 hit the 768-token generation limit (91 tasks). These are cases where the model generated truncated garbage code, which gets pasted into the repair prompt and distorts the hidden-state deltas. See "Data cleaning" note below Analysis 6.

Cosine similarity between per-transition directions:

|  | 1->2 | 2->3 | 3->4 |
|---|---|---|---|
| **0->1** | 0.342 | 0.370 | 0.183 |
| **1->2** | — | 0.807 | 0.824 |
| **2->3** | — | — | 0.732 |

Pooled (all transitions): 147 success deltas, 549 fail deltas, direction norm 90.86.

*Interpretation*: After data cleaning, the 0->1 direction's cosine alignment with later steps dropped substantially (0.18-0.37 vs 0.66-0.76 before cleaning), suggesting the old alignment was partly driven by the maxed-out cases. Later-step directions (1→2, 2→3, 3→4) remain mutually consistent (cosine 0.73-0.82) but have n_success = 2-13, so they are not independently load-bearing.

The 0->1 transition is by far the best-supported (n=128 success vs n=108 failure on clean data). Everything downstream treats it as the primary evidence.

##### Analysis 3 — Convergence (directional consistency + projection per step)

*What*: For tasks with 3+ attempts, compute consecutive deltas and measure (a) cosine similarity between consecutive deltas within the same trajectory ("does the model keep moving the same way across repair steps?"), and (b) the scalar projection of each delta onto the pooled repair direction.

*Results*:

- Directional consistency (pass trajectories): cos = -0.137 ± 0.245 (n=37 delta pairs)
- Directional consistency (fail trajectories): cos = -0.230 ± 0.312 (n=498 delta pairs)

Projection onto repair direction by step (pass / fail means):

| Step | Pass proj | Fail proj |
|---|---|---|
| 0 -> 1 | 91.83 | 104.29 |
| 1 -> 2 | 6.91 | -8.82 |
| 2 -> 3 | 8.32 | -1.06 |
| 3 -> 4 | 29.38 | -1.89 |

*Interpretation*: **No convergence.** Consecutive deltas are slightly *anti*-correlated for both groups. The model does not smoothly move toward any destination; each repair step is nearly independent of the previous one in direction. The projection table shows the expected pattern (pass positive, fail negative) at steps 1->2 and beyond, but with opposite-sign separation at step 0->1 where both are strongly positive — that step is dominated by the shared prompt-structure shift that pushes everything forward regardless of outcome.

##### Analysis 4 — Full-data PCA trajectory visualization

*What*: Fit PCA(n=2) on all 1494 attempt hidden states pooled together. Plot each task as a connected path in PC1-PC2 space, green=eventually passes, red=never, gray=repetition loop.

*Results*: PC1 explains 38.7% of variance, PC2 explains 19.7%. Visually the plot shows a left-right cluster structure (attempt-0 points on one side, repair attempts on the other).

*Interpretation*: **This plot is contaminated.** Analysis 13 later shows that PC1 correlates with `prompt_tokens` at r = -0.765, meaning the dominant axis of variance is "attempt 0 has a short prompt, attempts 1+ have long prompts." The apparent structure is mostly prompt-length, not repair geometry. We keep the plot in the output for transparency but do not draw any conclusions from it.

##### Analysis 5 — Distance to attempt-0 pass centroid

*What*: Define the "success centroid" as the mean hidden state of tasks that passed on attempt 0 (70 tasks). For every task, compute Euclidean distance from this centroid at each attempt index. Plot mean distance curves per group.

*Results* (distance):

| Step | Pass | Fail | Loops |
|---|---|---|---|
| 0 | 69.3 | 72.8 | 81.7 |
| 1 | 134.0 | 139.4 | 144.0 |
| 2 | 142.5 | 144.3 | 143.8 |
| 3 | 145.7 | 146.8 | 143.1 |
| 4 | 146.6 | 147.3 | 143.7 |

*Interpretation*: **Dead.** All three curves jump from ~70 to ~140 between step 0 and step 1 and stay flat. Pass and fail are never more than ~5 units apart. The curve shape reflects the fact that attempt-0 states are in a different region of space than attempt-1+ states (the prompt-structure shift is a ~140-unit move in 2560-dim space), so "distance from attempt-0-pass centroid" is basically "are you attempt 0 or not." This analysis inherits the same confound as Analysis 4.

##### Analysis 6 — Permutation test on 0->1 repair direction

*What*: Address the core concern from Experiment 4 — "is the 0->1 repair direction real, or would shuffled labels produce similar norms?" Collect all 236 clean 0->1 deltas. Record the real norm 23.78. Then 1000 times, shuffle the pass/fail labels and recompute `‖mean(pass) - mean(fail)‖`. Compare the real norm to the permuted null distribution. A p-value of (# permuted norms >= real) / 1000.

*Results*:

- Real direction norm: **23.78**
- Permuted null: mean = 9.07, std = 1.43, max = 16.14
- **p-value = 0.000** (not a single permutation out of 1000 reached the real norm; the real norm sits ~10 standard deviations above the null mean)

Output image: `results/h2/trajectory_analysis/permutation_test_0to1.png`

*Interpretation*: The direction is statistically real. It is not a chance artifact of comparing two arbitrary subsets of the delta population. Whatever it encodes is specifically tied to the pass/fail label, not to any shared property of 0->1 transitions. Note that this test does NOT prove the direction encodes "understanding the fix" — it only rules out random chance. We still need to control for confounds like prompt length (done in Analyses 8 and 13).

**Data cleaning note**: All direction analyses (2, 6, 8, 11, 15-20) exclude two categories of tasks from 0->1 deltas: (1) repetition-loop tasks (47 tasks where the model generates the same broken output at the token limit across all 5 attempts), and (2) tasks where attempt 0 hit the 768-token generation limit without being a full repetition loop (91 additional tasks). These maxed-out generations are truncated garbage code that gets pasted into the repair prompt, creating massive prompt-length inflation (median delta_prompt_tokens ~900 for these vs ~350 for clean tasks). 83 of the 91 fail at attempt 1 (vs 46% fail rate in clean data), making them a contamination source. After filtering: 236 clean 0->1 transitions (128 success, 108 fail).

##### Analysis 7 (helper) — PCA reduction fitted on attempt-0 states only

*What*: A helper used by Analyses 8, 9, 10. Fit PCA(n=100) on the 444 attempt-0 hidden states, then project ALL attempts (0-4) into that 100-dim subspace. The design goal: define a coordinate system that cannot encode prompt structure, because PCA was fit only on states that all have the same prompt structure (attempt-0 prompts). Attempts 1+ are projected INTO this space but never influence the axes.

*Result*: 100 components capture 93.2% of attempt-0 variance.

*Caveat found later in Analysis 13-style checks*: even the attempt-0 PCA is *partially* contaminated by prompt length. A follow-up check showed PC1 of the attempt-0 PCA correlates with `prompt_tokens` at r=0.509 (because different difficulty tasks have different prompt lengths even at attempt 0). So this coordinate system is cleaner than the full-data one (Analysis 4), but it is not prompt-length-free. This is relevant for Analyses 8/9/10.

##### Analysis 8 — Repair direction recomputed in attempt-0-fitted PCA space

*What*: Redo Analysis 2 in the 100-dim attempt-0 PCA subspace.

*Results*:

| Transition | ‖direction‖ (raw) | ‖direction‖ (PCA-100) |
|---|---|---|
| 0 -> 1 | 23.78 | 16.81 |
| 1 -> 2 | 55.65 | (not computed) |
| 2 -> 3 | 38.52 | (not computed) |
| 3 -> 4 | 85.79 | (not computed) |

*Interpretation*: The 0->1 direction survives dimensionality reduction (23.78 → 16.81). Later-step norms are large but unreliable (n_success = 2-13). The 0->1 cosine alignment with later steps dropped after data cleaning (0.34-0.37 vs previously 0.66-0.76), suggesting the old alignment was partly driven by the maxed-out cases. Later-step directions (1→2, 2→3, 3→4) remain mutually consistent (cosine 0.73-0.82) but are not independently load-bearing due to tiny success counts.

##### Analysis 9 — Convergence in PCA-reduced space

*What*: Redo Analysis 3 on PCA-reduced records.

*Results*: Consistency (pass): cos = -0.137, (fail): cos = -0.230. Projection pattern per step similar to Analysis 3 (pass positive at later steps, fail negative).

*Interpretation*: Same conclusion as Analysis 3. No convergence. The lack of directional consistency is not an artifact of high-dimensional noise — it survives reduction to 100 dimensions.

##### Analysis 10 — Distance to first-pass centroid (PCA space)

*What*: Redo Analysis 5 with two changes: (a) work in the PCA-reduced space, and (b) compute the centroid from the hidden state at the FIRST passing attempt of each eventually-passing task (231 tasks), rather than only from attempt-0 passes. The motivation: "the region of representation space associated with the model finally getting it right" is better defined by all 231 first-pass states than by the 70 trivially-easy attempt-0 passes.

*Results*:

| Step | Pass | Fail | Loops |
|---|---|---|---|
| 0 | 92.8 | 95.5 | 99.3 |
| 1 | 50.5 | 62.9 | 81.5 |
| 2 | 68.4 | 74.0 | 80.1 |
| 3 | 78.1 | 77.8 | 78.6 |
| 4 | 67.6 | 79.5 | 78.1 |

*Interpretation*: Step 1 shows a ~12-unit gap between pass (50.5) and fail (62.9) — interesting compared to Analysis 5's flat curves. But the gap is not monotonic (it shrinks at step 2, reappears at step 4 with only n=3 pass), the std bands overlap heavily, and the low-N later steps make the pattern noisy. **Not a strong finding.** Best described as "a hint of structure at step 1 that does not persist."

##### Analysis 11 — Per-task projection histogram (0->1 deltas onto repair direction)

*What*: For each 0->1 delta, compute its scalar projection onto the unit pooled repair direction. Plot overlaid histograms: green = attempt 1 passed, red = attempt 1 failed. Report Cohen's d.

*Results*:

- Pass deltas: mean projection = 120.61, std = 13.96, n = 136
- Fail deltas: mean projection = 102.66, std = 22.88, n = 191
- **Cohen's d = 0.910** (large effect size by convention)

Output image: `results/h2/trajectory_analysis/projection_histogram_0to1.png`

*Interpretation*: After data cleaning, Cohen's d dropped from 0.91 to 0.32 — the large effect size was inflated by the maxed-out cases. On the clean 236 tasks, the pass and fail projection distributions overlap substantially (means 123.9 vs 119.3). The direction still has a statistically significant magnitude (permutation test p<0.001, Analysis 6) and stable inclination (split-half cosine 0.756, Analysis 16), but the per-task projection separation is modest.

##### Analysis 12 — Focused PCA plot (attempt 0 -> attempt 1 transitions only)

*What*: Take only the 327 non-loop tasks with at least 2 attempts. Fit PCA(n=2) on just the attempt-0 states. Project both attempt-0 and attempt-1 states into this 2D space. Plot circles for attempt-0, triangles for attempt-1, arrows from 0 to 1, colored by attempt-1 outcome (green pass / red fail). Avoids the clutter of the Analysis-4 global plot.

*Result*: PC1 = 23.7% variance, PC2 = 13.0% variance, 327 tasks plotted (136 pass, 191 fail). Note: Analysis 12 plots all non-loop tasks including maxed-out ones; the data cleaning filter applies only to direction analyses.

*Interpretation*: **Weak.** There is a mild left-right tendency (more green on the upper-left, more red on the right) but the separation is far from clean, and 2 components only explain 36.7% of attempt-0 variance. Cannot be used as standalone evidence.

##### Analysis 13 — Prompt-length correlation (the confound check)

*What*: Take the full-data PCA from Analysis 4. For every attempt across all tasks, compute Pearson correlation between its PC1 (and PC2) coordinate and its `prompt_tokens` count.

*Results*:

| Axis | Variance | Pearson r with prompt_tokens | p-value |
|---|---|---|---|
| PC1 | 38.7% | **-0.765** | 4.4e-287 |
| PC2 | 19.7% | 0.127 | 8.8e-07 |

Output image: `results/h2/trajectory_analysis/pc_vs_prompt_tokens.png`

*Interpretation*: **This is the most important negative finding of Experiment 5.** PC1 of the full-data PCA — the single most explanatory axis (38.7% of all variance) — is dominantly encoding prompt length. The big left-right structure in Analysis 4 is mostly "attempt 0 has a short prompt, attempts 1+ have longer prompts." PC2 is clean (r=0.127). This retroactively invalidates any interpretation of Analysis 4's plot that relied on PC1 as "task/repair geometry." It also motivates the use of attempt-0-fitted PCA in Analyses 8/9/10 (since that PCA cannot contain the cross-attempt prompt-length axis by construction — though it still partially correlates with prompt length across tasks, r=0.509 at PC1, because harder tasks tend to have longer problem descriptions).

##### Analysis 14 — Residualized probe (formal prompt-length control)

*What*: Formalized version of the earlier ad-hoc control test. Trains three probes on attempt-0 hidden states (444 samples, same train/test split as Analysis 1):
1. **Raw**: standard probe on unchanged hidden states
2. **Residualized**: for each of the 2560 dimensions, fit `h_dim = slope * prompt_tokens + intercept` on the train set only, subtract predictions from both sets. This removes all linear prompt-length information.
3. **Prompt-only**: just `prompt_tokens` as the single feature (floor — how well can prompt length alone predict pass/fail?)

*Results*:

| Probe | Test AUC | CV AUC |
|---|---|---|
| Raw | 0.927 | 0.923 |
| Residualized | 0.909 | 0.888 |
| Prompt-only | 0.786 | 0.746 |

AUC drop from residualization: -0.018. Signal above prompt-only: +0.123.

*Interpretation*: The residualized probe loses only ~2 AUC points. Prompt length alone is a decent predictor (AUC 0.786) because in this dataset longer prompts tend to be harder tasks, but the hidden state carries 12.3 AUC points of signal beyond prompt length. **The probe's ability to predict correctness is real and not reducible to prompt structure.** Note: the prompt-only baseline being 0.786 is a property of this dataset's difficulty-length correlation, not a general property of the model.

##### Analysis 15 — Direction residualized against delta_prompt_tokens

*What*: For each 0->1 delta (h_attempt1 - h_attempt0), compute delta_prompt_tokens = prompt_tokens[attempt_1] - prompt_tokens[attempt_0]. For each of the 2560 dimensions, regress delta_dim on delta_prompt_tokens and subtract the prediction. This removes the component of the hidden-state shift that is linearly explained by "the prompt grew." Then recompute the repair direction on the residualized deltas and run a permutation test (1000 shuffles of pass/fail labels, same procedure as Analysis 6).

*Results*:

| Metric | Raw | Residualized |
|---|---|---|
| Direction norm | 23.78 | 16.62 |
| Norm retention | - | 69.9% |
| Permutation null mean | 9.07 | 8.71 |
| Permutation null std | 1.43 | 1.33 |
| p-value | <0.001 | <0.001 |

N = 236 (128 success, 108 fail). Output image: `results/h2/trajectory_analysis/direction_residualized_permtest.png`

*Interpretation*: After data cleaning (removing maxed-out attempt-0 cases), residualization only reduces the norm by 30% (from 23.78 to 16.62), compared to 68% before cleaning. This means the old 68% drop was driven by the garbage cases, not by a genuine prompt-length confound. The residualized norm (16.62) is highly significant (p<0.001, ~6 sigma above null).

**Caveat on residualization**: The residualization regresses each hidden-state dimension on `delta_prompt_tokens` (how much the prompt grew from attempt 0 to 1). This removes the linear effect of prompt length growth, but `delta_prompt_tokens` is entangled with the content of the feedback (longer wrong code → longer repair prompt). Removing prompt length does not remove the meaning of what's in the prompt. The residualized numbers are a sensitivity check, not the primary result. The primary evidence is the raw permutation test (norm 23.78, p<0.001) on clean data.

##### Analysis 16 — Split-half stability test (direction inclination)

*What*: Tests whether the direction's *orientation* (not just magnitude) is a stable property. Procedure:
1. Take 236 clean 0->1 deltas (128 pass, 108 fail)
2. Randomly split in half (stratified by label so each half has ~64 pass, ~54 fail)
3. Compute direction_A on half A, direction_B on half B
4. Measure cosine(direction_A, direction_B)
5. Repeat 1000 times with different random splits -> distribution of real cosines
6. Null distribution: shuffle pass/fail labels randomly, then do the same split-and-compare, 1000 times. This answers: "if pass/fail labels were meaningless, would random directions from two halves still be consistent?"

*Results*:

| Distribution | Mean cosine | Std | Min | Max |
|---|---|---|---|---|
| Real splits (1000) | **0.756** | 0.070 | 0.433 | 0.886 |
| Null / shuffled labels (1000) | -0.009 | 0.217 | - | - |

p < 0.001 (zero null cosines exceeded the real mean of 0.756).

Output image: `results/h2/trajectory_analysis/split_half_stability.png`

*Interpretation*: The direction is moderately stable across random data splits — two independent halves agree on direction with cosine ~0.76 on average, far above the null (~0). This is a real geometric property of the data. The signal is moderate: cosine 0.756 means the two halves agree on roughly 75% of the direction's orientation. The worst-case split still gives cosine 0.433.

##### Analysis 17 — Permutation test in PCA-100 space

*What*: Same permutation test as Analysis 6 but on PCA-100 reduced deltas (PCA fitted on attempt-0 states only). 1000 label shuffles.

*Results*:

| Metric | Value |
|---|---|
| Real direction norm (PCA-100) | 16.81 |
| Null mean | 7.52 |
| Null std | 1.37 |
| Null max | 14.72 |
| p-value | <0.001 |

N = 236 (128 success, 108 fail). Output image: `results/h2/trajectory_analysis/permutation_test_pca_0to1.png`

*Interpretation*: The direction is ~2.2x the null mean and exceeds the null max. Significant in the reduced space.

##### Analysis 18 — Permutation test in PCA-100 space (residualized)

*What*: Same as Analysis 17 but with delta_prompt_tokens regressed out of the PCA-100 deltas before testing. This is the most conservative test: dimensionality-reduced AND prompt-length-residualized.

*Results*:

| Metric | Value |
|---|---|
| Residualized direction norm (PCA-100) | 11.93 |
| Null mean | 7.28 |
| Null std | 1.31 |
| Null max | 13.96 |
| p-value | 0.003 |

N = 236 (128 success, 108 fail). Output image: `results/h2/trajectory_analysis/permutation_test_pca_resid_0to1.png`

*Interpretation*: After data cleaning, this result improved from p=0.018 to p=0.003. The old weak result was caused by the garbage cases distorting both the PCA space and the residualization. See also Analysis 19 below for the PCA-on-deltas variant.

**Note on Analyses 17/18 PCA design**: The PCA here was fitted on attempt-0 hidden states, but we are studying deltas (attempt1 - attempt0). There is no guarantee that the dimensions where repair signal lives are the same dimensions that vary most across tasks at the initial prompt. If the repair-relevant dimensions are quiet at attempt 0, PCA discards them. Analysis 19 tests this directly.

##### Analysis 19 — Permutation test with PCA fitted on deltas

*What*: Same permutation test as Analyses 17/18, but PCA is fitted on the 236 clean 0->1 deltas themselves instead of on attempt-0 states. This is the most natural space for studying deltas — the dimensionality reduction captures the axes where the deltas actually vary, not where the initial prompts vary. Runs both raw and residualized (delta_prompt_tokens removed) variants.

**Caveat**: This is somewhat circular — PCA fitted on deltas will tend to preserve the dominant patterns in the deltas, which includes the repair direction itself. The permutation test still controls for this (null is computed in the same PCA space with shuffled labels), but the result should not be taken as independent confirmation.

*Results*:

| Variant | Real norm | Null mean | Null std | p-value |
|---|---|---|---|---|
| Raw | 23.71 | 8.77 | 1.48 | <0.001 |
| Residualized | 16.52 | 8.40 | 1.38 | <0.001 |

PCA: 100 components, 93.9% variance explained. N = 236 (128 success, 108 fail).

Output image: `results/h2/trajectory_analysis/permutation_test_pca_deltas.png`

*Interpretation*: PCA-on-deltas recovers the same numbers as the full 2560-dim tests (Analysis 6: norm 23.78, Analysis 15 residualized: norm 16.62). This confirms that the attempt-0 PCA design is suboptimal for delta analysis but the signal is robust across PCA choices.

**Full magnitude comparison table:**

| Analysis | PCA space | Residualized? | Real norm | Null mean | p-value |
|---|---|---|---|---|---|
| 6 | None (2560-dim) | No | 23.78 | 9.07 | <0.001 |
| 15 | None (2560-dim) | Yes | 16.62 | 8.71 | <0.001 |
| 17 | Attempt-0 (100-dim) | No | 16.81 | 7.52 | <0.001 |
| 18 | Attempt-0 (100-dim) | Yes | 11.93 | 7.28 | 0.003 |
| 19 | Deltas (100-dim) | No | 23.71 | 8.77 | <0.001 |
| 19 | Deltas (100-dim) | Yes | 16.52 | 8.40 | <0.001 |

All results on 236 clean tasks (128 success, 108 fail). The primary evidence for magnitude is Analysis 6 (raw permutation test, 2560-dim, p<0.001). The residualized variants (Analyses 15, 18, 19) are sensitivity checks — see caveat in Analysis 15 about why residualization is not fully justified conceptually.

##### Analysis 20 — Split-half stability (residualized, sensitivity check)

*What*: Residualizes the 0→1 deltas against delta_prompt_tokens (via LinearRegression, same as Analysis 15) before running the split-half stability test. 1000 stratified splits + 1000 null (shuffled labels). N = 236 (128 success, 108 fail).

*Results*:

| Variant | Mean cosine | Std | Min | Max | Null mean | p-value |
|---|---|---|---|---|---|---|
| Raw (Analysis 16) | 0.756 | 0.070 | 0.433 | 0.886 | -0.009 | <0.001 |
| Residualized | 0.575 | 0.101 | 0.180 | 0.774 | -0.012 | <0.001 |

Output image: `results/h2/trajectory_analysis/split_half_stability_residualized.png`

*Interpretation*: On clean data, the residualization reduces the cosine from 0.756 to 0.575 — a modest drop (24%), unlike the dramatic drop seen before data cleaning (0.932 → 0.496 on the dirty 327 tasks). The residualized signal is still highly significant (p<0.001).

**Same caveat as Analysis 15**: residualizing against delta_prompt_tokens removes the linear effect of prompt length, but prompt length is entangled with the content of the feedback (the wrong code, the error message). The model's hidden state doesn't just encode "how many tokens I read" — it encodes a compressed understanding of what those tokens say. Removing length without removing content is an incomplete correction that may remove real signal. The primary result is Analysis 16 (raw cosine 0.756, p<0.001) on clean data.

#### 5.3 Consolidated summary — Experiment 5 (updated after data cleaning and Analyses 14-20)

**Data cleaning**: Direction analyses exclude tasks where attempt 0 hit the 768-token generation limit. These 91 tasks (plus 47 repetition loops) generated truncated garbage code that gets pasted into the repair prompt, creating extreme prompt-length inflation. 83 of the 91 fail at attempt 1 (vs 46% fail rate in clean data). After filtering: 236 clean 0->1 transitions (128 success, 108 fail). The probe analyses (Analyses 1, 14) are unaffected — they use attempt-0 hidden states only, no repair deltas.

**What worked (solid, defensible):**

1. **The probe replicates at scale** on 444 mixed-difficulty tasks: attempt-0 test AUC 0.927, CV AUC 0.923. Same number as Experiment 2, different dataset, larger sample. (Analysis 1)

2. **The probe survives prompt-length residualization**: AUC drops from 0.927 to 0.909 when prompt length is linearly removed from every hidden-state dimension. Prompt-only baseline is 0.786; the residualized probe is 12.3 AUC points above that. **The probe is not merely reading prompt length.** (Analysis 14)

3. **The 0->1 repair direction magnitude is statistically significant**: on 236 clean tasks, the contrastive direction norm is 23.78 (p<0.001, Analysis 6). The permutation test shuffles pass/fail labels 1000 times; no shuffle reaches the real norm (~10 sigma above null mean 9.07). PCA variants confirm: PCA-on-deltas gives norm 23.71 (p<0.001, Analysis 19), PCA-on-attempt-0 gives norm 16.81 (p<0.001, Analysis 17). See full comparison table in Analysis 19. Residualization sensitivity check: norm drops to 16.62 (70% retention, p<0.001, Analysis 15), but this correction is conceptually questionable — see caveat in Analysis 15.

4. **The direction inclination is moderately stable**: split-half cosine 0.756 (p<0.001, Analysis 16). Two random halves of the data independently discover directions that agree ~75% on orientation. Residualization sensitivity check: cosine drops to 0.575 (p<0.001, Analysis 20) — same caveat applies. Later-step directions (1→2, 2→3, 3→4) are mutually consistent (cosine 0.73-0.82) but have n_success = 2-13, so they are not independently load-bearing. The 0->1 direction's cosine alignment with later steps is weak (0.18-0.37), suggesting the 0->1 direction is partly specific to the first repair transition. (Analyses 2, 16, 20)

**What did not work (honest negatives):**

5. **No convergence.** Directional consistency between consecutive deltas within the same trajectory is slightly negative for both pass (cos = -0.137) and fail (cos = -0.230) groups. The model does not progressively move toward a solution in representation space. Each repair step is nearly independent of the previous one. (Analyses 3, 9)

6. **Pass rates collapse after attempt 1** (15.8% -> 36.4% -> 6.7% -> 2.7% -> 1.4%). Error feedback helps dramatically once, then hits a ceiling. Consistent with "no iterative refinement."

7. **Analysis 4 (full-data PCA trajectory plot) is contaminated**: PC1 (r=-0.765 with prompt_tokens) is mostly encoding prompt length, not repair geometry. The triangle/arrow structure in that plot is largely an artifact. (Analysis 13)

8. **Analysis 5 (distance to attempt-0-pass centroid) inherits the same confound** and shows no separation between pass and fail distance curves after step 1.

9. **Analysis 10 (distance to first-pass centroid in PCA space)** showed a modest step-1 gap (50.5 vs 62.9) that does not persist and is not strong enough to build a claim on.

10. **Analysis 12 (focused PCA plot)** shows weak visual separation but only explains 36.7% of attempt-0 variance in 2D — not actionable.

11. **Cohen's d dropped from 0.91 to 0.32 after data cleaning** (Analysis 11). The large effect size was inflated by the maxed-out cases. On clean data, per-task projection separation is modest.

**Gaps addressed by Analyses 14-20** (previously listed as open questions):

- ~~No formal residualization script~~ -> Analysis 14 (probe) and Analysis 15 (direction). Both survive, but residualization caveat noted.
- ~~No permutation test in PCA-reduced space~~ -> Analysis 17 (p<0.001) and Analysis 18 with residualization (p=0.003).
- ~~No test for direction inclination stability~~ -> Analysis 16 (split-half cosine=0.756, p<0.001).
- ~~No F1 scores reported~~ -> Added to Analysis 1 table with explanation of threshold artifact.
- ~~Direction universality language too strong~~ -> Corrected: 0->1 direction is weakly aligned with later steps (cosine 0.18-0.37).
- ~~Data quality concern about maxed-out generations~~ -> Filtered out 91 tasks where attempt 0 hit token limit.

**Remaining gaps:**

- Later-step analyses (1->2, 2->3, 3->4) have n_success = 2-13 — not independently load-bearing.
- No comparison against simpler pre-generation baselines (e.g., token-level uncertainty) to contextualize the probe's performance.

#### 5.4 Proposed unified insight (DRAFT — updated after data cleaning)

> **Layer 35 of Qwen3-4B carries a static, decodable signal about upcoming code correctness, and the 0->1 repair transition has a characteristic geometric direction — but the model does not iteratively follow this direction across repair attempts.**
>
> - A linear probe on layer-35 hidden states at the moment the model finishes reading a prompt (attempt 0) predicts whether the upcoming generation will pass all tests, with AUC 0.927 raw / 0.909 after linearly removing prompt length (prompt-only baseline: 0.786). This replicates and strengthens Experiment 2 on 4.4x more data across the full LCB difficulty range, with a formal confound control that Experiment 2 did not have. (Analyses 1, 14)
>
> - Beyond a static "will it work" signal, there is a **repair-success direction** at the 0->1 transition: on 236 clean tasks, the contrastive direction `mean(success_deltas) - mean(failure_deltas)` has norm 23.78 (p<0.001, ~10 sigma above null, Analysis 6). The direction's inclination is moderately stable across random data splits (split-half cosine 0.756, p<0.001, Analysis 16). The model has no memory between attempts — each attempt is a fresh forward pass on the repair prompt. The direction reflects the model's *comprehension* of the error feedback: when the feedback is informative enough to enable a fix, the hidden-state shift points in a characteristic direction. (Analyses 2, 6, 16)
>
> - However, this direction is **not a trajectory the model follows iteratively**. Consecutive repair steps within the same task are slightly anticorrelated in direction (cos ~ -0.2 for both pass and fail trajectories), there is no distance-to-success convergence, and raw pass rates collapse after the first repair attempt (36% -> 7% -> 3% -> 1%). The error feedback appears to provide a one-shot benefit: the first repair attempt re-reads the problem with additional context and either lands in the success region or does not. Subsequent attempts do not compound. (Analyses 3, 9)
>
> - Together, Experiments 2 and 5 support a picture in which layer-35 of a 4B-parameter code model holds a *snapshot* representation of "is this code going to be correct," and the repair process is a discrete re-read with a characteristic activation-space shift — not a continuous reasoning trajectory. For practical inference-time scaling on models of this size, this would predict that investing in *the quality of the first error message* matters more than allowing many repair attempts.

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
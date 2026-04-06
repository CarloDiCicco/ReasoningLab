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

| Classifier | N | Pass rate | CV AUC | Test AUC |
|---|---|---|---|---|
| Pooled (all attempts) | 1494 | 15.5% | 0.919 | 0.936 |
| Attempt 0 only | 444 | 15.8% | 0.923 | 0.927 |
| Attempt 1 only | 374 | 36.4% | 0.895 | 0.831 |
| Attempt 2 only | 238 | 6.7% | 0.709 | 0.741 |
| Attempt 3 only | 222 | 2.7% | 0.305 | 0.977 |
| Attempt 4 only | 216 | 1.4% | NaN | 0.953 |

*Interpretation*: The attempt-0 probe replicates Experiment 2 at essentially the same AUC (~0.92), on a larger and more diverse dataset. The pooled probe reaches AUC 0.936 because it has 3x more data. Attempt 1 still has real signal (CV AUC 0.895). Attempts 2-4 are unreliable — the CV AUC collapses or goes NaN because the positive class drops to 2-7% and cross-validation folds end up with zero positives. F1 is 0.0 at attempts 2-4: the classifier just predicts "fail" for everything, which is trivially ~97% accurate. **This is data starvation at the positive-class level, not a model-state change.**

##### Analysis 2 — Repair-success direction (RepE-style, per transition + pooled)

*What*: For each transition `k -> k+1` where attempt `k` failed, compute `delta_k = h_{k+1} - h_k` (both captured at layer 35, immediately after reading the prompt and before generation). Group deltas by transition index. For each group, compute `direction = mean(deltas | next passes) - mean(deltas | next fails)`. Measure the L2 norm of each direction, and the cosine similarity between directions at different transitions (the "consistency check": is the repair-success direction a stable geometric property or does it change at every step?).

*Results*:

| Transition | N success | N failure | ‖direction‖ |
|---|---|---|---|
| 0 -> 1 | 136 | 191 | 40.54 |
| 1 -> 2 | 16 | 175 | 43.70 |
| 2 -> 3 | 6 | 169 | 27.02 |
| 3 -> 4 | 3 | 166 | 60.91 |

Cosine similarity between per-transition directions:

|  | 1->2 | 2->3 | 3->4 |
|---|---|---|---|
| **0->1** | 0.757 | 0.727 | 0.660 |
| **1->2** | — | 0.805 | 0.840 |
| **2->3** | — | — | 0.683 |

Pooled (all transitions): 161 success deltas, 701 fail deltas, direction norm 79.30.

*Interpretation*: All four per-transition directions have non-trivial norms and all pairwise cosines are in the 0.66-0.84 range. That is unusual for random vectors in 2560-dim space, where expected cosine is near zero. The direction has the same geometric signature across every repair step. Later-step estimates (1->2, 2->3, 3->4) have weak support (n=3-16 successes), so by themselves they would not be trustworthy — but the cosine with the well-supported 0->1 direction (n=136) is still 0.66-0.76, which argues that the later-step directions are consistent with the 0->1 one rather than random.

The 0->1 transition is by far the best-supported (n=136 success vs n=191 failure). Everything downstream treats it as the primary evidence.

##### Analysis 3 — Convergence (directional consistency + projection per step)

*What*: For tasks with 3+ attempts, compute consecutive deltas and measure (a) cosine similarity between consecutive deltas within the same trajectory ("does the model keep moving the same way across repair steps?"), and (b) the scalar projection of each delta onto the pooled repair direction.

*Results*:

- Directional consistency (pass trajectories): cos = -0.185 ± 0.194 (n=37 delta pairs)
- Directional consistency (fail trajectories): cos = -0.253 ± 0.248 (n=498 delta pairs)

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

*What*: Address the core concern from Experiment 4 — "is the 0->1 repair direction real, or would shuffled labels produce similar norms?" Collect all 327 0->1 deltas. Record the real norm 40.54. Then 1000 times, shuffle the pass/fail labels and recompute `‖mean(pass) - mean(fail)‖`. Compare the real norm to the permuted null distribution. A p-value of (# permuted norms >= real) / 1000.

*Results*:

- Real direction norm: **40.54**
- Permuted null: mean = 8.48, std = 1.48, max = 14.997
- **p-value = 0.000** (not a single permutation out of 1000 reached the real norm; the real norm sits ~21 standard deviations above the null mean)

Output image: `results/h2/trajectory_analysis/permutation_test_0to1.png`

*Interpretation*: The direction is statistically real. It is not a chance artifact of comparing two arbitrary subsets of the delta population. Whatever it encodes is specifically tied to the pass/fail label, not to any shared property of 0->1 transitions. Note that this test does NOT prove the direction encodes "understanding the fix" — it only rules out random chance. We still need to control for confounds like prompt length (done in Analyses 8 and 13).

##### Analysis 7 (helper) — PCA reduction fitted on attempt-0 states only

*What*: A helper used by Analyses 8, 9, 10. Fit PCA(n=100) on the 444 attempt-0 hidden states, then project ALL attempts (0-4) into that 100-dim subspace. The design goal: define a coordinate system that cannot encode prompt structure, because PCA was fit only on states that all have the same prompt structure (attempt-0 prompts). Attempts 1+ are projected INTO this space but never influence the axes.

*Result*: 100 components capture 93.2% of attempt-0 variance.

*Caveat found later in Analysis 13-style checks*: even the attempt-0 PCA is *partially* contaminated by prompt length. A follow-up check showed PC1 of the attempt-0 PCA correlates with `prompt_tokens` at r=0.509 (because different difficulty tasks have different prompt lengths even at attempt 0). So this coordinate system is cleaner than the full-data one (Analysis 4), but it is not prompt-length-free. This is relevant for Analyses 8/9/10.

##### Analysis 8 — Repair direction recomputed in attempt-0-fitted PCA space

*What*: Redo Analysis 2 in the 100-dim attempt-0 PCA subspace.

*Results*:

| Transition | ‖direction‖ (raw) | ‖direction‖ (PCA-100) |
|---|---|---|
| 0 -> 1 | 40.54 | 26.69 |
| 1 -> 2 | 43.70 | 31.39 |
| 2 -> 3 | 27.02 | 18.27 |
| 3 -> 4 | 60.91 | 44.89 |

Cosine consistency (PCA space): 0->1 vs 1->2 = 0.717, 1->2 vs 2->3 = 0.827, 1->2 vs 3->4 = **0.880**, 2->3 vs 3->4 = 0.720. Pooled direction norm: 59.52.

*Interpretation*: The direction survives dimensionality reduction. Norms shrink (expected — we dropped 2460 dimensions, some of which carried signal), but the cosine consistency between *later*-step transitions actually do not disappear (e.g., 1->2 vs 3->4 goes from 0.840 to 0.880). This is the cleanest geometric result: removing the 2460 dimensions that did not show attempt-0 variance do not decrease the repair-direction consistency between 1->2, 2->3, and 3->4. The 0->1 transition drops slightly in cosine alignment with later steps, consistent with the interpretation that 0->1 carries an additional prompt-structure component that later transitions do not.

**Gap**: we did not run a permutation test in PCA space — the norm shrinkage (40.54 -> 26.69) cannot be directly compared to the 2560-dim null (mean 8.48). A permutation test on the PCA-reduced 0->1 deltas would close this gap and is a reasonable next step.

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

*Interpretation*: Both distributions are entirely positive (all projections in the 60-140 range), because the 0->1 step includes the shared prompt-structure shift that pushes every delta forward along this direction. The pass and fail distributions are clearly offset (means differ by ~18) but visibly overlap in the 95-120 range. Cohen's d = 0.91 confirms a real mean difference, but the overlap rules out using a single projection value as a reliable per-task predictor. **Interpretation**: the "pass" deltas are longer projections along the repair direction than "fail" deltas — the shift is not just present but *bigger in magnitude* when the model is about to succeed.

##### Analysis 12 — Focused PCA plot (attempt 0 -> attempt 1 transitions only)

*What*: Take only the 327 non-loop tasks with at least 2 attempts. Fit PCA(n=2) on just the attempt-0 states. Project both attempt-0 and attempt-1 states into this 2D space. Plot circles for attempt-0, triangles for attempt-1, arrows from 0 to 1, colored by attempt-1 outcome (green pass / red fail). Avoids the clutter of the Analysis-4 global plot.

*Result*: PC1 = 23.7% variance, PC2 = 13.0% variance, 327 tasks (136 pass, 191 fail).

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

#### 5.3 Follow-up control test — residualizing out prompt length from the probe

After seeing Analysis 13, we ran a control experiment outside the 13-analysis script to answer the question directly: *"Is the Experiment 2 probe just learning prompt length?"*

*Method*: For each of the 2560 hidden-state dimensions at attempt 0, fit a univariate linear regression on `prompt_tokens` and subtract the fitted line. This removes the linear effect of prompt length from every dimension. The resulting "residual" hidden states have *zero* linear correlation with `prompt_tokens` on every PC (verified: r = 0.000 for PC1-PC5 after residualization). Then train the standard probe on these residuals.

*Results*:

| Predictor | CV AUC | Test AUC |
|---|---|---|
| `prompt_tokens` alone | 0.761 | 0.786 |
| Hidden states (original, baseline) | 0.923 | 0.927 |
| Hidden states with `prompt_tokens` regressed out | **0.890** | **0.879** |

*Interpretation*: Prompt length by itself is a non-trivial predictor (AUC 0.76) — shorter prompts tend to be easier problems. But the probe on residualized hidden states still achieves AUC ~0.88. The drop from 0.92 to 0.88 says prompt length was contributing about 4 points of AUC to the original probe. The remaining ~12 points of AUC above prompt-length-alone (0.88 vs 0.76) is signal that cannot be reduced to prompt length by any linear function. **The probe is not merely reading prompt length. The signal is predominantly beyond prompt structure.**

This also partially rehabilitates the repair direction: the direction is computed from deltas, and every 0->1 delta shares approximately the same prompt-length change, so the contrastive subtraction cancels out most of the prompt-length contribution even without explicit residualization. Combined with the permutation test (Analysis 6) and the residualized probe, prompt length cannot be the sole explanation of the direction.

#### 5.4 Consolidated summary — Experiment 5

**What worked (solid, defensible):**

1. **The probe replicates at scale** on 444 mixed-difficulty tasks: attempt-0 test AUC 0.927, CV AUC 0.923. Same number as Experiment 2, different dataset, larger sample. (Analysis 1)

2. **The probe survives prompt-length residualization**: AUC drops from 0.927 to 0.879 when prompt length is linearly removed from every hidden-state dimension. Prompt length contributes ~4 AUC points; the remaining ~12 AUC points above prompt-length-alone (0.76) is real signal beyond prompt structure. (Follow-up control test)

3. **The 0->1 repair direction is statistically significant**: real norm 40.54 vs permuted null mean 8.48 (std 1.48, max 14.997), p < 0.001 with 1000 permutations. Not a chance artifact. (Analysis 6)

4. **The direction has a stable geometric signature across later repair steps** (cosine similarity 0.83-0.88 between 1->2, 2->3, 3->4 in the PCA-reduced space). It is not one-shot; it is a reproducible property of "the shift that happens when the model is about to produce correct code." (Analyses 2, 8)

5. **The direction encodes both inclination and magnitude**: Cohen's d = 0.91 on projection values of individual 0->1 deltas onto the pooled repair direction. The shift for pass deltas is not just in a characteristic direction, it is also *larger* in that direction than for fail deltas. (Analysis 11)

**What did not work (honest negatives):**

6. **No convergence.** Directional consistency between consecutive deltas within the same trajectory is slightly negative for both pass (cos = -0.185) and fail (cos = -0.253) groups. The model does not progressively move toward a solution in representation space. Each repair step is nearly independent of the previous one. (Analyses 3, 9)

7. **Pass rates collapse after attempt 1** (15.8% -> 36.4% -> 6.7% -> 2.7% -> 1.4%). Error feedback helps dramatically once, then hits a ceiling. Consistent with "no iterative refinement."

8. **Analysis 4 (full-data PCA trajectory plot) is contaminated**: PC1 (r=-0.765 with prompt_tokens) is mostly encoding prompt length, not repair geometry. The triangle/arrow structure in that plot is largely an artifact. (Analysis 13)

9. **Analysis 5 (distance to attempt-0-pass centroid) inherits the same confound** and shows no separation between pass and fail distance curves after step 1.

10. **Analysis 10 (distance to first-pass centroid in PCA space)** showed a modest step-1 gap (50.5 vs 62.9) that does not persist and is not strong enough to build a claim on.

11. **Analysis 12 (focused PCA plot)** shows weak visual separation but only explains 36.7% of attempt-0 variance in 2D — not actionable.

**Gaps and things we did not do:**

- No permutation test in PCA-reduced space (Analysis 8 norms cannot be directly compared to Analysis 6 null).
- Can we confirm with the permutation test (or similar test) that also the direction inclination (not just the magnitude) of success/correct code is real and is not just random result? Would it make sense?
- The results and beliefs we have about direction of success both in terms of inclination and magnitude, can be deleted by the facts that the PC1 (fitted on all attempts data) encode the prompt structure? or maybe by the fact that even the PCA fitted on attept 0 data encodes partially the prompt length (that is related to the task difficulty) ? Does these two facts disrupt (even partially) our most important thesis from these experiment, indeed, about the inclination e magnitude of direction of success? Explain your answer well
- Would be good to write down a test (we did it fastly without writing it down) to prove that the prediction of the success generation of code by using the hidden state vector, so the  Experiment 2: H2 Probe — Logistic Classifier on Hidden States (2026-03-14) is not just reading the prompot length, that in our dataset is partially related to the prompt length 
- We should report the F1 scores for the analysis 2, as we done for Experiment 2: H2 Probe — Logistic Classifier on Hidden States (2026-03-14), explain their values, and if and why they change from the Experiment 2: H2 Probe — Logistic Classifier on Hidden States (2026-03-14)
- No per-error-type breakdown of "which error types lead to successful repair" (would help distinguish "the model genuinely cannot extract more info from later errors" from "sampling noise").
- No residualization of the repair direction itself against prompt length (the contrastive subtraction does it implicitly, but an explicit test would be cleaner).
- Later-step analyses (1->2, 2->3, 3->4) have n_success = 16, 6, 3 — these support but cannot by themselves establish the direction claim. All load-bearing evidence is the 0->1 transition.

#### 5.5 Proposed unified insight (DRAFT — to be discussed together)

> **Layer 35 of Qwen3-4B carries a static, decodable signal about upcoming code correctness, including a reproducible geometric "about-to-succeed" direction — but the model does not iteratively follow this direction across repair attempts.**
>
> - A linear probe on layer-35 hidden states at the moment the model finishes reading a prompt (attempt 0) predicts whether the upcoming generation will pass all tests, with AUC 0.927 raw / 0.879 after linearly removing prompt length. This replicates and strengthens Experiment 2 on 4.4x more data across the full LCB difficulty range, with a confound control that Experiment 2 did not have.
>
> - Beyond a static "will it work" signal, there is also a **repair-success direction**: when the model receives error feedback from a failed attempt, the contrastive shift `mean(h_next | next passes) - mean(h_next | next fails)` is a stable vector with non-random norm (permutation test p < 0.001 at 0->1, n=327) and with cosine similarity 0.83-0.88 between later repair steps in the PCA-reduced attempt-0 coordinate system. Individual pass deltas are longer projections along this direction than fail deltas (Cohen's d = 0.91). This suggests that the "moment the model understands the problem and is about to produce correct code" is accompanied by a characteristic, reproducible transformation of the hidden state — both in inclination (which direction it moves) and in magnitude (how far it moves).
>
> - However, this direction is **not a trajectory the model follows iteratively**. Consecutive repair steps within the same task are slightly anticorrelated in direction (cos ≈ -0.2 for both pass and fail trajectories), there is no distance-to-success convergence, and raw pass rates collapse after the first repair attempt (36% -> 7% -> 3% -> 1%). The error feedback appears to provide a one-shot benefit: the first repair attempt re-reads the problem with additional context and either lands in the success region or does not. Subsequent attempts do not compound.
>
> - Together, Experiments 2 and 5 support a picture in which layer-35 of a 4B-parameter code model holds a *snapshot* representation of "is this code going to be correct," and the repair process is a discrete re-read with a characteristic activation-space shift — not a continuous reasoning trajectory. For practical inference-time scaling on models of this size, this would predict that investing in *the quality of the first error message* matters more than allowing many repair attempts.

This draft insight matches the intuition you described: **the direction is present when the model is "about to produce correct code," it is different from the direction it takes when it is about to produce incorrect code, and this is what the model's hidden state encodes about its own understanding of the task.** It is important to be honest that we do NOT have evidence of iterative convergence or a coherent multi-step trajectory — only of a reproducible one-shot transformation, most strongly at the 0->1 step where the data is plentiful, and weakly supported at later steps where n is small but the geometric signature is consistent with the 0->1 direction.

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
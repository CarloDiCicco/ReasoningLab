# ReasoningLab

Mechanistic interpretability of code reasoning in Qwen3-4B-Instruct-25-07 LLM model.

**What this studies.** Large language models process text through stacked transformer layers, each producing a hidden-state vector (a high-dimensional numerical representation of the input so far). In a 36-layer model like Qwen3-4B-Instruct-25-07, layer 35 (the penultimate) carries the model's highest-level semantic representation before it has to worry about the constraints of human language, its integrated understanding of the prompt before the final projection to output token probabilities. The hidden state at layer 35 of the last prompt token, the penultimate layer's representation of the final input position, is the model's "final internal summary" of what it has read, captured *before* it begins generating any output. This study asks: what information about code correctness is already readable from that vector?

---

## Summary

- A linear probe on layer-35 hidden states predicts whether the model's next generation will pass unit tests **before a single output token is produced**, reaching a mean **test AUC of 0.917** (95% CI ±0.009) over 50 random 80/20 splits. After linearly removing the effect of prompt length from the hidden state, mean AUC is 0.889 (±0.012); a prompt-length-only baseline reaches 0.754 (±0.014). Trained and evaluated on 444 LiveCodeBench tasks spanning easy, medium, and hard difficulty.
- At the first repair attempt, the hidden-state shift from attempt 0 to attempt 1 follows a **reproducible geometric direction** that distinguishes successful repairs from failed ones, on two independent tests: a magnitude test (success-vs-failure mean difference of norm 23.78, ~10 standard deviations above a label-shuffled baseline) and an inclination test (split-half cosine 0.756 over 1000 stratified halves, both p < 0.001). However, all three observable covariates differing between groups (attempt-0 code length, prompt-length growth, failure type) jointly explain most of this signal: after conditional residualization against them, the direction norm drops from 23.78 to 9.67 (p = 0.069) and the split-half cosine collapses to 0.181 (p = 0.857). The raw signal is therefore best read as a *correlate* of repair success, partially mediated by repair-context structure, not as evidence of an isolated repair-comprehension direction in activation space.
- This signal also does **not** propagate into iterative refinement. Pass rates collapse after the first repair (15.8% → 36.4% → 6.7% → 2.7% → 1.4%), and later transitions have too few successful repairs (13, 4, 2) for any statistical test to recover a direction. No proper trajectory of repair, and no persistent "region of success" in activation space, was found in this study.

---

## Setup

- **Model**: `Qwen/Qwen3-4B-Instruct-2507`
- **Data**: 444 LiveCodeBench tasks (easy / medium / hard).
- **Capture point**: layer 35 (penultimate decoder layer) hidden state vector, last prompt token, single forward pass on the prompt, before any generation. Layer 35 was selected from an earlier layer sweep over layers 29–36 in which it was the consistent top performer on both AUC and F1.
- **Hardware**: single laptop, RTX 4050 6 GB.
- **Libraries**: `transformers`, `scikit-learn`, `numpy`.

---

## Result 1 — Correctness is linearly readable before generation

The hidden state at the last prompt token already encodes information about whether the upcoming generation will be correct, before any code is produced.

**Setup.** For each of the 444 LCB tasks, the layer-35 hidden state at the last prompt token is captured (one sample per task, so no task appears in both train and test, avoiding the inflated accuracy that would come from having multiple attempts of the same task split across sets). A standard three-step linear probe is trained over **50 random 80/20 splits** (mean ± 95% CI reported below). On each split, 5-fold cross-validation runs within the training portion only for hyperparameter selection (regularization strength C):

1. `StandardScaler` (fit on train only)
2. `PCA` retaining 95% of variance (fit on train only)
3. `LogisticRegression` with L2 regularization, `C` selected by the CV above (scored on `roc_auc`)

Three variants:

| Probe                                | Test AUC (mean ± 95% CI) | CV AUC (mean ± 95% CI) |
|---                                   |---                       |---                     |
| Raw hidden states                    | 0.917 ± 0.009            | 0.925 ± 0.003          |
| Residualized (prompt length removed) | 0.889 ± 0.012            | 0.885 ± 0.005          |
| Prompt length only (baseline)        | 0.754 ± 0.014            | 0.751 ± 0.004          |

**Why residualize.** Prompt length is itself correlated with task difficulty in this dataset (longer prompts tend to be the harder LCB tasks), so a probe could in principle "cheat" by reading off how many prompt tokens went in rather than anything mechanistic about the model's internal state. The residualized variant linearly regresses each of the 2560 hidden-state dimensions on `prompt_tokens` (fit on train only), subtracts the prediction from both train and test, and trains the same probe on what's left. This removes the linear contribution of prompt length from every dimension. The prompt-only baseline goes one step further and asks how well a probe can do using `prompt_tokens` alone, i.e. how much of the raw probe's AUC could in principle be explained by length. The 0.754 floor and the residualized 0.889 together show the hidden state carries roughly **14 AUC points of signal beyond prompt length**, before any output token is sampled.

---

## Result 2 — A reproducible direction of repair success

**Context.** The data comes from a "repair-B" policy: for each task, the model generates a solution (attempt 0), the code is executed against unit tests, and if it fails the model receives its own code + the error message + a one-line "fix the code" instruction and tries again, up to 5 total attempts. Each attempt is a fresh forward pass on the full prompt (no chat memory, no KV-cache carryover between attempts). The repair prompt structure is: `{original problem} + {previous code} + {error output} + {fix instruction}`.

**Setup.** For each task where attempt 0 fails and attempt 1 exists, compute the per-task hidden-state delta at layer 35, capturing what changes in the model's representation when it re-reads the problem after seeing its own failure:

```
delta_i = hidden_state_layer35(attempt_1_i) − hidden_state_layer35(attempt_0_i)
```

Each attempt is a **fresh forward pass** (the model has no state carried between attempts), so `delta_i` is entirely driven by what the model re-reads the second time: the failing code from attempt 0, plus the error message, plus the retry instruction. Split the deltas by whether attempt 1 passed, giving a success set (n = 128) and a failure set (n = 108) on 236 clean tasks. The candidate direction is the contrastive mean difference:

```
v = mean(success_deltas) − mean(failure_deltas)
```

**Why the contrastive construction is the appropriate one here.** Both groups read the same prompt template (problem + wrong code + error message + retry instruction), so components that are equally distributed across the two groups cancel in the difference `mean(success) − mean(failure)`. What survives is whatever differs between the groups: potentially repair-relevant signal, but also any repair-context covariates that are not balanced between successes and failures.

Two independent tests are run on `v`, in the full 2560-dimensional layer-35 space. Both compare the real result to an empirical **null distribution** (what the statistic would look like if the success/failure labels carried no information). For the magnitude test, the null is generated by shuffling the 128/108 labels across the 236 deltas 1000 times and recomputing the norm. For the inclination test, the real split-half cosine is compared against split-half cosines computed after shuffled-label assignments.

| Test                                                          | Real value | Null distribution         | p-value  |
|---                                                            |---         |---                        |---       |
| **Magnitude**: ‖v‖ in 2560-dim space                          | 23.78      | mean 9.07, max 16.14      | < 0.001  |
| **Inclination**: split-half cosine over 1000 stratified halves| 0.756      | mean −0.009               | < 0.001  |

The magnitude test asks whether the success/failure mean difference is larger than label noise can explain. It is, by roughly 10 standard deviations of the shuffled-label distribution (real 23.78 vs null mean 9.07, std 1.43), and the largest of 1000 shuffled norms (16.14) doesn't even reach the real norm. (0 of 1000 permutations exceeded the real norm; reported as p < 0.001).

The inclination test asks: if the 236 deltas are randomly split into two halves (each preserving the original 128/108 success/failure ratio) and the direction `v` is recomputed independently on each half, do the two directions agree? Over 1000 random splits the average cosine is 0.756, meaning the two halves consistently discover the same direction. Under the null (shuffled labels), the same procedure averages to ≈ 0.

**Sensitivity check: how much of this is driven by the repair context itself?**

The contrastive construction `mean(success) − mean(failure)` automatically cancels any covariate that is *equally distributed* across the success and failure groups. Whether residualizing further is meaningful depends on whether candidate covariates actually differ between the two groups. Three covariates were checked:

| Covariate                                       | Success mean (n=128) | Failure mean (n=108) | p-value  |
|---                                              |---                   |---                   |---       |
| `delta_prompt_tokens` (attempt 1 − attempt 0)   | 337.1                | 431.5                | < 0.001  |
| `code_length_attempt_0` (completion tokens)     | 188.8                | 290.9                | < 0.001  |
| `failure_type` of attempt 0                     | —                    | —                    | < 0.001  |

All three differ significantly. Tasks that get repaired successfully tend to come from shorter, less messy attempt-0 code and a different distribution of failure types (more assertion errors, fewer runtime errors). The contrastive subtraction does *not* cancel these covariates, because they are not equally distributed across the two groups.

Residualizing each of the 2560 hidden-state delta dimensions against a 4-column covariate matrix, fitted per-task: the two continuous variables (`delta_prompt_tokens`, `code_length_attempt_0`) plus 2 binary columns encoding `failure_type` (3 observed levels: assertion, runtime, syntax; one-hot with assertion as baseline, giving 2 dummy columns), then recomputing the direction:

| Statistic                | Raw     | Conditional residualization |
|---                       |---      |---                          |
| Direction norm           | 23.78   | 9.67 (40.7% retained)       |
| Permutation p-value      | < 0.001 | 0.069                       |
| Split-half cosine        | 0.756   | 0.181                       |
| Split-half p-value       | < 0.001 | 0.857                       |

**The signal does not survive the full conditional residualization.** The residual direction is not statistically distinguishable from noise after controlling for the differentially-distributed observable covariates. This should not be read as the raw signal being "false": the raw direction passes both statistical tests by wide margins. 0 of 1000 permutations reached the real norm, and the split-half cosine (0.756) is far above the null mean (0.009). The honest reading: the observed geometric direction is a *correlate* of repair success, substantially mediated by what differs in the repair context (shorter wrong code → shorter repair prompt → different geometric shift), not evidence of an isolated repair-comprehension direction in activation space.

---

## A note on what repair looks like beyond the first attempt

The Result 2 direction is statistically solid at the **first** repair step (0 → 1), where there are 128 successes and 108 failures, enough data for both the magnitude and inclination tests. At later transitions (1→2, 2→3, 3→4), the number of successful repairs drops to **13, 4, and 2** respectively. With so few successes, no contrastive direction can be meaningfully estimated, regardless of whether one exists.

Pass rates across the 5 attempts:

| Attempt | Pass rate |
|---      |---        |
| 0       | 15.8%     |
| 1       | 36.4%     |
| 2       |  6.7%     |
| 3       |  2.7%     |
| 4       |  1.4%     |

The bulk of the gain happens at attempt 1 and never recovers. The first piece of error feedback delivers most of the information the model can use; subsequent retries yield diminishing returns. Extending the 0→1 direction analysis to later steps would require a dataset with far more successful multi-step repairs than this one provides.

---

## Further null results

- **Directional consistency across consecutive repair steps.** The cosine between `delta_{t→t+1}` and `delta_{t+1→t+2}` within the same trajectory averages to ≈ −0.14 (pass) and ≈ −0.23 (fail), slightly anti-correlated, despite the limited data at later steps.
- **Distance-to-success-centroid.** Tracked the distance from each attempt's hidden state to the centroid of attempt-0 passing tasks; found no monotonic convergence for failing trajectories, as curves for pass and fail stay essentially parallel from attempt 1 onward.

---

## Repo structure

```
ReasoningLab/
├── README.md                          ← this file
├── LICENSE                            ← Apache 2.0
├── scripts/
│   ├── analyze_trajectories.py        ← all reported numbers: probe, direction, permutation, split-half, convergence
│   ├── train_probe.py                 ← standalone probe trainer (layer sweeps, --residualize)
│   └── prepare_livecodebench.py       ← LCB data prep + utilities
├── src/reasoninglab/
│   ├── probing/                       ← reusable probe + hidden-state utilities
│   ├── policies/                      ← sampling / repair policies
│   ├── eval/                          ← pass@k evaluation harness
│   ├── tasks/                         ← benchmark adapters (LCB, HumanEval)
│   └── models/                        ← model loading / hidden-state hooks
├── configs/h2/                        ← run configs for the reported experiments
├── runs/
│   └── h2-trajectory-all/                          ← 444-task 5-attempt LCB trajectory data (Results 1, 2)
└── results/h2/
    ├── probe/metrics.json                         ← per-layer probe metrics
    └── trajectory_analysis/metrics.json           ← all direction / permutation / split-half metrics
```

---

## Reproduce

```bash
# Clone and set up (Python 3.11+)
git clone <repo-url> && cd ReasoningLab
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[probe]"        # installs scikit-learn and matplotlib

# Verify all reported numbers — reads pre-saved hidden states, no GPU required
python scripts/analyze_trajectories.py
# outputs results/h2/trajectory_analysis/metrics.json and PNG figures

# To re-run inference from scratch you additionally need:
#   pip install -e ".[model]"    # transformers, accelerate, bitsandbytes
#   Qwen/Qwen3-4B-Instruct-2507 weights (downloaded automatically via transformers)
#   an RTX-class GPU (experiments ran on RTX 4050 6 GB)
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

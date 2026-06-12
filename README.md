# Code Correctness Signals in LLM Hidden States: Pre-Generation Probing and Repair Geometry

**Paper:** arXiv link forthcoming.

**What this studies.** Large language models process text through stacked transformer layers, each producing a hidden state vector that numerically represents the input so far. This study reads the hidden state at the last prompt token of Qwen3-4B-Instruct-2507, captured on a single forward pass over the prompt before any output token is generated, and asks two questions. First, before the model writes anything, can the hidden state already predict whether the code it is about to generate will pass? Second, when the model is given a chance to repair a failed attempt, does the shift in its hidden state from the failed attempt to the repair carry a stable signature of whether the repair will succeed? The specific layer used is layer 29, an upper transformer block selected by nested cross-validation (CV) as described in Result 1, and all reported analyses use that layer.

---

## Summary

- On 444 LiveCodeBench tasks spanning easy, medium, and hard difficulty, a linear probe on layer-29 hidden states predicts whether the model's next generation will pass unit tests **before a single output token is produced**, reaching a mean **test area under the ROC curve (AUC) of 0.931** with a 95% confidence interval (CI) of ±0.008 over 50 outer splits. The probe layer is selected by nested CV, so the score is a leakage-free estimate. After linearly removing the effect of prompt length from the hidden state, mean AUC is 0.911 ± 0.010, well above a prompt-length-only baseline of 0.754 ± 0.014.
- At the first repair attempt, the hidden state shift from attempt 0 to attempt 1 follows a **statistically detectable geometric direction** that distinguishes successful repairs from failed ones, on two complementary tests. The magnitude test asks whether the gap between successful and failed repairs is larger than label noise can explain, and it is, sitting about 10 standard deviations above a label-shuffled baseline. The inclination test asks whether the same direction reappears when the data is split in half, and it does. Both reach p < 0.001. However, differences in observable repair-context covariates between the two groups jointly explain most of this signal. After conditional residualization against them the direction norm drops from 9.76 to 3.72 at p = 0.126 and the split-half cosine collapses to 0.140 at p = 0.942. The raw signal is therefore best read as a *correlate* of repair success, driven by repair-context structure, not as evidence of an isolated repair-comprehension direction in activation space.
- This signal also does **not** propagate into iterative refinement. Pass rates peak at the first repair, rising from 15.8% at attempt 0 to 36.4% at attempt 1, then collapse to 6.7%, 2.7%, and 1.4% at the later attempts. Later transitions have too few successful repairs, 13 then 4 then 2, for any statistical test to recover a direction.

---

## Setup

- **Model**: `Qwen/Qwen3-4B-Instruct-2507`
- **Data**: 444 LiveCodeBench tasks spanning easy, medium, and hard difficulty.
- **Capture point**: hidden state vector at the last prompt token, single forward pass on the prompt, before any generation. Hidden states are captured for the upper transformer blocks, layers 29 to 36. The specific layer used for all analyses is layer 29, selected by nested CV as described in Result 1.
- **Hardware**: single laptop, RTX 4050 6 GB.
- **Libraries**: `transformers`, `scikit-learn`, `numpy`.

---

## Result 1: Correctness is linearly readable before generation

The hidden state at the last prompt token already encodes information about whether the upcoming generation will be correct, before any code is produced.

**Setup.** For each of the 444 LCB tasks, the layer-29 hidden state at the last prompt token is captured. One sample per task means no task appears in both train and test, which avoids the inflated accuracy that would come from splitting multiple attempts of the same task across sets. The probe runs on the full 444-task set with no filtering. The cleaning filters described in Result 2 address a problem specific to the repair-direction analysis and do not apply here. A standard three-step linear probe is trained and evaluated over **50 random 80/20 splits**, with mean and 95% CI reported below. The probe pipeline is:

1. `StandardScaler`, fit on the train fold only.
2. `PCA` retaining 95% of variance, fit on the train fold only.
3. `LogisticRegression` with L2 regularization, strength `C`.

**Optimizing the hyperparameters.** Both the layer and `C` are hyperparameters, and picking them by the best test score of a per-layer sweep would leak the test set into model selection and inflate the reported number. Instead the layer and `C` are chosen jointly by nested CV. Within each of the 50 outer 80/20 splits, an inner 5-fold CV on the training portion selects the configuration, which is then evaluated once on the untouched outer-test fold. Layer 29 is the modal selection, chosen in 25 of 50 splits, with layer 30 a near-tie at 24. All reported AUCs are the leakage-free outer-test scores of this full select-then-fit procedure.

**Why residualize.** Prompt length is itself correlated with task difficulty in this dataset, since longer prompts tend to be the harder LCB tasks, so a probe could in principle "cheat" by reading off how many prompt tokens went in rather than anything mechanistic about the model's internal state. The residualized variant linearly regresses each of the 2560 hidden state dimensions on `prompt_tokens`, fit on train only, subtracts the prediction from both train and test, and trains the same probe on what's left. This removes the linear contribution of prompt length from every dimension. The prompt-only baseline goes one step further and asks how well a model fit on `prompt_tokens` alone can do, giving a reference for how far length by itself gets you.

Three configurations, reported as the mean outer-test AUC of the nested CV procedure:

| Configuration                        | Outer-test AUC (mean ± 95% CI) |
|---                                   |---                             |
| Raw hidden states                    | 0.931 ± 0.008                  |
| Residualized (prompt length removed) | 0.911 ± 0.010                  |
| Prompt length only (baseline)        | 0.754 ± 0.014                  |

The residualized 0.911 sits about **16 AUC points above the 0.754 prompt-length floor**, so the hidden state carries a substantial pass/fail signal that is not reducible to prompt length. This signal is available before the model writes a single output token. The 0.754 baseline uses logistic regression. A random forest, a gradient boosting model, and a small multilayer perceptron, each refit on `prompt_tokens` alone using the same nested CV procedure, reach 0.735, 0.736, and 0.753, none of them beating the logistic baseline. Prompt length alone does not approach the probe even with these more flexible models, so the gap is not an artifact of using a linear baseline.

---

## Result 2: A Geometric Direction of Repair Success

**Context.** The data comes from a "repair-B" policy. For each task the model generates a solution at attempt 0, the code is executed against unit tests, and if it fails the model is re-prompted with the failure information and tries again, up to 5 total attempts. Each attempt is a fresh forward pass on the full prompt, with no state carried over from previous attempts. The repair prompt structure is `{original problem} + {failure-type sentence} + {previous code} + {error output} + {fix instruction}`, where the failure-type sentence is a single line of the form "Your previous attempt produced a `{failure_type}` error", with `failure_type` drawn from the verifier's taxonomy of assertion, runtime, syntax, timeout, plus a rare unknown fallback.

**Setup.** For each task where attempt 0 fails and attempt 1 exists, compute the per-task hidden state delta at layer 29, capturing what changes in the model's representation when it re-reads the problem after seeing its own failure:

```
delta_i = hidden_state_layer29(attempt_1_i) − hidden_state_layer29(attempt_0_i)
```

Because no state is carried between attempts, `delta_i` is driven entirely by what the model re-reads on attempt 1, the repair prompt defined above. Splitting the deltas by whether attempt 1 passed gives a success set of 128 and a failure set of 108 on 236 clean tasks. Two filters produce this clean sample. The first removes whole trajectories where the model is stuck in a repetition loop across all five attempts, hitting the 768-token cap with broken code on every retry. All of these tasks fail at every attempt, so they contribute only failure-labeled deltas and would contaminate the success-versus-failure contrast. The second removes the remaining tasks where attempt 0 alone hit the 768-token cap with repetitive broken code, which is then pasted verbatim into the repair prompt. The vast majority of these tasks, 83 of 91, also fail at attempt 1, so they too would inflate the failure group and contaminate the contrast. Both filters therefore prevent a mechanical contamination that would have been almost entirely concentrated on the failure side. The candidate direction is the contrastive mean difference:

```
v = mean(success_deltas) − mean(failure_deltas)
```

**Why the contrastive construction is the appropriate one here.** Both groups read the same prompt template of problem, wrong code, error message, and retry instruction, so components whose mean is equal across the two groups cancel in the difference `mean(success) − mean(failure)`. What survives is whatever differs in group mean between them, which is potentially repair-relevant signal but also any repair-context covariates whose means are not balanced between successes and failures.

Two complementary tests are run on `v`, in the full 2560-dimensional layer-29 space. Both compare the real result to an empirical **null distribution**, that is, what the statistic would look like if the success/failure labels carried no information. For the magnitude test, the null is generated by shuffling the 128/108 labels across the 236 deltas 1000 times and recomputing the direction's norm ‖v‖ each time. For the inclination test, the real split-half cosine is compared against split-half cosines computed after shuffled-label assignments.

| Test                                                          | Real value | Null distribution         | p-value  |
|---                                                            |---         |---                        |---       |
| **Magnitude**: ‖v‖ in 2560-dim space                          | 9.76       | mean 3.64                 | < 0.001  |
| **Inclination**: split-half cosine over 1000 stratified halves| 0.767      | mean −0.011               | < 0.001  |

The magnitude test asks whether the success/failure mean difference is larger than label noise can explain. It is, by roughly 10 standard deviations of the shuffled-label distribution, with the real norm of 9.76 against a null mean of 3.64 and std 0.59. The largest of 1000 shuffled norms, 6.58, does not even reach the real norm. None of the 1000 permutations exceeded the real norm, so p < 0.001. The inclination test asks whether the direction is stable. If the 236 deltas are randomly split into two halves, each preserving the original 128/108 success/failure ratio, and the direction `v` is recomputed independently on each half, do the two directions agree? Over 1000 random splits the average cosine is 0.767, meaning the two halves consistently discover the same direction. Under the shuffled-label null, the same procedure averages to about 0.

**Sensitivity check: how much of this is driven by the repair context itself?**

The contrastive construction `mean(success) − mean(failure)` is a difference of group means, so it automatically cancels the linear contribution of any covariate whose mean is *equal* across the success and failure groups. For a one-hot-encoded categorical covariate this is equivalent to equal category proportions across groups. Whether residualizing further is meaningful therefore depends on whether candidate covariates actually differ in their group means between the two groups. Three covariates were checked:

| Covariate                                       | Success mean (n=128) | Failure mean (n=108) | p-value  |
|---                                              |---                   |---                   |---       |
| `delta_prompt_tokens` (attempt 1 − attempt 0)   | 337.1                | 431.5                | < 0.001  |
| `code_length_attempt_0` (completion tokens)     | 188.8                | 290.9                | < 0.001  |
| `failure_type` of attempt 0                     | —                    | —                    | < 0.001  |

All three differ significantly, by Welch's t-test for the two continuous covariates and a chi-squared test for `failure_type`. Tasks that get repaired successfully tend to come from shorter attempt-0 code and a different distribution of failure types, with more runtime errors and fewer assertion errors. The contrastive subtraction does *not* cancel these covariates, because their group means differ between successes and failures.

Each of the 2560 hidden state delta dimensions is residualized against a 4-column covariate matrix, fitted per task. The matrix holds the two continuous variables `delta_prompt_tokens` and `code_length_attempt_0`, plus 2 binary columns that one-hot encode `failure_type`. There are three observed levels, assertion, runtime, and syntax, with assertion as the baseline, which gives 2 dummy columns. The contrastive direction is recomputed on the residualized deltas and the two tests are rerun:

| Statistic                | Raw     | Residualized                |
|---                       |---      |---                          |
| Direction norm           | 9.76    | 3.72                        |
| Magnitude p-value        | < 0.001 | 0.126                       |
| Split-half cosine        | 0.767   | 0.140                       |
| Split-half p-value       | < 0.001 | 0.942                       |

**The signal does not survive the full conditional residualization.** The residual direction is not statistically distinguishable from noise after controlling for the observable covariates whose group means differ. This should not be read as the raw signal being "false", since the raw direction passes both statistical tests by wide margins. The honest reading is that the observed geometric direction is a *correlate* of repair success, driven by what differs in the repair context. There is no evidence of an isolated repair-comprehension direction in activation space.

**Note.** The two continuous covariates partially overlap in information content, since `delta_prompt_tokens` contains `code_length_attempt_0` with a Pearson r of 0.995. The resulting overfitting bias is under one percentage point of in-sample R². An extra regression column absorbs on average about 1/n of the variance even when it carries no real information, and the redundant column is a single parameter against 236 data points, so its spurious contribution is about 0.4 percentage points. The measured value agrees, since adding the redundant `delta_prompt_tokens` column on top of the other three raises the mean in-sample R² by under one percentage point. This is small enough that it does not affect the results.

---

## A note on what repair looks like beyond the first attempt

The Result 2 direction is statistically solid at the **first** repair step, from attempt 0 to attempt 1, where there are 128 successes and 108 failures, enough data for both the magnitude and inclination tests. At the later transitions from 1 to 2, 2 to 3, and 3 to 4, the number of successful repairs drops to **13, 4, and 2** respectively. With so few successes, no contrastive direction can be meaningfully estimated, regardless of whether one exists.

Pass rates across the 5 attempts:

| Attempt | Pass rate |
|---      |---        |
| 0       | 15.8%     |
| 1       | 36.4%     |
| 2       |  6.7%     |
| 3       |  2.7%     |
| 4       |  1.4%     |

The bulk of the gain happens at attempt 1 and never recovers. The first piece of error feedback delivers most of the information the model can use, and subsequent retries yield diminishing returns. Extending the attempt 0 to 1 direction analysis to later steps would require a dataset with far more successful multi-step repairs than this one provides.

---

## Limitations and further null results

- **Single model.** All experiments use one model, Qwen3-4B-Instruct-2507, on a single GPU. Replication across model families is left for future work.
- **Captured layer range.** Hidden states were captured only for the upper blocks, layers 29 to 36. The probe AUC peaks at the lower edge of that range, so a layer below 29 cannot be ruled out as comparable or better. The repair-direction verdict is, however, stable across layers 29, 30, and 35, making it unlikely, though unproven, that a nearby lower layer would change it.
- **Data starvation at later repairs.** The 1 to 2, 2 to 3, and 3 to 4 transitions have only 13, 4, and 2 successful repairs, too few to estimate a direction, so claims beyond the first repair are out of scope.
- **Covariate-set composition.** Two continuous covariates share information, `delta_prompt_tokens` contains `code_length_attempt_0` at Pearson r = 0.995. The overfitting bias is negligible because the redundant column is a single parameter against 236 data points, but a cleaner non-overlapping set would have needed re-running inference, which was beyond the available compute.
- **Directional consistency across repair steps (null result).** Within a single trajectory, consecutive deltas `delta_{t→t+1}` and `delta_{t+1→t+2}` have an average cosine of about −0.16 for eventually-passing tasks and −0.25 for eventually-failing tasks, slightly anti-correlated in both cases. This statistic needs trajectories with at least three attempts, which only exist at the data-starved transitions beyond the first repair, so the result is inconclusive.
- **Distance to the success region (null result).** The distance from each attempt's hidden state to the centroid of attempt-0 passing tasks, the "success region", was tracked across attempts. The curves for eventually-passing and eventually-failing tasks track each other closely from attempt 1 onward, with neither approaching the success region, so no monotonic convergence was observed.

---

## Repo structure

```
ReasoningLab/
├── README.md                          ← this file
├── LICENSE                            ← Apache 2.0
├── scripts/
│   ├── analyze_trajectories.py        ← Result 2 and the further null results: direction, permutation, split-half, convergence, distance (layer 29)
│   ├── layer_nested_cv.py             ← nested CV layer+C selection for the probe (raw and --residualize)
│   ├── prompt_only_nested_cv.py       ← prompt-length baseline under the same nested CV procedure
│   ├── prompt_only_nonlinear.py       ← prompt-length baseline across logistic, random forest, gradient boosting, MLP
│   ├── layer_sweep_multisplit.py      ← descriptive per-layer sweep (the layer landscape)
│   ├── train_probe.py                 ← standalone probe trainer (single-split, --residualize)
│   ├── verify_covariate_redundancy.py ← backs the Result 2 Note: Pearson r and the in-sample R² gain of the redundant covariate
│   └── prepare_livecodebench.py       ← LCB data prep + utilities
├── src/reasoninglab/
│   ├── probing/                       ← reusable probe + hidden state utilities
│   ├── policies/                      ← sampling / repair policies
│   ├── eval/                          ← pass@k evaluation harness
│   ├── tasks/                         ← benchmark adapters (LCB, HumanEval)
│   └── models/                        ← model loading / hidden state hooks
├── configs/h2/                        ← run configs for the reported experiments
├── runs/
│   └── h2-trajectory-all/                          ← 444-task 5-attempt LCB trajectory data (Results 1, 2)
└── results/h2/
    ├── probe_lcb444_multisplit/metrics.json       ← per-layer sweep (layer landscape)
    ├── probe_lcb444_nested_cv_grid/metrics.json   ← nested CV raw-probe selection + outer-test AUC
    ├── probe_lcb444_nested_cv_grid_resid/metrics.json ← nested CV residualized-probe results
    ├── prompt_only_nested_cv/metrics.json         ← prompt-length baseline (nested CV)
    ├── prompt_only_nonlinear/metrics.json         ← prompt-length baseline across model families (robustness check)
    └── trajectory_analysis/metrics.json           ← repair-direction / permutation / split-half / convergence (layer 29)
```

---

## Reproduce

```bash
# Clone and set up (Python 3.11+)
git clone <repo-url> && cd ReasoningLab
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[probe]"        # installs scikit-learn and matplotlib

# Verify all reported numbers. Reads pre-saved hidden states, no GPU required.
python scripts/analyze_trajectories.py          # repair-direction numbers (Result 2), layer 29
python scripts/layer_sweep_multisplit.py runs/h2-trajectory-all/hidden_states   # per-layer landscape
python scripts/layer_nested_cv.py runs/h2-trajectory-all/hidden_states          # raw probe: nested CV layer selection (Result 1)
python scripts/layer_nested_cv.py runs/h2-trajectory-all/hidden_states --residualize  # residualized probe
python scripts/prompt_only_nested_cv.py runs/h2-trajectory-all/hidden_states     # prompt-length baseline
python scripts/prompt_only_nonlinear.py runs/h2-trajectory-all/hidden_states     # prompt-length baseline across model families
python scripts/verify_covariate_redundancy.py    # Result 2 Note numbers: Pearson r = 0.995 and the R² gain
# each writes a metrics.json under results/h2/ (and PNG figures where applicable)

# To re-run inference from scratch you additionally need:
#   pip install -e ".[model]"    # transformers, accelerate, bitsandbytes
#   Qwen/Qwen3-4B-Instruct-2507 weights (downloaded automatically via transformers)
#   an RTX-class GPU (experiments ran on RTX 4050 6 GB)
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

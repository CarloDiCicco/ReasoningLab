# Code Correctness Signals in LLM Hidden States: Pre-Generation Probing and Repair Geometry

**📄 Paper:** [arXiv:2606.14530](https://arxiv.org/abs/2606.14530)

This repository contains the code, data, and analysis scripts for the paper above.

---

## Summary

Large language models encode rich information in their hidden states. This work asks whether code correctness is legible in the hidden states of Qwen3-4B-Instruct-2507, before it generates and as it repairs a failed attempt, studied on 444 LiveCodeBench tasks. It reports two findings connected by a single confound-control tool: residualization. First, the correctness of the model's first-attempt code is linearly decodable from the prompt-final hidden state, with a leakage-free held-out AUC of 0.955 ± 0.006 across 50 outer splits. After the linear effect of prompt length is removed from each hidden state dimension, the probe still reaches 0.940 ± 0.009, well above a prompt-length baseline of 0.720 ± 0.015. Second, on 246 cleaned cases where the model attempts to repair a failed first attempt, the hidden state shift from the failing attempt to its repair carries a robust geometric direction of repair success, significant on both a magnitude and a split-half test against label-shuffled nulls. This direction remains significant on both tests after a conditional residualization against three repair-context covariates that differ between successful and failed repairs, indicating a genuine repair-success signal that the observable repair context does not account for. The contribution is as much methodological as empirical, a confound-control diagnostic that reports each signal only to the extent it survives the control.

---

## Repo structure

```
ReasoningLab/
├── README.md                          ← this file
├── LICENSE                            ← Apache 2.0
├── paper/                             ← LaTeX source of the paper
├── scripts/
│   ├── analyze_trajectories.py        ← Result 2 (direction, permutation, split-half) plus the §5 null-result analyses (convergence, distance); computes more than the paper reports and writes diagnostic figures that are not committed (layer 30)
│   ├── layer_nested_cv.py             ← nested CV layer+C selection for the probe (raw and --residualize)
│   ├── prompt_only_nested_cv.py       ← prompt-length baseline under the same nested CV procedure
│   ├── prompt_only_nonlinear.py       ← prompt-length baseline (logistic, random forest, gradient boosting, MLP), same nested CV procedure
│   ├── layer_sweep_multisplit.py      ← descriptive per-layer sweep (the layer landscape)
│   ├── train_probe.py                 ← standalone probe trainer on a single train/test split (--residualize); a dev utility, not used for the reported numbers
│   ├── verify_covariate_redundancy.py ← backs the Result 2 Note: Pearson r and the in-sample R² gain of the redundant covariate
│   └── prepare_livecodebench.py       ← LCB data prep + utilities
├── src/reasoninglab/
│   ├── probing/                       ← probe + hidden state utilities
│   ├── policies/                      ← sampling / repair policies
│   ├── eval/                          ← evaluation harness
│   ├── tasks/                         ← benchmark adapters (LCB, HumanEval)
│   └── models/                        ← model loading / hidden state hooks
├── configs/                           ← generation config for the paper run
│   └── trajectory-repair_b-CORRECTED.yaml
├── runs/
│   └── h2-trajectory/                 ← 444-task 5-attempt LCB trajectory data used in the paper (Results 1, 2); hidden_states/*.npz drive all analysis, no GPU needed to reproduce
└── results/h2/
    ├── probe_lcb444_multisplit/metrics.json          ← per-layer sweep (layer landscape)
    ├── probe_nested_cv_grid/metrics.json             ← nested CV raw-probe selection + outer-test AUC
    ├── probe_nested_cv_grid_resid/metrics.json       ← nested CV residualized-probe selection + outer-test AUC
    ├── prompt_only_nested_cv/metrics.json            ← prompt-length baseline (nested CV, outer-test AUC)
    ├── prompt_only_nonlinear/metrics.json            ← prompt-length baseline across model families (same nested CV, outer-test AUC)
    └── trajectory_analysis/metrics.json              ← repair-direction / permutation / split-half / convergence (layer 30)
```

---

## Reproduce

The committed run `runs/h2-trajectory/` holds the `hidden_states/*.npz` files
behind every number in the paper. There are two ways in, depending on whether
you want to regenerate the data or just analyze it.

```bash
# Clone and set up (Python 3.11+)
git clone <repo-url> && cd ReasoningLab
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[probe]"        # installs scikit-learn and matplotlib
```

**Path A: analyze the committed hidden states (no GPU, exact paper numbers).**
The repository ships the hidden states used in the paper, and every script
defaults to `runs/h2-trajectory/` at layer 30, so these commands reproduce the
reported numbers exactly without any inference or extra flags.

```bash
HS=runs/h2-trajectory/hidden_states

# Result 1: pre-generation probe
python scripts/layer_sweep_multisplit.py $HS               # per-layer landscape (Table 1)
python scripts/layer_nested_cv.py $HS                      # nested CV layer+C selection, raw probe
python scripts/layer_nested_cv.py $HS --residualize        # residualized probe
python scripts/prompt_only_nested_cv.py $HS                # prompt-length baseline
python scripts/prompt_only_nonlinear.py $HS                # prompt-length baseline across model families

# Result 2: repair-direction geometry
python scripts/analyze_trajectories.py                     # direction, permutation, split-half, convergence
python scripts/verify_covariate_redundancy.py              # Pearson r = 0.924 and the redundant-covariate R² gain
```

**Path B: regenerate the hidden states from scratch (needs a GPU).** This
re-runs the model and rebuilds the run directory that Path A analyzes. The raw
LiveCodeBench task file (`data/lcb_all_FIXED.jsonl`, about 1 GB) is too large to
commit and is rebuilt with `scripts/prepare_livecodebench.py`. Because
generation is sampled, a fresh run produces numbers that are aligned with the
paper but not identical; for the exact reported numbers use Path A.

```bash
pip install -e ".[model]"                          # transformers, accelerate, bitsandbytes
python scripts/prepare_livecodebench.py            # rebuild data/lcb_all_FIXED.jsonl (~1 GB, not committed)
# then run generation with configs/trajectory-repair_b-CORRECTED.yaml
# (Qwen/Qwen3-4B-Instruct-2507 weights download automatically; experiments ran on an NVIDIA DGX)
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

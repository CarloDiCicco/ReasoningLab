# Code Correctness Is Linearly Decodable from LLM Hidden States Before Generation

**📄 Paper:** [arXiv:2606.14530](https://arxiv.org/abs/2606.14530)

This repository contains the code, data, and analysis scripts for the paper above.

---

## Summary

Large language models encode rich information in their hidden states. This work asks whether the correctness of code that Qwen3-4B-Instruct-2507 has not yet generated is already legible in its hidden states, evaluated on a set of 444 tasks from LiveCodeBench. The correctness of the model's first-attempt code is linearly decodable from the hidden state at the final prompt token, captured before any output token is generated, with a leakage-free held-out AUC of 0.881 ± 0.008 across 50 outer splits. To assess whether this signal is explained by prompt length, each hidden state dimension is residualized with respect to the linear effect of prompt length. The probe still achieves an AUC of 0.842 ± 0.010, substantially above a prompt-length baseline of 0.657 ± 0.014, and none of the nonlinear prompt-length models tested improves upon this baseline. A companion question about the model's self-repair behavior could not be answered because successful repairs following a failed first attempt are too rare in this setting to sustain a reliable geometric analysis. The contribution is both empirical and methodological, providing evidence that pre-generation hidden states contain a robust signal of eventual code correctness, together with a confound-control diagnostic that quantifies how much of that signal survives adjustment for prompt length.

---

## Repo structure

```
ReasoningLab/
├── README.md                          ← this file
├── LICENSE                            ← Apache 2.0
├── paper/                             ← LaTeX source of the paper
├── scripts/
│   ├── layer_nested_cv.py             ← nested CV layer+C selection for the probe (raw and --residualize)
│   ├── prompt_only_nested_cv.py       ← prompt-length baseline under the same nested CV procedure
│   ├── prompt_only_nonlinear.py       ← the same baseline across model families (logistic, random forest, gradient boosting, MLP)
│   ├── layer_sweep_multisplit.py      ← descriptive per-layer sweep (the layer landscape)
│   ├── repair_summary.py              ← repair-trajectory summary; source of the recovery counts and per-attempt pass rates the paper reports as a null result
│   ├── train_probe.py                 ← standalone probe trainer on a single train/test split (--residualize); a dev utility, not used for the reported numbers
│   └── prepare_livecodebench.py       ← LCB data prep + utilities
├── src/reasoninglab/
│   ├── probing/                       ← probe + hidden state utilities
│   ├── policies/                      ← sampling / repair policies
│   ├── eval/                          ← evaluation harness
│   ├── tasks/                         ← task loading + schema validation
│   └── models/                        ← model loading / prompt hidden state extraction
├── configs/                           ← generation config for the paper run
│   └── trajectory-repair_b.yaml
├── runs/
│   └── h2-trajectory/                 ← 444-task 5-attempt LCB trajectory data behind the paper
└── results/h2/
    ├── probe_lcb444_multisplit/metrics.json          ← per-layer sweep (layer landscape)
    ├── probe_nested_cv_grid/metrics.json             ← nested CV raw-probe selection + outer-test AUC
    ├── probe_nested_cv_grid_resid/metrics.json       ← nested CV residualized-probe selection + outer-test AUC
    ├── prompt_only_nested_cv/metrics.json            ← prompt-length baseline (nested CV, outer-test AUC)
    ├── prompt_only_nonlinear/metrics.json            ← prompt-length baseline across model families (same nested CV, outer-test AUC)
    └── repair_summary/metrics.json                   ← recovery counts + per-attempt pass rates
```

---

## Reproduce

The committed run `runs/h2-trajectory/` holds the data behind every number in
the paper. There are two ways in, depending on whether you want to regenerate
the data or just analyze it.

```bash
# Clone and set up (Python 3.11+)
git clone <repo-url> && cd ReasoningLab
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[probe]"        # installs scikit-learn
```

**Path A: analyze the committed hidden states (no GPU, exact paper numbers).**
The repository ships the hidden states used in the paper, so these commands
reproduce the reported numbers exactly without any inference or extra flags.
Each writes its `metrics.json` to the matching `results/h2/` directory, rewriting
the committed copy in place, so an empty `git diff` after a run is the check that
the numbers reproduced.

```bash
HS=runs/h2-trajectory/hidden_states

python scripts/layer_sweep_multisplit.py $HS               # per-layer landscape
python scripts/layer_nested_cv.py $HS                      # nested CV layer+C selection, raw probe
python scripts/layer_nested_cv.py $HS --residualize        # residualized probe
python scripts/prompt_only_nested_cv.py $HS                # prompt-length baseline
python scripts/prompt_only_nonlinear.py $HS                # prompt-length baseline across model families
python scripts/repair_summary.py                           # recovery counts, per-attempt pass rates
```

**Path B: regenerate the hidden states from scratch (needs a GPU).** This
re-runs the model and rebuilds the run directory that Path A analyzes. The
LiveCodeBench task file the run reads (`data/lcb_all_FIXED.jsonl`, about 1 GB)
is too large to commit and must be rebuilt first with
`scripts/prepare_livecodebench.py`, which pulls the problems and their test
suites from the LiveCodeBench release. Because generation is sampled, a fresh
run produces numbers that are aligned with the paper but not identical; for the
exact reported numbers use Path A.

```bash
pip install -e ".[model,data]"                     # transformers, accelerate, bitsandbytes, datasets

# rebuild the task file (~1 GB, not committed); covers the easy, medium and hard
# tiers the paper uses, so pass --difficulty explicitly and name the output
python scripts/prepare_livecodebench.py \
    --difficulty easy,medium,hard \
    --output data/lcb_all_FIXED.jsonl

# run generation; rebuilds runs/h2-trajectory/
# (Qwen/Qwen3-4B-Instruct-2507 weights download automatically; experiments ran on an NVIDIA DGX)
python -m reasoninglab.cli --config configs/trajectory-repair_b.yaml
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

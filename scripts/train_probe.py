#!/usr/bin/env python3
"""Train an H2 linear probe to predict pass/fail from model hidden states.

Loads .npz files saved by the experiment runner, applies PCA for
dimensionality reduction, and trains a logistic regression classifier.

Runs two analyses:
  1. Per-layer: separate probe for each transformer layer (which depth
     carries the most pass/fail signal?)
  2. Concatenated: all layers concatenated into one feature vector
     (best-effort classifier).

Usage:
    python scripts/train_probe.py runs/h2-baseline-probe-data_<ts>/hidden_states
    python scripts/train_probe.py runs/h2-baseline-probe-data_<ts>/hidden_states --pca-variance 0.99
    python scripts/train_probe.py runs/h2-baseline-probe-data_<ts>/hidden_states --cross-domain

Requires: pip install -e ".[probe]"   (installs scikit-learn)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from reasoninglab.probing.data import ProbeSample, build_feature_matrix, load_samples
from reasoninglab.probing.probe import ProbeResult, train_probe


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train H2 linear probe from hidden states.",
    )
    parser.add_argument(
        "hidden_states_dir",
        help="Directory containing .npz files from a run with return_hidden_states=true.",
    )
    parser.add_argument(
        "--output-dir", default="results/h2/probe",
        help="Directory for metrics.json and probe_concat.joblib (default: results/h2/probe).",
    )
    parser.add_argument(
        "--pca-variance", type=float, default=0.95,
        help="Cumulative variance threshold for PCA component selection (default: 0.95).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data reserved for test set (default: 0.2).",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of stratified CV folds for C tuning (default: 5).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for splits and CV (default: 0).",
    )
    parser.add_argument(
        "--cross-domain", action="store_true", # default is False, but --cross-domain in the call by terminal is gonna be like is described in the help
        help="Train on HumanEval, test on LCB (instead of random stratified split).",
    )
    return parser.parse_args()


# ── Printing ─────────────────────────────────────────────────────────────────

def _print_data_summary(samples: list[ProbeSample], layers: list[int]) -> None:
    n = len(samples)
    n_pass = sum(1 for s in samples if s.passed)
    n_fail = n - n_pass

    # counting how many samples you have from each different dataset (like "HumanEval" or "LCB")
    sources: dict[str, int] = {}
    for s in samples:
        sources[s.source] = sources.get(s.source, 0) + 1

    hidden_dim = samples[0].layer_vectors[layers[0]].shape[0] if layers else 0

    print()
    print("── Data summary ───────────────────────────────────────────────")
    print(f"  samples    : {n}")
    print(f"  passed     : {n_pass} ({n_pass / n:.1%})")
    print(f"  failed     : {n_fail} ({n_fail / n:.1%})")
    print(f"  sources    : {sources}")
    print(f"  layers     : {layers}")
    print(f"  hidden_dim : {hidden_dim}")
    print("───────────────────────────────────────────────────────────────")
    print()


def _print_split_info(
    label: str,
    y: np.ndarray,
) -> None:
    n = len(y)
    n_pass = int(y.sum())
    print(f"  {label:6s}: {n} samples  (pass={n_pass}, fail={n - n_pass}, rate={n_pass / n:.1%})")


def _print_results_table(results: list[ProbeResult]) -> None:
    print()
    print("── Probe results ──────────────────────────────────────────────")
    header = f"  {'Layer':<8} {'PCA_k':>5} {'Var%':>6} {'CV AUC':>8} {'Test AUC':>9} {'Acc':>6} {'F1':>6} {'C':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for r in results:
        layer_str = str(r.layer)
        print(
            f"  {layer_str:<8} {r.n_features_pca:>5} {r.pca_variance_explained:>5.1%}"
            f" {r.cv_auc_mean:>7.3f}±{r.cv_auc_std:.3f}"
            f" {r.test_auc:>8.3f} {r.test_accuracy:>6.3f} {r.test_f1:>6.3f} {r.best_C:>8.3f}"
        )
    print("───────────────────────────────────────────────────────────────")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    hs_dir = Path(args.hidden_states_dir)
    out_dir = Path(args.output_dir)

    # 1. Load samples.
    print(f"[probe] Loading samples from {hs_dir} ...", flush=True)
    samples = load_samples(hs_dir)
    if not samples:
        print("[probe] ERROR: no .npz files found.")
        return

    # Layer 0 is the raw token embedding lookup. We verified empirically that
    # every prompt ends with the same token: Qwen3's chat template always appends
    # "<|im_start|>assistant\n" as the generation prompt suffix, so the last token
    # is identical across all 388 samples. Same token ID → same embedding vector
    # → zero variance across samples → zero information for pass/fail prediction.
    # We exclude it here before any analysis so that all remaining layer vectors
    # stay shape-compatible (all 2560-dim) for future averaging or concat strategies.
    layers = [l for l in sorted(samples[0].layer_vectors.keys()) if l != 0]
    print(
        "[probe] Layer 0 excluded: verified that every prompt ends with the same token "
        "('<|im_start|>assistant\\n' suffix from Qwen3 chat template) → "
        "embedding vector is identical across all samples → zero variance, zero signal."
    )

    _print_data_summary(samples, layers)

    # 2. Train/test split.
    if args.cross_domain:
        # Train on HumanEval, test on LCB.
        train_samples = [s for s in samples if s.source == "HumanEval"]
        test_samples = [s for s in samples if s.source == "LCB"]
        if not train_samples or not test_samples:
            print("[probe] ERROR: cross-domain split has empty train or test set.")
            return
        print("[probe] Cross-domain split: train=HumanEval, test=LCB")
    else:
        # Stratified random split.
        y_all = np.array([int(s.passed) for s in samples])
        train_idx, test_idx = train_test_split(
            np.arange(len(samples)),
            test_size=args.test_size,
            stratify=y_all,
            random_state=args.seed,
        )
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]
        print(f"[probe] Stratified random split (test_size={args.test_size}):")

    y_train_all = np.array([int(s.passed) for s in train_samples])
    y_test_all = np.array([int(s.passed) for s in test_samples])
    _print_split_info("train", y_train_all)
    _print_split_info("test", y_test_all)
    print()

    all_results: list[ProbeResult] = []

    # 3. Per-layer analysis.
    # For each transformer layer separately: how predictive is each layer alone?
    # This tells us WHERE in the network pass/fail information is encoded.
    # We collect all pipelines so we can save them after the results table.
    layer_pipelines: dict[int, object] = {}  # layer_idx → fitted Pipeline
    print("[probe] Per-layer analysis:", flush=True)
    for layer_idx in layers:
        # Build a feature matrix using only this single layer (shape: n × 2560).
        X_train, y_train = build_feature_matrix(train_samples, layers=[layer_idx])
        X_test, y_test = build_feature_matrix(test_samples, layers=[layer_idx])
        result, layer_pipeline = train_probe(
            X_train, y_train, X_test, y_test,
            pca_variance=args.pca_variance,
            cv_folds=args.cv_folds,
            seed=args.seed,
            layer_label=layer_idx,
        )
        all_results.append(result)
        layer_pipelines[layer_idx] = layer_pipeline
        print(f"  layer {layer_idx}: AUC={result.test_auc:.3f} (PCA k={result.n_features_pca})", flush=True)

    # 4. Concatenated analysis.
    # All layers stacked into one vector (shape: n × 9*2560 = n × 23040).
    # This is the "best effort" probe — uses maximum information.
    print("[probe] Concatenated analysis:", flush=True)
    X_train, y_train = build_feature_matrix(train_samples)  # layers=None → all layers
    X_test, y_test = build_feature_matrix(test_samples)
    concat_result, concat_pipeline = train_probe(  # concat_pipeline saved below
        X_train, y_train, X_test, y_test,
        pca_variance=args.pca_variance,
        cv_folds=args.cv_folds,
        seed=args.seed,
        layer_label="concat",
    )
    all_results.append(concat_result)
    print(f"  concat: AUC={concat_result.test_auc:.3f} (PCA k={concat_result.n_features_pca})", flush=True)

    # 5. Print results table.
    _print_results_table(all_results)

    # 6. Save outputs.
    out_dir.mkdir(parents=True, exist_ok=True)

    # All per-layer + concat metrics in one JSON (easy to read / plot later).
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        # ProbeResult fields like n_train/n_test may be numpy int64 — convert to native.
        records = [{k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
                    for k, v in asdict(r).items()} for r in all_results]
        json.dump(records, f, indent=2)
    print(f"[probe] Metrics saved to {metrics_path}")

    # Save all pipelines (scaler + PCA + logreg) as serialized objects.
    # To use one later on new hidden states:
    #   probe = joblib.load("results/h2/probe/probe_layer_35.joblib")
    #   proba = probe.predict_proba(X_new)[:, 1]  # pass probability for each sample
    for layer_idx, pipeline in layer_pipelines.items():
        pipeline_path = out_dir / f"probe_layer_{layer_idx}.joblib"
        joblib.dump(pipeline, pipeline_path)
        print(f"[probe] Pipeline saved to {pipeline_path}")

    pipeline_path = out_dir / "probe_concat.joblib"
    joblib.dump(concat_pipeline, pipeline_path)
    print(f"[probe] Pipeline saved to {pipeline_path}")


if __name__ == "__main__":
    main()
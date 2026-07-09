#!/usr/bin/env python3
"""Multi-split per-layer probe sweep.

Reuses the same pipeline as train_probe.py / analyze_trajectories.py:
StandardScaler -> PCA(0.95 variance) -> LogisticRegression(L2), with 5-fold
inner CV on C in {1e-3, 1e-2, 1e-1, 1, 10}. Runs N stratified 80/20 splits
per layer and aggregates mean +/- 95% CI of test AUC and CV AUC.

Output:
  - console table
  - <output_dir>/metrics.json with per-layer summaries and per-split lists

Usage:
  python scripts/layer_sweep_multisplit.py runs/h2-trajectory/hidden_states \
      --output-dir results/h2/probe_lcb444_multisplit --n-splits 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reasoninglab.probing.data import build_feature_matrix, load_samples
from reasoninglab.probing.probe import train_probe


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-split per-layer probe sweep.")
    p.add_argument("hidden_states_dir",
                   help="Directory of .npz files (one per task).")
    p.add_argument("--output-dir", default="results/h2/probe_lcb444_multisplit",
                   help="Where to write metrics.json.")
    p.add_argument("--n-splits", type=int, default=50)
    p.add_argument("--pca-variance", type=float, default=0.95)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Layers to sweep. Default: all available except 0.")
    return p.parse_args()


def _summary(arr: list[float]) -> dict:
    a = np.array(arr)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)),
        "ci95_half": float(1.96 * a.std(ddof=1) / np.sqrt(len(a))),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep] Loading samples from {args.hidden_states_dir} ...")
    samples = load_samples(Path(args.hidden_states_dir))
    available_layers = sorted(samples[0].layer_vectors.keys())
    layers = args.layers if args.layers else [l for l in available_layers if l != 0]
    print(f"[sweep] {len(samples)} samples, layers to sweep: {layers}, "
          f"n_splits={args.n_splits}, PCA={args.pca_variance}, CV={args.cv_folds}")

    results: dict[int, dict] = {}

    for layer in layers:
        print(f"\n[sweep] Layer {layer} ...", flush=True)
        X, y = build_feature_matrix(samples, [layer])
        test_aucs: list[float] = []
        cv_aucs: list[float] = []

        for seed in range(args.n_splits):
            from sklearn.model_selection import train_test_split
            idx = np.arange(len(y))
            train_idx, test_idx = train_test_split(
                idx, test_size=0.2, stratify=y, random_state=seed)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            result, _ = train_probe(
                X_train, y_train, X_test, y_test,
                pca_variance=args.pca_variance,
                cv_folds=args.cv_folds,
                seed=seed,
                layer_label=f"L{layer}_s{seed}",
            )
            test_aucs.append(float(result.test_auc))
            cv_aucs.append(float(result.cv_auc_mean))
            if (seed + 1) % 10 == 0:
                print(f"  layer {layer}: {seed+1}/{args.n_splits} done", flush=True)

        results[layer] = {
            "test_auc": test_aucs,
            "cv_auc": cv_aucs,
            "test_auc_summary": _summary(test_aucs),
            "cv_auc_summary": _summary(cv_aucs),
        }

    # ---- Console table ----
    print("\n")
    print("=" * 76)
    print(f"{'Layer':<6} {'Test AUC (mean ± 95% CI)':<28} {'CV AUC (mean ± 95% CI)':<28}")
    print("-" * 76)
    for layer in layers:
        ts = results[layer]["test_auc_summary"]
        cs = results[layer]["cv_auc_summary"]
        print(f"{layer:<6} "
              f"{ts['mean']:.3f} ± {ts['ci95_half']:.3f}  "
              f"[{ts['min']:.3f}, {ts['max']:.3f}]   "
              f"{cs['mean']:.3f} ± {cs['ci95_half']:.3f}")
    print("=" * 76)

    # ---- Save JSON ----
    payload = {
        "n_splits": args.n_splits,
        "pca_variance": args.pca_variance,
        "cv_folds": args.cv_folds,
        "layers": layers,
        "n_samples": len(samples),
        "per_layer": results,
    }
    out_path = out_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[sweep] Saved {out_path}")


if __name__ == "__main__":
    main()

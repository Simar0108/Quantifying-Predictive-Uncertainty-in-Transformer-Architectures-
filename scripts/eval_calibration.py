#!/usr/bin/env python3
"""
Compute reliability diagram data and ECE from MC Dropout and optionally baseline predictions.
Expects results from eval_mc_dropout.py and optionally eval_baseline.py.
Usage:
  python scripts/eval_calibration.py --predictions results/mc_predictions.npz [--bins 10] [--out results/calibration.npz]
  python scripts/eval_calibration.py --predictions results/mc_predictions.npz --baseline results/baseline_predictions.npz --out results/calibration.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _confidences_and_correct(mean_pos: np.ndarray, pred: np.ndarray, labels: np.ndarray):
    """Binary: confidence = P(predicted class). correct = (pred == labels)."""
    confidences = np.where(pred == 1, mean_pos, 1.0 - mean_pos)
    correct = (pred == labels).astype(np.float64)
    return confidences, correct


def reliability_diagram(confidences: np.ndarray, correct: np.ndarray, num_bins: int = 10):
    """
    Bin samples by confidence and return bin accuracies and bin confidences.
    confidences: predicted P(correct class) or P(positive) for binary.
    correct: 0/1 whether prediction was correct.
    """
    bins = np.linspace(0, 1, num_bins + 1)
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi) if i < num_bins - 1 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_accs[i] = correct[mask].mean()
            bin_confs[i] = confidences[mask].mean()
            bin_counts[i] = mask.sum()
    return bin_accs, bin_confs, bin_counts, bins


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, num_bins: int = 10) -> float:
    bin_accs, bin_confs, bin_counts, _ = reliability_diagram(confidences, correct, num_bins)
    total = bin_counts.sum()
    if total == 0:
        return 0.0
    ece = np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / total
    return float(ece)


def main():
    parser = argparse.ArgumentParser(description="Calibration from MC and optionally baseline predictions")
    parser.add_argument("--predictions", type=str, default="results/mc_predictions.npz", help="MC Dropout npz from eval_mc_dropout.py")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline npz from eval_baseline.py (optional)")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--out", type=str, default="results/calibration.npz")
    args = parser.parse_args()

    data = np.load(args.predictions, allow_pickle=True)
    mean_pos = data["id_mean_positive"]
    pred = data["id_pred"]
    labels = data["id_labels"]
    confidences, correct = _confidences_and_correct(mean_pos, pred, labels)

    ece = expected_calibration_error(confidences, correct, num_bins=args.bins)
    bin_accs, bin_confs, bin_counts, bins = reliability_diagram(confidences, correct, num_bins=args.bins)

    save_dict = {
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bins": bins,
        "ece": ece,
    }

    if args.baseline:
        bl_data = np.load(args.baseline, allow_pickle=True)
        bl_mean = bl_data["id_mean_positive"]
        bl_pred = bl_data["id_pred"]
        bl_labels = bl_data["id_labels"]
        bl_conf, bl_correct = _confidences_and_correct(bl_mean, bl_pred, bl_labels)
        baseline_ece = expected_calibration_error(bl_conf, bl_correct, num_bins=args.bins)
        bl_bin_accs, bl_bin_confs, bl_bin_counts, _ = reliability_diagram(bl_conf, bl_correct, num_bins=args.bins)
        save_dict["baseline_bin_accs"] = bl_bin_accs
        save_dict["baseline_bin_confs"] = bl_bin_confs
        save_dict["baseline_bin_counts"] = bl_bin_counts
        save_dict["baseline_ece"] = baseline_ece
        print(f"Baseline ECE = {baseline_ece:.4f}")

    print(f"MC Dropout ECE = {ece:.4f}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **save_dict)
    print(f"Saved calibration data to {out_path}")


if __name__ == "__main__":
    main()

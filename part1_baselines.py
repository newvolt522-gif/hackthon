"""
Part 1: Baselines for shortage and full.
- Persistence: next-30m state == current state.
- Rule-based: threshold on current ratio; continuous score = 1 - ratio.
Metrics on Validation and Test: Precision/Recall/F1/ROC-AUC/PR-AUC/confusion matrix.
Writes reports/baseline_results.csv.
"""
import json
import os
import csv

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)

BASE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816"
REPORTS = os.path.join(BASE, "reports")
os.makedirs(REPORTS, exist_ok=True)

with open(os.path.join(BASE, "feature_columns.json"), encoding="utf-8") as f:
    FEATS = json.load(f)["feature_columns"]

# CSV: col0=label, then features in FEATS order.
IDX_AVAIL_RATIO = 1 + FEATS.index("available_ratio")      # current available_ratio
IDX_EMPTY_RATIO = 1 + FEATS.index("empty_dock_ratio")     # current empty_dock_ratio

SHORTAGE_THR = 0.15
FULL_THR = 0.10


def load(path):
    arr = np.loadtxt(path, delimiter=",")
    y = arr[:, 0].astype(int)
    return arr, y


def metrics(y_true, y_pred, y_score):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def run(task, ratio_idx, ratio_thr):
    rows = []
    for split in ["validation", "test"]:
        path = os.path.join(BASE, "model_data", task, f"{split}.csv")
        arr, y = load(path)
        cur_ratio = arr[:, ratio_idx]
        # current risk state (persistence) = current ratio <= thr
        cur_state = (cur_ratio <= ratio_thr).astype(int)

        # A. Persistence: pred = current state; score = 1 - ratio (continuous)
        score = 1.0 - cur_ratio
        m = metrics(y, cur_state, score)
        m.update({"task": task, "baseline": "persistence", "split": split})
        rows.append(m)

        # B. Rule-based: same threshold rule but reported as a rule baseline.
        # (Distinct interpretation: fire when current ratio below a rule cutoff.)
        rule_pred = (cur_ratio <= ratio_thr).astype(int)
        m2 = metrics(y, rule_pred, score)
        m2.update({"task": task, "baseline": "rule_based", "split": split})
        rows.append(m2)
    return rows


def main():
    all_rows = []
    all_rows += run("shortage", IDX_AVAIL_RATIO, SHORTAGE_THR)
    all_rows += run("full", IDX_EMPTY_RATIO, FULL_THR)

    cols = ["task", "baseline", "split", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "tp", "fp", "tn", "fn"]
    out = os.path.join(REPORTS, "baseline_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in cols})

    # Print summary.
    print(f"{'task':9} {'baseline':12} {'split':11} "
          f"{'P':>6} {'R':>6} {'F1':>6} {'ROC':>6} {'PR':>6}  TP/FP/TN/FN")
    for r in all_rows:
        print(f"{r['task']:9} {r['baseline']:12} {r['split']:11} "
              f"{r['precision']:.3f} {r['recall']:.3f} {r['f1']:.3f} "
              f"{r['roc_auc']:.3f} {r['pr_auc']:.3f}  "
              f"{r['tp']}/{r['fp']}/{r['tn']}/{r['fn']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

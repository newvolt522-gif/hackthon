"""
Part 4: Threshold analysis on VALIDATION (0.10..0.70).
Part 5: Final TEST evaluation at the fixed, validation-chosen threshold.

Rules:
- Threshold chosen ONLY from validation.
- Two thresholds reported per task: math-best F1, and an operationally sensible one.
- Test is scored once at the chosen operational threshold (no test tuning).
Writes reports/threshold_analysis.csv, reports/shortage_results.csv, reports/full_results.csv.
"""
import csv
import os

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

BASE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816"
MODELS = os.path.join(BASE, "models")
REPORTS = os.path.join(BASE, "reports")
os.makedirs(REPORTS, exist_ok=True)

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]


def load(task, split):
    proba = np.load(os.path.join(MODELS, f"{task}_{split}_proba.npy"))
    y = np.load(os.path.join(MODELS, f"{task}_{split}_y.npy"))
    return proba, y


def eval_at(y, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "threshold": thr,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "tp": int(((y == 1) & (pred == 1)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
    }


def main():
    thr_rows = []
    chosen = {}  # task -> (math_best_thr, operational_thr)

    for task in ["shortage", "full"]:
        proba, y = load(task, "validation")
        rows = [eval_at(y, proba, t) for t in THRESHOLDS]
        for r in rows:
            r2 = dict(r); r2["task"] = task; r2["split"] = "validation"
            thr_rows.append(r2)

        # math-best F1
        best = max(rows, key=lambda r: r["f1"])
        math_best = best["threshold"]

        # operational threshold:
        if task == "shortage":
            # prioritize recall & F1 for early warning: pick lowest threshold whose
            # precision still >= 0.70 (avoid alert flooding) while maximizing recall.
            candidates = [r for r in rows if r["precision"] >= 0.70]
            oper = min(candidates, key=lambda r: r["threshold"])["threshold"] if candidates else math_best
        else:
            # full is rare: balance precision & false alarms. Require precision >= 0.50
            # (<=1 false alarm per true alarm) then take highest recall among those.
            candidates = [r for r in rows if r["precision"] >= 0.50]
            oper = max(candidates, key=lambda r: r["recall"])["threshold"] if candidates else math_best

        chosen[task] = (math_best, oper)
        print(f"\n[{task}] validation threshold sweep:")
        print(f"{'thr':>5} {'P':>7} {'R':>7} {'F1':>7}   TP/FP/TN/FN")
        for r in rows:
            mark = ""
            if r["threshold"] == math_best: mark += " <=F1best"
            if r["threshold"] == oper: mark += " <=OPER"
            print(f"{r['threshold']:.2f} {r['precision']:.4f} {r['recall']:.4f} "
                  f"{r['f1']:.4f}   {r['tp']}/{r['fp']}/{r['tn']}/{r['fn']}{mark}")
        print(f"  -> math-best F1 threshold = {math_best}")
        print(f"  -> operational threshold  = {oper}")

    # write threshold analysis
    cols = ["task", "split", "threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
    with open(os.path.join(REPORTS, "threshold_analysis.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in thr_rows:
            w.writerow({k: r[k] for k in cols})

    # ---- Part 5: TEST at fixed operational threshold ----
    for task in ["shortage", "full"]:
        math_best, oper = chosen[task]
        out_rows = []
        for split in ["validation", "test"]:
            proba, y = load(task, split)
            m = eval_at(y, proba, oper)
            m["roc_auc"] = roc_auc_score(y, proba)
            m["pr_auc"] = average_precision_score(y, proba)
            m["task"] = task; m["split"] = split
            m["chosen_threshold"] = oper; m["math_best_threshold"] = math_best
            out_rows.append(m)
            print(f"\n[{task}] {split} @thr={oper}: P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"F1={m['f1']:.4f} ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f} "
                  f"TP/FP/TN/FN={m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")

        cols2 = ["task", "split", "chosen_threshold", "math_best_threshold",
                 "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "tn", "fn"]
        with open(os.path.join(REPORTS, f"{task}_results.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols2); w.writeheader()
            for r in out_rows:
                w.writerow({k: r[k] for k in cols2})

    # persist chosen thresholds
    with open(os.path.join(BASE, "_chosen_thresholds.txt"), "w") as f:
        for task, (mb, op) in chosen.items():
            f.write(f"{task},math_best={mb},operational={op}\n")
    print("\nWrote threshold_analysis.csv, shortage_results.csv, full_results.csv")


if __name__ == "__main__":
    main()

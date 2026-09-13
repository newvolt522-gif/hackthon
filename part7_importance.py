"""
Part 7: Feature importance (gain + weight) for both models.
The SageMaker XGBoost model stores features as f0..f14; map back to names via
feature_columns.json order. Writes reports/feature_importance_{task}.csv.
"""
import csv
import json
import os

import xgboost as xgb

BASE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816"
MODELS = os.path.join(BASE, "models")

with open(os.path.join(BASE, "feature_columns.json"), encoding="utf-8") as f:
    FEATS = json.load(f)["feature_columns"]
NAME = {f"f{i}": FEATS[i] for i in range(len(FEATS))}


def importance(task):
    b = xgb.Booster()
    b.load_model(os.path.join(MODELS, f"{task}.model"))
    gain = b.get_score(importance_type="gain")
    weight = b.get_score(importance_type="weight")
    rows = []
    for fid in [f"f{i}" for i in range(len(FEATS))]:
        rows.append({
            "feature": NAME[fid],
            "gain": round(gain.get(fid, 0.0), 4),
            "weight": int(weight.get(fid, 0)),
        })
    rows.sort(key=lambda r: r["gain"], reverse=True)
    return rows


def main():
    for task in ["shortage", "full"]:
        rows = importance(task)
        out = os.path.join(BASE, "reports", f"feature_importance_{task}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["feature", "gain", "weight"]); w.writeheader()
            w.writerows(rows)
        print(f"\n=== {task}: top features by gain ===")
        print(f"{'feature':34} {'gain':>10} {'weight':>8}")
        for r in rows:
            print(f"{r['feature']:34} {r['gain']:>10.3f} {r['weight']:>8}")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

"""
Part 6: Early-warning KPIs on the TEST period, per station in time order.

Definitions (event = the target is 1 at a given 30-min slot t):
- We evaluate the model's alert at slot t-1 (issued 30 min before slot t), because
  the model at slot s predicts the state at s+1. So an alert "for slot t" is the
  prediction made at slot t-1.
- An "event onset" is the first slot of a contiguous run of target==1 (per station).
- Lead time = how far before the onset the model's first alert (within a lookback
  window ending at the onset) fired. Each alert predicts one slot ahead, so if the
  model raises the alert-for-slot at k consecutive slots ending at the onset, the
  earliest lead time = k*30 min. We cap the lookback so we only credit a warning
  that is part of the run leading into the event.
- Coverage = fraction of event onsets that had >=1 alert in the slot immediately
  before onset (i.e. predicted onset 30 min ahead).
- False alarm rate = fraction of alerts (predicted-positive slots) that were NOT
  followed by an actual event in the targeted slot.

We compare XGBoost (operational threshold) vs the persistence baseline.
Writes nothing new; prints KPI table (consumed by the report step).
"""
import os
from datetime import timedelta

import numpy as np
import polars as pl

BASE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816"
MODELS = os.path.join(BASE, "models")
PROC = os.path.join(BASE, "data", "processed", "youbike_features.parquet")

STEP_MIN = 30

CHOSEN = {}
with open(os.path.join(BASE, "_chosen_thresholds.txt")) as f:
    for line in f:
        parts = line.strip().split(",")
        task = parts[0]
        op = float(parts[2].split("=")[1])
        CHOSEN[task] = op


def test_frame():
    """Rebuild the TEST split rows in the SAME order used to write test.csv."""
    df = pl.read_parquet(PROC)
    d0 = df["ts30"].min().date()
    test_start = d0 + timedelta(days=11)
    test = df.filter(pl.col("ts30").dt.date() >= test_start)
    # The CSVs were written from this same frame WITHOUT extra sorting, so the
    # row order equals the parquet order. Keep it identical.
    return test


def kpi_for(task, target_col, proba, thr, test):
    y = test[target_col].to_numpy().astype(int)
    assert len(y) == len(proba), (len(y), len(proba))
    alert = (proba >= thr).astype(int)  # alert issued at slot s, predicts s+1

    # Build per-station ordered arrays.
    stations = test["station"].to_numpy()
    ts = test["ts30"].to_numpy()

    # Group indices by station preserving order.
    order = np.arange(len(y))
    df = pl.DataFrame({
        "idx": order, "station": stations, "ts": ts,
        "y": y, "alert": alert, "proba": proba,
    }).sort(["station", "ts"])

    lead_times = []          # minutes of lead for covered onsets
    onsets = 0
    covered = 0

    # iterate per station
    for (st,), sub in df.group_by(["station"], maintain_order=True):
        yy = sub["y"].to_numpy()
        aa = sub["alert"].to_numpy()
        n = len(yy)
        # onset = y[i]==1 and (i==0 or y[i-1]==0)
        for i in range(n):
            if yy[i] == 1 and (i == 0 or yy[i - 1] == 0):
                onsets += 1
                # alert predicting slot i is issued at slot i-1 (aa[i-1]).
                # lead time = count consecutive alerts ending at i-1 going backward,
                # but only while those slots are BEFORE the event onset.
                if i - 1 >= 0 and aa[i - 1] == 1:
                    covered += 1
                    k = 0
                    j = i - 1
                    while j >= 0 and aa[j] == 1 and yy[j] == 0:
                        k += 1
                        j -= 1
                    lead_times.append(k * STEP_MIN)

    # false alarm rate: alerts at slot s predict s+1; false if y[s+1]!=1.
    # Align alert[s] with y[s+1] within station.
    fa_total = 0
    fa_false = 0
    for (st,), sub in df.group_by(["station"], maintain_order=True):
        yy = sub["y"].to_numpy()
        aa = sub["alert"].to_numpy()
        n = len(yy)
        for s in range(n - 1):
            if aa[s] == 1:
                fa_total += 1
                if yy[s + 1] != 1:
                    fa_false += 1
    far = fa_false / fa_total if fa_total else float("nan")

    lt = np.array(lead_times) if lead_times else np.array([0])
    return {
        "task": task,
        "onsets": onsets,
        "covered": covered,
        "coverage": covered / onsets if onsets else float("nan"),
        "lead_mean": float(lt.mean()),
        "lead_median": float(np.median(lt)),
        "lead_p90": float(np.percentile(lt, 90)),
        "false_alarm_rate": far,
        "alerts_total": fa_total,
    }


def main():
    test = test_frame()
    print(f"Test frame rows: {test.height}")

    results = []
    for task, target_col in [("shortage", "future_shortage_30m"),
                             ("full", "future_full_30m")]:
        proba = np.load(os.path.join(MODELS, f"{task}_test_proba.npy"))
        thr = CHOSEN[task]

        # XGBoost KPIs
        xgb_kpi = kpi_for(task, target_col, proba, thr, test)
        xgb_kpi["model"] = "xgboost"

        # Persistence baseline: alert = current-state risk.
        # current shortage state = available_ratio<=0.15 ; full = empty_dock_ratio<=0.10
        cur_col = "available_ratio" if task == "shortage" else "empty_dock_ratio"
        cur_thr = 0.15 if task == "shortage" else 0.10
        cur_state = (test[cur_col].to_numpy() <= cur_thr).astype(float)
        base_kpi = kpi_for(task, target_col, cur_state, 0.5, test)
        base_kpi["model"] = "persistence"

        results += [xgb_kpi, base_kpi]

    # print
    hdr = ["task", "model", "onsets", "coverage", "lead_mean", "lead_median",
           "lead_p90", "false_alarm_rate", "alerts_total"]
    print("\n" + " ".join(f"{h:>16}" for h in hdr))
    for r in results:
        print(" ".join(f"{str(round(r[h],4) if isinstance(r[h],float) else r[h]):>16}" for h in hdr))

    # save
    import csv
    with open(os.path.join(BASE, "reports", "kpi_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr + ["covered"]); w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in hdr + ["covered"]})
    print("\nWrote reports/kpi_results.csv")


if __name__ == "__main__":
    main()

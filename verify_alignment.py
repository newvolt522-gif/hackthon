import os
from datetime import timedelta
import numpy as np
import polars as pl

BASE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816"
PROC = os.path.join(BASE, "data", "processed", "youbike_features.parquet")

df = pl.read_parquet(PROC)
d0 = df["ts30"].min().date()
test_start = d0 + timedelta(days=11)
test = df.filter(pl.col("ts30").dt.date() >= test_start)

for task, tgt in [("shortage", "future_shortage_30m"), ("full", "future_full_30m")]:
    csv_labels = np.loadtxt(os.path.join(BASE, "model_data", task, "test.csv"), delimiter=",")[:, 0].astype(int)
    parquet_labels = test[tgt].to_numpy().astype(int)
    same_len = len(csv_labels) == len(parquet_labels)
    identical = same_len and bool((csv_labels == parquet_labels).all())
    print(f"[{task}] csv_n={len(csv_labels)} parquet_n={len(parquet_labels)} "
          f"same_len={same_len} labels_identical={identical}")

    # also check a couple feature columns align (available_ratio idx1)
    csv_arr = np.loadtxt(os.path.join(BASE, "model_data", task, "test.csv"), delimiter=",", max_rows=1000)
    par_ar = test["available_ratio"].to_numpy()[:1000]
    feat_ok = np.allclose(csv_arr[:, 1], par_ar, atol=1e-9)
    print(f"       available_ratio matches (first 1000): {feat_ok}")

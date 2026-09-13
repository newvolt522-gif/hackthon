"""
Tasks 5-8: Feature engineering (no leakage), targets, time-based split, outputs.

Leakage rules:
- Features use ONLY current or PAST data:
    lag_k  = shift(+k)  over station (past)
    delta  = current - past
    station_hist_avg = EXPANDING mean of PAST available_ratio (shift then cummean)
- Targets use the NEXT 30-min slot: shift(-1) over station (future).
- Rows where target is null (each station's last slot) are dropped.
- No future value ever feeds a feature.
"""
import json
import os

import polars as pl

CACHE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816\data\raw_cache.parquet"
OUT_DIR = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816"
PROC_PARQUET = os.path.join(OUT_DIR, "data", "processed", "youbike_features.parquet")

# Thresholds chosen from task 4 (ratio-based).
SHORTAGE_RATIO_THR = 0.15   # available_ratio <= 0.15 -> shortage risk
FULL_RATIO_THR = 0.10       # empty_dock_ratio <= 0.10 -> full risk

# Feature order (also written to feature_columns.json).
FEATURE_COLUMNS = [
    "available_ratio",
    "empty_dock_ratio",
    "available_bikes",
    "empty_docks",
    "capacity",
    "lag_1_available",
    "lag_1_empty_docks",
    "delta_30m_available",
    "delta_30m_empty_docks",
    "delta_60m_available",
    "delta_60m_empty_docks",
    "hour",
    "day_of_week",
    "is_weekend",
    "station_hist_avg_available_ratio",
]


def build_grid(df):
    df = df.with_columns(pl.col("ts").dt.truncate("30m").alias("ts30"))
    return (
        df.sort("ts")
        .group_by(["station", "ts30"])
        .agg(
            pl.col("capacity").last(),
            pl.col("available_bikes").last(),
            pl.col("empty_docks").last(),
            pl.col("district").last(),
        )
        .sort(["station", "ts30"])
    )


def main():
    df = pl.read_parquet(CACHE)
    grid = build_grid(df)

    # Base ratios (current-time; safe). Guard divide-by-zero (capacity always >0 here).
    grid = grid.with_columns(
        (pl.col("available_bikes") / pl.col("capacity")).alias("available_ratio"),
        (pl.col("empty_docks") / pl.col("capacity")).alias("empty_dock_ratio"),
    )

    # ---- Features (PAST/CURRENT only), computed per station in time order ----
    grid = grid.with_columns(
        # lag 1 step (30 min ago)
        pl.col("available_bikes").shift(1).over("station").alias("lag_1_available"),
        pl.col("empty_docks").shift(1).over("station").alias("lag_1_empty_docks"),
        # lag 2 steps (60 min ago) - used for 60m delta
        pl.col("available_bikes").shift(2).over("station").alias("_lag2_available"),
        pl.col("empty_docks").shift(2).over("station").alias("_lag2_empty_docks"),
    )
    grid = grid.with_columns(
        (pl.col("available_bikes") - pl.col("lag_1_available")).alias("delta_30m_available"),
        (pl.col("empty_docks") - pl.col("lag_1_empty_docks")).alias("delta_30m_empty_docks"),
        (pl.col("available_bikes") - pl.col("_lag2_available")).alias("delta_60m_available"),
        (pl.col("empty_docks") - pl.col("_lag2_empty_docks")).alias("delta_60m_empty_docks"),
    )

    # Time features (from current timestamp; safe).
    grid = grid.with_columns(
        pl.col("ts30").dt.hour().alias("hour"),
        pl.col("ts30").dt.weekday().alias("day_of_week"),  # 1=Mon..7=Sun
    )
    grid = grid.with_columns(
        (pl.col("day_of_week") >= 6).cast(pl.Int64).alias("is_weekend"),
    )

    # Station historical average supply state = expanding mean of PAST available_ratio.
    # shift(1) first so the CURRENT row's value is NOT included -> no leakage.
    grid = grid.with_columns(
        pl.col("available_ratio")
        .shift(1)
        .cum_sum()
        .over("station")
        .alias("_cum_sum_ratio"),
    )
    grid = grid.with_columns(
        (pl.int_range(0, pl.len()).over("station")).alias("_row_idx_in_station"),
    )
    grid = grid.with_columns(
        pl.when(pl.col("_row_idx_in_station") > 0)
        .then(pl.col("_cum_sum_ratio") / pl.col("_row_idx_in_station"))
        .otherwise(None)
        .alias("station_hist_avg_available_ratio"),
    )

    # ---- Targets: state at the NEXT 30-min slot (future) ----
    grid = grid.with_columns(
        pl.col("available_ratio").shift(-1).over("station").alias("_next_available_ratio"),
        pl.col("empty_dock_ratio").shift(-1).over("station").alias("_next_empty_dock_ratio"),
        # verify the next row is exactly +30min (guards station-boundary leakage)
        pl.col("ts30").shift(-1).over("station").alias("_next_ts30"),
    )
    grid = grid.with_columns(
        ((pl.col("_next_ts30") - pl.col("ts30")).dt.total_minutes() == 30).alias("_next_is_valid"),
    )
    grid = grid.with_columns(
        pl.when(pl.col("_next_is_valid"))
        .then((pl.col("_next_available_ratio") <= SHORTAGE_RATIO_THR).cast(pl.Int64))
        .otherwise(None)
        .alias("future_shortage_30m"),
        pl.when(pl.col("_next_is_valid"))
        .then((pl.col("_next_empty_dock_ratio") <= FULL_RATIO_THR).cast(pl.Int64))
        .otherwise(None)
        .alias("future_full_30m"),
    )

    # Drop helper cols.
    drop_cols = [c for c in grid.columns if c.startswith("_")]
    grid = grid.drop(drop_cols)

    # Labeled dataset: need targets present AND all features present.
    # Feature nulls come from lags at each station's first 1-2 rows.
    labeled = grid.drop_nulls(
        subset=FEATURE_COLUMNS + ["future_shortage_30m", "future_full_30m"]
    )
    print(f"Grid rows: {grid.height}")
    print(f"Labeled rows (features + targets non-null): {labeled.height}")
    print(f"Dropped (warm-up lags + last slot per station): {grid.height - labeled.height}")

    # Save the full processed feature table (parquet).
    os.makedirs(os.path.dirname(PROC_PARQUET), exist_ok=True)
    labeled.write_parquet(PROC_PARQUET)
    print(f"Wrote {PROC_PARQUET}")

    # ---- Time-based split (no random). Early / mid / late by ts30 date ----
    # 14 days -> Train days 1-9, Validation days 10-11, Test days 12-14 (~64/14/21%).
    d0 = labeled["ts30"].min()
    # boundaries as datetimes
    from datetime import timedelta
    day = timedelta(days=1)
    start_date = d0.date()
    val_start = start_date + timedelta(days=9)   # 2026-03-10
    test_start = start_date + timedelta(days=11)  # 2026-03-12

    train = labeled.filter(pl.col("ts30").dt.date() < val_start)
    val = labeled.filter(
        (pl.col("ts30").dt.date() >= val_start) & (pl.col("ts30").dt.date() < test_start)
    )
    test = labeled.filter(pl.col("ts30").dt.date() >= test_start)

    def rng(name, d):
        print(f"  {name}: {d.height} rows | {d['ts30'].min()} -> {d['ts30'].max()}")

    print("\nTime-based split:")
    rng("Train", train)
    rng("Validation", val)
    rng("Test", test)

    # ---- Write SageMaker CSVs: no header, label first, numeric features only ----
    def write_sm_csv(frame, target, path):
        cols = [target] + FEATURE_COLUMNS  # label first
        frame.select(cols).write_csv(path, include_header=False)

    for tgt, sub in [("future_shortage_30m", "shortage"), ("future_full_30m", "full")]:
        base = os.path.join(OUT_DIR, "model_data", sub)
        os.makedirs(base, exist_ok=True)
        write_sm_csv(train, tgt, os.path.join(base, "train.csv"))
        write_sm_csv(val, tgt, os.path.join(base, "validation.csv"))
        write_sm_csv(test, tgt, os.path.join(base, "test.csv"))
        # positive rate report
        for nm, d in [("train", train), ("val", val), ("test", test)]:
            pos = d[tgt].mean()
            print(f"  [{sub}] {nm} positive rate: {100*pos:.2f}%  (n={d.height})")

    # ---- feature_columns.json ----
    fc_path = os.path.join(OUT_DIR, "feature_columns.json")
    with open(fc_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "targets": {
                    "future_shortage_30m": f"available_ratio_next_30m <= {SHORTAGE_RATIO_THR}",
                    "future_full_30m": f"empty_dock_ratio_next_30m <= {FULL_RATIO_THR}",
                },
                "csv_format": "no header; column 0 = label; columns 1..n = features in listed order",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote {fc_path}")


if __name__ == "__main__":
    main()

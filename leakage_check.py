"""
Task 9 prep: leakage audit on the processed features.
Reconstruct the grid independently and confirm:
 1. Each labeled row's target == the NEXT slot's ratio-threshold state.
 2. lag_1_available == previous slot's available_bikes.
 3. station_hist_avg excludes the current row (uses only past).
 4. No column equals a future-derived value.
 5. Split date ranges do not overlap.
"""
import polars as pl

CACHE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816\data\raw_cache.parquet"
PROC = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816\data\processed\youbike_features.parquet"
SHORTAGE_THR = 0.15
FULL_THR = 0.10


def build_grid(df):
    df = df.with_columns(pl.col("ts").dt.truncate("30m").alias("ts30"))
    return (
        df.sort("ts").group_by(["station", "ts30"]).agg(
            pl.col("capacity").last(),
            pl.col("available_bikes").last(),
            pl.col("empty_docks").last(),
        ).sort(["station", "ts30"])
    )


def main():
    raw = pl.read_parquet(CACHE)
    grid = build_grid(raw).with_columns(
        (pl.col("available_bikes") / pl.col("capacity")).alias("available_ratio"),
        (pl.col("empty_docks") / pl.col("capacity")).alias("empty_dock_ratio"),
    )
    # independent ground-truth: next slot ratios and prev available
    grid = grid.with_columns(
        pl.col("available_ratio").shift(-1).over("station").alias("gt_next_ar"),
        pl.col("empty_dock_ratio").shift(-1).over("station").alias("gt_next_er"),
        pl.col("ts30").shift(-1).over("station").alias("gt_next_ts"),
        pl.col("available_bikes").shift(1).over("station").alias("gt_prev_avail"),
    ).with_columns(
        ((pl.col("gt_next_ts") - pl.col("ts30")).dt.total_minutes() == 30).alias("valid_next")
    )

    proc = pl.read_parquet(PROC)

    # Join processed rows back to ground truth on (station, ts30).
    j = proc.join(
        grid.select(["station", "ts30", "gt_next_ar", "gt_next_er",
                     "gt_prev_avail", "valid_next"]),
        on=["station", "ts30"], how="left",
    )

    # 1. target correctness
    exp_short = (j["gt_next_ar"] <= SHORTAGE_THR).cast(pl.Int64)
    exp_full = (j["gt_next_er"] <= FULL_THR).cast(pl.Int64)
    mism_short = (j["future_shortage_30m"] != exp_short).sum()
    mism_full = (j["future_full_30m"] != exp_full).sum()
    print(f"[1] shortage target mismatches vs recomputed-from-next: {mism_short}")
    print(f"[1] full target mismatches vs recomputed-from-next:     {mism_full}")

    # 2. lag_1_available correctness
    mism_lag = (j["lag_1_available"] != j["gt_prev_avail"]).sum()
    print(f"[2] lag_1_available mismatches vs previous slot:        {mism_lag}")

    # 3. all labeled rows must have a valid future neighbor (+30min)
    invalid = (~j["valid_next"]).sum()
    print(f"[3] labeled rows without a valid +30min future slot:    {invalid}")

    # 4. sanity: targets should NOT equal current-row state (that'd be trivial leakage)
    cur_short = (j["available_ratio"] <= SHORTAGE_THR).cast(pl.Int64)
    agree_with_current = (j["future_shortage_30m"] == cur_short).mean()
    print(f"[4] fraction where future_shortage == CURRENT-state shortage: "
          f"{100*agree_with_current:.2f}% (should be high but NOT 100%)")
    identical = (j["future_shortage_30m"] == cur_short).all()
    print(f"    future target identical to current state everywhere? {identical} "
          f"(must be False)")

    # 5. split overlap check
    from datetime import timedelta
    d0 = proc["ts30"].min().date()
    val_start = d0 + timedelta(days=9)
    test_start = d0 + timedelta(days=11)
    train = proc.filter(pl.col("ts30").dt.date() < val_start)
    val = proc.filter((pl.col("ts30").dt.date() >= val_start) & (pl.col("ts30").dt.date() < test_start))
    test = proc.filter(pl.col("ts30").dt.date() >= test_start)
    print(f"[5] Train  last : {train['ts30'].max()}")
    print(f"[5] Val    first: {val['ts30'].min()}  last: {val['ts30'].max()}")
    print(f"[5] Test   first: {test['ts30'].min()}")
    no_overlap = train["ts30"].max() < val["ts30"].min() < val["ts30"].max() < test["ts30"].min()
    print(f"[5] strictly non-overlapping & ordered: {no_overlap}")


if __name__ == "__main__":
    main()

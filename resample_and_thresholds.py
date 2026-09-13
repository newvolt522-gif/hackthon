"""
Task 3: Resample to per-station 30-min grid; report missing ratio (no over-imputing).
Task 4: Analyze target threshold options for shortage & full risk.
"""
import polars as pl

CACHE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816\data\raw_cache.parquet"


def build_grid(df):
    """Collapse to (station, ts30). If a slot has >1 raw row, take the last by ts."""
    df = df.with_columns(pl.col("ts").dt.truncate("30m").alias("ts30"))
    # one row per (station, ts30): keep the latest raw ts within the slot
    grid = (
        df.sort("ts")
        .group_by(["station", "ts30"])
        .agg(
            pl.col("capacity").last(),
            pl.col("available_bikes").last(),
            pl.col("empty_docks").last(),
            pl.col("district").last(),
            pl.col("lon").last(),
            pl.col("lat").last(),
        )
        .sort(["station", "ts30"])
    )
    return grid


def main():
    df = pl.read_parquet(CACHE)
    grid = build_grid(df)
    print(f"Grid rows (station x 30-min slot): {grid.height}")

    # ---- Task 3: missingness vs a complete per-station grid ----
    # Full grid = each station's own [first_slot, last_slot] at 30-min steps.
    span = grid.group_by("station").agg(
        pl.col("ts30").min().alias("first"),
        pl.col("ts30").max().alias("last"),
        pl.len().alias("n_present"),
    )
    # expected slots within each station's own active window
    span = span.with_columns(
        (((pl.col("last") - pl.col("first")).dt.total_minutes() // 30) + 1).alias("n_expected")
    )
    span = span.with_columns(
        (pl.col("n_expected") - pl.col("n_present")).alias("n_missing_internal")
    )
    total_present = span["n_present"].sum()
    total_expected_within_window = span["n_expected"].sum()
    internal_missing = span["n_missing_internal"].sum()
    print("\n=== Task 3: Missingness (within each station's active window) ===")
    print(f"Total present rows:            {total_present}")
    print(f"Total expected (within window): {total_expected_within_window}")
    print(f"Internal missing slots:        {internal_missing} "
          f"({100*internal_missing/total_expected_within_window:.4f}%)")
    print("Stations with internal gaps:",
          span.filter(pl.col('n_missing_internal') > 0).height)

    # Missingness vs the GLOBAL grid (full 14 days for every station).
    global_first = grid["ts30"].min()
    global_last = grid["ts30"].max()
    n_global_slots = (((global_last - global_first).total_seconds() // 1800) + 1)
    n_stations = grid["station"].n_unique()
    global_expected = int(n_global_slots * n_stations)
    print(f"\nGlobal grid: {n_stations} stations x {int(n_global_slots)} slots "
          f"= {global_expected} cells")
    print(f"Present: {grid.height} "
          f"-> missing vs global {100*(global_expected-grid.height)/global_expected:.2f}% "
          f"(driven by 3 late-start stations)")

    # ---- Task 4: threshold analysis ----
    grid = grid.with_columns(
        (pl.col("available_bikes") / pl.col("capacity")).alias("available_ratio"),
        (pl.col("empty_docks") / pl.col("capacity")).alias("empty_dock_ratio"),
    )

    print("\n=== Task 4: SHORTAGE candidates (few bikes to borrow) ===")
    for thr in [0, 1, 2, 3]:
        p = 100 * grid.filter(pl.col("available_bikes") <= thr).height / grid.height
        print(f"  A. available_bikes <= {thr}: {p:.2f}% of rows")
    for r in [0.05, 0.10, 0.15, 0.20]:
        p = 100 * grid.filter(pl.col("available_ratio") <= r).height / grid.height
        print(f"  B. available_ratio <= {r:.2f}: {p:.2f}% of rows")

    print("\n=== Task 4: FULL candidates (few docks to return) ===")
    for thr in [0, 1, 2, 3]:
        p = 100 * grid.filter(pl.col("empty_docks") <= thr).height / grid.height
        print(f"  A. empty_docks <= {thr}: {p:.2f}% of rows")
    for r in [0.05, 0.10, 0.15, 0.20]:
        p = 100 * grid.filter(pl.col("empty_dock_ratio") <= r).height / grid.height
        print(f"  B. empty_dock_ratio <= {r:.2f}: {p:.2f}% of rows")

    # capacity distribution: helps decide count vs ratio
    print("\n=== Capacity distribution (why ratio may be fairer) ===")
    print(grid.select(
        pl.col("capacity").quantile(0.05).alias("p05"),
        pl.col("capacity").quantile(0.25).alias("p25"),
        pl.col("capacity").median().alias("p50"),
        pl.col("capacity").quantile(0.75).alias("p75"),
        pl.col("capacity").quantile(0.95).alias("p95"),
    ))


if __name__ == "__main__":
    main()

"""
Task 2: Data quality report (10 checks). Lists anomalies; deletes nothing.
Reads the cached UTF-8 parquet produced in task 1.
"""
import polars as pl

CACHE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816\data\raw_cache.parquet"

pl.Config(tbl_rows=20, tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=40)


def main():
    df = pl.read_parquet(CACHE)

    print("#" * 60)
    print("# 1. Row count")
    print("#" * 60)
    print(f"Total rows: {df.height}")

    print("\n" + "#" * 60)
    print("# 2. Column names & dtypes")
    print("#" * 60)
    for n, d in zip(df.columns, df.dtypes):
        print(f"  {n}: {d}")

    print("\n" + "#" * 60)
    print("# 3. Missing values per column")
    print("#" * 60)
    nulls = df.null_count()
    print(nulls)
    # parse failures for ts specifically
    ts_null = df.filter(pl.col("ts").is_null())
    print(f"Rows where ts failed to parse: {ts_null.height}")

    print("\n" + "#" * 60)
    print("# 4. Duplicate rows")
    print("#" * 60)
    full_dupes = df.height - df.unique().height
    # duplicates on the key (station, ts) after flooring to 30-min grid
    df = df.with_columns(pl.col("ts").dt.truncate("30m").alias("ts30"))
    key_dupes = (
        df.group_by(["station", "ts30"]).len().filter(pl.col("len") > 1)
    )
    print(f"Fully identical rows: {full_dupes}")
    print(f"(station, 30-min-slot) combos with >1 row: {key_dupes.height}")
    print("Sample of duplicated keys:")
    print(key_dupes.sort("len", descending=True).head(5))

    print("\n" + "#" * 60)
    print("# 5. Rows per station")
    print("#" * 60)
    per_station = df.group_by("station").len().sort("len")
    print(f"stations: {per_station.height}")
    print(f"min={per_station['len'].min()}, max={per_station['len'].max()}, "
          f"median={per_station['len'].median()}")
    full = 14 * 48
    incomplete = per_station.filter(pl.col("len") < full)
    print(f"Stations with < {full} rows (incomplete coverage): {incomplete.height}")
    print(incomplete.head(10))

    print("\n" + "#" * 60)
    print("# 6. Per-station time coverage range")
    print("#" * 60)
    cov = df.group_by("station").agg(
        pl.col("ts").min().alias("first"),
        pl.col("ts").max().alias("last"),
        pl.col("ts").n_unique().alias("n_snapshots"),
    )
    global_first = df["ts"].min()
    global_last = df["ts"].max()
    print(f"Global span: {global_first} -> {global_last}")
    late_start = cov.filter(pl.col("first") > global_first).sort("first", descending=True)
    early_end = cov.filter(pl.col("last") < global_last).sort("last")
    print(f"Stations starting later than global first: {late_start.height}")
    print(late_start.head(5))
    print(f"Stations ending earlier than global last: {early_end.height}")
    print(early_end.head(5))

    print("\n" + "#" * 60)
    print("# 7. Time interval regularity (on 30-min grid, per station)")
    print("#" * 60)
    # For each station, diffs between consecutive 30-min slots.
    sorted_df = df.select(["station", "ts30"]).unique().sort(["station", "ts30"])
    gaps = sorted_df.with_columns(
        pl.col("ts30").diff().over("station").alias("gap")
    ).drop_nulls("gap")
    gap_counts = gaps.group_by("gap").len().sort("len", descending=True)
    print("Distribution of gaps between consecutive 30-min slots:")
    print(gap_counts.head(10))
    irregular = gaps.filter(pl.col("gap") != pl.duration(minutes=30))
    print(f"Gaps != 30min (i.e. missing slots somewhere): {irregular.height}")

    print("\n" + "#" * 60)
    print("# 8. Unreasonable values")
    print("#" * 60)
    bad_cap = df.filter(pl.col("capacity") <= 0)
    bad_avail = df.filter((pl.col("available_bikes") < 0) | (pl.col("available_bikes") > pl.col("capacity")))
    bad_empty = df.filter((pl.col("empty_docks") < 0) | (pl.col("empty_docks") > pl.col("capacity")))
    print(f"capacity <= 0: {bad_cap.height}")
    print(f"available_bikes < 0 or > capacity: {bad_avail.height}")
    print(f"empty_docks < 0 or > capacity: {bad_empty.height}")
    if bad_avail.height:
        print(bad_avail.head(5))
    if bad_empty.height:
        print(bad_empty.head(5))

    print("\n" + "#" * 60)
    print("# 9. available + empty == capacity ?")
    print("#" * 60)
    df = df.with_columns(
        (pl.col("available_bikes") + pl.col("empty_docks")).alias("sum_ae")
    )
    mismatch = df.filter(pl.col("sum_ae") != pl.col("capacity"))
    print(f"Rows where available+empty != capacity: {mismatch.height} "
          f"({100*mismatch.height/df.height:.2f}%)")
    if mismatch.height:
        diff_dist = mismatch.with_columns(
            (pl.col("sum_ae") - pl.col("capacity")).alias("diff")
        ).group_by("diff").len().sort("len", descending=True)
        print("Distribution of (available+empty - capacity):")
        print(diff_dist.head(10))
        print(mismatch.select(
            ["station", "ts", "capacity", "available_bikes", "empty_docks", "sum_ae"]
        ).head(5))

    print("\n" + "#" * 60)
    print("# 10. Longitude / latitude outliers")
    print("#" * 60)
    # New Taipei roughly lon 121.3-122.0, lat 24.6-25.4
    lon_bad = df.filter((pl.col("lon") < 121.0) | (pl.col("lon") > 122.2))
    lat_bad = df.filter((pl.col("lat") < 24.5) | (pl.col("lat") > 25.5))
    print(f"lon outside [121.0, 122.2]: {lon_bad.height}")
    print(f"lat outside [24.5, 25.5]: {lat_bad.height}")
    # Also: stations whose coords vary over time (should be fixed)
    coord_var = df.group_by("station").agg(
        pl.col("lon").n_unique().alias("n_lon"),
        pl.col("lat").n_unique().alias("n_lat"),
    ).filter((pl.col("n_lon") > 1) | (pl.col("n_lat") > 1))
    print(f"Stations with varying coordinates over time: {coord_var.height}")
    print(coord_var.head(5))


if __name__ == "__main__":
    main()

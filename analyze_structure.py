"""
Task 1: Understand the full dataset structure before building the pipeline.
Reads the Big5-encoded CSV, converts to a UTF-8 parquet cache, and reports:
- row count, timestamp cadence, per-station snapshot counts, value ranges.
"""
import os
import polars as pl

SRC = r"C:\Users\20040\OneDrive\桌面\黑客松\A_交通局-資料集\新北AWS黑克松競賽0301-14.csv 的副本.csv"
CACHE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816\data\raw_cache.parquet"

RENAME = {
    "日期": "ts_raw",
    "城市": "city",
    "行政區": "district",
    "場站名稱": "station",
    "總車柱數": "capacity",
    "可借車數": "available_bikes",
    "可還位數": "empty_docks",
    "經度": "lon",
    "緯度": "lat",
}


def load_utf8_parquet():
    """Decode Big5 -> write UTF-8 parquet once, reuse afterwards."""
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if os.path.exists(CACHE):
        return pl.read_parquet(CACHE)

    # Decode the whole file from Big5 into a UTF-8 temp, then let polars parse.
    tmp_utf8 = CACHE + ".utf8.csv"
    with open(SRC, "r", encoding="big5", errors="replace") as fin, \
         open(tmp_utf8, "w", encoding="utf-8", newline="") as fout:
        for line in fin:
            fout.write(line)

    df = pl.read_csv(tmp_utf8)
    df = df.rename(RENAME)
    df = df.with_columns(
        pl.col("ts_raw").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).alias("ts")
    )
    df.write_parquet(CACHE)
    os.remove(tmp_utf8)
    return df


def main():
    df = load_utf8_parquet()
    print(f"Total rows: {df.height}")
    print(f"Columns: {df.columns}")

    # Timestamp span
    ts_min = df["ts"].min()
    ts_max = df["ts"].max()
    n_ts = df["ts"].n_unique()
    print(f"\nTimestamp span: {ts_min}  ->  {ts_max}")
    print(f"Distinct timestamps: {n_ts}")

    # Cadence: diffs between consecutive distinct snapshot timestamps
    uniq_ts = df.select("ts").unique().sort("ts")
    diffs = uniq_ts.select(pl.col("ts").diff().alias("d")).drop_nulls()
    diff_counts = (
        diffs.group_by("d").len().sort("len", descending=True).head(15)
    )
    print("\nTop gaps between consecutive snapshot timestamps:")
    print(diff_counts)

    # Stations
    n_stations = df["station"].n_unique()
    print(f"\nDistinct stations (by name): {n_stations}")
    # Is station name unique per (district)? check duplicates of names
    per_station = df.group_by("station").len().sort("len", descending=True)
    print("\nRows per station (top 5 / bottom 5):")
    print(per_station.head(5))
    print(per_station.tail(5))
    print(f"Rows-per-station: min={per_station['len'].min()}, "
          f"max={per_station['len'].max()}, "
          f"median={per_station['len'].median()}")

    # Value ranges
    print("\nValue ranges:")
    print(df.select(
        pl.col("capacity").min().alias("cap_min"),
        pl.col("capacity").max().alias("cap_max"),
        pl.col("available_bikes").min().alias("avail_min"),
        pl.col("available_bikes").max().alias("avail_max"),
        pl.col("empty_docks").min().alias("empty_min"),
        pl.col("empty_docks").max().alias("empty_max"),
        pl.col("lon").min().alias("lon_min"),
        pl.col("lon").max().alias("lon_max"),
        pl.col("lat").min().alias("lat_min"),
        pl.col("lat").max().alias("lat_max"),
    ))

    # Same station name across different districts / coords?
    name_district = df.select("station", "district").unique()
    dup_names = (
        name_district.group_by("station").len().filter(pl.col("len") > 1)
    )
    print(f"\nStation names appearing in >1 district: {dup_names.height}")
    print(dup_names.head(10))


if __name__ == "__main__":
    main()

import io
import polars as pl

PATH = r"C:\Users\20040\OneDrive\桌面\黑客松\A_交通局-資料集\新北AWS黑克松競賽0301-14.csv 的副本.csv"

# Read a sample of lines decoded from Big5.
lines = []
with open(PATH, "r", encoding="big5", errors="replace") as f:
    for i, line in enumerate(f):
        lines.append(line)
        if i >= 30:
            break

sample = "".join(lines)
df = pl.read_csv(io.StringIO(sample), infer_schema_length=30)
print("=== dtypes ===")
for n, d in zip(df.columns, df.dtypes):
    print(f"  {n!r}: {d}")

print("\n=== sample rows ===")
with pl.Config(tbl_cols=-1, tbl_width_chars=300, fmt_str_lengths=40):
    print(df.head(15))

# Look at distinct date-like values to understand cadence.
print("\n=== distinct 日期 values in sample ===")
print(df.select(pl.col("日期").unique()).to_series().to_list())

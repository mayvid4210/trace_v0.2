import pandas as pd

df = pd.read_parquet("data/processed/resistance_data.parquet")

df["speed_bin"] = pd.cut(
    df["Speed"],
    bins=[140, 180, 220, 260, 300, 340, 360]
)

result = df.groupby("speed_bin", observed=True)["RPM"].agg(
    ["count", "mean", "median", "std"]
)

print(result.to_string())
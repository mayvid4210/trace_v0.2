import pandas as pd

df = pd.read_parquet("data/processed/resistance_data.parquet")

df["speed_bin"] = pd.cut(
    df["Speed"],
    bins=[100, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360]
)

result = df.groupby("speed_bin", observed=True)["resistance_force_n"].agg(
    ["count", "mean", "median"]
)

print(result.to_string())
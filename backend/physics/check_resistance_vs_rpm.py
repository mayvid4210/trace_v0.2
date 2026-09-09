import pandas as pd

df = pd.read_parquet("data/processed/resistance_data.parquet")

df["speed_bin"] = pd.cut(
    df["Speed"],
    bins=[140, 180, 220, 260, 300, 340, 360]
)

for label, group in df.groupby("speed_bin", observed=True):
    corr = group["RPM"].corr(group["resistance_force_n"])
    print(f"{label}: n={len(group)}, RPM-vs-resistance correlation = {corr:.3f}")
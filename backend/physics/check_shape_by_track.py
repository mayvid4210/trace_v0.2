import pandas as pd

df = pd.read_parquet("data/processed/resistance_data.parquet")

df["speed_bin"] = pd.cut(
    df["Speed"],
    bins=[100, 140, 180, 220, 260, 300, 340, 360]
)

for track in df["GrandPrix"].unique():
    sub = df[df["GrandPrix"] == track]
    print(f"\n--- {track} ---")
    print(
        sub.groupby("speed_bin", observed=True)["resistance_force_n"]
        .agg(["count", "median"])
        .to_string()
    )
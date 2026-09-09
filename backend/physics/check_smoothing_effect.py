import pandas as pd

df = pd.read_parquet("data/processed/resistance_data.parquet")

df["resistance_raw_n"] = df["total_mass_kg"] * (-df["Acceleration_raw"])

df["speed_bin"] = pd.cut(
    df["Speed"],
    bins=[100, 140, 180, 220, 260, 300, 340, 360]
)

result = df.groupby("speed_bin", observed=True)[
    ["resistance_force_n", "resistance_raw_n"]
].median()

print(result.to_string())
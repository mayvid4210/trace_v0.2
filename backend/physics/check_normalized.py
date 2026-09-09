import pandas as pd

df = pd.read_parquet("data/processed/resistance_data.parquet")

df["force_per_v2"] = df["resistance_force_n"] / (df["Speed_ms"] ** 2)

df["speed_bin"] = pd.cut(
    df["Speed"], bins=[140, 160, 180, 200, 220, 240, 260, 280, 300, 320]
)

print(
    df.groupby("speed_bin", observed=True)["force_per_v2"]
    .agg(["count", "mean", "median"])
    .to_string()
)

print("\nMass range:")
print(df["total_mass_kg"].describe())

import pandas as pd


df = pd.read_parquet(
    "data/processed/resistance_data.parquet"
)

print("\n========== DRS RESISTANCE CHECK ==========")

print("\nDRS distribution:")
print(
    df["DRS"].value_counts(dropna=False)
)

print("\nResistance by DRS state:")

print(
    df.groupby("DRS")["resistance_force_n"].agg(
        [
            "count",
            "mean",
            "median",
            "std"
        ]
    )
)

print("\nResistance by speed and DRS:")

df["speed_bin"] = pd.cut(
    df["Speed"],
    bins=[
        140,
        160,
        180,
        200,
        220,
        240,
        260,
        280,
        300,
        320,
        350
    ]
)

result = (
    df.groupby(
        ["speed_bin", "DRS"],
        observed=True
    )["resistance_force_n"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std"
        ]
    )
)

print(result)
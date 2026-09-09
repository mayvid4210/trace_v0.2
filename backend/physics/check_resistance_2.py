import pandas as pd


df = pd.read_parquet(
    "data/processed/resistance_data.parquet"
)

print("\n========== CLEAN RESISTANCE CHECK ==========")

print("\nSamples:")
print(len(df))

print("\nSpeed bins:")

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

result = df.groupby(
    "speed_bin",
    observed=True
)["resistance_force_n"].agg(
    [
        "count",
        "mean",
        "median",
        "std"
    ]
)

print(result)
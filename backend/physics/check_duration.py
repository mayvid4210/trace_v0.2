import pandas as pd

df = pd.read_parquet("data/processed/resistance_data.parquet")

# how many consecutive rows (by lap) are actually in this coasting filter?
df = df.sort_values(["GrandPrix", "SessionName", "Driver", "LapNumber", "Time"])

df["group_id"] = (
    df.groupby(["GrandPrix", "SessionName", "Driver", "LapNumber"])
    .cumcount()
)

run_lengths = df.groupby(["GrandPrix", "SessionName", "Driver", "LapNumber"]).size()

print(run_lengths.describe())
print("\n% of laps with only 1-2 coasting points:")
print((run_lengths <= 2).mean())
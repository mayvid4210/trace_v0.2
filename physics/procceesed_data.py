import pandas as pd

for name in ["resistance_curve", "resistance_data", "resistance_analysis",
             "physics_data", "physics_data_with_force", "physics_data_with_regime"]:
    try:
        df = pd.read_parquet(f"data/processed/{name}.parquet")
        print(f"\n=== {name} ===")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head(3))
    except Exception as e:
        print(f"\n=== {name} === FAILED: {e}")

laps = pd.read_parquet("data/processed/lap_features.parquet")
print(laps.columns.tolist())
print(laps[["Driver", "lap", "air_temperature", "fuel_estimate"]].head())        
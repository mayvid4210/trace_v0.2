import pandas as pd
df = pd.read_parquet("data/processed/telemetry_with_acceleration.parquet")
print(df.columns.tolist())
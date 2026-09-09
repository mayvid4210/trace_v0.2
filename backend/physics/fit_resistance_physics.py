import numpy as np
import pandas as pd

INPUT_FILE = "data/processed/resistance_data.parquet"
OUTPUT_FILE = "data/processed/resistance_curve.parquet"

def build_resistance_curve(df):
    df = df.dropna(subset=["Speed_ms", "resistance_force_n"])
    df["speed_bin"] = pd.cut(df["Speed_ms"], bins=np.arange(35, 100, 5.0), include_lowest=True)

    curve = (
        df.groupby("speed_bin", observed=True)
        .agg(speed_ms=("Speed_ms", "mean"),
             resistance_force_n=("resistance_force_n", "median"),
             samples=("resistance_force_n", "count"))
        .reset_index(drop=True)
    )
    curve = curve[curve["samples"] >= 5].sort_values("speed_ms").reset_index(drop=True)
    return curve

if __name__ == "__main__":
    df = pd.read_parquet(INPUT_FILE)
    curve = build_resistance_curve(df)
    print(curve.to_string(index=False))
    curve.to_parquet(OUTPUT_FILE, index=False)
    print("\nSaved:", OUTPUT_FILE)
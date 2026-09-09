import pandas as pd
import numpy as np


data = pd.read_parquet(
    "data/processed/physics_data.parquet"
)

df = data[
    (data["Throttle"] <= 2.0) &
    (data["Brake"] == False) &
    (data["Speed_ms"] > 25.0) &
    (data["Acceleration_smoothed"] < 0) &
    (data["Acceleration_smoothed"] > -6.0)
].copy()

df = df.dropna(
    subset=[
        "Speed_ms",
        "Acceleration_smoothed",
        "total_mass_kg",
        "air_temperature"
    ]
)

df["resistance_force_n"] = (
    df["total_mass_kg"]
    * (-df["Acceleration_smoothed"])
)

df["speed_squared"] = (
    df["Speed_ms"] ** 2
)

print("\n========== RESISTANCE DATA CHECK ==========")

print("\nSamples:")
print(len(df))

print("\nSpeed:")
print(df["Speed"].describe())

print("\nAcceleration:")
print(df["Acceleration_smoothed"].describe())

print("\nResistance:")
print(df["resistance_force_n"].describe())

print("\nResistance by speed bin:")

df["speed_bin"] = pd.cut(
    df["Speed"],
    bins=[
        80,
        100,
        120,
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

summary = df.groupby(
    "speed_bin",
    observed=True
)["resistance_force_n"].agg(
    ["count", "mean", "median", "std"]
)

print(summary)
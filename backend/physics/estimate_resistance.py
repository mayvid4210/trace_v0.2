import numpy as np
import pandas as pd


def find_coasting_samples(df):

    df = df.copy()

    coasting = df[
        (df["Throttle"] <= 2.0) &
        (df["Brake"] == False) &
        (df["Speed_ms"] > 25.0) &
        (df["Acceleration_smoothed"] < 0) &
        (df["Acceleration_smoothed"] > -6.0)
    ].copy()

    return coasting


def estimate_resistance(df):

    coasting = find_coasting_samples(df)

    coasting = coasting.dropna(
        subset=[
            "Speed_ms",
            "Acceleration_smoothed",
            "total_mass_kg",
            "air_temperature"
        ]
    )

    if coasting.empty:
        raise ValueError("No valid coasting samples found.")

    coasting["deceleration"] = (
        -coasting["Acceleration_smoothed"]
    )

    coasting["v_squared"] = (
        coasting["Speed_ms"] ** 2
    )

    coasting["mass_accel_force"] = (
        coasting["total_mass_kg"]
        * coasting["deceleration"]
    )

    return coasting


if __name__ == "__main__":

    df = pd.read_parquet(
        "data/processed/physics_data.parquet"
    )

    coasting = estimate_resistance(df)

    print("\n========== COASTING SAMPLES ==========")

    print("\nNumber of coasting samples:")
    print(len(coasting))

    print("\nSample:")
    print(
        coasting[
            [
                "GrandPrix",
                "SessionName",
                "Driver",
                "LapNumber",
                "Speed",
                "Throttle",
                "Brake",
                "Acceleration_smoothed",
                "total_mass_kg",
                "deceleration",
                "mass_accel_force"
            ]
        ].head(20).to_string(index=False)
    )

    print("\nDeceleration statistics:")
    print(
        coasting["deceleration"].describe()
    )

    print("\nSpeed statistics:")
    print(
        coasting["Speed"].describe()
    )
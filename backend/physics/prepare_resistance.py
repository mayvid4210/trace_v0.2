import numpy as np
import pandas as pd


INPUT_FILE = "data/processed/physics_data.parquet"
OUTPUT_FILE = "data/processed/resistance_data.parquet"

MIN_RUN_LENGTH = 5
GROUP_KEYS = ["GrandPrix", "SessionName", "Driver", "LapNumber"]


def prepare_resistance_data(df):

    df = df.copy()
    df = df.sort_values(GROUP_KEYS + ["Time"])

    df["is_coasting"] = (
        (df["Throttle"] <= 0.0) &
        (df["Brake"] == False) &
        (df["DRS"] == 0.0) &
        (df["nGear"] >= 7) &
        (df["Speed_ms"] >= 40.0)
    )

    df["run_change"] = (
        df.groupby(GROUP_KEYS)["is_coasting"]
        .transform(lambda x: (x != x.shift()).cumsum())
    )

    df["run_id"] = (
        df["GrandPrix"].astype(str) + "_" +
        df["SessionName"].astype(str) + "_" +
        df["Driver"].astype(str) + "_" +
        df["LapNumber"].astype(str) + "_" +
        df["run_change"].astype(str)
    )

    run_lengths = df.groupby("run_id").size()
    df["run_length"] = df["run_id"].map(run_lengths)

    df = df[
        df["is_coasting"] &
        (df["run_length"] >= MIN_RUN_LENGTH)
    ].copy()

    df["row_in_run"] = df.groupby("run_id").cumcount()
    df = df[
        (df["row_in_run"] > 0) &
        (df["row_in_run"] < df["run_length"] - 1)
    ].copy()

    df = df[
        (df["Acceleration_raw"] < 0) &
        (df["Acceleration_raw"] > -5.0)
    ].copy()

    df = df.dropna(
        subset=["Speed_ms", "Acceleration_raw", "total_mass_kg", "air_temperature"]
    )

    if df.empty:
        raise ValueError("No valid resistance samples found.")

    df["resistance_force_n"] = df["total_mass_kg"] * (-df["Acceleration_raw"])

    q1 = df["resistance_force_n"].quantile(0.25)
    q3 = df["resistance_force_n"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    print("\nResistance force IQR filtering:")
    print(f"Q1: {q1:.2f} N")
    print(f"Q3: {q3:.2f} N")
    print(f"Lower bound: {lower:.2f} N")
    print(f"Upper bound: {upper:.2f} N")

    before = len(df)
    df = df[
        (df["resistance_force_n"] >= lower) &
        (df["resistance_force_n"] <= upper)
    ].copy()
    print(f"Samples before: {before}")
    print(f"Samples after: {len(df)}")
    print(f"Removed: {before - len(df)}")

    df["observed_deceleration_force_n"] = df["resistance_force_n"]
    df["speed_squared"] = df["Speed_ms"] ** 2
    df["air_temperature_k"] = df["air_temperature"] + 273.15
    df["air_density"] = 101325.0 / (287.05 * df["air_temperature_k"])

    return df


if __name__ == "__main__":

    print("\n========== PREPARED RESISTANCE DATA ==========")

    data = pd.read_parquet(INPUT_FILE)
    resistance = prepare_resistance_data(data)

    print("\nSamples:")
    print(len(resistance))
    print("\nSpeed range:")
    print(resistance["Speed"].min(), "to", resistance["Speed"].max(), "km/h")
    print("\nObserved deceleration force statistics:")
    print(resistance["observed_deceleration_force_n"].describe())

    resistance.to_parquet(OUTPUT_FILE, index=False)
    print("\nSaved:")
    print(OUTPUT_FILE)
import pandas as pd
import numpy as np


INPUT_FILE = "data/processed/telemetry_with_drs.parquet"
OUTPUT_FILE = "data/processed/telemetry_with_acceleration.parquet"

def flag_coast_points(telemetry_df, throttle_thresh=2, brake_thresh=0.5):
    """
    A 'coast' point = no throttle, no brake.
    On these points, the ONLY forces acting are drag + rolling resistance —
    no engine force, no braking force to confound the fit.
    """
    df = telemetry_df.copy()
    df["is_coast"] = (
        (df["Throttle"] <= throttle_thresh) &
        (df["Brake"] <= brake_thresh)
    )
    return df



def main():

    print("Loading telemetry...")

    df = pd.read_parquet(INPUT_FILE)

    print("Rows:", len(df))

    # Convert speed from km/h to m/s
    df["Speed_ms"] = df["Speed"] / 3.6

    # Make sure telemetry is ordered correctly
    group_cols = [
        "GrandPrix",
        "SessionName",
        "Driver",
        "LapNumber"
    ]

    df = df.sort_values(group_cols + ["Time"]).reset_index(drop=True)
        # Flag coast points (no throttle, no brake) — needed for Step 30 aero regression
    df = flag_coast_points(df)

    # Calculate time difference within each lap
    dt = (
        df.groupby(group_cols)["Time"]
        .diff()
        .dt.total_seconds()
    )

    # Calculate speed difference within each lap
    dv = (
        df.groupby(group_cols)["Speed_ms"]
        .diff()
    )

    # Calculate acceleration
    df["Acceleration_ms2"] = dv / dt

    # Remove invalid values caused by zero/negative time intervals
    df.loc[
        (dt <= 0) | (~np.isfinite(df["Acceleration_ms2"])),
        "Acceleration_ms2"
    ] = np.nan

    # Clip extreme telemetry spikes
    df.loc[
        df["Acceleration_ms2"].abs() > 15,
        "Acceleration_ms2"
    ] = np.nan

    print("\nAcceleration statistics:")

    print(
        df["Acceleration_ms2"].describe()
    )

    print("\nMissing acceleration:")

    print("\nCoast point counts:")
    print(df["is_coast"].value_counts())

    print("\nCoast points with valid acceleration:")
    print(df[df["is_coast"] & df["Acceleration_ms2"].notna()].shape[0])

    print(
        df["Acceleration_ms2"].isna().sum()
    )

    print("\nSaving...")

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n========== COMPLETE ==========")

    print(
        "Saved to:",
        OUTPUT_FILE
    )

    print(
        "Final shape:",
        df.shape
    )

    print(
        "Final columns:",
        df.columns.tolist()
    )


if __name__ == "__main__":
    main()
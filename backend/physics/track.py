import pandas as pd


def add_track_length(df):
    df = df.copy()

    # Max distance reached within EACH lap
    per_lap_length = (
        df.groupby(["GrandPrix", "SessionName", "Driver", "LapNumber"])["Distance"]
        .max()
        .reset_index(name="lap_distance_m")
    )

    # Median across laps per circuit/session — robust to one-off glitches
    track_length = (
        per_lap_length.groupby(["GrandPrix", "SessionName"])["lap_distance_m"]
        .median()
        .rename("track_length_m")
        .reset_index()
    )

    df = df.merge(
        track_length,
        on=["GrandPrix", "SessionName"],
        how="left"
    )

    return df


if __name__ == "__main__":

    telemetry = pd.read_parquet(
        "data/processed/telemetry_with_acceleration.parquet"
    )

    result = add_track_length(telemetry)

    print("\nTrack length by circuit:")
    print(
        result[["GrandPrix", "SessionName", "track_length_m"]]
        .drop_duplicates()
    )
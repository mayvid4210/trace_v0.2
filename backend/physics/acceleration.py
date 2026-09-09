import pandas as pd


def add_smoothed_acceleration(df):
    df = df.copy()

    df["Acceleration_raw"] = df["Acceleration_ms2"]

    df["Acceleration_smoothed"] = (
        df.groupby(
            [
                "GrandPrix",
                "SessionName",
                "Driver",
                "LapNumber"
            ]
        )["Acceleration_raw"]
        .transform(
            lambda x: x.rolling(
                window=5,
                center=True,
                min_periods=1
            ).median()
        )
    )

    return df


if __name__ == "__main__":

    telemetry = pd.read_parquet(
        "data/processed/telemetry_with_acceleration.parquet"
    )

    telemetry = add_smoothed_acceleration(
        telemetry
    )

    print("\nAcceleration check:")
    print(
        telemetry[
            [
                "Speed",
                "Acceleration_raw",
                "Acceleration_smoothed"
            ]
        ].head(20)
    )

    print("\nRaw acceleration:")
    print(
        telemetry["Acceleration_raw"].describe()
    )

    print("\nSmoothed acceleration:")
    print(
        telemetry["Acceleration_smoothed"].describe()
    )

    telemetry.to_parquet(
        "data/processed/telemetry_with_smoothed_acceleration.parquet",
        index=False
    )

    print(
        "\nSaved: "
        "data/processed/telemetry_with_smoothed_acceleration.parquet"
    )
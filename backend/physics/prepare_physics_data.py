import pandas as pd

KEYS = [
    "GrandPrix",
    "SessionName",
    "Driver",
    "LapNumber"
]


def prepare_physics_data():

    telemetry = pd.read_parquet(
        "data/processed/telemetry_with_smoothed_acceleration.parquet"
    )

    laps = pd.read_parquet(
        "data/processed/lap_features.parquet"
    )

    if "LapNumber" not in laps.columns and "lap" in laps.columns:
        laps = laps.rename(columns={"lap": "LapNumber"})

    lap_columns = KEYS + [
        "fuel_estimate",
        "track_temperature",
        "air_temperature",
        "track_grip",
        "tyre_age"
    ]

    laps = laps[lap_columns].copy()

    physics = telemetry.merge(
        laps,
        on=KEYS,
        how="left"
    )

    physics["total_mass_kg"] = (
        798.0 + physics["fuel_estimate"]
    )

    print("\n========== PHYSICS DATA ==========")

    print("\nShape:")
    print(physics.shape)

    print("\nColumns:")
    print(physics.columns.tolist())

    print("\nMissing values:")
    print(
        physics[
            [
                "Acceleration_smoothed",
                "fuel_estimate",
                "total_mass_kg",
                "track_temperature",
                "air_temperature"
            ]
        ].isna().sum()
    )

    print("\nSample:")

    print(
        physics[
            [
                "GrandPrix",
                "SessionName",
                "Driver",
                "LapNumber",
                "Speed",
                "Acceleration_smoothed",
                "fuel_estimate",
                "total_mass_kg",
                "Throttle",
                "Brake",
                "DRS"
            ]
        ].head(20).to_string(index=False)
    )

    physics.to_parquet(
        "data/processed/physics_data.parquet",
        index=False
    )

    print(
        "\nSaved:"
        " data/processed/physics_data.parquet"
    )


if __name__ == "__main__":
    prepare_physics_data()
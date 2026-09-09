import pandas as pd

try:
    from physics.parameters import CAR_MASS_KG
except ModuleNotFoundError:
    from parameters import CAR_MASS_KG


def add_vehicle_mass(df):

    df = df.copy()

    df["total_mass_kg"] = (
        CAR_MASS_KG
        + df["fuel_estimate"]
    )

    return df


if __name__ == "__main__":

    laps = pd.read_parquet(
        "data/processed/lap_features.parquet"
    )

    result = add_vehicle_mass(laps)

    print("\nMass calculation:")

    print(
        result[
            [
                "GrandPrix",
                "SessionName",
                "Driver",
                "lap",
                "fuel_estimate",
                "total_mass_kg"
            ]
        ].head(20)
    )

    print("\nMass statistics:")

    print(
        result["total_mass_kg"].describe()
    )
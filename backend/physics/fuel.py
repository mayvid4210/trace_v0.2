import pandas as pd


MIN_CAR_MASS_KG = 798.0  # FIA 2024 minimum race mass, excludes fuel


def add_total_mass(df):
    """
    Turns fuel_estimate (kg of fuel remaining) into total_mass_kg
    (car + driver + fuel) — the exact input vehicle.py expects.

    Assumes df already has a 'fuel_estimate' column
    (built earlier in feature_engineering.py via add_fuel_estimate_by_session).
    """

    df = df.copy()

    if "fuel_estimate" not in df.columns:
        raise ValueError(
            "fuel_estimate column not found — run add_fuel_estimate_by_session first"
        )

    df["total_mass_kg"] = MIN_CAR_MASS_KG + df["fuel_estimate"]

    return df


if __name__ == "__main__":

    features = pd.read_parquet(
        "data/processed/lap_features.parquet"
    )

    result = add_total_mass(features)

    print("\nTotal mass estimate:")
    print(
        result[
            [
                "GrandPrix",
                "Driver",
                "lap",
                "fuel_estimate",
                "total_mass_kg"
            ]
        ].head(15)
    )

    print("\nTotal mass statistics:")
    print(result["total_mass_kg"].describe())

    print("\nSanity check — should be between 798 and 908:")
    print("Min:", result["total_mass_kg"].min())
    print("Max:", result["total_mass_kg"].max())
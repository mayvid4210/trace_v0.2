import pandas as pd


INPUT_FILE = "data/processed/physics_data.parquet"
OUTPUT_FILE = "data/processed/physics_data_with_force.parquet"


def add_longitudinal_force(df):

    df = df.copy()

    df["longitudinal_force_n"] = (
        df["total_mass_kg"]
        * df["Acceleration_smoothed"]
    )

    return df


def main():

    print("\n========== LONGITUDINAL FORCE ==========")

    df = pd.read_parquet(
        INPUT_FILE
    )

    print("\nOriginal shape:")
    print(df.shape)

    df = add_longitudinal_force(
        df
    )

    print("\nForce samples:")

    print(
        df[
            [
                "GrandPrix",
                "SessionName",
                "Driver",
                "LapNumber",
                "Speed",
                "Acceleration_smoothed",
                "total_mass_kg",
                "longitudinal_force_n"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nForce statistics:")

    print(
        df["longitudinal_force_n"].describe()
    )

    print("\nMissing force values:")

    print(
        df["longitudinal_force_n"].isna().sum()
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
import pandas as pd


INPUT_FILE = (
    "data/processed/"
    "physics_data_with_force.parquet"
)

OUTPUT_FILE = (
    "data/processed/"
    "physics_data_with_regime.parquet"
)


def classify_regimes(df):

    df = df.copy()

    df["Throttle"] = pd.to_numeric(
        df["Throttle"],
        errors="coerce"
    )

    df["Brake"] = df["Brake"].astype(bool)

    df["regime"] = "UNKNOWN"

    braking = (
        df["Brake"]
    )

    coasting = (
        (df["Throttle"] <= 5)
        &
        (~df["Brake"])
        &
        (df["Acceleration_smoothed"] < 0)
    )

    powered = (
        (df["Throttle"] > 5)
        &
        (~df["Brake"])
    )

    df.loc[
        braking,
        "regime"
    ] = "BRAKING"

    df.loc[
        coasting,
        "regime"
    ] = "COASTING"

    df.loc[
        powered,
        "regime"
    ] = "POWERED"

    return df


def main():

    print("\n========== OPERATING REGIMES ==========")

    df = pd.read_parquet(
        INPUT_FILE
    )

    df = classify_regimes(
        df
    )

    print("\nRegime counts:")

    print(
        df["regime"].value_counts(
            dropna=False
        )
    )

    print("\nExamples:")

    print(
        df[
            [
                "Speed",
                "Throttle",
                "Brake",
                "Acceleration_smoothed",
                "regime"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
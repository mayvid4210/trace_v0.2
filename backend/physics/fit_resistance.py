import numpy as np
import pandas as pd


INPUT_FILE = "data/processed/resistance_data.parquet"
OUTPUT_FILE = "data/processed/resistance_curve.parquet"


def fit_resistance_curve(df):

    df = df.copy()

    df = df.dropna(
        subset=[
            "Speed_ms",
            "resistance_force_n",
            "air_density"
        ]
    )

    df = df[
        (df["Speed_ms"] >= 40.0) &
        (df["resistance_force_n"] > 0)
    ].copy()

    # Convert speed into bins.
    #
    # This gives us a stable average resistance
    # instead of fitting directly to extremely noisy
    # individual telemetry samples.

    df["speed_bin"] = pd.cut(
        df["Speed_ms"],
        bins=np.arange(25, 101, 2.0),
        include_lowest=True
    )

    curve = (
        df.groupby(
            "speed_bin",
            observed=True
        )
        .agg(
            speed_ms=("Speed_ms", "mean"),
            resistance_force_n=(
                "resistance_force_n",
                "median"
            ),
            samples=(
                "resistance_force_n",
                "count"
            )
        )
        .reset_index()
    )

    # Remove bins with very little data.
    curve = curve[
        curve["samples"] >= 20
    ].copy()

    # IMPORTANT:
    # speed_bin is a pandas Interval object.
    # PyArrow cannot reliably write this column.
    # We do not need it in the final model anyway.

    curve = curve[
        [
            "speed_ms",
            "resistance_force_n",
            "samples"
        ]
    ].copy()

    curve = curve.sort_values(
        "speed_ms"
    ).reset_index(drop=True)

    return curve


if __name__ == "__main__":

    print(
        "\n========== FITTING TELEMETRY RESISTANCE CURVE =========="
    )

    data = pd.read_parquet(
        INPUT_FILE
    )

    print("\nInput samples:")
    print(len(data))

    curve = fit_resistance_curve(
        data
    )

    print("\nCurve points:")
    print(len(curve))

    print("\n========== RESISTANCE CURVE ==========")

    print(
        curve.to_string(
            index=False
        )
    )

    curve.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT_FILE)
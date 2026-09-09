
import pandas as pd
import numpy as np


def main():

    path = "data/processed/resistance_data.parquet"

    df = pd.read_parquet(path)

    print("\n========== RESISTANCE ANALYSIS ==========")

    print("\nTotal samples:")
    print(len(df))

    # --------------------------------------------------
    # BASIC FILTERS
    # --------------------------------------------------

    df = df.dropna(
        subset=[
            "Speed_ms",
            "air_density",
            "resistance_force_n",
            "Acceleration_smoothed",
            "total_mass_kg",
            "DRS"
        ]
    ).copy()

    print("\nSamples after NaN removal:")
    print(len(df))

    # --------------------------------------------------
    # CALCULATE NORMALIZED RESISTANCE
    # --------------------------------------------------

    df["rho_v2"] = (
        df["air_density"] *
        df["Speed_ms"] ** 2
    )

    df["force_per_rho_v2"] = (
        df["resistance_force_n"] /
        df["rho_v2"]
    )

    # --------------------------------------------------
    # SPEED BINS
    # --------------------------------------------------

    bins = [
        140,
        160,
        180,
        200,
        220,
        240,
        260,
        280,
        300,
        320,
        340,
        360
    ]

    labels = [
        "140-160",
        "160-180",
        "180-200",
        "200-220",
        "220-240",
        "240-260",
        "260-280",
        "280-300",
        "300-320",
        "320-340",
        "340-360"
    ]

    df["speed_bin"] = pd.cut(
        df["Speed"],
        bins=bins,
        labels=labels
    )

    print("\n========== RESISTANCE BY SPEED ==========")

    speed_stats = (
        df.groupby(
            "speed_bin",
            observed=True
        )[
            [
                "Speed_ms",
                "resistance_force_n",
                "force_per_rho_v2"
            ]
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std"
            ]
        )
    )

    print(speed_stats.to_string())

    # --------------------------------------------------
    # DRS ANALYSIS
    # --------------------------------------------------

    print("\n========== DRS ANALYSIS ==========")

    print("\nDRS counts:")
    print(
        df["DRS"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nNormalized resistance by DRS:")

    drs_stats = (
        df.groupby("DRS")[
            [
                "resistance_force_n",
                "force_per_rho_v2"
            ]
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std"
            ]
        )
    )

    print(drs_stats.to_string())

    # --------------------------------------------------
    # HIGH-SPEED CLEAN SAMPLE
    # --------------------------------------------------

    high = df[
        (df["Speed"] >= 180) &
        (df["Speed"] <= 320)
    ].copy()

    print("\n========== HIGH SPEED SAMPLE ==========")

    print("Samples:")
    print(len(high))

    print("\nNormalized resistance:")
    print(
        high["force_per_rho_v2"].describe()
    )

    # --------------------------------------------------
    # ROBUST OUTLIER FILTER
    # --------------------------------------------------

    q1 = high["force_per_rho_v2"].quantile(0.25)
    q3 = high["force_per_rho_v2"].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    clean = high[
        (high["force_per_rho_v2"] >= lower) &
        (high["force_per_rho_v2"] <= upper)
    ].copy()

    print("\n========== CLEAN HIGH-SPEED SAMPLE ==========")

    print("Samples:")
    print(len(clean))

    print("\nRemoved:")
    print(len(high) - len(clean))

    print("\nNormalized resistance:")
    print(
        clean["force_per_rho_v2"].describe()
    )

    # --------------------------------------------------
    # DRS HIGH-SPEED SAMPLE
    # --------------------------------------------------

    print("\n========== HIGH-SPEED DRS ==========")

    high_drs = (
        clean.groupby("DRS")[
            [
                "Speed_ms",
                "resistance_force_n",
                "force_per_rho_v2"
            ]
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std"
            ]
        )
    )

    print(high_drs.to_string())

    # --------------------------------------------------
    # CORRELATIONS
    # --------------------------------------------------

    print("\n========== CORRELATIONS ==========")

    corr_columns = [
        "Speed_ms",
        "resistance_force_n",
        "air_density",
        "total_mass_kg",
        "Acceleration_smoothed",
        "tyre_age",
        "track_temperature",
        "air_temperature"
    ]

    print(
        df[corr_columns]
        .corr()["resistance_force_n"]
        .sort_values()
        .to_string()
    )

    # --------------------------------------------------
    # SAVE CLEAN DATA
    # --------------------------------------------------

    clean.to_parquet(
        "data/processed/resistance_analysis.parquet",
        index=False
    )

    print("\nSaved:")
    print(
        "data/processed/resistance_analysis.parquet"
    )


if __name__ == "__main__":
    main()

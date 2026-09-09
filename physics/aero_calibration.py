import numpy as np
import pandas as pd

from constants import GRAVITY, FIA_MIN_MASS_KG


def build_regression_table(telemetry_df, laps_df, rho_by_row):
    """
    Build the table needed to fit CdA and Crr from coast-phase telemetry.

    telemetry_df: your telemetry_with_acceleration.parquet, already has
                  is_coast, Acceleration_ms2, Speed_ms
    laps_df:      your lap features table, needs fuel_estimate per Driver+LapNumber
    rho_by_row:   a pandas Series of air density values, same length/index
                  as telemetry_df (we'll build this in the next step)
    """
    df = telemetry_df[telemetry_df["is_coast"]].copy()

    df = df.merge(
        laps_df[["Driver", "LapNumber", "fuel_estimate"]],
        on=["Driver", "LapNumber"],
        how="left"
    )

    df["mass_kg"] = FIA_MIN_MASS_KG + df["fuel_estimate"].fillna(0)
    df["rho"] = rho_by_row

    df = df.dropna(subset=["Acceleration_ms2", "Speed_ms", "mass_kg", "rho"])

    # Only decelerating coast points are physically valid for this fit
    df = df[df["Acceleration_ms2"] < 0]

    df["target"] = df["mass_kg"] * df["Acceleration_ms2"]
    df["reg1"] = 0.5 * df["rho"] * df["Speed_ms"] ** 2
    df["reg2"] = df["mass_kg"] * GRAVITY

    return df


def fit_cda_crr(reg_table):
    """
    Fit m*a = -CdA*(0.5*rho*v^2) - Crr*(m*g) via least squares, no intercept.
    Returns (CdA, Crr).
    """
    X = reg_table[["reg1", "reg2"]].values
    y = reg_table["target"].values

    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)

    CdA = -coeffs[0]
    Crr = -coeffs[1]
    return CdA, Crr
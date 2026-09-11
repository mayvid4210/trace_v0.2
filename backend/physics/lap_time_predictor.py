"""Additive historical/backward and forward lap-time predictor.

This module integrates only validated empirical relationships:

* Race-only dry-compound tyre degradation rates from ``tyre.py``.
* Circuit fuel-mass slopes measured in ``fuel_effect.py``.
* Session-matched sector wetness proxies from ``track_state.py``.

The result is an empirical lap-time predictor, not a first-principles lap
simulator. ``vehicle.py`` is intentionally not converted into lap time here:
there is no validated force-to-lap-time or lap-energy mapping in this project.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


SECTORS = ["sector_1", "sector_2", "sector_3"]
VALIDATED_TYRE_RATES_S_PER_LAP = {
    "SOFT": 0.0627,
    "MEDIUM": 0.0447,
    "HARD": 0.1060,
}
FUEL_EFFECT_S_PER_KG = {
    "Bahrain": 0.0357,
    "Canada": 0.0261,
}

from pathlib import Path

try:
    from config import resolve_path, PROCESSED_DATA_DIR
except ModuleNotFoundError:
    try:
        from backend.config import resolve_path, PROCESSED_DATA_DIR
    except ModuleNotFoundError:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

        def resolve_path(p):
            p = Path(p)
            return p if p.is_absolute() else PROJECT_ROOT / p

FUEL_EFFECT_FILE = PROCESSED_DATA_DIR / "fuel_effect.parquet"
TRACK_STATE_FILE = PROCESSED_DATA_DIR / "track_state_data.parquet"
BACKTEST_OUTPUT_FILE = PROCESSED_DATA_DIR / "lap_time_backtest.parquet"


@dataclass
class CircuitCalibration:
    dry_baseline_lap_time_s: float
    reference_mass_kg: float


class LapTimePredictor:
    """Predict lap time from circuit, tyre, mass, and sector wetness."""

    def __init__(self, calibrations):
        self.calibrations = calibrations

    @classmethod
    def from_artifacts(cls, fuel_df, track_df):
        """Build circuit references from the validated saved artifacts."""
        keys = ["Year", "GrandPrix", "SessionName", "Driver", "LapNumber"]
        wetness_columns = [
            f"{sector}_wetness_proxy" for sector in SECTORS
        ]
        sector_time_columns = [
            f"{sector}_s" for sector in SECTORS
        ]
        track_columns = keys + [
            "TrackStatus", "Rainfall",
            *sector_time_columns,
            *wetness_columns,
        ]
        track = track_df[track_columns].copy()
        track = track.rename(columns={"Rainfall": "track_rainfall"})
        merged = fuel_df.merge(track, on=keys, how="inner")

        merged["dry_baseline_lap_time_s"] = merged.apply(
            lambda row: sum(
                row[f"{sector}_s"] - row[f"{sector}_wetness_proxy"]
                for sector in SECTORS
            )
            if all(
                pd.notna(row[f"{sector}_wetness_proxy"])
                for sector in SECTORS
            )
            else np.nan,
            axis=1,
        )

        calibrations = {}
        for grand_prix, group in merged.groupby("GrandPrix"):
            dry = group[
                (group["SessionName"] == "Race")
                & (group["track_rainfall"] == False)
                & (group["TrackStatus"].astype(str) == "1")
            ].dropna(
                subset=["dry_baseline_lap_time_s", "total_mass_kg"]
            )
            if dry.empty:
                continue

            reference_mass = dry["total_mass_kg"].median()
            for compound, compound_rows in dry.groupby("Compound"):
                tyre_rate = VALIDATED_TYRE_RATES_S_PER_LAP.get(
                    compound,
                    0.0,
                )
                fuel_rate = FUEL_EFFECT_S_PER_KG[grand_prix]
                adjusted_lap_time = (
                    compound_rows["LapTime_s"]
                    - tyre_rate * (compound_rows["TyreLife"] - 1.0)
                    - fuel_rate * (
                        compound_rows["total_mass_kg"] - reference_mass
                    )
                )
                calibrations[(grand_prix, compound)] = CircuitCalibration(
                    dry_baseline_lap_time_s=adjusted_lap_time.median(),
                    reference_mass_kg=reference_mass,
                )

            calibrations[(grand_prix, "__default__")] = CircuitCalibration(
                dry_baseline_lap_time_s=dry[
                    "dry_baseline_lap_time_s"
                ].median(),
                reference_mass_kg=reference_mass,
            )

        return cls(calibrations)

    def predict_lap_time(
        self,
        circuit,
        compound,
        tyre_age,
        fuel_mass_kg,
        sector_wetness=None,
    ):
        """Predict lap time using reference-relative additive terms.

        ``fuel_mass_kg`` is converted to total mass using the project constant
        of 798 kg before applying the validated total-mass slope. This avoids
        treating the 798 kg base as fuel while still using the measured slope.
        Wetness is applied only when supplied; dry laps should pass ``None``.
        """
        calibration = self.calibrations.get(
            (circuit, compound),
            self.calibrations.get((circuit, "__default__")),
        )
        if calibration is None:
            raise KeyError(
                f"No calibration available for circuit {circuit}"
            )

        total_mass_kg = 798.0 + float(fuel_mass_kg)
        tyre_rate = VALIDATED_TYRE_RATES_S_PER_LAP.get(compound, 0.0)
        fuel_rate = FUEL_EFFECT_S_PER_KG[circuit]
        wetness_penalty = 0.0
        if sector_wetness is not None:
            wetness_penalty = sum(
                value for value in sector_wetness
                if value is not None and pd.notna(value) and value > 0
            )

        return (
            calibration.dry_baseline_lap_time_s
            + tyre_rate * (float(tyre_age) - 1.0)
            + fuel_rate * (total_mass_kg - calibration.reference_mass_kg)
            + wetness_penalty
        )


def build_backtest_frame(fuel_df, track_df, predictor):
    """Join observations and produce predictions for historical laps."""
    keys = ["Year", "GrandPrix", "SessionName", "Driver", "LapNumber"]
    wetness_columns = [
        f"{sector}_wetness_proxy" for sector in SECTORS
    ]
    track = track_df[
        keys + ["Rainfall", *wetness_columns]
    ].copy().rename(columns={"Rainfall": "track_rainfall"})
    merged = fuel_df.merge(track, on=keys, how="inner")

    def predict_row(row):
        wetness = None
        if row["track_rainfall"] == True:
            wetness = [row[column] for column in wetness_columns]
        return predictor.predict_lap_time(
            circuit=row["GrandPrix"],
            compound=row["Compound"],
            tyre_age=row["TyreLife"],
            fuel_mass_kg=row["total_mass_kg"] - 798.0,
            sector_wetness=wetness,
        )

    merged["predicted_lap_time_s"] = merged.apply(predict_row, axis=1)
    merged["prediction_error_s"] = (
        merged["LapTime_s"] - merged["predicted_lap_time_s"]
    )
    return merged


def print_backtest_metrics(df):
    """Print overall and grouped MAE/RMSE for the historical backtest."""
    def report(label, subset):
        if subset.empty:
            print(f"{label}: no rows")
            return
        error = subset["prediction_error_s"]
        print(
            f"{label}: n={len(subset)} "
            f"MAE={error.abs().mean():.3f}s "
            f"RMSE={np.sqrt(np.mean(error**2)):.3f}s"
        )

    report("ALL", df)
    for circuit, group in df.groupby("GrandPrix"):
        report(circuit, group)
    for compound, group in df.groupby("Compound"):
        report(compound, group)
    for label, group in df.groupby(df["Rainfall"].eq(True)):
        report("RAIN" if label else "DRY", group)


if __name__ == "__main__":
    fuel = pd.read_parquet(FUEL_EFFECT_FILE)
    track = pd.read_parquet(TRACK_STATE_FILE)
    predictor = LapTimePredictor.from_artifacts(fuel, track)

    print("===== CIRCUIT CALIBRATIONS =====")
    for (circuit, compound), calibration in predictor.calibrations.items():
        print(
            f"{circuit} {compound}: "
            f"baseline={calibration.dry_baseline_lap_time_s:.3f}s "
            f"reference_mass={calibration.reference_mass_kg:.3f}kg"
        )

    print("\n===== HISTORICAL BACKTEST =====")
    backtest = build_backtest_frame(fuel, track, predictor)
    print_backtest_metrics(backtest)
    print("\n===== RACE-ONLY BACKTEST =====")
    print_backtest_metrics(
        backtest[backtest["SessionName"] == "Race"]
    )
    backtest.to_parquet(BACKTEST_OUTPUT_FILE, index=False)
    print(f"\nSaved: {BACKTEST_OUTPUT_FILE}")
